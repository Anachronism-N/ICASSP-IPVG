#!/usr/bin/env python3
"""
E1 scoring and preregistered config selection (docs/43 section 3). Thin wrapper: no metric is re-implemented.

  vbench  : official VBench-I2V code, the same 8 dims / call as scripts/vbench_eval.py (suite wm), with the dev-set
            first frame (eval/e1_devset_20261021/images/<id>.png) as custom_image_path. Needs a GPU; run with
            /dockerdata/venv_vbench/bin/python (rank0 only has that env). Resume-safe: valid dim files are skipped.
            -> eval/e1_devset_20261021/scores/<config>__<dim>_eval_results.json
  gtref   : scripts/gt_ref_metrics.py read/sample/motion/K and the same DreamSim(CPU) computation, against the real
            continuation of each dev window. CPU. -> eval/e1_devset_20261021/gtref/<config>.json
  summarize: per-config means, per-dataset means, paired differences vs default s5_c5 (case bootstrap CI, descriptive
            only), Total* via scripts/summarize_eval.total (camera_motion excluded, dynamic weight 0.5) and the
            preregistered selection rule below. CPU. -> eval/e1_devset_20261021/e1_summary.json (aggregates only)

Usage:
  CUDA_VISIBLE_DEVICES=0 /dockerdata/venv_vbench/bin/python scripts/score_e1.py vbench s5_c5 [s3_c4 ...]
  THREADS=8 /dockerdata/venv_vbench/bin/python scripts/score_e1.py gtref s5_c5 [...]
  /opt/conda/envs/torch-base/bin/python scripts/score_e1.py summarize [--partial]
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
from build_e1_jobs import CFGS, DEFAULT, SHIFTS, config_name  # noqa: E402
from summarize_eval import DIMS, total  # noqa: E402

ROOT = "/srv/wm_teacher"
E1 = f"{ROOT}/eval/e1_devset_20261021"
VBREPO = "/dockerdata/wm_eval/VBench"
WORK_BASE = "/dockerdata/wm_eval/work"
CONFIGS = [config_name(s, c) for s in SHIFTS for c in CFGS] + ["z0_s5_c5", "z2_s5_c5"]  # E1b CFG-Zero* variants
E3_ARMS = ["armA_orig", "armB_fixed", "armC_fixed_anchor"]  # E3 three-arm dev-set eval (same scorer, root eval/e3_devset_eval_20261024)
DEFAULT_CONFIG = config_name(*DEFAULT)
GT_KEYS = ("dreamsim", "dreamsim_last", "motion_ratio")
# Preregistered rule (docs/43 section 3). Deltas are config minus default on dev-set means; 1pp = 0.01.
RULE = {
    "motion_smoothness_min_delta": -0.003,   # motion_smoothness >= default - 0.3pp
    "subject_consistency_min_delta": -0.010,  # subject_consistency >= default - 1.0pp
    "dreamsim_max_delta": 0.005,             # GT DreamSim <= default + 0.005
    "motion_ratio_max_delta": 0.05,          # motion error |log(gen/GT)| <= default + 0.05
}
EPS = 1e-12  # boundary values count as passing
BOOT_B = 2000
BOOT_SEED = 20261021
SENSITIVITY_EXCLUDE = ("egoexo4d",)


def load_cases(root=E1):
    return [json.loads(line) for line in open(f"{root}/cases.jsonl")]


def video_path(config, case, root=E1):
    return f"{root}/videos/{config}/{case['id']}.mp4"


def finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- scoring (GPU / CPU workers)
def stage(config, cases, root=E1):
    """Local copies named <id>-0.mp4 + custom full_info (as vbench_eval.prepare_wm)."""
    work = f"{WORK_BASE}/{os.path.basename(root.rstrip('/'))}/{config}"
    vdir = f"{work}/videos"
    os.makedirs(vdir, exist_ok=True)
    info = []
    for c in cases:
        src, dst = video_path(config, c, root), f"{vdir}/{c['id']}-0.mp4"
        if not os.path.isfile(src) or os.path.getsize(src) == 0:
            raise RuntimeError(f"Missing video {config}/{c['id']}")
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copyfile(src, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        info.append({"prompt_en": c["id"], "dimension": DIMS, "custom_image_path": c["image"]})
    full_info = f"{work}/full_info.json"
    with open(full_info, "w") as f:
        json.dump(info, f)
    return vdir, full_info


def cmd_vbench(configs, root=E1):
    import torch
    from vbench_eval import valid_result
    cases = load_cases(root)
    expected = {c["id"] + "-0.mp4" for c in cases}
    out = f"{root}/scores"
    os.makedirs(out, exist_ok=True)
    os.chdir(VBREPO)
    sys.path.insert(0, VBREPO)
    from vbench2_beta_i2v import VBenchI2V
    for config in configs:
        vdir, full_info = stage(config, cases, root)
        vb = VBenchI2V(torch.device("cuda"), full_info, out)
        for dim in DIMS:
            path = f"{out}/{config}__{dim}_eval_results.json"
            if os.path.exists(path) and valid_result(json.load(open(path))[dim], expected):
                continue
            vb.evaluate(videos_path=vdir, name=f"{config}__{dim}", dimension_list=[dim],
                        resolution="16-9", mode="vbench_standard")
            if not valid_result(json.load(open(path))[dim], expected):
                raise RuntimeError(f"Incomplete/invalid {config}/{dim}")
            print(f"[vbench] {config}/{dim} done", flush=True)
    print("[done]", flush=True)


def cmd_gtref(configs, root=E1):
    import numpy as np
    import torch
    import gt_ref_metrics as g
    from dreamsim import dreamsim
    cases = load_cases(root)
    os.makedirs(f"{root}/gtref", exist_ok=True)
    os.chdir(VBREPO)  # dreamsim weights live in ./models (as gt_ref_metrics.main)
    model, preprocess = dreamsim(pretrained=True, device="cpu", cache_dir="./models")
    gt_cache = {}
    for config in configs:
        out = f"{root}/gtref/{config}.json"
        res = json.load(open(out)) if os.path.exists(out) else {}
        for c in cases:
            vp = video_path(config, c, root)
            if c["id"] in res:
                continue
            if c["id"] not in gt_cache:
                gt_cache[c["id"]] = g.read(c["gt"])
            gf, gfps = gt_cache[c["id"]]
            vf, vfps = g.read(vp)
            dur = min(len(gf) / gfps, len(vf) / vfps) - 1e-3
            a = [x.resize((256, 144)) for x in g.sample(gf, gfps, dur)]
            b = [x.resize((256, 144)) for x in g.sample(vf, vfps, dur)]
            with torch.no_grad():
                d = model(torch.cat([preprocess(x) for x in a]), torch.cat([preprocess(x) for x in b])).flatten().tolist()
            res[c["id"]] = {"dataset": c["dataset"], "dreamsim": float(np.mean(d)), "dreamsim_last": d[-1],
                            "motion_ratio": abs(math.log(g.motion(b) / g.motion(a))), "dur": dur}
            with open(out + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(out + ".tmp", out)
        print(f"[gtref] {config}: {len(res)}/{len(cases)}", flush=True)


# ---------------------------------------------------------------- aggregation and selection (CPU)
def per_video_value(dim, value):
    """Per-video VBench values on the aggregate scale (VBench divides imaging_quality by 100; dynamic is bool)."""
    return float(value) / 100 if dim == "imaging_quality" else float(value)


def load_config_rows(config, cases, root=E1):
    """{case_id: {dim..., dreamsim, dreamsim_last, motion_ratio}}; raises unless coverage is exact and finite."""
    ids = {c["id"] for c in cases}
    rows = {i: {} for i in ids}
    for dim in DIMS:
        path = f"{root}/scores/{config}__{dim}_eval_results.json"
        if not os.path.isfile(path):
            raise RuntimeError(f"{config}/{dim}: missing score")
        result = json.load(open(path))[dim]
        got = {}
        for v in result[1]:
            got[os.path.basename(v["video_path"])[:-len("-0.mp4")]] = v["video_results"]
        if set(got) != ids or len(result[1]) != len(ids) or not all(finite(x) for x in got.values()):
            raise RuntimeError(f"{config}/{dim}: incomplete/non-finite coverage")
        for i, x in got.items():
            rows[i][dim] = per_video_value(dim, x)
        mean = sum(rows[i][dim] for i in ids) / len(ids)
        if not math.isclose(mean, float(result[0]), rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"{config}/{dim}: per-video mean != VBench aggregate")
    path = f"{root}/gtref/{config}.json"
    ref = json.load(open(path)) if os.path.isfile(path) else {}
    if set(ref) != ids or not all(all(finite(v.get(k)) for k in GT_KEYS) for v in ref.values()):
        raise RuntimeError(f"{config}: incomplete/non-finite GT-reference metrics")
    for i in ids:
        rows[i].update({k: float(ref[i][k]) for k in GT_KEYS})
        rows[i]["total"] = total({d: rows[i][d] for d in DIMS})[0]
    return rows


def mean_stats(rows):
    ids = sorted(rows)
    m = {k: sum(rows[i][k] for i in ids) / len(ids) for k in DIMS + list(GT_KEYS)}
    m["total"], m["i2v"], m["quality"] = total({d: m[d] for d in DIMS})
    m["n"] = len(ids)
    return m


def paired(rows, base_rows, keys, b=BOOT_B, seed=BOOT_SEED):
    ids = sorted(rows)
    rng = random.Random(seed)
    out = {}
    for k in keys:
        d = [rows[i][k] - base_rows[i][k] for i in ids]
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(b))
        out[k] = {"mean": sum(d) / len(d), "ci95": [boots[int(0.025 * b)], boots[int(0.975 * b) - 1]],
                  "better_cases": sum((x < 0) if k in GT_KEYS else (x > 0) for x in d), "n": len(d)}
    return out


def guard_checks(s, d, rule=RULE):
    return {
        "motion_smoothness": s["motion_smoothness"] - d["motion_smoothness"] >= rule["motion_smoothness_min_delta"] - EPS,
        "subject_consistency": s["subject_consistency"] - d["subject_consistency"] >= rule["subject_consistency_min_delta"] - EPS,
        "dreamsim": s["dreamsim"] - d["dreamsim"] <= rule["dreamsim_max_delta"] + EPS,
        "motion_ratio": s["motion_ratio"] - d["motion_ratio"] <= rule["motion_ratio_max_delta"] + EPS,
    }


def parse_config(name):
    """Grid configs are s<shift>_c<cfg>; E1b CFG-Zero* variants report their base grid point."""
    if name.startswith("z"):
        return parse_config(name.split("_", 1)[1])
    s, c = name[1:].split("_c")
    return int(s), int(c)


def select_config(stats, default=DEFAULT_CONFIG, rule=RULE):
    """stats: {config: {'total', 'motion_smoothness', 'subject_consistency', 'dreamsim', 'motion_ratio'}}.
    Highest Total* among configs passing all guards (default always passes); ties -> default, then closest grid
    point to default, then lower shift, lower cfg. Falls back to default if nothing passes."""
    if default not in stats:
        raise ValueError("Default config missing")
    d = stats[default]
    checks = {name: guard_checks(s, d, rule) for name, s in stats.items()}
    passing = [name for name in stats if all(checks[name].values())]
    ds, dc = parse_config(default)

    def key(name):
        sh, cf = parse_config(name)
        return (-round(stats[name]["total"], 12), name != default, abs(sh - ds) + abs(cf - dc), sh, cf)

    chosen = min(passing, key=key) if passing else default
    return {"selected": chosen, "passing": sorted(passing, key=key), "checks": checks, "rule": rule,
            "default": default, "fallback_to_default": not passing}


def cmd_summarize(partial, root=E1):
    cases = load_cases(root)
    by_ds = defaultdict(list)
    for c in cases:
        by_ds[c["dataset"]].append(c["id"])
    rows, missing = {}, {}
    for config in CONFIGS:
        try:
            rows[config] = load_config_rows(config, cases, root)
        except RuntimeError as e:
            missing[config] = str(e)
    if missing and not partial:
        raise SystemExit("Incomplete E1 scores:\n" + "\n".join(f"{k}: {v}" for k, v in missing.items()))
    stats = {k: mean_stats(v) for k, v in rows.items()}
    keys = DIMS + list(GT_KEYS) + ["total"]
    summary = {"protocol": "E1 docs/43 section 3", "n_cases": len(cases), "configs": CONFIGS,
               "default": DEFAULT_CONFIG, "missing": missing, "means": stats,
               "per_dataset": {cfg: {ds: mean_stats({i: r[i] for i in ids}) for ds, ids in sorted(by_ds.items())}
                               for cfg, r in rows.items()}}
    if DEFAULT_CONFIG in rows:
        summary["paired_vs_default"] = {cfg: paired(r, rows[DEFAULT_CONFIG], keys)
                                        for cfg, r in rows.items() if cfg != DEFAULT_CONFIG}
    summary["selection"] = select_config(stats) if not missing else {"selected": None, "reason": "incomplete"}
    # sensitivity (lead 2026-10-21): egoexo4d parents are unresolved -> same analysis without them; descriptive,
    # the preregistered selection above stays the primary result
    summary["sensitivity_excluding"] = {}
    for ds in SENSITIVITY_EXCLUDE:
        keep = {c["id"] for c in cases if c["dataset"] != ds}
        sub = {cfg: {i: r[i] for i in keep} for cfg, r in rows.items()}
        sub_stats = {k: mean_stats(v) for k, v in sub.items()}
        entry = {"n_cases": len(keep), "means": sub_stats}
        if DEFAULT_CONFIG in sub:
            entry["paired_vs_default"] = {cfg: paired(r, sub[DEFAULT_CONFIG], keys)
                                          for cfg, r in sub.items() if cfg != DEFAULT_CONFIG}
        entry["selection"] = select_config(sub_stats) if not missing else {"selected": None, "reason": "incomplete"}
        summary["sensitivity_excluding"][ds] = entry
    out = f"{root}/e1_summary{'_partial' if missing else ''}.json"
    with open(out + ".tmp", "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(out + ".tmp", out)
    print("| config | Total* | dynamic | aesthetic | imaging | motion_smooth | subject | DreamSim | motion_err |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cfg, s in stats.items():
        print(f"| {cfg} | {100 * s['total']:.2f} | {100 * s['dynamic_degree']:.2f} | {100 * s['aesthetic_quality']:.2f} | "
              f"{100 * s['imaging_quality']:.2f} | {100 * s['motion_smoothness']:.2f} | "
              f"{100 * s['subject_consistency']:.2f} | {s['dreamsim']:.4f} | {s['motion_ratio']:.3f} |")
    print(f"selected: {summary['selection'].get('selected')} -> {out}")
    for ds, e in summary["sensitivity_excluding"].items():
        print(f"sensitivity without {ds} (n={e['n_cases']}): selected {e['selection'].get('selected')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("vbench", "gtref"):
        p = sub.add_parser(name)
        p.add_argument("configs", nargs="+", choices=CONFIGS + E3_ARMS)
        p.add_argument("--root", default=E1, help="dev-set dir with cases.jsonl and videos/<config>/ (smoke only)")
    p = sub.add_parser("summarize")
    p.add_argument("--partial", action="store_true", help="report complete configs only; no selection")
    p.add_argument("--root", default=E1, help="dev-set dir (tests/smoke only)")
    a = ap.parse_args()
    if a.cmd == "vbench":
        cmd_vbench(a.configs, a.root)
    elif a.cmd == "gtref":
        cmd_gtref(a.configs, a.root)
    else:
        cmd_summarize(a.partial, a.root)


if __name__ == "__main__":
    main()
