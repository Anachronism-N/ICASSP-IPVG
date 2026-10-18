#!/usr/bin/env python3
"""
Summarize VBench-I2V scores (official leaderboard normalization and weights, camera_motion excluded)
for the public suite and for WM-Bench, plus the official leaderboard rows recomputed the same way.
Writes eval/summary.json and eval/summary.md.
"""
import argparse
import csv
import glob
import math
import json
import os
from collections import defaultdict

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval"
DIMS = ["i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]
LB_COL = {"i2v_subject": "Video-Image Subject Consistency", "i2v_background": "Video-Image Background Consistency",
          "subject_consistency": "Subject Consistency", "background_consistency": "Background Consistency",
          "motion_smoothness": "Motion Smoothness", "dynamic_degree": "Dynamic Degree",
          "aesthetic_quality": "Aesthetic Quality", "imaging_quality": "Imaging Quality"}
NORM = {"i2v_subject": (0.1462, 1.0), "i2v_background": (0.2615, 1.0), "subject_consistency": (0.1462, 1.0),
        "background_consistency": (0.2615, 1.0), "motion_smoothness": (0.7060, 0.9975),
        "dynamic_degree": (0.0, 1.0), "aesthetic_quality": (0.0, 1.0), "imaging_quality": (0.0, 1.0)}
WEIGHT = {d: 1.0 for d in DIMS}
WEIGHT["dynamic_degree"] = 0.5
I2V = ["i2v_subject", "i2v_background"]
QUAL = [d for d in DIMS if d not in I2V]
LABEL = {"gt": "真实视频 (GT)", "base": "Wan2.2-TI2V-5B (ours, base)", "round1": "Round1 LoRA",
         "round2": "Round2 LoRA", "round3": "Round3 LoRA", "round4": "Round4 LoRA (aes-filtered)", "round5": "Round5 LoRA (motion+aes, short-cap mix)", "full1": "Full-FT 5B (FT1)", "ft2": "Full-FT 5B (FT2, base init + general data)", "m_full1_a03": "0.7·base + 0.3·FT1 (weight merge)", "m_full1_a05": "0.5·base + 0.5·FT1 (weight merge)", "m_full1_a07": "0.3·base + 0.7·FT1 (weight merge)", "cogvideox": "CogVideoX-5B-I2V",
         "hunyuan": "HunyuanVideo-I2V", "wan21": "Wan2.1-I2V-14B-720P"}


def scores(d):
    """returns {model: {dim: (mean, [(video_path, score), ...])}}"""
    out = defaultdict(dict)
    for f in glob.glob(f"{d}/scores/*__*_eval_results.json"):
        model, dim = os.path.basename(f)[:-len("_eval_results.json")].split("__", 1)
        r = json.load(open(f))[dim]
        per = [(x["video_path"], x["video_results"]) for x in r[1]] if isinstance(r[1], list) else []
        out[model][dim] = (r[0], per)
    return out


def total(dimvals):
    def norm(d):
        lo, hi = NORM[d]
        return (dimvals[d] - lo) / (hi - lo) * WEIGHT[d]
    if not all(d in dimvals for d in DIMS):
        return None, None, None
    q = sum(norm(d) for d in QUAL) / sum(WEIGHT[d] for d in QUAL)
    i = sum(norm(d) for d in I2V) / sum(WEIGHT[d] for d in I2V)
    return (q + i) / 2, i, q


def table(rows, title):
    hdr = ["模型", "Total*", "I2V*", "Quality"] + DIMS
    md = [f"### {title}", "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for name, dv in rows:
        t, i, q = total(dv)
        f = lambda v: "-" if v is None else f"{100 * v:.2f}"
        md.append("| " + " | ".join([name, f(t), f(i), f(q)] + [f(dv.get(d)) for d in DIMS]) + " |")
    return md


def label(m):
    """'round3_po' -> 'Round3 LoRA + prompt-optimized'"""
    if m.endswith("_po"):
        return LABEL.get(m[:-3], m[:-3]) + " + prompt-optimized"
    return LABEL.get(m, m)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_jobs(paths):
    groups = defaultdict(dict)
    errors = []
    for path in paths:
        for line in open(path):
            job = json.loads(line)
            parts = os.path.relpath(job["out"], EV).split(os.sep)
            suite = parts[0]
            groups[(suite, job["model"])][job["out"]] = job
    for (suite, model), jobs in groups.items():
        missing = [p for p in jobs if not os.path.isfile(p) or os.path.getsize(p) == 0]
        if missing:
            errors.append(f"{suite}/{model}: {len(missing)}/{len(jobs)} missing/empty outputs")
        if suite in ("wm_bench", "vbench_i2v"):
            cases = [json.loads(line) for line in open(f"{EV}/{suite}/cases.jsonl")]
            for dim in DIMS:
                expected = {c["id"] + "-0.mp4" for c in cases if dim in c.get("dims", DIMS)}
                score_path = f"{EV}/{suite}/scores/{model}__{dim}_eval_results.json"
                if not os.path.isfile(score_path):
                    errors.append(f"{suite}/{model}/{dim}: missing score")
                    continue
                result = json.load(open(score_path))[dim]
                actual = [os.path.basename(v["video_path"]) for v in result[1]]
                if (len(actual) != len(expected) or set(actual) != expected or not finite(result[0])
                        or not all(finite(v.get("video_results")) for v in result[1])):
                    errors.append(f"{suite}/{model}/{dim}: incomplete/non-finite score coverage")
            if suite == "wm_bench":
                p = f"{EV}/{suite}/gtref_{model}.json"
                reference = json.load(open(p)) if os.path.isfile(p) else {}
                if (set(reference) != {c["id"] for c in cases}
                        or not all(all(finite(v.get(k)) for k in ("dreamsim", "dreamsim_last", "motion_ratio", "dur"))
                                   for v in reference.values())):
                    errors.append(f"{suite}/{model}: incomplete/non-finite GT-reference metrics")
        elif suite == "wmb":
            p = f"{EV}/wmb/results/{model}.json"
            if not os.path.isfile(p):
                errors.append(f"wmb/{model}: missing judge result")
                continue
            result = json.load(open(p))
            expected = {os.path.splitext(os.path.basename(v))[0] for v in jobs}
            if len(expected) != 350 or set(result["preds"]) != expected:
                errors.append(f"wmb/{model}: incomplete 350-case judge coverage")
            for name, count in (("instruction", 350), ("physical_laws", 1750), ("common_sense", 700)):
                values = result["accs"][name]
                if len(values) != count or not all(math.isfinite(float(v)) for v in values):
                    errors.append(f"wmb/{model}/{name}: incomplete/non-finite aggregate")
        elif suite == "physiq":
            d = f"{EV}/physiq/results/physics-IQ-benchmark/results/{model}-op-run_01"
            if not os.path.isfile(d + "_metrics.json") or not os.path.isfile(d + ".csv"):
                errors.append(f"physiq/{model}: missing score or per-scenario table")
                continue
            metrics = json.load(open(d + "_metrics.json"))
            rows = list(csv.DictReader(open(d + ".csv")))
            if len(jobs) != 198 or len(rows) != 66 or len({r["scenario"] for r in rows}) != 66:
                errors.append(f"physiq/{model}: incomplete 66-scenario/198-view coverage")
            if not math.isfinite(float(metrics["final_score_orig"])):
                errors.append(f"physiq/{model}: non-finite score")
        else:
            errors.append(f"Unknown benchmark suite: {suite}")
    if not groups:
        errors.append("No jobs were provided")
    if errors:
        raise RuntimeError("Evaluation incomplete:\n" + "\n".join(errors))
    print(f"Validated {sum(map(len, groups.values()))} outputs and {len(groups)} benchmark/model groups")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-jobs", action="append", default=[],
                        help="Require complete outputs and benchmark scores for this job manifest before updating summary")
    args = parser.parse_args()
    if args.validate_jobs:
        validate_jobs(args.validate_jobs)
    res = {}
    md = ["# 评测汇总（VBench-I2V 官方维度与归一化；*不含 camera_motion）", "",
          "> 本地主要为单 seed 评测；细小分差不代表可靠提升。`prompt-optimized` 是独立输入增强协议，",
          "> 当前扩写抽查发现物体属性和物理终局偏移，不保证与原始 prompt 语义等价。",
          "> 只在同一 raw 或同一扩写协议内比较模型；不能把 raw→po 涨分归因于模型训练收益。",
          "> Total* 为不含 camera_motion 的8维重算值，非榜单原始 Total。",
          "> 例如 DreamX 快照原始 Total 为90.49%，本表同8维重算为90.79%。", ""]
    # public suite
    vb = scores(f"{EV}/vbench_i2v")
    rows = [(label(m) + "（本地复现）", {d: v[0] for d, v in vb[m].items()}) for m in sorted(vb)]
    lb = list(csv.DictReader(open(f"{EV}/vbench_i2v_leaderboard_20261011.csv")))
    keep = ["DreamX-World-1.0", "Wan2.2-TI2V-5B", "Wan2.2-I2V-A14B", "Wan2.1-I2V-14B", "HunyuanVideo-I2V", "CogVideoX-5b-I2V",
            "Step-Video-TI2V", "Magi-1", "LTX-2-I2V", "Gen-4-I2V", "DynamiCrafter-1024"]
    for r in lb:
        if any(k in r["Model Name (clickable)"] for k in keep):
            dv = {d: float(r[LB_COL[d]].strip("%")) / 100 for d in DIMS if r[LB_COL[d]] not in ("nan", "")}
            rows.append((r["Model Name (clickable)"] + "（官方榜单）", dv))
    md += table(rows, "公开基准 VBench-I2V（355 对图文，16:9，本地 1 seed / 榜单 5 seed）") + [""]
    res["vbench_i2v"] = {n: dv for n, dv in rows}
    # in-domain
    wm = scores(f"{EV}/wm_bench")
    order = [m for m in ["gt", "base", "round1", "round2", "round3", "round4", "round5", "full1", "ft2", "m_full1_a03", "m_full1_a05", "m_full1_a07", "cogvideox", "wan21", "hunyuan"] if m in wm]
    rows = [(LABEL.get(m, m), {d: v[0] for d, v in wm[m].items()}) for m in order]
    md += table(rows, "领域内基准 WM-Bench（108 条留出 5s 片段，GT 首帧 + caption）") + [""]
    res["wm_bench"] = {n: dv for n, dv in rows}
    # GT-referenced metrics (gt_ref_metrics.py): closeness to the real continuation
    g_rows = []
    for m in ["base", "round1", "round2", "round3", "round4", "round5", "full1", "ft2", "m_full1_a03", "m_full1_a05", "m_full1_a07", "cogvideox", "wan21", "hunyuan"]:
        p = f"{EV}/wm_bench/gtref_{m}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if not r:
            continue
        mean = lambda k: sum(v[k] for v in r.values()) / len(r)
        g_rows.append((m, len(r), mean("dreamsim"), mean("dreamsim_last"), mean("motion_ratio")))
    if g_rows:
        md += ["### WM-Bench 与真实后续的接近程度（越低越好）", "",
               "| 模型 | n | DreamSim 距离（均值） | DreamSim（末帧） | 运动量误差 \\|log(gen/GT)\\| |",
               "|---|---|---|---|---|"]
        md += [f"| {LABEL.get(m, m)} | {n} | {a:.4f} | {b:.4f} | {c:.3f} |" for m, n, a, b, c in g_rows]
        md.append("")
        res["wm_gtref"] = {m: {"n": n, "dreamsim": a, "dreamsim_last": b, "motion_err": c} for m, n, a, b, c in g_rows}
    # extra public world-model benchmarks: Physics-IQ (original) and WorldModelBench
    pq = {}
    for f in glob.glob(f"{EV}/physiq/results/physics-IQ-benchmark/results/*-op-run_01_metrics.json"):
        m = os.path.basename(f)[: -len("-op-run_01_metrics.json")]
        metrics = json.load(open(f))
        pq[m] = round(100 * round(float(metrics["final_score_orig"]), 4), 2)
    wb = {}
    for f in glob.glob(f"{EV}/wmb/results/*.json"):
        a = json.load(open(f))["accs"]
        mean = lambda v: sum(v) / len(v)
        i, p, c = mean(a["instruction"]), mean(a["physical_laws"]), mean(a["common_sense"])
        wb[os.path.basename(f)[:-5]] = (i, 5 * p, 2 * c, i + 5 * p + 2 * c)  # official: instr(0-3) + 5 physics + 2 common-sense checks
    if pq or wb:
        ms = sorted(set(pq) | set(wb), key=lambda m: (m.endswith("_po"), m))
        md += ["### 公开世界模型基准：Physics-IQ（Original，1 seed）与 WorldModelBench（350 条，满分≈10）", "",
               "| 模型 | Physics-IQ ↑ | WMB 总分 ↑ | 指令遵循 /3 | 物理规律 /5 | 常识 /2 |", "|---|---|---|---|---|---|"]
        for m in ms:
            w = wb.get(m)
            md.append(f"| {label(m)} | {pq[m]:.1f} | " if m in pq else f"| {label(m)} | - | ")
            md[-1] += (f"{w[3]:.2f} | {w[0]:.2f} | {w[1]:.2f} | {w[2]:.2f} |" if w else "- | - | - | - |")
        md += ["", "评测口径限制：prompt-optimized 为本地扩写条件下的探索结果；抽查发现部分扩写增加未给定的物理结局或改变对象属性，因此不等同于原始官方协议。不要用 po 对 raw 的涨幅宣称模型物理能力提升。",
               "参考（官方榜单）：Physics-IQ Original — Wan2.2 38.3，Sora2 42.3；WorldModelBench — KLING 9.10，CogVideoX-I2V 7.08", ""]
        res["physiq"] = pq
        res["wmb"] = {m: dict(zip(["instruction", "physical_laws", "common_sense", "total"], v)) for m, v in wb.items()}
    # per-dataset dynamic degree / i2v_subject on WM-Bench
    cases = {json.loads(l)["id"]: json.loads(l)["dataset"] for l in open(f"{EV}/wm_bench/cases.jsonl")}
    for dim in ["dynamic_degree", "i2v_subject", "imaging_quality"]:
        per_ds = defaultdict(dict)
        for m in order:
            if dim not in wm[m]:
                continue
            acc = defaultdict(list)
            for vp, s in wm[m][dim][1]:
                cid = os.path.basename(vp).rsplit("-", 1)[0]
                value = float(s) / 100 if dim == "imaging_quality" else float(s)
                acc[cases.get(cid, "?")].append(value)
            for ds, v in acc.items():
                per_ds[ds][m] = sum(v) / len(v)
        if per_ds:
            md += [f"#### WM-Bench 分数据集：{dim}", "", "| 数据集 | " + " | ".join(order) + " |", "|" + "---|" * (len(order) + 1)]
            for ds in sorted(per_ds):
                md.append(f"| {ds} | " + " | ".join(f"{100 * per_ds[ds][m]:.1f}" if m in per_ds[ds] else "-" for m in order) + " |")
            md.append("")
    json.dump(res, open(f"{EV}/summary.json", "w"), indent=1, ensure_ascii=False)
    open(f"{EV}/summary.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
