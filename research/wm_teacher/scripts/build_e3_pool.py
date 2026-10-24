#!/usr/bin/env python3
"""
E3 training-pool prefilter (docs/43 section 5), CPU only.

Pool = teacher namespace candidates (32346 windows, video,prompt). Removed by parent source (rule and code imported
from prepare_e1_devset, not re-implemented): the E1 dev set (cases.jsonl parent keys) plus every E1 exclusion source
(WM108, fixed val120, P1, C1, M1, training smoke, 2026-10-17 inference dev) recomputed with load_exclusions().
Then the prefilter rules below (constants) are applied to existing signals; a row whose required signal is missing is
counted under "missing:<signal>" and is NOT passed. Per parent-key cap and matrix share cap are applied last, in a
deterministic priority order (aesthetic desc, then video path).

Signals: final_5s_landscape.jsonl (diff_median, scene_cut, num_frames), round4 aes_scores_*.jsonl (aes),
ft2/cam_scores.jsonl (cam_shift), eval/diag/cuts_ft2_train_full.jsonl (cuts, flicker), OpenVidHD.csv
(aesthetic score, motion score; matched by file name) for openvidgen. Optional --cam-supplement computes the same
cam_shift (scripts/score_camera.score, identical algorithm) for rows that only lack cam_shift, into a NEW jsonl.

Only new files: training_metadata/e3_pool_20261021/{prefilter.csv, rejected.csv, summary.json[, cam_supplement.jsonl]}
Usage: /opt/conda/envs/torch-base/bin/python scripts/build_e3_pool.py [--cam-supplement --procs 16] [--dry-run]
"""
import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = "/srv/wm_teacher"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dev = _load("prepare_e1_devset")  # parent_keys / easycom_index / load_exclusions / dataset_of

CANDIDATES = dev.CANDIDATES
E1_CASES = f"{ROOT}/eval/e1_devset_20261021/cases.jsonl"
E1_SUMMARY = f"{ROOT}/eval/e1_devset_20261021/devset_summary.json"
FINAL_5S = f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl"
AES_GLOB = f"{ROOT}/training_metadata/round4/aes_scores_*.jsonl"
CAM_SCORES = f"{ROOT}/training_metadata/ft2/cam_scores.jsonl"
CUTS = f"{ROOT}/eval/diag/cuts_ft2_train_full.jsonl"
OPENVID_CSV = "/dockerdata/wm_eval/openvid/data/train/OpenVidHD.csv"
OUT_DIR = f"{ROOT}/training_metadata/e3_pool_20261021"

# ---- prefilter rules (pre-registered constants) ----
EXCLUDED_DATASETS = frozenset({"mira"})
OPENVID_DATASETS = frozenset({"openvidgen"})  # scored with OpenVidHD.csv aesthetic/motion
N_FRAMES = 121
CUTS_MAX = 0
FLICKER_SOURCE_Q = 0.80        # flicker <= q80 of the same source (computed over that source's full candidate pool)
CAM_SHIFT_MAX = 8.0            # px per 4 frames (score_camera.py)
AES_MIN_INDOMAIN = 5.8
DIFF_MEDIAN_MIN = 1.5
AES_MIN_OPENVID = 6.3
OPENVID_MOTION_SOURCE_Q = 0.40  # OpenVid motion score >= q40 of the openvidgen candidate pool
PER_PARENT_MAX = 2
MATRIX_MAX_SHARE = 0.10
RULES_INDOMAIN = ("n_frames", "scene_cut", "cuts", "flicker", "cam_shift", "aes", "diff_median")
RULES_OPENVID = ("n_frames", "scene_cut", "cuts", "flicker", "cam_shift", "ov_aes", "ov_motion")
SIGNAL_COLUMNS = ["n_frames", "scene_cut", "diff_median", "aes", "cam_shift", "cam_source", "cuts", "flicker",
                  "flicker_src_q80", "ov_aes", "ov_motion", "ov_motion_src_q40"]


def quantile(xs, q):
    """Linear-interpolated quantile (numpy default method)."""
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def openvid_name(rel):
    stem = re.sub(r"_w\d+$", "", os.path.splitext(os.path.basename(rel))[0])
    return re.sub(r"^OpenVidHD_part_\d+\.zip_", "", stem) + ".mp4"


def parent_key_str(keys):
    return "|".join(sorted(keys))


def check_rule(rule, s, src_q):
    """-> 'pass' | 'fail' | 'missing'. s: signal dict of one row; src_q: per-source thresholds."""
    v = s.get(rule)
    if rule == "flicker":
        thr = src_q.get("flicker_q80")
        if v is None or thr is None:
            return "missing"
        return "pass" if v <= thr else "fail"
    if rule == "ov_motion":
        thr = src_q.get("ov_motion_q40")
        if v is None or thr is None:
            return "missing"
        return "pass" if v >= thr else "fail"
    if v is None:
        return "missing"
    ok = {"n_frames": lambda x: x == N_FRAMES,
          "scene_cut": lambda x: x is False,
          "cuts": lambda x: x <= CUTS_MAX,
          "cam_shift": lambda x: x <= CAM_SHIFT_MAX,
          "aes": lambda x: x >= AES_MIN_INDOMAIN,
          "diff_median": lambda x: x >= DIFF_MEDIAN_MIN,
          "ov_aes": lambda x: x >= AES_MIN_OPENVID}[rule](v)
    return "pass" if ok else "fail"


def evaluate(dataset, s, src_q):
    """All rule outcomes for one row: (passed, {rule: outcome}, first_reason or None).
    Every rule is evaluated (no short circuit) so per-rule counts are independent; first_reason follows rule order."""
    if dataset in EXCLUDED_DATASETS:
        return False, {}, "excluded_dataset"
    rules = RULES_OPENVID if dataset in OPENVID_DATASETS else RULES_INDOMAIN
    out = {r: check_rule(r, s, src_q) for r in rules}
    first = next((f"{o}:{r}" for r, o in out.items() if o != "pass"), None)
    return first is None, out, first


def priority(row):
    a = row.get("ov_aes") if row["dataset"] in OPENVID_DATASETS else row.get("aes")
    return (-(a if a is not None else -1e9), row["video"])


def apply_caps(rows, per_parent=PER_PARENT_MAX, matrix_share=MATRIX_MAX_SHARE):
    """rows need 'video','dataset','parent_keys' (set) and aes/ov_aes. Greedy in priority order:
    a row is kept iff each of its parent keys has < per_parent kept rows (keys of multi-key rows all count).
    Then matrix is trimmed (lowest priority first) so matrix <= matrix_share of the final total.
    Returns (kept, {'parent_cap': n, 'matrix_cap': n})."""
    used = Counter()
    kept, n_parent = [], 0
    for r in sorted(rows, key=priority):
        if any(used[k] >= per_parent for k in r["parent_keys"]):
            n_parent += 1
            continue
        for k in r["parent_keys"]:
            used[k] += 1
        kept.append(r)
    matrix = [r for r in kept if r["dataset"] == "matrix"]
    other = len(kept) - len(matrix)
    if matrix_share >= 1:
        return kept, {"parent_cap": n_parent, "matrix_cap": 0}
    max_matrix = math.floor(matrix_share * other / (1 - matrix_share) + 1e-9)
    drop = {r["video"] for r in matrix[max_matrix:]}  # matrix already in priority order
    kept = [r for r in kept if r["video"] not in drop]
    return kept, {"parent_cap": n_parent, "matrix_cap": len(drop)}


# ---------------- signal loaders ----------------
def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_final(want, path=FINAL_5S):
    """{video: {diff_median, scene_cut, n_frames}} for wanted videos; string-slices the path before json parsing."""
    out, key = {}, '"video_path": "'
    with open(path) as f:
        for line in f:
            i = line.find(key)
            if i < 0:
                continue
            j = line.find('"', i + len(key))
            v = line[i + len(key):j]
            if v not in want:
                continue
            d = json.loads(line)
            out[v] = {"diff_median": d.get("diff_median"), "scene_cut": d.get("scene_cut"),
                      "n_frames": d.get("num_frames")}
    return out


def load_aes(want, pattern=AES_GLOB):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for d in load_jsonl(f):
            if d.get("video") in want and d.get("aes") is not None:
                out[d["video"]] = float(d["aes"])
    return out


def load_cam(want, paths):
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for d in load_jsonl(p):
            if d.get("video") in want and d.get("cam_shift") is not None:
                out.setdefault(d["video"], (float(d["cam_shift"]), os.path.basename(p)))
    return out


def load_cuts(want, path=CUTS):
    out = {}
    for d in load_jsonl(path):
        p = d.get("path", "")
        rel = os.path.relpath(p, ROOT) if p.startswith("/") else p
        if rel in want:
            out[rel] = {"cuts": d.get("cuts"), "flicker": d.get("flicker")}
    return out


def load_openvid(names, path=OPENVID_CSV):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["video"] in names:
                try:
                    out[r["video"]] = (float(r["aesthetic score"]), float(r["motion score"]))
                except (TypeError, ValueError):
                    pass
    return out


def frame_count(rel):
    """Extra sources are absent from final_5s_landscape.jsonl; read the real frame count via ffprobe."""
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", f"{ROOT}/{rel}"],
                         capture_output=True, text=True, timeout=60)
    return int(out.stdout.strip()) if out.stdout.strip().isdigit() else None


def build_signals(rows, final, aes, cam, cuts, ov, frame_cache_path=None):
    import subprocess
    fc = {}
    if frame_cache_path and os.path.exists(frame_cache_path):
        fc = {d["video"]: d for d in load_jsonl(frame_cache_path)}
    # Parallel ffprobe for rows missing n_frames (extra sources); cached rows are skipped.
    todo = [r["video"] for r in rows if final.get(r["video"], {}).get("n_frames") is None
            and r["video"] not in fc]
    if todo:
        from multiprocessing import Pool
        print(f"[frames] ffprobe {len(todo)} extra-source clips", flush=True)
        with Pool(min(48, os.cpu_count() or 8)) as p:
            counts = p.map(frame_count, todo, chunksize=16)
        for v, n_ in zip(todo, counts):
            fc[v] = {"video": v, "n_frames": n_}
    sig = {}
    for r in rows:
        v = r["video"]
        f, c = final.get(v, {}), cuts.get(v, {})
        o = ov.get(openvid_name(v)) if r["dataset"] in OPENVID_DATASETS else None
        cs = cam.get(v)
        n = f.get("n_frames")
        sc = f.get("scene_cut")
        if n is None and v in fc:
            n = fc[v]["n_frames"]
        if sc is None:
            # scene_cut is unset for extra sources; cuts>0 from scan_cuts is the scene-cut surrogate
            sc = bool(c.get("cuts")) if c.get("cuts") is not None else None
        sig[v] = {"n_frames": n, "scene_cut": sc, "diff_median": f.get("diff_median"),
                  "aes": aes.get(v), "cam_shift": cs[0] if cs else None, "cam_source": cs[1] if cs else None,
                  "cuts": c.get("cuts"), "flicker": c.get("flicker"),
                  "ov_aes": o[0] if o else None, "ov_motion": o[1] if o else None}
    if frame_cache_path:
        os.makedirs(os.path.dirname(frame_cache_path), exist_ok=True)
        with open(frame_cache_path + ".tmp", "w") as fh:
            for d in fc.values():
                fh.write(json.dumps(d) + "\n")
        os.replace(frame_cache_path + ".tmp", frame_cache_path)
    return sig


def source_thresholds(rows, sig):
    by = defaultdict(lambda: {"flicker": [], "ov_motion": []})
    for r in rows:
        s = sig[r["video"]]
        if s["flicker"] is not None:
            by[r["dataset"]]["flicker"].append(s["flicker"])
        if s["ov_motion"] is not None:
            by[r["dataset"]]["ov_motion"].append(s["ov_motion"])
    return {ds: {"flicker_q80": quantile(v["flicker"], FLICKER_SOURCE_Q),
                 "ov_motion_q40": quantile(v["ov_motion"], OPENVID_MOTION_SOURCE_Q)} for ds, v in by.items()}


def excluded_parent_keys(ec):
    """E1 exclusion sources (recomputed exactly as prepare_e1_devset.main) + the E1 dev-set cases' parent keys."""
    keys, per = set(), {}
    for name, paths in dev.load_exclusions().items():
        k = set()
        for p in paths:
            k |= dev.parent_keys(p, ec)[0]
        per[name] = len(k)
        keys |= k
    e1 = set()
    for d in load_jsonl(E1_CASES):
        e1 |= set(d["parent_keys"])
        e1 |= dev.parent_keys(d["video"], ec)[0]
    per["e1_devset_cases"] = len(e1)
    return keys, e1, per


def cam_supplement(videos, out_path, procs):
    """Same cam_shift algorithm as training_metadata/ft2/cam_scores.jsonl (scripts/score_camera.score); resumable."""
    sc = _load("score_camera")
    from multiprocessing import Pool
    done = {d["video"] for d in load_jsonl(out_path)} if os.path.exists(out_path) else set()
    todo = sorted(set(videos) - done)
    print(f"[cam] {len(videos)} need cam_shift, {len(todo)} to score", flush=True)
    if not todo:
        return
    with open(out_path, "a") as f, Pool(procs) as p:
        for i, r in enumerate(p.imap_unordered(sc.score, todo, chunksize=4), 1):
            f.write(json.dumps(r) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"[cam] {i}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--cam-supplement", action="store_true",
                    help="compute cam_shift (score_camera algorithm) for rows that pass every other rule but lack it")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for name in ("prefilter.csv", "rejected.csv", "summary.json"):
        if not a.dry_run and os.path.exists(f"{a.out_dir}/{name}"):
            sys.exit(f"Refusing to overwrite {a.out_dir}/{name}")
    cands = list(csv.DictReader(open(CANDIDATES, newline="")))
    rows = [{"video": r["video"], "prompt": r["prompt"], "dataset": dev.dataset_of(r["video"])} for r in cands]
    want = {r["video"] for r in rows}
    ec = dev.easycom_index()
    excl, e1_keys, per_source = excluded_parent_keys(ec)
    e1_summary = json.load(open(E1_SUMMARY))
    stats = {"pool_by_dataset": Counter(r["dataset"] for r in rows)}

    kept_rows, excl_counts = [], Counter()
    for r in rows:
        keys, resolved = dev.parent_keys(r["video"], ec)
        r["parent_keys"], r["parent_resolved"] = set(keys), resolved
        hit_e1, hit_src = bool(keys & e1_keys), bool(keys & excl)
        if hit_e1 or hit_src:
            excl_counts[r["dataset"]] += 1
            stats.setdefault("excluded_by", Counter())["e1_devset" if hit_e1 else "e1_exclusion_sources"] += 1
            continue
        kept_rows.append(r)

    print("[load] signals ...", flush=True)
    final = load_final(want)
    aes = load_aes(want)
    sup_path = f"{a.out_dir}/cam_supplement.jsonl"
    cam = load_cam(want, [CAM_SCORES, sup_path])
    cuts = load_cuts(want)
    ov = load_openvid({openvid_name(r["video"]) for r in rows if r["dataset"] in OPENVID_DATASETS})
    sig = build_signals(rows, final, aes, cam, cuts, ov, frame_cache_path=f"{a.out_dir}/frame_counts.jsonl")
    src_q = source_thresholds(rows, sig)  # thresholds over each source's FULL candidate pool

    def run_rules():
        passed, rejected = [], []
        first, any_fail, any_missing = Counter(), Counter(), Counter()
        for r in kept_rows:
            s = sig[r["video"]]
            ok, out, reason = evaluate(r["dataset"], s, src_q.get(r["dataset"], {}))
            for rule, o in out.items():
                if o == "fail":
                    any_fail[f"{r['dataset']}:{rule}"] += 1
                elif o == "missing":
                    any_missing[f"{r['dataset']}:{rule}"] += 1
            if ok:
                passed.append(dict(r, **s))
            else:
                first[reason] += 1
                rejected.append((r, reason, out))
        return passed, rejected, first, any_fail, any_missing

    passed, rejected, first, any_fail, any_missing = run_rules()
    if a.cam_supplement:
        need = [r["video"] for r, reason, out in rejected
                if [k for k, o in out.items() if o != "pass"] == ["cam_shift"] and out["cam_shift"] == "missing"]
        if need:
            os.makedirs(a.out_dir, exist_ok=True)
            cam_supplement(need, sup_path, a.procs)
            cam = load_cam(want, [CAM_SCORES, sup_path])
            sig = build_signals(rows, final, aes, cam, cuts, ov, frame_cache_path=f"{a.out_dir}/frame_counts.jsonl")
            passed, rejected, first, any_fail, any_missing = run_rules()

    final_rows, cap = apply_caps(passed)
    final_rows.sort(key=lambda r: r["video"])
    q = {ds: {k: (round(v, 4) if v is not None else None) for k, v in t.items()} for ds, t in sorted(src_q.items())}
    summary = {
        "protocol": "E3 pool prefilter v1 (docs/43 section 5)",
        "candidates_csv": os.path.relpath(CANDIDATES, ROOT), "candidates_csv_sha256": dev.sha256_file(CANDIDATES),
        "candidate_rows": len(rows),
        "rules": {"excluded_datasets": sorted(EXCLUDED_DATASETS), "n_frames": N_FRAMES, "scene_cut": False,
                  "cuts_max": CUTS_MAX, "flicker_le_source_q": FLICKER_SOURCE_Q, "cam_shift_max": CAM_SHIFT_MAX,
                  "indomain": {"aes_min": AES_MIN_INDOMAIN, "diff_median_min": DIFF_MEDIAN_MIN, "rules": RULES_INDOMAIN},
                  "openvidgen": {"openvid_aes_min": AES_MIN_OPENVID, "openvid_motion_ge_source_q": OPENVID_MOTION_SOURCE_Q,
                                 "rules": RULES_OPENVID},
                  "per_parent_max": PER_PARENT_MAX, "matrix_max_share": MATRIX_MAX_SHARE,
                  "missing_signal": "counted as missing:<rule>, never passed",
                  "cap_priority": "aes desc (OpenVid aes for openvidgen), then video path"},
        "parent_exclusion": {"e1_exclusion_source_parent_keys": per_source,
                             "recomputed_source_keys_total": len(excl),
                             "e1_summary_excluded_parent_keys_total": e1_summary.get("excluded_parent_keys_total"),
                             "e1_cases_parent_keys": len(e1_keys),
                             "excluded_windows_by_dataset": dict(excl_counts),
                             "excluded_by": dict(stats.get("excluded_by", {}))},
        "signal_coverage_full_pool": {k: sum(sig[r["video"]][k] is not None for r in rows)
                                      for k in ("n_frames", "scene_cut", "diff_median", "aes", "cam_shift", "cuts",
                                                "flicker", "ov_aes", "ov_motion")},
        "source_thresholds": q,
        "pool_by_dataset": dict(stats["pool_by_dataset"]),
        "after_parent_exclusion_by_dataset": dict(Counter(r["dataset"] for r in kept_rows)),
        "first_reason_counts": dict(first.most_common()),
        "fail_counts_by_dataset_rule": dict(sorted(any_fail.items())),
        "missing_counts_by_dataset_rule": dict(sorted(any_missing.items())),
        "rows_with_any_missing_signal": sum(1 for _, _, out in rejected if "missing" in out.values()),
        "passed_rules": len(passed), "passed_rules_by_dataset": dict(Counter(r["dataset"] for r in passed)),
        "cap_removed": cap,
        "final_total": len(final_rows), "final_by_dataset": dict(sorted(Counter(r["dataset"] for r in final_rows).items())),
        "final_parent_keys": len({k for r in final_rows for k in r["parent_keys"]}),
        "final_unresolved_parent": sum(not r["parent_resolved"] for r in final_rows),
        "cam_supplement_rows_in_final": sum(r.get("cam_source") == "cam_supplement.jsonl" for r in final_rows),
    }
    if a.dry_run:
        print(json.dumps(summary, indent=1))
        return
    os.makedirs(a.out_dir, exist_ok=True)
    cols = ["video", "prompt", "dataset", "parent_key", "parent_resolved"] + SIGNAL_COLUMNS
    with open(f"{a.out_dir}/prefilter.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in final_rows:
            q_ = src_q.get(r["dataset"], {})
            w.writerow({**{k: r.get(k) for k in cols}, "parent_key": parent_key_str(r["parent_keys"]),
                        "flicker_src_q80": q_.get("flicker_q80"), "ov_motion_src_q40": q_.get("ov_motion_q40")})
    with open(f"{a.out_dir}/rejected.csv", "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "dataset", "first_reason", "outcomes"])
        for r, reason, out in sorted(rejected, key=lambda x: x[0]["video"]):
            w.writerow([r["video"], r["dataset"], reason, json.dumps(out, sort_keys=True)])
    summary["prefilter_csv_sha256"] = dev.sha256_file(f"{a.out_dir}/prefilter.csv")
    with open(f"{a.out_dir}/summary.json", "x") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
