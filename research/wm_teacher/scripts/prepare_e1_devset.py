#!/usr/bin/env python3
"""
E1 independent development set (docs/43 section 3).

Stratified by the 17 datasets of the training candidate pool, fixed RNG seed, at most PER_DATASET windows per
dataset, at most one window per parent source. Every window whose parent source also appears in any of the
EXCLUSION_SOURCES (WM108, the fixed val120 used by eval_val_loss.py, P1/C1/M1/training-smoke cases incl. the P1
candidate/review lists, the 2026-10-17 inference dev cases) is removed before sampling.

Parent rule (conservative: when in doubt merge, so more windows are excluded):
  YouTube-derived sets (celebvhq, vfhq, hoigen1m, openvidgen, openvidhd) -> youtube:<video id> (shared namespace)
  charades -> video id; direct ADT -> sequence number, AEA -> loc/script/seq (recs merged), Hot3D -> participant; easycom -> every Session_* directory
  that holds the basename (basenames are reused across sessions); egodex -> task (without -partN) + episode;
  h2o -> subject + scene; matrix -> recording timestamp; mira -> match id; noxi -> session number;
  robomind -> robot + episode (all cameras merged); seamless -> V/S session token; ukdash -> recording day.
  hoigen1m non-YouTube "<n>_scene-<k>" -> clip n; openvidgen magictime_<id>_<k> -> magictime id.
  egoexo4d basenames lost their take mapping -> basename key, marked unresolved (reported, not claimed clean).
VBench-I2V conditions on external public images that have no parent id in this namespace -> 0 exclusions by design.

Only new files are written: private copy under /dockerdata/wm_tmp/e1_devset_20261021 and the shared copy
(first frames + manifest) under eval/e1_devset_20261021. Existing outputs are never overwritten.
CPU only. Usage: /opt/conda/envs/torch-base/bin/python scripts/prepare_e1_devset.py [--dry-run]
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

ROOT = "/srv/wm_teacher"
TMP = "/dockerdata/wm_tmp"
CANDIDATES = f"{ROOT}/training_metadata/teacher_namespace_candidates_20261020_v1/namespace_candidates.csv"
EASYCOM_SESSIONS = f"{ROOT}/dataset/egocentric_interaction/easycom/extracted/Main/Video_Compressed"
SHARED_OUT = f"{ROOT}/eval/e1_devset_20261021"
PRIVATE_OUT = f"{TMP}/e1_devset_20261021"
VAL_CSV = f"{ROOT}/training_metadata/round2/round2_val.csv"
VAL_N = 120
SEED = 20261021
PER_DATASET = 6
# name -> glob patterns; every source must yield >= 1 internal video path (guards against silently empty sources)
EXCLUSION_SOURCES = {
    "wm108": [f"{ROOT}/eval/wm_bench/cases.jsonl"],
    "fixed_val120": [VAL_CSV],  # only the 120 rows eval_val_loss.py uses, see fixed_val120_paths()
    "teacher_P1": [f"{TMP}/teacher_p1_cfg_20261019_run01/jobs.input.jsonl",
                   f"{TMP}/teacher_opt_candidates_20261019.json",
                   f"{TMP}/teacher_opt_review_20261019/*.json*"],
    "teacher_C1": [f"{TMP}/teacher_C1_20261020_run01/jobs.frozen.jsonl"],
    "teacher_M1": [f"{TMP}/teacher_m1_20261020_run01/mechanism_manifest.json"],
    "training_smoke": [f"{ROOT}/training_metadata/teacher_gradient_smoke_20261020_v1/input_manifest.json"],
    "inference_dev_20261017": [f"{ROOT}/eval/diag/inference_dev_cases.json"],
}
PATH_RE = re.compile(r"wan22_5s/([a-z0-9]+)_5s/[^\"\s,]+?\.mp4")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_paths(text):
    """Internal 5s-window paths, normalised to 'dataset/wan22_5s/<ds>_5s/...mp4'."""
    return {"dataset/" + m.group(0) for m in PATH_RE.finditer(text)}


def dataset_of(rel):
    return rel.split("/")[2][:-len("_5s")]


def easycom_index(directory=EASYCOM_SESSIONS):
    index = defaultdict(set)
    for path in glob.glob(f"{directory}/Session_*/*.mp4"):
        index[os.path.splitext(os.path.basename(path))[0]].add(os.path.basename(os.path.dirname(path)))
    return index


def parent_keys(rel, easycom=None):
    """(frozenset of parent keys, resolved flag). Two windows share a parent iff their key sets intersect."""
    ds = dataset_of(rel)
    parts = rel.split("/")
    stem = re.sub(r"_w\d+$", "", os.path.splitext(parts[-1])[0])
    key = None
    if ds == "celebvhq":
        key = f"youtube:{stem}"
    elif ds == "vfhq":
        m = re.fullmatch(r"Clip\+(.+)\+P\d+\+C\d+\+F\d+-\d+", stem)
        key = m and f"youtube:{m.group(1)}"
    elif ds == "hoigen1m":
        m = re.fullmatch(r"(.+)-\d+_\d+_\d+\.\d+-\d+_\d+_\d+\.\d+", stem)
        key = m and f"youtube:{m.group(1)}"
        if not m:
            m = re.fullmatch(r"(\d+)_scene-\d+", stem)
            key = m and f"hoigen1m:clip:{m.group(1)}"
    elif ds == "openvidgen":
        m = re.fullmatch(r"(.+)_\d+_\d+to\d+", stem)
        key = m and f"youtube:{m.group(1)}"
        if not m:
            m = re.fullmatch(r"magictime_(\d+)_\d+", stem)
            key = m and f"magictime:{m.group(1)}"
    elif ds == "openvidhd":
        m = re.fullmatch(r"OpenVidHD_part_\d+\.zip_(.+)_\d+_\d+to\d+", stem)
        key = m and f"youtube:{m.group(1)}"
    elif ds == "charades":
        m = re.fullmatch(r"([A-Z0-9]+)_1280x704_24fps", stem)
        key = m and f"charades:video:{m.group(1)}"
    elif ds == "direct":
        m = re.match(r"ADT_.*_seq(\d+)_", stem)
        m2 = re.match(r"AriaEverydayActivities_[\d.]+_(loc\d+_script\d+_seq\d+)_rec\d+", stem)
        if m:
            key = f"aria:ADT:seq{int(m.group(1))}"
        elif m2:  # rec1/rec2 are simultaneous wearers of one sequence -> merged
            key = f"aria:AEA:{m2.group(1)}"
        else:
            m = re.match(r"Hot3D[A-Za-z]+_v[\d.]+_(P\d+)_", stem)
            key = m and f"aria:Hot3D:{m.group(1)}"
    elif ds == "easycom":
        m = re.fullmatch(r"(\d+-\d+-\d+)_1280x704_24fps", stem)
        sessions = (easycom or {}).get(m.group(1)) if m else None
        if sessions:
            return frozenset(f"easycom:{s}" for s in sessions), True
    elif ds == "egodex":
        m = re.fullmatch(r"camera_(ep\d+)", stem)
        task = re.sub(r"-part\d+$", "", parts[3])
        key = m and f"egodex:{task}:{m.group(1)}"
    elif ds == "h2o":
        m = re.fullmatch(r"(subject\d+)_(?:ego_)?([a-z]\d+)_\d+_cam\d+", stem)
        key = m and f"h2o:session:{m.group(1)}:{m.group(2)}"
    elif ds == "matrix":
        m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", stem)
        key = m and f"matrix:recording:{m.group(1)}"
    elif ds == "mira":
        m = re.search(r"processed_(.+?)_p\d+_c\d+$", stem)
        key = m and f"mira:match:{m.group(1)}"
    elif ds == "noxi":
        m = re.fullmatch(r"(\d+)_(?:novice|expert)", stem)
        key = m and f"noxi:session:{int(m.group(1))}"
    elif ds == "robomind":
        m = re.fullmatch(r"camera_[a-z_]+?_(ep\d+)", stem)
        key = m and f"robomind:{parts[3]}:{m.group(1)}"
    elif ds == "seamless":
        m = re.match(r"(V\d+)_(S\d+)_", stem)
        key = m and f"seamless:session_token:{m.group(1)}:{m.group(2)}"
    elif ds == "ukdash":
        m = re.match(r"(\d{6})_\d{6}_", stem)
        key = m and f"ukdash:day:{m.group(1)}"
    # egoexo4d (take mapping lost) and unmatched names stay unresolved
    if key:
        return frozenset({key}), True
    return frozenset({f"unresolved:{ds}:{stem}"}), False


def fixed_val120_paths(val_csv=VAL_CSV, n=VAL_N):
    """Same row choice as scripts/eval_val_loss.py (defaults --val-csv round2_val.csv --n 120)."""
    rows = list(csv.DictReader(open(val_csv, newline="")))
    step = max(1, len(rows) // n)
    return [r["video"] for r in rows[::step][:n]]


def load_exclusions(sources=EXCLUSION_SOURCES):
    """{source name: set of internal window paths}; raises if a source is missing or yields no path."""
    out = {}
    for name, patterns in sources.items():
        files = sorted({f for p in patterns for f in glob.glob(p)})
        if not files:
            raise FileNotFoundError(f"Exclusion source {name} has no files: {patterns}")
        paths = set()
        for f in files:
            if name == "fixed_val120":
                text = "\n".join(fixed_val120_paths(f))
                if len(set(fixed_val120_paths(f))) != VAL_N:
                    raise ValueError("fixed val120 does not contain 120 unique rows")
            else:
                with open(f, errors="replace") as fh:
                    text = fh.read()
            paths |= extract_paths(text)
        if not paths:
            raise ValueError(f"Exclusion source {name} yielded no internal video path")
        out[name] = paths
    return out


def select_devset(candidates, excluded_keys, seed=SEED, per_dataset=PER_DATASET, easycom=None, usable=None):
    """candidates: [{'video','prompt'}]. Returns (selected rows, stats); deterministic in (seed, set of rows).
    usable(row) -> bool is consulted in draw order (decode check); rejected rows are skipped and recorded."""
    rows = sorted(candidates, key=lambda r: r["video"])
    by_ds = defaultdict(lambda: defaultdict(list))
    stats = {"pool_by_dataset": Counter(), "excluded_windows_by_dataset": Counter(),
             "eligible_parents_by_dataset": {}, "unresolved_parent_selected": 0, "unusable_skipped": []}
    for r in rows:
        ds = dataset_of(r["video"])
        stats["pool_by_dataset"][ds] += 1
        keys, resolved = parent_keys(r["video"], easycom)
        if keys & excluded_keys:
            stats["excluded_windows_by_dataset"][ds] += 1
            continue
        by_ds[ds][min(keys)].append((r, keys, resolved))
    rng = random.Random(seed)
    selected, used_keys = [], set()
    for ds in sorted(by_ds):
        parents = sorted(by_ds[ds])
        stats["eligible_parents_by_dataset"][ds] = len(parents)
        rng.shuffle(parents)
        picked = 0
        for p in parents:
            if picked >= per_dataset:
                break
            windows = by_ds[ds][p]
            r, keys, resolved = windows[rng.randrange(len(windows))]
            if keys & used_keys:  # e.g. the same YouTube video under two datasets
                continue
            if usable is not None and not usable(r):
                stats["unusable_skipped"].append(r["video"])
                continue
            used_keys |= keys
            selected.append(dict(r, dataset=ds, parent_keys=sorted(keys), parent_resolved=resolved))
            stats["unresolved_parent_selected"] += not resolved
            picked += 1
    stats["selected_by_dataset"] = Counter(s["dataset"] for s in selected)
    return selected, stats


def probe_video(path):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 0)
        width, height = s.codec_context.width, s.codec_context.height
        n = sum(1 for _ in c.decode(video=0))
    return {"n_frames": n, "fps": fps, "width": width, "height": height}


def first_frame_png(video, dst):
    """Frame 0 decoded exactly as gen_jobs.load_image / vbench_eval.prepare_wm do (imageio)."""
    import imageio.v2 as imageio
    from PIL import Image
    r = imageio.get_reader(video)
    Image.fromarray(r.get_data(0)).save(dst)
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="selection + statistics only, write nothing")
    a = ap.parse_args()
    if not a.dry_run:
        for d in (SHARED_OUT, PRIVATE_OUT):
            if os.path.exists(d):
                sys.exit(f"Refusing to overwrite existing {d}")
    candidates = list(csv.DictReader(open(CANDIDATES, newline="")))
    ec = easycom_index()
    exclusions = load_exclusions()
    excluded_keys, per_source = set(), {}
    for name, paths in exclusions.items():
        keys, unresolved = set(), 0
        for p in paths:
            k, resolved = parent_keys(p, ec)
            keys |= k
            unresolved += not resolved
        per_source[name] = {"window_paths": len(paths), "parent_keys": len(keys),
                            "unresolved_parent_paths": unresolved,
                            "datasets": dict(Counter(dataset_of(p) for p in paths)),
                            "candidate_windows_excluded": sum(bool(parent_keys(r["video"], ec)[0] & keys)
                                                              for r in candidates)}
        excluded_keys |= keys
    probes = {}

    def usable(r):
        try:
            probes[r["video"]] = probe_video(f"{ROOT}/{r['video']}")
        except Exception as e:  # undecodable source -> skipped and recorded
            print(f"[skip] {r['video']}: {type(e).__name__}", flush=True)
            return False
        return probes[r["video"]]["n_frames"] >= 2 and probes[r["video"]]["fps"] > 0

    selected, stats = select_devset(candidates, excluded_keys, easycom=ec, usable=None if a.dry_run else usable)
    summary = {
        "protocol": "E1 dev set v1 (docs/43 section 3)", "seed": SEED, "per_dataset_max": PER_DATASET,
        "candidates_csv": os.path.relpath(CANDIDATES, ROOT), "candidates_csv_sha256": sha256_file(CANDIDATES),
        "candidate_rows": len(candidates), "datasets": len(stats["pool_by_dataset"]),
        "exclusion_sources": per_source, "excluded_parent_keys_total": len(excluded_keys),
        "excluded_candidate_windows_total": sum(stats["excluded_windows_by_dataset"].values()),
        "vbench_i2v": "external public images, no internal parent id -> 0 exclusions by design; "
                      "never used for selection",
        "pool_by_dataset": dict(stats["pool_by_dataset"]),
        "excluded_windows_by_dataset": dict(stats["excluded_windows_by_dataset"]),
        "eligible_parents_by_dataset": stats["eligible_parents_by_dataset"],
        "selected_by_dataset": dict(stats["selected_by_dataset"]), "selected_total": len(selected),
        "unresolved_parent_selected": stats["unresolved_parent_selected"],
        "unusable_skipped": len(stats["unusable_skipped"]),
        "parent_rule_note": "egoexo4d parents unresolved (basename only); easycom by Session directory index",
    }
    if a.dry_run:
        print(json.dumps(summary, indent=1))
        return
    os.makedirs(f"{PRIVATE_OUT}/images", mode=0o700)
    os.makedirs(f"{SHARED_OUT}/images", mode=0o700)
    cases = []
    for s in selected:
        cid = f"{s['dataset']}__{os.path.splitext(os.path.basename(s['video']))[0]}"
        video = f"{ROOT}/{s['video']}"
        private_png = f"{PRIVATE_OUT}/images/{cid}.png"
        shared_png = f"{SHARED_OUT}/images/{cid}.png"
        first_frame_png(video, private_png)
        shutil.copyfile(private_png, shared_png)
        image_sha = sha256_file(private_png)
        if sha256_file(shared_png) != image_sha:
            raise RuntimeError(f"Shared first-frame copy mismatch: {cid}")
        cases.append({"id": cid, "dataset": s["dataset"], "video": s["video"], "gt": video,
                      "video_sha256": sha256_file(video), "image": shared_png, "image_sha256": image_sha,
                      "prompt": s["prompt"], "prompt_sha256": sha256_text(s["prompt"]),
                      "parent_keys": s["parent_keys"], "parent_resolved": s["parent_resolved"],
                      **probes[s["video"]]})
    if len({c["id"] for c in cases}) != len(cases):
        raise RuntimeError("Duplicate case id")
    text = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases)
    for d in (PRIVATE_OUT, SHARED_OUT):
        with open(f"{d}/cases.jsonl", "x") as f:
            f.write(text)
    summary["cases_jsonl_sha256"] = sha256_text(text)
    summary["unusable_skipped_paths"] = stats["unusable_skipped"]
    for d in (PRIVATE_OUT, SHARED_OUT):
        with open(f"{d}/devset_summary.json", "x") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
