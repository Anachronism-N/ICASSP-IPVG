"""Round 4 mixture: aesthetic-filtered quality tuning.

Per dataset, among the scored pool (training_metadata/round4/aes_scores_*.jsonl, val sequences
already excluded), drop that dataset's least-moving --min-motion-q fraction (by diff_median) and keep the windows with
the highest LAION aesthetic score, up to the Round-2 share scaled to --n-total. At most
--per-seq windows per source sequence to keep diversity.
Output: training_metadata/round4/round4_train.csv (video,prompt) + round4_stats.json
"""
import argparse
import csv
import glob
import json
import random
import statistics as st
from collections import defaultdict

ROOT = "/srv/wm_teacher"
OUT = f"{ROOT}/training_metadata/round4"
SHARE = {  # same relative mix as round 2
    "hoigen1m": 20000, "celebvhq": 12000, "mira": 12000, "egoexo4d": 10000, "charades": 10000,
    "noxi": 8000, "direct": 8000, "vfhq": 6000, "matrix": 6000, "seamless": 3000,
    "easycom": 2800, "openvidhd": 300, "h2o": 196,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=30000)
    ap.add_argument("--min-motion-q", type=float, default=0.2, help="per-dataset diff_median quantile floor")
    ap.add_argument("--per-seq", type=int, default=4)
    a = ap.parse_args()

    aes = {}
    for f in glob.glob(f"{OUT}/aes_scores_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            aes[r["video"]] = r["aes"]
    meta = {}
    for l in open(f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl"):
        m = json.loads(l)
        if m["video_path"] in aes:
            meta[m["video_path"]] = m
    scale = a.n_total / sum(SHARE.values())
    rows, stats = [], {}
    print(f"{'dataset':10s} {'scored':>7s} {'moving':>7s} {'target':>7s} {'taken':>6s} {'aes_all':>7s} {'aes_kept':>8s}")
    for ds, share in SHARE.items():
        cand = [m for m in meta.values() if m["source_dataset"] == ds]
        # relative motion floor: conversational sets (noxi/seamless) are static by nature,
        # so drop each dataset's own least-moving fraction instead of one global threshold
        dm = sorted((m.get("diff_median") or 0) for m in cand)
        floor = dm[int(a.min_motion_q * (len(dm) - 1))] if dm else 0
        moving = [m for m in cand if (m.get("diff_median") or 0) >= floor]
        moving.sort(key=lambda m: -aes[m["video_path"]])
        target = round(share * scale)
        per_seq, got = defaultdict(int), []
        for m in moving:
            if len(got) >= target:
                break
            if per_seq[m["source_sequence_id"]] >= a.per_seq:
                continue
            per_seq[m["source_sequence_id"]] += 1
            got.append(m)
        a_all = st.mean(aes[m["video_path"]] for m in cand) if cand else 0
        a_kept = st.mean(aes[m["video_path"]] for m in got) if got else 0
        stats[ds] = dict(scored=len(cand), moving=len(moving), target=target, taken=len(got),
                         aes_all=round(a_all, 3), aes_kept=round(a_kept, 3))
        print(f"{ds:10s} {len(cand):7d} {len(moving):7d} {target:7d} {len(got):6d} {a_all:7.3f} {a_kept:8.3f}")
        rows += got
    random.Random(4).shuffle(rows)
    with open(f"{OUT}/round4_train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        for m in rows:
            w.writerow([m["video_path"], m["prompt"]])
    json.dump(dict(args=vars(a), per_dataset=stats, total=len(rows)), open(f"{OUT}/round4_stats.json", "w"), indent=1)
    print("total", len(rows), "->", f"{OUT}/round4_train.csv")


if __name__ == "__main__":
    main()
