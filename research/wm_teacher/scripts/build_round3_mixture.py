#!/usr/bin/env python3
"""
Round-3 mixture: fresh 5s windows never used in round 2 (train or val), same dataset targets and
round-robin-over-sources strategy as round 2, val sequences held out exactly as in round 2
(round-2 val sequences are excluded entirely, so WM-Bench stays unseen).

Optional motion-aware sampling (--min-motion X): drop windows whose median inter-frame
difference (diff_median from reslice_5s.py, 64x36 gray, 0-255 scale) is below X, i.e. near-static
clips, which push the model toward low dynamic degree.

Usage: python build_round3_mixture.py --n-total 60000 [--min-motion 0.5] [--out round3_train.csv]
"""
import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

ROOT = "/srv/wm_teacher"
SRC = f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl"
R2 = f"{ROOT}/training_metadata/round2"
SHARE = {  # same relative mix as round 2 (round2 targets / 96,555)
    "hoigen1m": 20000, "celebvhq": 12000, "mira": 12000, "egoexo4d": 10000, "charades": 10000,
    "noxi": 8000, "direct": 8000, "vfhq": 6000, "matrix": 6000, "seamless": 3000,
    "easycom": 2800, "openvidhd": 300, "h2o": 196,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=60000)
    ap.add_argument("--min-motion", type=float, default=0.0)
    ap.add_argument("--out", default=f"{R2}/../round3/round3_train.csv")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    used = set()
    val_seqs = set()
    for name in ("round2_train.csv", "round2_val.csv"):
        for r in csv.DictReader(open(f"{R2}/{name}")):
            used.add(r["video"])
            if name == "round2_val.csv":
                val_seqs.add(os.path.basename(r["video"]).rsplit("_w", 1)[0])

    by = defaultdict(lambda: defaultdict(list))
    reuse = defaultdict(lambda: defaultdict(list))
    stat = Counter()
    for line in open(SRC):
        m = json.loads(line)
        ds = m["source_dataset"]
        if ds not in SHARE:
            continue
        stat[f"{ds}_all"] += 1
        vp = m["video_path"]
        seq = os.path.basename(vp).rsplit("_w", 1)[0]
        if seq in val_seqs or m.get("scene_cut"):
            continue
        if vp in used:  # kept aside: refills datasets that round 2 already exhausted
            reuse[ds][m["source_sequence_id"]].append((vp, m["prompt"]))
            continue
        if a.min_motion and (m.get("diff_median") or 0) < a.min_motion:
            stat[f"{ds}_static"] += 1
            continue
        by[ds][m["source_sequence_id"]].append((vp, m["prompt"]))

    scale = a.n_total / sum(SHARE.values())
    rows = []
    def take(pool, want):
        seqs = sorted(pool)
        rng.shuffle(seqs)
        for s in seqs:
            rng.shuffle(pool[s])
        got, k = [], 0
        while len(got) < want and any(k < len(pool[s]) for s in seqs):
            for s in seqs:
                if k < len(pool[s]) and len(got) < want:
                    got.append(pool[s][k])
            k += 1
        return got

    print(f"{'dataset':10s} {'fresh':>8s} {'static':>7s} {'target':>7s} {'fresh_t':>7s} {'reused':>7s}")
    for ds, share in SHARE.items():
        target = round(share * scale)
        got = take(by[ds], target)
        again = take(reuse[ds], target - len(got)) if len(got) < target else []
        rows += got + again
        print(f"{ds:10s} {sum(map(len, by[ds].values())):8d} {stat[ds + '_static']:7d} {target:7d} "
              f"{len(got):7d} {len(again):7d}")
    rng.shuffle(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        w.writerows(rows)
    print(f"total={len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
