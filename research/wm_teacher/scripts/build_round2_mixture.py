#!/usr/bin/env python3
"""
Build round-2 training mixture (5s / 121-frame windows, Qwen3.5 dense captions) for Wan2.2 TI2V LoRA.

Source: dataset/wan22_5s/final_5s_landscape.jsonl (written by finalize_5s.py; captions already
sanitized and >= 60 words). Landscape 1280x704 only, same as round 1.
Strategy (same as round 1):
  - per-dataset targets, single source <= ~20%
  - drop windows flagged scene_cut
  - round-robin over source_sequence_id -> max source diversity
  - val held out by source_sequence_id (~2% of sequences), no train/val leakage
Outputs: training_metadata/round2/round2_train.csv, round2_val.csv  (video,prompt)
"""
import csv
import json
import os
import random
from collections import defaultdict

ROOT = "/srv/wm_teacher"
SRC = f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl"
OUT = f"{ROOT}/training_metadata/round2"
TARGETS = {
    "hoigen1m": 20000,
    "celebvhq": 12000,
    "mira": 12000,
    "egoexo4d": 10000,
    "charades": 10000,
    "noxi": 8000,
    "direct": 8000,
    "vfhq": 6000,
    "matrix": 6000,
    "seamless": 3000,
    "easycom": 2800,
    "openvidhd": 300,
    "h2o": 196,
}
VAL_SEQ_FRACTION = 0.02
SEED = 42


def take(by_seq, seqs, want):
    got, k = [], 0
    while len(got) < want:
        added = False
        for s in seqs:
            if k < len(by_seq[s]):
                got.append(by_seq[s][k])
                added = True
                if len(got) >= want:
                    break
        if not added:
            break
        k += 1
    return got


def main():
    rng = random.Random(SEED)
    data = defaultdict(lambda: defaultdict(list))
    n_cut = defaultdict(int)
    with open(SRC) as f:
        for line in f:
            r = json.loads(line)
            ds = r["source_dataset"]
            if ds not in TARGETS:
                continue
            if r.get("scene_cut"):
                n_cut[ds] += 1
                continue
            data[ds][r["source_sequence_id"]].append((r["video_path"], r["prompt"]))

    train, val = [], []
    print(f"{'dataset':10s} {'windows':>8s} {'seqs':>7s} {'cut':>6s} {'train':>7s} {'val':>5s}")
    for ds, target in TARGETS.items():
        by_seq = data[ds]
        for s in by_seq.values():
            rng.shuffle(s)  # spread picks along each source instead of always its first windows
        seqs = sorted(by_seq)
        rng.shuffle(seqs)
        n_val = max(1, int(len(seqs) * VAL_SEQ_FRACTION)) if len(seqs) > 50 else 0
        tr = take(by_seq, seqs[n_val:], target)
        va = take(by_seq, seqs[:n_val], max(10, target // 50)) if n_val else []
        train += tr
        val += va
        print(f"{ds:10s} {sum(map(len, by_seq.values())):8d} {len(seqs):7d} {n_cut[ds]:6d} {len(tr):7d} {len(va):5d}")
    rng.shuffle(train)

    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("round2_train.csv", train), ("round2_val.csv", val)):
        with open(f"{OUT}/{name}", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "prompt"])
            w.writerows(rows)
    print(f"train={len(train)} val={len(val)} -> {OUT}")


if __name__ == "__main__":
    main()
