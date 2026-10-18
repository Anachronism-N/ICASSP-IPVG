#!/usr/bin/env python3
"""
Build round-1 training mixture CSV for Wan2.2 TI2V LoRA.

Sources: training_metadata/recaptioned/*_recaptioned.jsonl (VLM captions, verified non-empty)
Strategy:
  - per-dataset targets (see TARGETS), single source <= ~25%
  - round-robin across source_sequence_id: first pass takes 1 clip per sequence,
    second pass 2nd clip, ... -> maximum source diversity, no adjacent-clip dominance
  - val split held out by source_sequence_id (~2% of sequences), no cross-split leakage
  - MIRA: prompt length filter (>= 40 chars) to drop degenerate template captions
  - full path existence check (threaded)
Outputs:
  training_metadata/round1_train.csv  (video,prompt)
  training_metadata/round1_val.csv    (video,prompt)
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = "/srv/wm_teacher"
SRC = os.path.join(ROOT, "training_metadata", "recaptioned")
OUT_TRAIN = os.path.join(ROOT, "training_metadata", "round1_train.csv")
OUT_VAL = os.path.join(ROOT, "training_metadata", "round1_val.csv")

TARGETS = {
    "hoigen1m": 15000,
    "egoexo4d": 8000,
    "charades": 6000,
    "noxi": 6000,
    "direct": 5000,
    "mira": 5000,
    "celebvhq": 5000,
    "matrix": 3000,
    "vfhq": 2500,
    "dexycb": 2400,
    "easycom": 1500,
    "h2o": 297,
}
MIN_PROMPT_LEN = {"mira": 40}  # dataset-specific floor
VAL_SEQ_FRACTION = 0.02
SEED = 42


def load_dataset(name):
    path = os.path.join(SRC, f"{name}_recaptioned.jsonl")
    by_seq = defaultdict(list)
    n_total = n_bad = 0
    min_len = MIN_PROMPT_LEN.get(name, 1)
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            prompt = (d.get("prompt") or "").strip()
            video = d.get("video_path") or ""
            if not video or len(prompt) < min_len:
                n_bad += 1
                continue
            by_seq[d.get("source_sequence_id") or video].append((video, prompt))
    return by_seq, n_total, n_bad


def sample_dataset(name, target, rng):
    by_seq, n_total, n_bad = load_dataset(name)
    seqs = sorted(by_seq.keys())
    rng.shuffle(seqs)

    # hold out val sequences
    n_val = max(1, int(len(seqs) * VAL_SEQ_FRACTION)) if len(seqs) > 50 else 0
    val_seqs, train_seqs = set(seqs[:n_val]), seqs[n_val:]

    # round-robin: pass k takes the k-th clip of each sequence
    def take(seqlist, want):
        got = []
        k = 0
        while len(got) < want:
            added = False
            for s in seqlist:
                clips = by_seq[s]
                if k < len(clips):
                    got.append(clips[k])
                    added = True
                    if len(got) >= want:
                        break
            if not added:
                break
            k += 1
        return got

    train = take(train_seqs, target)
    val = take(sorted(val_seqs), max(10, target // 50))
    rng.shuffle(train)
    return train, val, n_total, n_bad, len(by_seq)


def main():
    rng = random.Random(SEED)
    all_train, all_val = [], []
    print(f"{'dataset':12s} {'total':>9s} {'seqs':>8s} {'sampled':>8s} {'val':>5s} {'dropped':>8s}")
    for name, target in TARGETS.items():
        train, val, n_total, n_bad, n_seq = sample_dataset(name, target, rng)
        all_train.extend(train)
        all_val.extend(val)
        print(f"{name:12s} {n_total:9d} {n_seq:8d} {len(train):8d} {len(val):5d} {n_bad:8d}", flush=True)

    # existence check
    print(f"\nChecking existence of {len(all_train)+len(all_val)} files ...", flush=True)
    pairs = all_train + all_val
    with ThreadPoolExecutor(32) as ex:
        exists = list(ex.map(lambda vp: os.path.exists(os.path.join(ROOT, vp[0])), pairs))
    n_missing = sum(1 for e in exists if not e)
    all_train = [vp for vp, e in zip(all_train, exists[:len(all_train)]) if e]
    all_val = [vp for vp, e in zip(all_val, exists[len(all_train):]) if e]

    for path, rows in ((OUT_TRAIN, all_train), (OUT_VAL, all_val)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "prompt"])
            w.writerows(rows)
    print(f"\ntrain={len(all_train)} -> {OUT_TRAIN}")
    print(f"val={len(all_val)} -> {OUT_VAL}")
    print(f"missing files dropped: {n_missing}")
    if n_missing > len(pairs) * 0.05:
        print("WARNING: >5% missing files, check dataset dirs!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
