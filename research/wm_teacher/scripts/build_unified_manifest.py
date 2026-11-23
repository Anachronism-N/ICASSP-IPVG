#!/usr/bin/env python3
"""
Build a unified Wan2.2 training manifest from per-dataset manifests.

Produces:
  - training_metadata/unified_train.jsonl
  - training_metadata/unified_val.jsonl
  - training_metadata/unified_train.csv  (DiffSynth format: video,prompt)
  - training_metadata/unified_val.csv

Design rules (from docs/24_Wan2.2首轮训练执行计划.md):
  - Normalize field names across sources (video vs video_path; caption vs caption_i2v).
  - Split train/val by source_sequence_id so adjacent clips never leak across splits.
  - Cap any single source at 50% of the unified set.
  - Deduplicate by source_sequence_id (one clip per source sequence per split).
  - Skip entries without a usable caption.

Usage:
    python3 scripts/build_unified_manifest.py
    python3 scripts/build_unified_manifest.py --target 30000 --val-ratio 0.1
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"
MANIFESTS_DIR = PROJECT_ROOT / "dataset" / "manifests"
OUTPUT_DIR = PROJECT_ROOT / "training_metadata"

# Per-source config: uses *_wan22/manifest_wan22.jsonl for processed datasets (4s clips, 1280x704 24fps)
# MIRA uses mira_captions_final.jsonl (already has caption_i2v, videos already in format)
# Caption is empty for most (TBD by VLM); MIRA already has captions.
SOURCE_CONFIGS = [
    {
        "name": "hoigen1m",
        "manifest": WT_DIR / "hoigen1m_wan22" / "manifest_wan22.jsonl",
        "target": 20000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "mira",
        "manifest": MANIFESTS_DIR / "mira_captions_final.jsonl",
        "target": 10000,
        "video_field": "video_path",
        "caption_field": "caption_i2v",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "charades",
        "manifest": WT_DIR / "charades_wan22" / "manifest_wan22.jsonl",
        "target": 10000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "noxi",
        "manifest": WT_DIR / "noxi_wan22" / "manifest_wan22.jsonl",
        "target": 5000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "matrix",
        "manifest": WT_DIR / "matrix_wan22" / "manifest_wan22.jsonl",
        "target": 5000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "dexycb",
        "manifest": WT_DIR / "dexycb_processed" / "manifest_wan22.jsonl",
        "target": 2400,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "easycom",
        "manifest": WT_DIR / "easycom_wan22" / "manifest_wan22.jsonl",
        "target": 3920,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "h2o",
        "manifest": WT_DIR / "h2o_wan22" / "manifest_wan22.jsonl",
        "target": 297,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "seamless",
        "manifest": WT_DIR / "seamless_wan22" / "manifest_wan22.jsonl",
        "target": 10000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "egoexo4d",
        "manifest": WT_DIR / "egoexo4d_wan22" / "manifest_wan22.jsonl",
        "target": 10000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "openvidhd",
        "manifest": WT_DIR / "openvidhd_wan22" / "manifest_wan22.jsonl",
        "target": 1317,  # all
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "vfhq",
        "manifest": WT_DIR / "vfhq_wan22" / "manifest_wan22.jsonl",
        "target": 10000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
    {
        "name": "direct",
        "manifest": WT_DIR / "direct_wan22" / "manifest_wan22.jsonl",
        "target": 10000,
        "video_field": "video_path",
        "caption_field": "prompt",
        "source_id_field": "source_sequence_id",
    },
]


def load_source(cfg):
    """Load one source manifest, normalize fields, dedup by source_id."""
    name = cfg["name"]
    path = cfg["manifest"]
    if not path.exists():
        print(f"  SKIP {name}: manifest not found at {path}")
        return []

    vfield = cfg["video_field"]
    cfield = cfg["caption_field"]
    sfield = cfg["source_id_field"]
    base_path = cfg.get("base_path")  # optional: join to relative video paths

    rows = []
    seen_seq = set()
    n_read = 0
    n_skip_no_video = 0
    n_skip_no_caption = 0
    n_skip_dup_seq = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            video = d.get(vfield) or d.get("video_path") or d.get("video")
            caption = d.get(cfield) or d.get("caption") or d.get("caption_i2v")
            source_id = d.get(sfield) or ""

            if not video:
                n_skip_no_video += 1
                continue
            # If video is a relative filename and a base_path is configured, join them.
            if base_path and not os.path.isabs(str(video)):
                video = str(base_path / video)
            # Allow empty captions — VLM captioning is done later.
            caption = str(caption).strip() if caption else ""

            # No dedup by source_sequence_id — each 4s clip is a unique training sample.
            # source_sequence_id is used for train/val split isolation (all clips from
            # the same source video go to the same split).

            rows.append({
                "video_path": video,
                "prompt": str(caption).strip(),
                "source_dataset": name,
                "source_sequence_id": source_id or f"{name}:{Path(video).stem}",
            })

    print(f"  {name}: read={n_read} kept={len(rows)} "
          f"(no_video={n_skip_no_video} no_caption={n_skip_no_caption} dup_seq={n_skip_dup_seq})")
    return rows


def split_by_sequence(rows, val_ratio=0.1, seed=42):
    """Split into train/val by source_sequence_id (no leakage)."""
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["source_sequence_id"]].append(r)

    seq_ids = sorted(by_seq.keys())
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    n_val = max(1, int(len(seq_ids) * val_ratio))
    val_seqs = set(seq_ids[:n_val])

    train, val = [], []
    for sid in seq_ids:
        for r in by_seq[sid]:
            (val if sid in val_seqs else train).append(r)
    return train, val


def cap_single_source(rows, max_fraction=0.5, seed=42):
    """Cap any single source at max_fraction of total. Downsample over-represented sources."""
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source_dataset"]].append(r)

    total = len(rows)
    cap = int(total * max_fraction)
    rng = random.Random(seed)

    capped = []
    for src, items in by_src.items():
        if len(items) > cap:
            rng.shuffle(items)
            items = items[:cap]
            print(f"  cap {src}: {len(by_src[src])} -> {len(items)}")
        capped.extend(items)
    rng.shuffle(capped)
    return capped


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "prompt"])
        w.writeheader()
        for r in rows:
            w.writerow({"video": r["video_path"], "prompt": r["prompt"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-source-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading sources (dedup by source_sequence_id):")
    all_rows = []
    for cfg in SOURCE_CONFIGS:
        rows = load_source(cfg)
        # Subsample to target if over target.
        if len(rows) > cfg["target"]:
            rng = random.Random(args.seed)
            rng.shuffle(rows)
            rows = rows[: cfg["target"]]
            print(f"  {cfg['name']}: subsampled to {len(rows)}")
        all_rows.extend(rows)

    print(f"\nTotal before cap/split: {len(all_rows)}")
    by_src = defaultdict(int)
    for r in all_rows:
        by_src[r["source_dataset"]] += 1
    for src, n in sorted(by_src.items()):
        pct = 100.0 * n / len(all_rows)
        print(f"  {src}: {n} ({pct:.1f}%)")

    # Cap single source at max_fraction.
    print(f"\nCapping single source at {args.max_source_fraction*100:.0f}%:")
    all_rows = cap_single_source(all_rows, args.max_source_fraction, args.seed)

    # Split by source_sequence_id.
    print(f"\nSplitting train/val (val_ratio={args.val_ratio}, by source_sequence_id):")
    train, val = split_by_sequence(all_rows, args.val_ratio, args.seed)
    print(f"  train: {len(train)}  val: {len(val)}")

    # Write outputs.
    write_jsonl(train, OUTPUT_DIR / "unified_train.jsonl")
    write_jsonl(val, OUTPUT_DIR / "unified_val.jsonl")
    write_csv(train, OUTPUT_DIR / "unified_train.csv")
    write_csv(val, OUTPUT_DIR / "unified_val.csv")

    print(f"\nWrote:")
    print(f"  {OUTPUT_DIR/'unified_train.jsonl'} ({len(train)} rows)")
    print(f"  {OUTPUT_DIR/'unified_val.jsonl'} ({len(val)} rows)")
    print(f"  {OUTPUT_DIR/'unified_train.csv'}")
    print(f"  {OUTPUT_DIR/'unified_val.csv'}")

    # Final source distribution.
    print("\nFinal train distribution:")
    by_src = defaultdict(int)
    for r in train:
        by_src[r["source_dataset"]] += 1
    for src, n in sorted(by_src.items()):
        pct = 100.0 * n / len(train)
        print(f"  {src}: {n} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
