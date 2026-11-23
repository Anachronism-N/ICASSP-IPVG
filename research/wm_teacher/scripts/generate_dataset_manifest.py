#!/usr/bin/env python3
"""
Generate a manifest jsonl listing all video clips in a wan22 dataset directory.
Used for stage 2 full-dataset captioning (recaption_unified.py reads this).

Output format: one json per line with video_path, source_dataset, source_sequence_id.

Usage:
    python3 scripts/generate_dataset_manifest.py --dataset noxi
    python3 scripts/generate_dataset_manifest.py --dataset charades
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"

DATASET_DIRS = {
    "noxi": (WT_DIR / "noxi_wan22", "noxi"),
    "charades": (WT_DIR / "charades_wan22", "charades"),
    "easycom": (WT_DIR / "easycom_wan22", "easycom"),
    "mira": (WT_DIR / "mira_processed", "mira"),
}


def generate_manifest(ds_name, output_path):
    video_dir, ds_label = DATASET_DIRS[ds_name]
    if not video_dir.exists():
        print(f"ERROR: {video_dir} not found")
        return 0

    print(f"Scanning {video_dir} for .mp4 files ...")
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(video_dir):
            for fname in sorted(files):
                if not fname.endswith(".mp4"):
                    continue
                fpath = os.path.join(root, fname)
                # Store relative to PROJECT_ROOT for consistency with unified manifest
                rel = os.path.relpath(fpath, PROJECT_ROOT)
                # Derive source_sequence_id from filename stem
                stem = Path(fname).stem
                # Strip clip suffix (_NNN) for source_sequence_id
                parts = stem.rsplit("_", 1)
                if parts[0] and parts[1].isdigit():
                    seq_id = f"{ds_label}:{parts[0]}"
                else:
                    seq_id = f"{ds_label}:{stem}"
                f.write(json.dumps({
                    "video_path": rel,
                    "source_dataset": ds_label,
                    "source_sequence_id": seq_id,
                }, ensure_ascii=False) + "\n")
                count += 1
                if count % 10000 == 0:
                    print(f"  {count} ...")
    print(f"Total: {count} clips -> {output_path}")
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    ap.add_argument("--output", default=None, help="Output manifest path")
    args = ap.parse_args()

    out = args.output or str(PROJECT_ROOT / "training_metadata" / f"manifest_{args.dataset}_full.jsonl")
    generate_manifest(args.dataset, out)


if __name__ == "__main__":
    main()
