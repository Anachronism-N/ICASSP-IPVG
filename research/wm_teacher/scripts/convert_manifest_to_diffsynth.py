#!/usr/bin/env python3
"""
Convert wan22_training manifest.jsonl to DiffSynth-Studio metadata CSV format.

DiffSynth-Studio expects a CSV with columns:
  - video: relative path from dataset_base_path to the video file
  - prompt: text caption

For I2V (Wan2.2-TI2V-5B), the first frame of the video is automatically used
as the input_image (controlled by --extra_inputs "input_image").

Usage:
    python3 scripts/convert_manifest_to_diffsynth.py \
        --manifest dataset/wan22_training/hoigen1m_processed/manifest.jsonl \
        --base-path dataset/wan22_training/hoigen1m_processed \
        --output training_metadata/hoigen1m_metadata.csv

    # Convert ALL datasets at once:
    python3 scripts/convert_manifest_to_diffsynth.py \
        --all --output-dir training_metadata/
"""

import argparse
import csv
import json
import os
from pathlib import Path


def convert_single(manifest_path: Path, base_path: Path, output_path: Path):
    """Convert a single manifest.jsonl to DiffSynth-Studio metadata CSV."""
    rows = []
    skipped = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            video_abs = entry.get("video_path", "")
            caption = entry.get("caption", "")

            if not video_abs or not caption:
                skipped += 1
                continue

            # Make video path relative to base_path
            try:
                video_rel = os.path.relpath(video_abs, str(base_path))
            except ValueError:
                # Different drives, use absolute
                video_rel = video_abs

            rows.append({"video": video_rel, "prompt": caption})

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Converted {manifest_path.name}: {len(rows)} rows → {output_path.name}")
    print(f"  Skipped: {skipped} (missing video or caption)")
    return len(rows)


def convert_all(data_root: Path, output_dir: Path):
    """Convert all processed dataset manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = data_root / "dataset" / "wan22_training"
    total = 0

    for d in sorted(processed_dir.glob("*_processed")):
        manifest = d / "manifest.jsonl"
        if not manifest.exists():
            print(f"Skipping {d.name}: no manifest.jsonl")
            continue

        dataset_name = d.name.replace("_processed", "")
        output_path = output_dir / f"{dataset_name}_metadata.csv"
        n = convert_single(manifest, d, output_path)
        total += n

    print(f"\nTotal: {total} training samples across all datasets")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Wan2.2 training manifest to DiffSynth-Studio metadata CSV"
    )
    parser.add_argument("--manifest", type=Path, help="Path to manifest.jsonl")
    parser.add_argument("--base-path", type=Path, help="Dataset base path (videos directory parent)")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--all", action="store_true", help="Convert all datasets")
    parser.add_argument("--output-dir", type=Path, default=Path("training_metadata"),
                        help="Output directory for all-datasets mode")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/srv/wm_teacher"),
                        help="Project root directory")

    args = parser.parse_args()

    if args.all:
        convert_all(args.data_root, args.output_dir)
    else:
        if not all([args.manifest, args.base_path, args.output]):
            parser.error("--manifest, --base-path, and --output are required (or use --all)")
        convert_single(args.manifest, args.base_path, args.output)


if __name__ == "__main__":
    main()
