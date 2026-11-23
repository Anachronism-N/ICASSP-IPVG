#!/usr/bin/env python3
"""
Generate smoke test metadata CSV for Wan2.2 training.

Uses the already-processed training data:
  - 5,000 samples from hoigen1m_processed (subsampled)
  - 476 samples from seamless_processed (all)

Outputs: training_metadata/smoke_combined_metadata.csv

Usage:
    python3 scripts/prepare_smoke_test.py
"""

import csv
import json
import os
import random
from pathlib import Path


PROJECT_ROOT = Path("/srv/wm_teacher")
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "wan22_training"
OUTPUT_DIR = PROJECT_ROOT / "training_metadata"
HOIGEN_MANIFEST = PROCESSED_DIR / "hoigen1m_processed" / "manifest.jsonl"
SEAMLESS_MANIFEST = PROCESSED_DIR / "seamless_processed" / "manifest.jsonl"
SMOKE_CSV = OUTPUT_DIR / "smoke_combined_metadata.csv"

HOIGEN_SMOKE_SAMPLES = 5000
SEAMLESS_SMOKE_SAMPLES = 476  # All of them
RANDOM_SEED = 42


def load_and_subsample_manifest(manifest_path: Path, max_samples: int) -> list[dict]:
    """Load a manifest JSONL and subsample randomly."""
    entries = []
    abs_path = str((PROCESSED_DIR / manifest_path.parent.name).resolve())

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # Keep only video and prompt fields
            video_path = entry.get("video_path", "")
            caption = entry.get("caption", "")
            if video_path and caption:
                entries.append({"video_abs": video_path, "prompt": caption})

    if len(entries) > max_samples:
        random.seed(RANDOM_SEED)
        entries = random.sample(entries, max_samples)

    return entries


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading HOIGen-1M manifest (106K entries)...")
    hoigen_entries = load_and_subsample_manifest(HOIGEN_MANIFEST, HOIGEN_SMOKE_SAMPLES)
    print(f"  Selected {len(hoigen_entries)} HOIGen samples")

    print("Loading Seamless Interaction manifest...")
    seamless_entries = load_and_subsample_manifest(SEAMLESS_MANIFEST, SEAMLESS_SMOKE_SAMPLES)
    print(f"  Selected {len(seamless_entries)} Seamless samples")

    all_entries = hoigen_entries + seamless_entries

    # For smoke test, we use hoigen1m_processed as the base path
    # and write all entries relative to it.
    # Since seamless videos are in a different directory, we use absolute paths.
    # (DiffSynth-Studio supports absolute paths when dataset_base_path is empty-ish)

    # Strategy: Use PROJECT_ROOT as dataset_base_path
    # Then all paths are relative to PROJECT_ROOT
    base_path = str(PROJECT_ROOT)
    rows = []

    for entry in all_entries:
        try:
            video_rel = os.path.relpath(entry["video_abs"], base_path)
        except ValueError:
            video_rel = entry["video_abs"]

        # Check if video file exists
        video_full = os.path.join(base_path, video_rel)
        if not os.path.exists(video_full):
            # Try the absolute path directly
            if os.path.exists(entry["video_abs"]):
                video_rel = entry["video_abs"]
            else:
                print(f"WARNING: Video not found: {entry['video_abs']}")
                continue

        rows.append({"video": video_rel, "prompt": entry["prompt"]})

    with open(SMOKE_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSmoke test metadata written to: {SMOKE_CSV}")
    print(f"Total samples: {len(rows)}")
    print(f"  HOIGen: {len(hoigen_entries)}")
    print(f"  Seamless: {min(len(seamless_entries), SEAMLESS_SMOKE_SAMPLES)}")


if __name__ == "__main__":
    main()
