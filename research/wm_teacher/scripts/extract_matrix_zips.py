#!/usr/bin/env python3
"""
Extract videos from Matrix zips and prepare for transcoding.

Extracts MP4 videos from a subset of Matrix zips to a flat directory,
then generates a manifest for the transcode_and_clip.py script.

Usage:
    python3 scripts/extract_matrix_zips.py --num-zips 10 --output-dir dataset/wan22_training/matrix_extracted
"""

import argparse
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")
MATRIX_DATA = PROJECT_ROOT / "dataset" / "general_action" / "matrix_dataset" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-zips", type=int, default=10, help="Number of zips to extract from")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dataset" / "wan22_training" / "matrix_extracted")
    ap.add_argument("--max-videos", type=int, default=0, help="Max videos to extract (0=all from selected zips)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find available zips (sorted, skip incomplete/corrupt)
    zips = sorted(MATRIX_DATA.glob("part_*.zip"))
    # Filter to valid zips (size > 1GB)
    valid_zips = [z for z in zips if z.stat().st_size > 1_000_000_000]
    print(f"Available zips: {len(valid_zips)} (of {len(zips)} total)")

    # Select subset
    selected = valid_zips[: args.num_zips]
    print(f"Selected {len(selected)} zips for extraction")

    # Extract videos
    manifest_entries = []
    total_extracted = 0
    for zi, zip_path in enumerate(selected, 1):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                video_members = [m for m in zf.namelist() if m.endswith(".mp4")]
                print(f"  [{zi}/{len(selected)}] {zip_path.name}: {len(video_members)} videos")

                for member in video_members:
                    if args.max_videos > 0 and total_extracted >= args.max_videos:
                        break

                    # Extract to flat directory
                    out_name = f"{zip_path.stem}_{Path(member).name}"
                    out_path = args.output_dir / out_name

                    if not out_path.exists():
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())

                    manifest_entries.append({
                        "video_path": str(out_path),
                        "source": "matrix",
                        "zip": zip_path.name,
                        "original_member": member,
                    })
                    total_extracted += 1

        except Exception as e:
            print(f"  [{zi}/{len(selected)}] {zip_path.name}: ERROR {e}")
            continue

        if args.max_videos > 0 and total_extracted >= args.max_videos:
            break

    # Write manifest
    manifest_path = args.output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nExtracted {total_extracted} videos -> {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
