#!/usr/bin/env python3
"""
Generate manifest.jsonl for OpenVidHD processed videos.

Scans `dataset/wan22_training/openvidhd_processed/*.mp4` and parses filenames
of the form `{youtube_id}_{clip_num}_{start}to{end}.mp4` to produce a manifest
compatible with the caption + unified-manifest pipeline.

Output: dataset/wan22_training/openvidhd_processed/manifest.jsonl

Usage:
    python3 scripts/generate_openvidhd_manifest.py
    python3 scripts/generate_openvidhd_manifest.py --dry-run
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")
VIDEO_DIR = PROJECT_ROOT / "dataset" / "wan22_training" / "openvidhd_processed"
OUTPUT = VIDEO_DIR / "manifest.jsonl"

# Two filename patterns coexist in openvidhd_processed/:
#   1. OpenVidHD YouTube clips: {youtube_id}_{clip_num}_{start}to{end}.mp4
#      youtube_id may contain `-` and `_`, so we anchor on the trailing two `_`-separated tokens.
#   2. MagicTime clips: magictime_{numeric_id}[_{clip_num}].mp4  (no time range)
YT_CLIP_RE = re.compile(r"^(?P<ytid>.+)_(?P<clip>\d+)_(?P<start>\d+)to(?P<end>\d+)\.mp4$")
MAGICTIME_RE = re.compile(r"^magictime_(?P<mid>\d+)(?P<suffix>(?:_\d+)*)\.mp4$")


def parse_filename(name):
    m = YT_CLIP_RE.match(name)
    if m:
        return {
            "youtube_id": m.group("ytid"),
            "clip_num": int(m.group("clip")),
            "time_start": int(m.group("start")),
            "time_end": int(m.group("end")),
        }
    m = MAGICTIME_RE.match(name)
    if m:
        # MagicTime clips: trailing _\d+ suffixes are clip/sub-clip indices.
        suffix = m.group("suffix")
        clip_num = 0
        if suffix:
            # Take the last numeric token as clip_num.
            parts = [int(p) for p in suffix.strip("_").split("_") if p]
            if parts:
                clip_num = parts[-1]
        return {
            "youtube_id": f"magictime_{m.group('mid')}",
            "clip_num": clip_num,
            "time_start": -1,
            "time_end": -1,
        }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print stats only, do not write")
    args = ap.parse_args()

    if not VIDEO_DIR.exists():
        raise SystemExit(f"Video dir not found: {VIDEO_DIR}")

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(videos)} MP4 files in {VIDEO_DIR}")

    rows = []
    n_fail = 0
    for vp in videos:
        parsed = parse_filename(vp.name)
        if not parsed:
            n_fail += 1
            print(f"  WARN: cannot parse {vp.name}")
            continue
        rows.append({
            "video_path": str(vp.resolve()),
            "source": "openvidhd",
            "youtube_id": parsed["youtube_id"],
            "clip_num": parsed["clip_num"],
            "time_start": parsed["time_start"],
            "time_end": parsed["time_end"],
        })

    print(f"Parsed: {len(rows)}, failed: {n_fail}")

    if args.dry_run:
        print("[--dry-run] not writing. Sample row:")
        if rows:
            print(f"  {json.dumps(rows[0], ensure_ascii=False)}")
        return

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
