#!/usr/bin/env python3
"""
Generate a training manifest for a processed *_wan22 directory.

Scans a directory of Wan2.2-ready clips (1280x704 24fps 4s MP4) and produces
a JSONL manifest with:
  - video_path: absolute path to the clip
  - source_dataset: dataset name
  - source_sequence_id: derived from clip filename (original video stem)
  - prompt: empty string (to be filled by VLM captioning)
  - width, height, fps, duration_sec, num_frames

Usage:
    python3 scripts/generate_wan22_manifest.py --dir dataset/wan22_training/charades_wan22 --source charades
    python3 scripts/generate_wan22_manifest.py --all
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/srv/wm_teacher")


def probe_clip(path):
    """Quick ffprobe for a single clip."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        d = json.loads(r.stdout)
        s = d.get("streams", [{}])[0]
        fps_num, fps_den = (s.get("r_frame_rate") or "0/1").split("/")
        fps = round(float(fps_num) / float(fps_den), 2) if float(fps_den) else 0
        dur = float(d.get("format", {}).get("duration", 0))
        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": fps,
            "duration_sec": round(dur, 2),
            "num_frames": int(s.get("nb_frames", 0) or 0),
        }
    except Exception:
        return None


def derive_source_id(filename, source):
    """Derive a source_sequence_id from a clip filename.

    Clips are named: {original_video_stem}_{NNN}.mp4 or {original_video_stem}.mp4
    The source_sequence_id is the original video stem (without the _NNN suffix).
    """
    stem = Path(filename).stem
    # Remove trailing _NNN (3-digit segment index)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 3:
        return parts[0]
    return stem


def generate_for_dir(dir_path, source_name, output_manifest=None, skip_probe=True):
    """Generate manifest for a single *_wan22 directory.

    If skip_probe=True, assumes all clips are 1280x704 24fps 4s (the output format
    of transcode_and_clip.py). This is ~1000x faster than probing each clip.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"SKIP (not a dir): {dir_path}")
        return 0

    clips = sorted(dir_path.glob("*.mp4"))
    if not clips:
        print(f"SKIP (no mp4): {dir_path}")
        return 0

    if output_manifest is None:
        output_manifest = dir_path / "manifest_wan22.jsonl"

    total = len(clips)
    print(f"Generating manifest for {source_name}: {total} clips -> {output_manifest.name}")

    rows = []
    for i, clip in enumerate(clips, 1):
        if skip_probe:
            # Use known format values (output of transcode_and_clip.py)
            row = {
                "video_path": str(clip),
                "source_dataset": source_name,
                "source_sequence_id": f"{source_name}:{derive_source_id(clip.name, source_name)}",
                "prompt": "",  # TBD — VLM captioning later
                "width": 1280,
                "height": 704,
                "fps": 24,
                "duration_sec": 4.0,
                "num_frames": 96,
            }
        else:
            info = probe_clip(str(clip))
            if info is None:
                continue
            row = {
                "video_path": str(clip),
                "source_dataset": source_name,
                "source_sequence_id": f"{source_name}:{derive_source_id(clip.name, source_name)}",
                "prompt": "",
                "width": info["width"],
                "height": info["height"],
                "fps": info["fps"],
                "duration_sec": info["duration_sec"],
                "num_frames": info["num_frames"],
            }
        rows.append(row)
        if i % 10000 == 0:
            print(f"  [{i}/{total}]")

    with open(output_manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Done: {len(rows)}/{total} clips -> {output_manifest}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate Wan2.2 manifest for processed clips")
    ap.add_argument("--dir", type=Path, help="Single directory to process")
    ap.add_argument("--source", type=str, help="Source dataset name")
    ap.add_argument("--all", action="store_true", help="Process all *_wan22 directories")
    args = ap.parse_args()

    if args.all:
        wt = PROJECT_ROOT / "dataset" / "wan22_training"
        total = 0
        for d in sorted(wt.glob("*_wan22")):
            source = d.name.replace("_wan22", "")
            # Also include already-ready datasets (MIRA, DexYCB) by pointing to their existing manifests
            n = generate_for_dir(d, source)
            total += n
        print(f"\nTotal: {total} clips across all *_wan22 directories")
    elif args.dir and args.source:
        generate_for_dir(args.dir, args.source)
    else:
        ap.error("Use --all or provide --dir and --source")


if __name__ == "__main__":
    main()
