#!/usr/bin/env python3
"""
Transcode and clip videos into Wan2.2-TI2V training format.

Target format:
  - Landscape bucket: 1280x704, 24fps, h264, 8s segments
  - Portrait bucket:  704x1280, 24fps, h264, 8s segments

Auto-detects whether a video needs transcoding (wrong resolution/fps) and/or
clipping (duration > max_segment_sec). Videos already in target format are
passed through (copy) for clipping only.

Usage:
    # Experiment: process a few samples from a manifest
    python3 scripts/transcode_and_clip.py \
        --manifest dataset/wan22_training/charades_processed/manifest.jsonl \
        --video-field video --base-path dataset/wan22_training/charades_processed \
        --output-dir /tmp/transcode_test/charades --max-samples 3

    # Full run
    python3 scripts/transcode_and_clip.py \
        --manifest dataset/wan22_training/hoigen1m_processed/manifest.jsonl \
        --video-field video_path --output-dir dataset/wan22_training/hoigen1m_clipped
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TARGET_FPS = 24
TARGET_SEGMENT_SEC = 4  # 4s segments (96 frames); loader uses first 49 frames (~2s)
NO_CLIP_THRESHOLD = 5   # videos <= 5s are not clipped; > 5s are segmented into 4s clips
TARGET_CODEC = "libopenh264"
TARGET_BITRATE = "5M"  # libopenh264 has no CRF, use bitrate
LANDSCAPE_W, LANDSCAPE_H = 1280, 704
PORTRAIT_W, PORTRAIT_H = 704, 1280
MIN_DURATION = 2.0
KEYFRAME_INTERVAL = 48  # 2s at 24fps, for clean segment splits


def ffprobe_video(path):
    """Return dict with width, height, fps, duration, codec or None."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,codec_name",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=60,
        )
        d = json.loads(r.stdout)
        s = d.get("streams", [{}])[0]
        fps_num, fps_den = (s.get("r_frame_rate") or "0/1").split("/")
        fps = float(fps_num) / float(fps_den) if float(fps_den) else 0.0
        dur = float(d.get("format", {}).get("duration", 0))
        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": round(fps, 2),
            "duration": round(dur, 2),
            "codec": s.get("codec_name", ""),
        }
    except Exception:
        return None


def needs_transcode(info):
    """Check if video needs transcoding (resolution or fps mismatch)."""
    w, h = info["width"], info["height"]
    fps = info["fps"]

    # Check if already in target format
    if (w, h) == (LANDSCAPE_W, LANDSCAPE_H) and abs(fps - TARGET_FPS) < 0.5:
        return False
    if (w, h) == (PORTRAIT_W, PORTRAIT_H) and abs(fps - TARGET_FPS) < 0.5:
        return False
    return True


def is_portrait(info):
    """Determine if video should go to portrait bucket."""
    return info["height"] > info["width"]


def process_video(video_path, output_dir, source_name, idx, clip_captions=None):
    """Transcode and/or clip a single video. Returns list of output clip paths."""
    info = ffprobe_video(video_path)
    if info is None or info["duration"] < MIN_DURATION:
        print(f"  SKIP (probe fail or too short): {video_path}")
        return []

    transcode = needs_transcode(info)
    needs_clip = info["duration"] > NO_CLIP_THRESHOLD

    # Determine target bucket
    portrait = is_portrait(info)
    target_w, target_h = (PORTRAIT_W, PORTRAIT_H) if portrait else (LANDSCAPE_W, LANDSCAPE_H)

    stem = Path(video_path).stem
    clips = []

    if not transcode and not needs_clip:
        # Already in target format, just copy
        out_path = output_dir / f"{stem}.mp4"
        if not out_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", video_path,
                 "-c", "copy", "-an", str(out_path)],
                timeout=60, check=False,
            )
        if out_path.exists() and out_path.stat().st_size > 0:
            clips.append(str(out_path))
        return clips

    # Build ffmpeg command — always re-encode with libopenh264 for exact segment boundaries
    portrait = is_portrait(info)
    target_w, target_h = (PORTRAIT_W, PORTRAIT_H) if portrait else (LANDSCAPE_W, LANDSCAPE_H)
    vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
          f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2")
    codec_args = ["-c:v", TARGET_CODEC, "-b:v", TARGET_BITRATE,
                   "-r", str(TARGET_FPS), "-g", str(KEYFRAME_INTERVAL),
                   "-vf", vf, "-an"]

    if needs_clip:
        # Segment into TARGET_SEGMENT_SEC clips (re-encode for exact boundaries)
        out_pattern = output_dir / f"{stem}_%03d.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video_path] + codec_args + [
            "-f", "segment", "-segment_time", str(TARGET_SEGMENT_SEC),
            "-reset_timestamps", "1", str(out_pattern),
        ]
        subprocess.run(cmd, timeout=600, check=False)
        # Collect generated clips
        for i in range(200):  # max 200 clips per video
            clip_path = output_dir / f"{stem}_{i:03d}.mp4"
            if clip_path.exists() and clip_path.stat().st_size > 0:
                # Verify clip duration >= MIN_DURATION
                clip_info = ffprobe_video(str(clip_path))
                if clip_info and clip_info["duration"] >= MIN_DURATION:
                    clips.append(str(clip_path))
                else:
                    clip_path.unlink(missing_ok=True)  # remove too-short tail
            else:
                break
    else:
        # Transcode only, no clip
        out_path = output_dir / f"{stem}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", video_path] + codec_args + [str(out_path)]
        subprocess.run(cmd, timeout=300, check=False)
        if out_path.exists() and out_path.stat().st_size > 0:
            clips.append(str(out_path))

    return clips


def main():
    ap = argparse.ArgumentParser(description="Transcode and clip videos for Wan2.2-TI2V")
    ap.add_argument("--manifest", type=Path, required=True, help="Input manifest JSONL")
    ap.add_argument("--video-field", default="video_path", help="Field name for video path")
    ap.add_argument("--base-path", type=Path, help="Base path for relative video paths")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--max-samples", type=int, default=0, help="Max videos to process (0=all)")
    ap.add_argument("--source-name", default="", help="Source dataset name for logging")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest and resolve paths
    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    # Build work items: (video_path, output_dir, source_name, idx)
    # NOTE: skip os.path.exists check here — it's catastrophically slow on shared FS
    # with 100K+ files. Let the worker handle missing files.
    work = []
    for i, r in enumerate(rows):
        vp = r.get(args.video_field) or r.get("video_path") or r.get("video", "")
        if not vp:
            continue
        vp = str(vp)
        if not os.path.isabs(vp) and args.base_path:
            vp = str(args.base_path / vp)
        work.append((vp, str(args.output_dir), args.source_name, i))

    print(f"Processing {len(work)} videos -> {args.output_dir} ({args.workers} workers)")

    total_clips = 0
    if args.workers <= 1:
        # Sequential
        for vp, out_dir, src, idx in work:
            clips = process_video(vp, Path(out_dir), src, idx)
            total_clips += len(clips)
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(work)}] {total_clips} clips so far")
    else:
        # Parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import functools
        fn = functools.partial(_process_wrapper, output_dir=str(args.output_dir), source_name=args.source_name)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fn, vp, i): i for i, (vp, _, _, _) in enumerate(work)}
            done = 0
            for fut in as_completed(futures):
                try:
                    clips = fut.result()
                    total_clips += len(clips)
                except Exception:
                    pass
                done += 1
                if done % 50 == 0:
                    print(f"  [{done}/{len(work)}] {total_clips} clips so far")

    print(f"\nDone: {len(work)} videos -> {total_clips} clips in {args.output_dir}")


def _process_wrapper(video_path, idx, output_dir="", source_name=""):
    """Wrapper for multiprocessing."""
    return process_video(video_path, Path(output_dir), source_name, idx)


if __name__ == "__main__":
    main()
