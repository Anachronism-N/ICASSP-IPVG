#!/usr/bin/env python3
"""
Convert H2O dataset image sequences to Wan2.2 format MP4 videos.

Structure:
  tar.gz → {subject}/{session}/{seq}/cam4/rgb256/{frame}.jpg
  Output: {subject}_{session}_{seq}_cam4.mp4
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Config
FRAMERATE_IN = 30
FRAMERATE_OUT = 24
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
MIN_FRAMES = 10
FFMPEG = "ffmpeg"

H2O_DIR = "/srv/wm_teacher/dataset/egocentric_interaction/h2o"
OUTPUT_DIR = "/srv/wm_teacher/dataset/wan22_training/h2o_processed"
EXTRACT_ROOT = "/tmp/h2o_extract"

# Tar archives to process
TAR_FILES = [
    os.path.join(H2O_DIR, "subject1_v1_1.tar.gz"),
    os.path.join(H2O_DIR, "subject1_ego_v1_1.tar.gz"),
]


def run_cmd(cmd, timeout=600):
    """Run a command and return success, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", str(e)


def extract_images(tar_path, extract_root):
    """Extract only rgb256 image directories from a tar.gz file."""
    tar_name = os.path.basename(tar_path).replace(".tar.gz", "")
    extract_dir = os.path.join(extract_root, tar_name)
    os.makedirs(extract_dir, exist_ok=True)

    print(f"[extract] Extracting {tar_path} → {extract_dir}")

    # Use tar to extract, stripping components to get flat session/seq/cam structure
    cmd = [
        "tar", "-xzf", tar_path,
        "-C", extract_dir,
        "--wildcards", "*/rgb256/*.jpg",
        "--ignore-failed-read",
        "--warning=no-file-changed",
    ]
    success, stdout, stderr = run_cmd(cmd, timeout=7200)
    if not success:
        print(f"[extract] WARNING: tar completed with issues for {tar_path}")
        print(f"[extract] stderr: {stderr[:500]}")

    # Verify extraction
    jpg_count = sum(1 for _ in Path(extract_dir).rglob("*.jpg"))
    print(f"[extract] Extracted {jpg_count} jpg files from {tar_name}")

    return extract_dir, jpg_count


def find_video_sequences(base_dir):
    """Find all rgb256 directories that represent video sequences.
    
    Returns list of (name, source_dir) tuples.
    """
    sequences = []
    for rgb_dir in sorted(Path(base_dir).rglob("rgb256")):
        if rgb_dir.is_dir():
            jpg_files = sorted(rgb_dir.glob("*.jpg"))
            if len(jpg_files) >= MIN_FRAMES:
                # Build a name from the path: subject_session_seq_cam4
                # Path is like: extract_root/tar_name/subject/session/seq/cam4/rgb256
                parts = rgb_dir.parts
                # Find cam4's index and build name from the parts above it
                try:
                    cam4_idx = parts.index("cam4")
                    name_parts = parts[cam4_idx - 3:cam4_idx + 1]  # subject, session, seq, cam4
                    seq_name = "_".join(name_parts)
                except (ValueError, IndexError):
                    # Fallback: use the path relative to extract root
                    rel_path = rgb_dir.relative_to(base_dir.parent.parent)
                    seq_name = "_".join(rel_path.parts[:-2])  # remove rgb256
                
                sequences.append((seq_name, str(rgb_dir), len(jpg_files)))
    
    return sequences


def convert_sequence(args):
    """Convert one image sequence to MP4."""
    seq_name, source_dir, num_frames, output_dir = args

    output_path = os.path.join(output_dir, f"{seq_name}.mp4")

    # Skip if already exists
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size > 1000:
            print(f"[skip] {seq_name}: already exists ({size} bytes)")
            return seq_name, output_path, "skipped"

    print(f"[convert] {seq_name}: {num_frames} frames → {output_path}")

    # ffmpeg command with padding and scaling
    vf_filter = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )

    # Build file list using concat demuxer for correct frame ordering
    # Sort by frame number (files are like 000054.jpg)
    jpg_files = sorted(Path(source_dir).glob("*.jpg"))

    # Use pattern glob approach - create symlinks or use concat
    # For ffmpeg image sequence input with glob pattern
    cmd = [
        FFMPEG, "-y",
        "-framerate", str(FRAMERATE_IN),
        "-pattern_type", "glob",
        "-i", os.path.join(source_dir, "*.jpg"),
        "-vf", vf_filter,
        "-r", str(FRAMERATE_OUT),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        "-threads", "2",
        output_path,
    ]

    success, stdout, stderr = run_cmd(cmd, timeout=600)
    if success and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[done] {seq_name}: {size_mb:.1f} MB")
        return seq_name, output_path, "converted"
    else:
        error_msg = stderr[-200:] if stderr else "unknown"
        print(f"[error] {seq_name}: {error_msg}")
        return seq_name, output_path, f"error: {error_msg}"


def write_manifest(manifest_path, results):
    """Write manifest.jsonl with video_path and source."""
    entries = []
    for seq_name, output_path, status in results:
        if status in ("converted", "skipped"):
            entries.append({
                "video_path": output_path,
                "source": f"h2o/{seq_name}",
                "frames_source": seq_name,
            })

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[manifest] Written {len(entries)} entries to {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true", help="Skip tar extraction")
    parser.add_argument("--workers", type=int, default=16, help="Parallel ffmpeg workers")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(EXTRACT_ROOT, exist_ok=True)

    all_sequences = []

    if not args.skip_extract:
        for tar_path in TAR_FILES:
            if not os.path.exists(tar_path):
                print(f"[warn] tar not found: {tar_path}")
                continue
            
            extract_dir, jpg_count = extract_images(tar_path, EXTRACT_ROOT)
            if jpg_count == 0:
                print(f"[warn] No jpg files extracted from {tar_path}")
                continue

            sequences = find_video_sequences(extract_dir)
            print(f"[info] Found {len(sequences)} sequences in {os.path.basename(tar_path)}")
            for name, src, nframes in sequences:
                print(f"  - {name}: {nframes} frames")
            all_sequences.extend(sequences)
    else:
        # Search existing extraction
        for subdir in sorted(Path(EXTRACT_ROOT).iterdir()):
            if subdir.is_dir():
                sequences = find_video_sequences(subdir)
                all_sequences.extend(sequences)

    print(f"\n[info] Total sequences found: {len(all_sequences)}")
    if not all_sequences:
        print("[error] No sequences found!")
        sys.exit(1)

    # Convert in parallel
    tasks = [(name, src, nframes, args.output_dir) for name, src, nframes in all_sequences]
    results = []

    print(f"\n[convert] Starting conversion with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_sequence, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Summary
    converted = sum(1 for _, _, s in results if s == "converted")
    skipped = sum(1 for _, _, s in results if s == "skipped")
    errors = sum(1 for _, _, s in results if s.startswith("error"))
    print(f"\n[summary] Total: {len(results)} | Converted: {converted} | Skipped: {skipped} | Errors: {errors}")

    # Write manifest
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    write_manifest(manifest_path, results)

    # Report errors
    if errors > 0:
        print("\n[errors] Failed sequences:")
        for name, path, status in results:
            if status.startswith("error"):
                print(f"  {name}: {status}")

    print("\n[done] Conversion complete!")


if __name__ == "__main__":
    main()
