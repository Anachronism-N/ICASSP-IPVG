#!/usr/bin/env python3
"""
Convert DexYCB image sequences to Wan2.2 format MP4 videos.

Structure: {subject}/{session}/{camera_id}/color_*.jpg
Output: {output_dir}/{subject}/{session}/{camera_id}.mp4

Skips sequences with fewer than 10 frames.
Generates manifest.jsonl with metadata.
"""

import os
import subprocess
import json
import argparse
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict
import time


def find_sessions(input_dir):
    """Walk through input_dir and find all {subject}/{session}/{camera_id} combinations."""
    work_items = []
    input_path = Path(input_dir)

    for subject_dir in sorted(input_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name

        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session = session_dir.name

            for camera_dir in sorted(session_dir.iterdir()):
                if not camera_dir.is_dir():
                    continue
                camera_id = camera_dir.name

                work_items.append({
                    'subject': subject,
                    'session': session,
                    'camera_id': camera_id,
                    'source_dir': str(camera_dir),
                })

    return work_items


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_camera(args):
    """Convert one camera's image sequence to MP4. Returns manifest entry or None."""
    item = args
    subject = item['subject']
    session = item['session']
    camera_id = item['camera_id']
    source_dir = item['source_dir']
    output_dir = item['output_dir']
    min_frames = item['min_frames']
    framerate_in = item['framerate_in']
    framerate_out = item['framerate_out']
    crf = item['crf']

    source_path = Path(source_dir).resolve()
    output_dir_abs = Path(output_dir).resolve()

    # Find all color_*.jpg files and sort them
    jpg_files = sorted(source_path.glob('color_*.jpg'))
    num_frames = len(jpg_files)

    if num_frames < min_frames:
        return {'status': 'skipped', 'reason': f'too few frames ({num_frames} < {min_frames})', 'num_frames': num_frames, **item}

    # Create output directory
    out_dir = output_dir_abs / subject / session
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / f'{camera_id}.mp4'

    # Skip if already exists
    if output_file.exists():
        return {
            'status': 'skipped',
            'reason': 'already exists',
            'video_path': str(output_file),
            'num_frames': num_frames,
            **item,
        }

    # ffmpeg command: use start_number + pattern to read color_*.jpg sequentially
    # Files are named color_000000.jpg, color_000001.jpg, etc.
    # ffmpeg will stop when it hits the first missing frame number
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(framerate_in),
        '-start_number', '0',
        '-i', 'color_%06d.jpg',
        '-vf', 'scale=1280:704:force_original_aspect_ratio=decrease,pad=1280:704:(ow-iw)/2:(oh-ih)/2',
        '-r', str(framerate_out),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', str(crf),
        '-an',
        '-loglevel', 'error',
        str(output_file),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(source_path),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes per video
        )
        if result.returncode == 0:
            return {
                'status': 'success',
                'video_path': str(output_file),
                'num_frames': num_frames,
                **item,
            }
        else:
            return {
                'status': 'error',
                'reason': result.stderr.strip() or 'Unknown ffmpeg error',
                **item,
            }
    except subprocess.TimeoutExpired:
        return {'status': 'error', 'reason': 'timeout', **item}
    except Exception as e:
        return {'status': 'error', 'reason': str(e), **item}


def convert_session(args):
    """Convert all cameras in one session. Returns list of manifest entries."""
    session_dir, output_dir, min_frames, framerate_in, framerate_out, crf = args

    session_path = Path(session_dir)
    results = []

    for camera_dir in sorted(session_path.iterdir()):
        if not camera_dir.is_dir():
            continue

        item = {
            'subject': session_path.parent.name,
            'session': session_path.name,
            'camera_id': camera_dir.name,
            'source_dir': str(camera_dir),
            'output_dir': output_dir,
            'min_frames': min_frames,
            'framerate_in': framerate_in,
            'framerate_out': framerate_out,
            'crf': crf,
        }

        result = convert_camera(item)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert DexYCB to Wan2.2 MP4')
    parser.add_argument('--input', '-i', required=True,
                        help='Input DexYCB directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for MP4 files')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of worker processes (default: 8)')
    parser.add_argument('--min-frames', type=int, default=10,
                        help='Minimum frames per sequence (default: 10)')
    parser.add_argument('--framerate-in', type=int, default=30,
                        help='Input framerate (default: 30)')
    parser.add_argument('--framerate-out', type=int, default=24,
                        help='Output framerate (default: 24)')
    parser.add_argument('--crf', type=int, default=23,
                        help='CRF quality (default: 23)')
    parser.add_argument('--manifest', type=str, default='manifest.jsonl',
                        help='Output manifest file name (default: manifest.jsonl)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List work items without converting')
    args = parser.parse_args()

    if not check_ffmpeg():
        print("ERROR: ffmpeg not found. Please install ffmpeg.")
        return 1

    # Collect all session directories
    input_path = Path(args.input)
    session_dirs = []
    for subject_dir in sorted(input_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_dirs.append(str(session_dir))

    print(f"Found {len(session_dirs)} sessions across {sum(1 for d in input_path.iterdir() if d.is_dir())} subjects")

    if args.dry_run:
        total_cameras = 0
        for sd in session_dirs:
            for cd in Path(sd).iterdir():
                if cd.is_dir():
                    total_cameras += 1
        print(f"Would process {total_cameras} camera directories")
        return 0

    # Prepare work items for multiprocessing - one per session
    work_items = [
        (sd, args.output, args.min_frames, args.framerate_in, args.framerate_out, args.crf)
        for sd in session_dirs
    ]

    print(f"Starting conversion with {args.workers} workers...")
    start_time = time.time()

    all_results = []
    completed = 0

    with Pool(processes=args.workers) as pool:
        for i, session_results in enumerate(pool.imap_unordered(convert_session, work_items)):
            all_results.extend(session_results)
            completed += 1

            # Count statuses
            statuses = defaultdict(int)
            for r in session_results:
                statuses[r['status']] += 1
            succ = statuses.get('success', 0)
            skip = statuses.get('skipped', 0)
            err = statuses.get('error', 0)
            print(f"[{completed}/{len(work_items)}] Sessions ({succ} succ, {skip} skip, {err} err)", flush=True)

    elapsed = time.time() - start_time

    # Write manifest
    manifest_path = Path(args.output) / args.manifest
    manifest_entries = []
    for r in all_results:
        if r['status'] == 'success':
            manifest_entries.append({
                'video_path': r['video_path'],
                'source': 'dexycb',
                'subject': r['subject'],
                'session': r['session'],
                'camera_id': r['camera_id'],
            })

    with open(manifest_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')

    # Summary
    statuses = defaultdict(int)
    total_frames = 0
    errors = []
    for r in all_results:
        statuses[r['status']] += 1
        if r['status'] == 'success':
            total_frames += r.get('num_frames', 0)
        if r['status'] == 'error':
            errors.append(r)

    print(f"\n{'='*60}")
    print(f"Conversion complete in {elapsed:.1f}s")
    print(f"  Success:  {statuses.get('success', 0)}")
    print(f"  Skipped:  {statuses.get('skipped', 0)}")
    print(f"  Errors:   {statuses.get('error', 0)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Manifest: {manifest_path}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e['subject']}/{e['session']}/{e['camera_id']}: {e.get('reason', 'unknown')}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return 0 if statuses.get('error', 0) == 0 else 1


if __name__ == '__main__':
    exit(main())
