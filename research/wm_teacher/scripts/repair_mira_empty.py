#!/usr/bin/env python3
"""
Repair empty MIRA 720p video files by re-extracting from source tars.

Scans mira_processed/ for empty (<1KB) .mp4 files, maps them back to their
source tar via mira_tar_index_train.json, extracts the original .mp4, and
transcodes to 720p using PyAV (no ffmpeg/ffprobe dependency).

Usage:
    python3 scripts/repair_mira_empty.py --dry-run   # preview only
    python3 scripts/repair_mira_empty.py              # actually repair
    python3 scripts/repair_mira_empty.py --workers 4  # parallel repair
"""
import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path("/srv/wm_teacher")
MIRA_PROCESSED = PROJECT_ROOT / "dataset" / "wan22_training" / "mira_processed"
TAR_INDEX_TRAIN = PROJECT_ROOT / "dataset" / "mira_tar_index_train.json"
TAR_INDEX_TEST = PROJECT_ROOT / "dataset" / "mira_tar_index_test.json"

# PyAV for transcoding (avoids ffmpeg/ffprobe dependency)
import av

TARGET_W, TARGET_H = 1280, 704
TARGET_FPS = 24


def find_empty_files(manifest_path, max_check=0):
    """Scan manifest for empty (<1KB) video files. Returns list of (video_path, source_seq_id)."""
    empty = []
    n = 0
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            vp = r["video_path"]
            if not os.path.isabs(vp):
                vp = str(PROJECT_ROOT / vp)
            try:
                if os.path.getsize(vp) < 1024:
                    empty.append((vp, r.get("source_sequence_id", "")))
            except OSError:
                empty.append((vp, r.get("source_sequence_id", "")))
            n += 1
            if max_check and n >= max_check:
                break
    return empty


def transcode_to_720p(src_path, dst_path):
    """Transcode video to 720p (1280x704) 24fps using PyAV."""
    try:
        inp = av.open(src_path)
        out = av.open(dst_path, 'w')
        in_stream = inp.streams.video[0]

        out_stream = out.add_stream('libx264', rate=TARGET_FPS)
        out_stream.width = TARGET_W
        out_stream.height = TARGET_H
        out_stream.pix_fmt = 'yuv420p'
        out_stream.options = {'crf': '23', 'preset': 'fast'}

        for frame in inp.decode(video=0):
            # Resize frame to target resolution
            frame = frame.reformat(width=TARGET_W, height=TARGET_H, format='yuv420p')
            out_stream.encode(frame)

        # Flush encoder
        for packet in out_stream.encode():
            out.mux(packet)

        inp.close()
        out.close()
        return True
    except Exception as e:
        print(f"  transcode failed: {e}", file=sys.stderr)
        return False


def repair_one(video_path, source_seq_id, tar_indices, dry_run=False):
    """Repair one empty file by extracting from tar and transcoding."""
    # source_seq_id format: "mira:2026-05-08T21-52-24Z-9fda7f_c00070.p1_720p"
    # Strip "mira:" prefix
    if source_seq_id.startswith("mira:"):
        source_seq_id = source_seq_id[5:]

    # Extract timestamp: 2026-05-08T21-52-24Z-9fda7f
    # and clip name: c00070.p1
    parts = source_seq_id.split("_")
    if len(parts) < 2:
        return ("fail", "bad source_seq_id format")

    # Timestamp is everything before the first "_c"
    ts_end = source_seq_id.find("_c")
    if ts_end < 0:
        return ("fail", "no _c in source_seq_id")
    timestamp = source_seq_id[:ts_end]

    # Find the tar file
    tar_path = None
    for idx in tar_indices:
        if timestamp in idx:
            tar_path = idx[timestamp]
            break

    if not tar_path or not os.path.exists(tar_path):
        return ("fail", f"tar not found for {timestamp}")

    # Original filename in tar: {source_seq_id without _720p}.mp4
    # source_seq_id: 2026-05-08T21-52-24Z-9fda7f_c00070.p1_720p
    # original:      2026-05-08T21-52-24Z-9fda7f_c00070.p1.mp4
    original_name = source_seq_id.replace("_720p", "") + ".mp4"

    if dry_run:
        return ("dry", f"would extract {original_name} from {Path(tar_path).name}")

    try:
        # Extract original from tar
        with tarfile.open(tar_path, 'r') as tar:
            try:
                member = tar.getmember(original_name)
            except KeyError:
                return ("fail", f"{original_name} not in tar")
            extracted = tar.extractfile(member)
            if not extracted:
                return ("fail", "extractfile returned None")

            # Write to temp file
            tmp_path = video_path + ".tmp"
            with open(tmp_path, 'wb') as f:
                f.write(extracted.read())

        # Transcode to 720p
        success = transcode_to_720p(tmp_path, video_path)
        os.unlink(tmp_path)

        if success and os.path.getsize(video_path) > 1024:
            return ("ok", f"repaired {Path(video_path).name}")
        else:
            return ("fail", "transcode failed")
    except Exception as e:
        return ("fail", str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Preview only, don't repair")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--max", type=int, default=0, help="Max files to repair (0=all)")
    args = ap.parse_args()

    # Load tar indices
    tar_indices = []
    for idx_path in [TAR_INDEX_TRAIN, TAR_INDEX_TEST]:
        if idx_path.exists():
            with open(idx_path) as f:
                tar_indices.append(json.load(f))
            print(f"Loaded {len(tar_indices[-1])} entries from {idx_path.name}")

    # Find empty files
    manifest = PROJECT_ROOT / "training_metadata" / "manifest_mira_full.jsonl"
    print(f"\nScanning {manifest} for empty files ...")
    empty_files = find_empty_files(manifest, max_check=args.max)
    print(f"Found {len(empty_files)} empty files")

    if not empty_files:
        print("Nothing to repair.")
        return

    if args.dry_run:
        print("\n=== Dry run (first 10) ===")
        for vp, seq in empty_files[:10]:
            status, msg = repair_one(vp, seq, tar_indices, dry_run=True)
            print(f"  [{status}] {Path(vp).name}: {msg}")
        return

    # Repair in parallel
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(repair_one, vp, seq, tar_indices): vp
                   for vp, seq in empty_files}
        from tqdm import tqdm
        with tqdm(total=len(empty_files), desc="repair", unit="file") as pbar:
            for fut in as_completed(futures):
                status, msg = fut.result()
                if status == "ok":
                    ok += 1
                else:
                    fail += 1
                    if fail <= 5 or fail % 50 == 0:
                        print(f"  FAIL: {msg}", file=sys.stderr)
                pbar.update(1)
                pbar.set_postfix(ok=ok, fail=fail)

    print(f"\nRepair complete: ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
