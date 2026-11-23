#!/usr/bin/env python3
"""
Process NoXi dataset videos into Wan2.2 training format.

NoXi is an egocentric dyadic interaction dataset. Each session has two
participants: "expert" (interviewer) and "novice" (interviewee). Both
have egocentric video recordings.

Input: zip files in train/val/test-base/test-additional splits
Output: 1280x704, 24fps, H.264 MP4 videos + manifest.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# === Configuration ===
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24
TARGET_PIX_FMT = "yuv420p"
TARGET_CODEC = "libx264"
TARGET_CRF = 18
TARGET_PRESET = "medium"

# Video files to look for inside each zip/session directory
VIDEO_PATTERNS = ["expert.video.mp4", "novice.video.mp4"]

# Annotation value encodings (from NoXi annotation CSV format: start;end;value;confidence)
GENDER_MAP = {
    "1": "female",
    "2": "male",
}
AGE_MAP = {
    "1": "18-25",
    "2": "18-25",
    "3": "26-35",
    "4": "36-45",
    "5": "46-55",
    "6": "56-65",
    "7": "66+",
}


def get_video_info(video_path):
    """Get video width, height, fps, duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split(",")
    if len(parts) < 3:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
        fps_str = parts[2]
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0
        else:
            fps = float(fps_str)
        duration = float(parts[3]) if len(parts) > 3 and parts[3] else None
        return {"width": width, "height": height, "fps": fps, "duration": duration}
    except (ValueError, IndexError):
        return None


def convert_video(input_path, output_path):
    """
    Convert video to Wan2.2 format:
    - Scale to 1280x704, padding with black bars to preserve aspect ratio
    - 24fps
    - H.264 encoding
    """
    filter_str = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={TARGET_FPS}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", TARGET_CODEC,
        "-crf", str(TARGET_CRF),
        "-preset", TARGET_PRESET,
        "-pix_fmt", TARGET_PIX_FMT,
        "-an",  # strip audio
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[-500:]}", file=sys.stderr)
        return False
    return True


def extract_annotation(zip_ref, session_id, role, annot_name):
    """Extract a small annotation file from zip."""
    path = f"{session_id}/{role}.{annot_name}"
    try:
        return zip_ref.read(path).decode("utf-8").strip()
    except (KeyError, UnicodeDecodeError):
        return None


def parse_annotation_csv(data):
    """Parse NoXi annotation CSV format: start;end;value;confidence"""
    if not data:
        return None
    parts = data.strip().split(";")
    if len(parts) >= 3:
        return parts[2].strip()
    return None


def read_gender_annotation(zip_ref, session_id, role):
    """Read gender from annotation CSV and map to label."""
    data = extract_annotation(zip_ref, session_id, role, "gender.annotation.csv")
    val = parse_annotation_csv(data)
    if val:
        return GENDER_MAP.get(val, val)
    return None


def read_age_annotation(zip_ref, session_id, role):
    """Read age group from annotation CSV and map to label."""
    data = extract_annotation(zip_ref, session_id, role, "age.annotation.csv")
    val = parse_annotation_csv(data)
    if val:
        return AGE_MAP.get(val, val)
    return None


def read_language_annotation(zip_ref, session_id):
    """Read language from annotation CSV."""
    try:
        data = zip_ref.read(f"{session_id}/language.annotation.csv").decode("utf-8").strip()
    except (KeyError, UnicodeDecodeError):
        return None
    return parse_annotation_csv(data)


def generate_caption(session_id, role, annotations):
    """Generate a template caption for the video."""
    lang = annotations.get("language", "unknown")
    gender = annotations.get("gender", "unknown")
    age = annotations.get("age")

    role_desc = {
        "expert": "interviewer",
        "novice": "interviewee",
    }

    parts = ["A person"]

    if gender != "unknown":
        parts = [f"A {gender} person"]

    if age is not None:
        parts.append(f"aged {age}")

    role_str = role_desc.get(role, role)
    parts.append(f"is participating as the {role_str}")
    parts.append("in a dyadic conversation or interview setting.")

    if lang != "unknown":
        parts.append(f"The conversation is conducted in {lang}.")

    parts.append(
        "The video captures an egocentric perspective of the interaction, "
        "showing the other participant from a first-person viewpoint."
    )

    return " ".join(parts)


def process_session(zip_path, split, output_dir):
    """
    Process one session zip file. Extract and convert both expert and novice videos.

    Returns list of manifest entries.
    """
    session_id = Path(zip_path).stem  # e.g., "001"
    entries = []

    print(f"Processing session {session_id} ({split})...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Collect annotations
        annotations = {}

        # Language (session-level)
        lang = read_language_annotation(zf, session_id)
        if lang:
            annotations["language"] = lang

        for role in ["expert", "novice"]:
            video_name = f"{role}.video.mp4"
            video_path_in_zip = f"{session_id}/{video_name}"

            if video_path_in_zip not in zf.namelist():
                print(f"  WARNING: {video_path_in_zip} not found in zip, skipping")
                continue

            # Read role-specific annotations
            role_annotations = dict(annotations)
            gender = read_gender_annotation(zf, session_id, role)
            if gender:
                role_annotations["gender"] = gender
            age = read_age_annotation(zf, session_id, role)
            if age:
                role_annotations["age"] = age

            # Extract video to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(zf.read(video_path_in_zip))
                tmp_path = tmp.name

            try:
                # Get source video info
                src_info = get_video_info(tmp_path)
                if src_info is None:
                    print(f"  ERROR: Cannot read video info for {video_path_in_zip}")
                    os.unlink(tmp_path)
                    continue

                print(f"  {role}: {src_info['width']}x{src_info['height']} @ {src_info['fps']:.2f}fps, "
                      f"{src_info.get('duration', '?'):.1f}s")

                # Convert
                output_name = f"{session_id}_{role}.mp4"
                output_path = os.path.join(output_dir, output_name)
                if not convert_video(tmp_path, output_path):
                    print(f"  ERROR: Conversion failed for {video_path_in_zip}")
                    os.unlink(tmp_path)
                    continue

                # Get output video info
                out_info = get_video_info(output_path)
                output_size = os.path.getsize(output_path)

                # Generate caption
                caption = generate_caption(session_id, role, role_annotations)

                entry = {
                    "video": os.path.abspath(output_path),
                    "caption": caption,
                    "session_id": session_id,
                    "role": role,
                    "split": split,
                    "source_width": src_info["width"],
                    "source_height": src_info["height"],
                    "source_fps": round(src_info["fps"], 2),
                    "target_width": out_info["width"] if out_info else TARGET_WIDTH,
                    "target_height": out_info["height"] if out_info else TARGET_HEIGHT,
                    "target_fps": out_info["fps"] if out_info else TARGET_FPS,
                    "duration_seconds": out_info.get("duration") if out_info else None,
                    "file_size_bytes": output_size,
                    "annotations": role_annotations,
                }
                entries.append(entry)
                print(f"  -> {output_name} ({output_size / 1024 / 1024:.1f} MB)")

            finally:
                os.unlink(tmp_path)

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Process NoXi dataset into Wan2.2 training format"
    )
    parser.add_argument(
        "--input-dir",
        default="/srv/wm_teacher/dataset/egocentric_interaction/noxi",
        help="Path to NoXi dataset root",
    )
    parser.add_argument(
        "--output-dir",
        default="/srv/wm_teacher/dataset/wan22_training/noxi_processed",
        help="Output directory for converted videos and manifest",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test-base", "test-additional"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="Max sessions to process per split (0 = all)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    all_entries = []
    total_sessions = 0
    total_videos = 0
    total_errors = 0

    for split in args.splits:
        split_dir = os.path.join(input_dir, split)
        if not os.path.isdir(split_dir):
            print(f"WARNING: Split directory not found: {split_dir}")
            continue

        zip_files = sorted([
            f for f in os.listdir(split_dir)
            if f.endswith(".zip")
        ])

        if args.max_sessions > 0:
            zip_files = zip_files[:args.max_sessions]

        print(f"\n{'='*60}")
        print(f"Split: {split} ({len(zip_files)} sessions)")
        print(f"{'='*60}")

        for zip_file in zip_files:
            zip_path = os.path.join(split_dir, zip_file)
            try:
                entries = process_session(zip_path, split, output_dir)
                all_entries.extend(entries)
                total_sessions += 1
                total_videos += len(entries)
            except Exception as e:
                print(f"  ERROR processing {zip_file}: {e}")
                total_errors += 1

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    # Summary
    total_size = sum(e["file_size_bytes"] for e in all_entries)
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Sessions processed: {total_sessions}")
    print(f"Videos converted:  {total_videos}")
    print(f"Errors:            {total_errors}")
    print(f"Total output size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Manifest written:  {manifest_path}")

    # Per-split breakdown
    print(f"\nPer-split breakdown:")
    for split in args.splits:
        split_entries = [e for e in all_entries if e["split"] == split]
        if split_entries:
            split_size = sum(e["file_size_bytes"] for e in split_entries)
            print(f"  {split}: {len(split_entries)} videos, {split_size / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
