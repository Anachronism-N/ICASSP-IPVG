#!/usr/bin/env python3
"""Build manifests for locally available archived video datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path


def write_rows(output: Path, rows) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, output)
    return count


def build_hoigen(args: argparse.Namespace) -> int:
    local_videos: dict[str, tuple[str, str]] = {}
    duplicates = set()
    archives = sorted(args.root.glob(args.archive_glob))
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if not member.lower().endswith(".mp4"):
                    continue
                filename = Path(member).name
                if filename in local_videos:
                    duplicates.add(filename)
                local_videos[filename] = (str(archive), member)
    if duplicates:
        raise RuntimeError(f"duplicate video names across archives: {len(duplicates)}")

    records: dict[str, dict] = {}
    with args.caption_csv.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            filename = Path(source["path"]).name
            location = local_videos.get(filename)
            if not location:
                continue
            caption = source.get("text", "").strip()
            if filename in records:
                variants = records[filename]["caption_variants"]
                if caption and caption not in variants:
                    variants.append(caption)
                continue
            archive_path, member = location
            match = re.match(r"^(.+?)-\d+:\d{2}:\d", filename)
            source_sequence_id = match.group(1) if match else Path(filename).stem
            records[filename] = {
                "sample_id": f"hoigen1m:{filename}",
                "source_dataset": "hoigen1m",
                "source_sequence_id": source_sequence_id,
                "archive_path": archive_path,
                "video_member": member,
                "video_path": None,
                "reference_image": None,
                "caption_detailed": caption,
                "caption_variants": [caption] if caption else [],
                "caption_i2v": None,
                "annotation_source": "dataset_caption",
                "fps": float(source["fps"]),
                "width": int(source["width"]),
                "height": int(source["height"]),
                "duration_sec": float(source["duration"]),
                "num_frames": int(source["num_frames"]),
                "split": "unassigned",
            }

    count = write_rows(args.output, (records[name] for name in sorted(records)))
    missing_captions = len(local_videos) - len(records)
    variant_rows = sum(len(record["caption_variants"]) for record in records.values())
    print(
        f"dataset=hoigen1m archives={len(archives)} local_videos={len(local_videos)} "
        f"manifest_rows={count} caption_variants={variant_rows} "
        f"videos_without_caption={missing_captions} output={args.output}"
    )
    return 1 if missing_captions else 0


def transcript_text(metadata: dict) -> str:
    return " ".join(
        item.get("transcript", "").strip()
        for item in metadata.get("metadata:transcript", [])
        if item.get("transcript", "").strip()
    )


def build_seamless(args: argparse.Namespace) -> int:
    archives = sorted(args.root.glob(args.archive_glob))

    def rows():
        for archive in archives:
            with tarfile.open(archive) as handle:
                members = {member.name: member for member in handle if member.isfile()}
                stems: dict[str, set[str]] = {}
                for name in members:
                    path = Path(name)
                    stems.setdefault(str(path.with_suffix("")), set()).add(path.suffix.lower())
                for stem, suffixes in sorted(stems.items()):
                    required = {".mp4", ".json", ".npz", ".wav"}
                    if not required.issubset(suffixes):
                        continue
                    json_member = members[f"{stem}.json"]
                    extracted = handle.extractfile(json_member)
                    if extracted is None:
                        continue
                    metadata = json.load(extracted)
                    yield {
                        "sample_id": f"seamless_interaction:{metadata.get('id', Path(stem).name)}",
                        "source_dataset": "seamless_interaction",
                        "archive_path": str(archive),
                        "video_member": f"{stem}.mp4",
                        "audio_member": f"{stem}.wav",
                        "motion_member": f"{stem}.npz",
                        "metadata_member": f"{stem}.json",
                        "video_path": None,
                        "audio_path": None,
                        "reference_image": None,
                        "transcript": transcript_text(metadata),
                        "transcript_segments": metadata.get("metadata:transcript", []),
                        "vad": metadata.get("metadata:vad", []),
                        "caption_i2v": None,
                        "annotation_source": "dataset_transcript_vad",
                        "split": "dev",
                    }

    count = write_rows(args.output, rows())
    print(f"dataset=seamless_interaction archives={len(archives)} manifest_rows={count} output={args.output}")
    return 0


def build_hoigen_smoke(args: argparse.Namespace) -> int:
    candidates = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            duration = float(row.get("duration_sec") or 0)
            if not args.min_duration <= duration <= args.max_duration:
                continue
            if not row.get("caption_detailed"):
                continue
            rank = hashlib.sha256(f"{args.seed}:{row['sample_id']}".encode()).hexdigest()
            candidates.append((rank, row))
    candidates.sort(key=lambda item: item[0])

    selected = []
    used_sources = set()
    for _, row in candidates:
        source = row.get("source_sequence_id") or row["sample_id"]
        if source in used_sources:
            continue
        selected.append(row)
        used_sources.add(source)
        if len(selected) == args.limit:
            break
    if len(selected) < args.limit:
        selected_ids = {row["sample_id"] for row in selected}
        for _, row in candidates:
            if row["sample_id"] in selected_ids:
                continue
            selected.append(row)
            if len(selected) == args.limit:
                break
    if len(selected) < args.limit:
        raise RuntimeError(f"only {len(selected)} eligible samples for requested limit {args.limit}")

    by_archive: dict[str, list[dict]] = {}
    for row in selected:
        by_archive.setdefault(row["archive_path"], []).append(row)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    for archive_path, rows in sorted(by_archive.items()):
        with zipfile.ZipFile(archive_path) as archive:
            for row in rows:
                member = row["video_member"]
                target = args.video_dir / Path(member).name
                info = archive.getinfo(member)
                if target.exists() and target.stat().st_size == info.file_size:
                    row["video_path"] = str(target)
                    continue
                temporary = target.with_suffix(target.suffix + ".part")
                with archive.open(info) as source, temporary.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"incomplete extraction for {member}")
                os.replace(temporary, target)
                row["video_path"] = str(target)

    count = write_rows(args.output, sorted(selected, key=lambda row: row["sample_id"]))
    print(
        f"dataset=hoigen1m_smoke rows={count} unique_sources={len(used_sources)} "
        f"video_dir={args.video_dir} output={args.output}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    hoigen = subparsers.add_parser("hoigen")
    hoigen.add_argument("--root", type=Path, required=True)
    hoigen.add_argument("--caption-csv", type=Path, required=True)
    hoigen.add_argument("--archive-glob", default="videos_part_*.zip")
    hoigen.add_argument("--output", type=Path, required=True)
    hoigen.set_defaults(func=build_hoigen)

    seamless = subparsers.add_parser("seamless")
    seamless.add_argument("--root", type=Path, required=True)
    seamless.add_argument("--archive-glob", default="*.tar")
    seamless.add_argument("--output", type=Path, required=True)
    seamless.set_defaults(func=build_seamless)

    smoke = subparsers.add_parser("hoigen-smoke")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--video-dir", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--limit", type=int, default=5000)
    smoke.add_argument("--seed", type=int, default=20260719)
    smoke.add_argument("--min-duration", type=float, default=2.0)
    smoke.add_argument("--max-duration", type=float, default=10.0)
    smoke.set_defaults(func=build_hoigen_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
