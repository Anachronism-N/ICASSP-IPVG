#!/usr/bin/env python3
"""Audit archive central directories without extracting their payloads."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections import Counter
from pathlib import Path


def inspect_zip(path: Path) -> dict:
    row = {
        "path": str(path),
        "filename": path.name,
        "archive_bytes": path.stat().st_size,
        "status": "central_directory_ok",
        "members": 0,
        "video_members": 0,
        "uncompressed_bytes": 0,
    }
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            row["members"] = len(infos)
            row["video_members"] = sum(
                info.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
                for info in infos
            )
            row["uncompressed_bytes"] = sum(info.file_size for info in infos)
    except (OSError, zipfile.BadZipFile) as error:
        row["status"] = "invalid_zip"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    zip_paths = []
    incomplete_paths = []
    for root in args.roots:
        zip_paths.extend(path for path in root.rglob("*.zip") if path.is_file())
        incomplete_paths.extend(path for path in root.rglob("*.incomplete") if path.is_file())
    rows = [inspect_zip(path) for path in sorted(set(zip_paths))]
    filename_counts = Counter(row["filename"] for row in rows)
    for row in rows:
        row["duplicate_filename"] = filename_counts[row["filename"]] > 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(temporary, args.output)

    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "zip_files": len(rows),
        "status": dict(status_counts),
        "duplicate_files": sum(row["duplicate_filename"] for row in rows),
        "duplicate_names": sum(count > 1 for count in filename_counts.values()),
        "incomplete_files": len(incomplete_paths),
        "archive_bytes": sum(row["archive_bytes"] for row in rows),
        "video_members": sum(row["video_members"] for row in rows),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 1 if status_counts.get("invalid_zip") else 0


if __name__ == "__main__":
    raise SystemExit(main())
