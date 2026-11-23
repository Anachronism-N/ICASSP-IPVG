#!/usr/bin/env python3
"""Build and execute resumable download queues from Project Aria URL exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not value:
        raise ValueError("empty path component")
    return value


def dataset_name(path: Path) -> str:
    suffix = "_download_urls"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def load_export(path: Path, artifact: str) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = []
    for sequence, assets in payload.get("sequences", {}).items():
        asset = assets.get(artifact)
        if not asset or not asset.get("download_url"):
            continue
        rows.append(
            {
                "dataset": dataset_name(path),
                "sequence": sequence,
                "artifact": artifact,
                "filename": asset["filename"],
                "url": asset["download_url"],
                "expected_bytes": int(asset.get("file_size_bytes") or 0),
                "sha1": asset.get("sha1sum"),
                "source_export": str(path),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def destination(root: Path, row: dict) -> Path:
    return (
        root
        / safe_component(row["dataset"])
        / safe_component(row["sequence"])
        / safe_component(row["filename"])
    )


def sha1sum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path, row: dict, verify_hash: bool = False) -> str:
    if not path.exists():
        return "partial" if path.with_suffix(path.suffix + ".part").exists() else "missing"
    expected = row.get("expected_bytes") or 0
    if expected and path.stat().st_size != expected:
        return "size_mismatch"
    if verify_hash and row.get("sha1") and sha1sum(path) != row["sha1"]:
        return "hash_mismatch"
    return "complete"


def append_status(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def download_one(row: dict, root: Path, timeout: int, retries: int) -> dict:
    target = destination(root, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = file_state(target, row, verify_hash=False)
    if state == "complete":
        return {"status": "skipped", "path": str(target), "bytes": target.stat().st_size}

    partial = target.with_suffix(target.suffix + ".part")
    error_message = "unknown error"
    # `retries` counts retries after the initial request, so zero still performs
    # one network attempt.
    for attempt in range(1, retries + 2):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "wm-dataset-downloader/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(row["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_code = getattr(response, "status", response.getcode())
                if offset and response_code != 206:
                    offset = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=CHUNK_SIZE)

            expected = row.get("expected_bytes") or 0
            if expected and partial.stat().st_size != expected:
                raise IOError(
                    f"size mismatch: received {partial.stat().st_size}, expected {expected}"
                )
            if row.get("sha1") and sha1sum(partial) != row["sha1"]:
                raise IOError("SHA1 mismatch")
            os.replace(partial, target)
            return {"status": "complete", "path": str(target), "bytes": target.stat().st_size}
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            error_message = f"{type(error).__name__}: {error}"
            if attempt <= retries:
                time.sleep(min(2**attempt, 30))
    return {"status": "failed", "path": str(target), "error": error_message}


def command_plan(args: argparse.Namespace) -> int:
    rows = []
    for input_path in args.inputs:
        source_rows = load_export(input_path, args.artifact)
        if args.limit_per_dataset:
            source_rows = source_rows[: args.limit_per_dataset]
        rows.extend(source_rows)
    rows.sort(key=lambda row: (row["dataset"], row["sequence"], row["filename"]))
    write_jsonl(args.output, rows)
    total_bytes = sum(row["expected_bytes"] for row in rows)
    print(f"queue={args.output} files={len(rows)} expected_bytes={total_bytes}")
    return 0


def command_audit(args: argparse.Namespace) -> int:
    counts: dict[str, int] = {}
    bytes_by_state: dict[str, int] = {}
    for row in read_jsonl(args.queue):
        path = destination(args.output_root, row)
        state = file_state(path, row, verify_hash=args.verify_hash)
        counts[state] = counts.get(state, 0) + 1
        bytes_by_state[state] = bytes_by_state.get(state, 0) + int(
            row.get("expected_bytes") or 0
        )
    print(json.dumps({"files": counts, "expected_bytes": bytes_by_state}, indent=2))
    return 0 if not any(key.endswith("mismatch") for key in counts) else 1


def command_download(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.queue)
    if args.max_items:
        rows = rows[: args.max_items]
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, args.output_root, args.timeout, args.retries): row
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            result = future.result()
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "dataset": row["dataset"],
                "sequence": row["sequence"],
                "filename": row["filename"],
                **result,
            }
            append_status(args.status_log, record)
            failures += result["status"] == "failed"
            print(
                f"[{completed}/{len(rows)}] {result['status']} "
                f"{row['dataset']}/{row['sequence']}/{row['filename']}"
            )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Build a queue from URL export JSON files")
    plan.add_argument("inputs", type=Path, nargs="+")
    plan.add_argument("--artifact", default="video_main_rgb")
    plan.add_argument("--limit-per-dataset", type=int)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(func=command_plan)

    audit = subparsers.add_parser("audit", help="Compare a queue with local files")
    audit.add_argument("--queue", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--verify-hash", action="store_true")
    audit.set_defaults(func=command_audit)

    download = subparsers.add_parser("download", help="Download queue entries with resume support")
    download.add_argument("--queue", type=Path, required=True)
    download.add_argument("--output-root", type=Path, required=True)
    download.add_argument("--status-log", type=Path, required=True)
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--retries", type=int, default=5)
    download.add_argument("--timeout", type=int, default=60)
    download.add_argument("--max-items", type=int)
    download.set_defaults(func=command_download)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
