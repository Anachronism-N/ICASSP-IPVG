#!/usr/bin/env python3
"""Read-only generation/resource monitoring with diagnostic snapshots and sample decoding.

Snapshots span start/end times, not a single coherent instant. Nonempty counts
and sample decodes never replace complete decoding and benchmark coverage gates.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

from score_merge_cpu import read_json, require, validate_decode, check_video

ROOT = Path(__file__).resolve().parents[1]
ERROR_MARKERS = ("] FAIL ", "Traceback (most recent call last)", "OutOfMemoryError", "[shard failed]")


def node_gpu(node, local_rank=0, run=subprocess.run):
    args = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"]
    host = node["ip"]
    require(re.fullmatch(r"[A-Za-z0-9_.:-]+", host) and not host.startswith("-"), "Invalid state SSH host")
    if node["rank"] != local_rank:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o",
                "StrictHostKeyChecking=accept-new", "-p", str(node.get("port", 36000)),
                node.get("user", "root") + "@" + host, *args]
    try:
        result = run(args, capture_output=True, text=True, timeout=20)
        return {"rank": node["rank"], "ip": host, "rc": result.returncode,
                "gpu": result.stdout.strip(), "error": result.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"rank": node["rank"], "ip": host, "error": "GPU query timeout"}
    except OSError as error:
        return {"rank": node["rank"], "ip": host, "error": type(error).__name__ + ": " + str(error)[-300:]}


def read_logs(paths, cursors, errors):
    """Read every appended byte, retaining partial lines and cumulative errors across polls."""
    for path in paths:
        stat = path.stat()
        cursor = cursors.setdefault(path.name, {"offset": 0, "inode": stat.st_ino,
                                               "partial": "", "partial_error": False, "sampling": False})
        if cursor["inode"] != stat.st_ino or stat.st_size < cursor["offset"]:
            cursor.update(offset=0, inode=stat.st_ino, partial="", partial_error=False)
        with path.open("rb") as handle:
            if cursor["offset"] and "tail" in cursor:
                handle.seek(max(0, cursor["offset"] - 64))
                if handle.read(min(64, cursor["offset"])).hex() != cursor["tail"]:
                    cursor.update(offset=0, partial="", partial_error=False)
            handle.seek(cursor["offset"])
            while chunk := handle.read(65536):
                text = cursor["partial"] + chunk.decode(errors="replace")
                lines = re.split(r"[\r\n]", text)
                for index, line in enumerate(lines[:-1]):
                    if any(marker in line for marker in ERROR_MARKERS) and not (index == 0 and cursor["partial_error"]):
                        errors.append({"log": path.name, "line": line[-400:]})
                cursor["sampling"] = cursor["sampling"] or "/50 [" in text
                cursor["partial_error"] = (len(lines) == 1 and cursor["partial_error"])
                cursor["partial"] = lines[-1]
                if any(marker in lines[-1] for marker in ERROR_MARKERS) and not cursor["partial_error"]:
                    errors.append({"log": path.name, "line": lines[-1][-400:]})
                    cursor["partial_error"] = True
            cursor["offset"] = handle.tell()
            handle.seek(max(0, cursor["offset"] - 64))
            cursor["tail"] = handle.read(min(64, cursor["offset"])).hex()
    return sum(cursors[path.name]["sampling"] for path in paths)


def snapshot(root, state, known, cursors, errors, sample_limit=4, local_rank=0,
             decode=check_video, query=node_gpu):
    started = time.strftime("%F %T %Z")
    with (root / "eval/jobs_merge_all.jsonl").open() as handle:
        jobs = [json.loads(line) for line in handle]
    groups = {}
    rows = []
    for job in jobs:
        groups.setdefault(job["model"], []).append(job)
        path = Path(job["out"])
        try:
            stat = path.stat()
        except FileNotFoundError:
            stat = None
        claim = Path(str(path) + ".claim")
        try:
            age = time.time() - claim.stat().st_mtime
        except FileNotFoundError:
            age = None
        rows.append((job, stat, age))
    fully_decoded = set()
    for model, model_jobs in groups.items():
        report = root / "eval/diag" / ("decode_" + model + "_20261018.json")
        if len(model_jobs) == len({job["out"] for job in model_jobs}) == 1011 and report.is_file():
            try:
                validate_decode(report, model, model_jobs)
                fully_decoded.add(model)
            except (RuntimeError, KeyError, TypeError, ValueError):
                pass
    complete = [(j, s) for j, s, _ in rows if s is not None and s.st_size > 0]
    by_model = {}
    for job, _ in complete:
        by_model[job["model"]] = by_model.get(job["model"], 0) + 1
    candidates = sorted(((j, s) for j, s in complete if j["out"] not in known and
                         j["model"] not in fully_decoded), key=lambda row: row[1].st_mtime_ns, reverse=True)
    decoded = []
    for job, _ in candidates[:sample_limit]:
        try:
            decode(Path(job["out"]))
            known.add(job["out"])
            decoded.append({"out": job["out"], "pass": True, "frames": 121})
        except Exception as error:
            decoded.append({"out": job["out"], "pass": False,
                            "error": type(error).__name__ + ":" + str(error)[:180]})
    tag = state["tag"]
    require(re.fullmatch(r"[A-Za-z0-9_-]+", tag), "Invalid generation log tag")
    logs = sorted((root / "dataset/download_logs").glob("gen_" + tag + "_s*.log"))
    sampling = read_logs(logs, cursors, errors)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        nodes = list(pool.map(lambda node: query(node, local_rank), state["nodes"]))
    finished = time.strftime("%F %T %Z")
    return {"at": finished, "snapshot_started_at": started, "snapshot_finished_at": finished,
            "tag": tag, "complete_nonempty": len(complete), "total": len(jobs), "by_model": by_model,
            "claims": sum(age is not None for _, _, age in rows),
            "stale_claims": sum(age is not None and age > 2400 for _, _, age in rows),
            "sample_decodes": decoded, "fully_decoded_models": sorted(fully_decoded),
            "checked_new_outputs": len(known), "generation_logs": len(logs), "sampling_seen_logs": sampling,
            "errors": list(errors), "nodes": nodes, "log_cursors": cursors, "sample_checked_outputs": sorted(known),
            "caveat": "Interval snapshot, not a coherent instant. Nonempty count/sample decoding only; complete output-matched decoding and benchmark coverage remain mandatory."}


def write_snapshot(root, report):
    directory = root / "eval/diag"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".merge_monitor-", delete=False) as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, directory / "merge_monitor_latest.json")
    with (directory / "merge_monitor_events.jsonl").open("a") as handle:
        json.dump(report, handle, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--state", type=Path, default=Path("eval/diag/merge_resume_20261018.json"))
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--cap-hours", type=float, default=72)
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    require(args.interval > 0 and args.cap_hours > 0 and args.sample_limit >= 0, "Invalid interval/cap/sample limit")
    root = args.root.resolve()
    state_path = args.state if args.state.is_absolute() else root / args.state
    state = read_json(state_path)
    latest = root / "eval/diag/merge_monitor_latest.json"
    previous = read_json(latest) if latest.exists() else {}
    previous = previous if previous.get("tag") == state["tag"] else {}
    known = set(previous.get("sample_checked_outputs", []))
    cursors, errors = previous.get("log_cursors", {}), previous.get("errors", [])
    started = time.monotonic()
    while time.monotonic() - started < args.cap_hours * 3600:
        report = snapshot(root, state, known, cursors, errors, args.sample_limit, args.local_rank)
        write_snapshot(root, report)
        print("[MONITOR]", report["at"], report["complete_nonempty"], "/", report["total"],
              "by_model", report["by_model"], "errors", len(report["errors"]), flush=True)
        if args.once:
            return
        time.sleep(min(args.interval, max(0, args.cap_hours * 3600 - (time.monotonic() - started))))
    print("MONITOR_CAP_REACHED", flush=True)


if __name__ == "__main__":
    main()
