#!/usr/bin/env python3
"""Read-only race audit. Never reconstructs a ledger or authorizes generation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def record(path):
    raw = path.read_bytes()
    stat = path.stat()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                mtime_ns=stat.st_mtime_ns)


def read_evidence(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        return dict(parse_error=str(exc))


def audit(plan_path, node_run_upper_seconds=None):
    plan_path = Path(plan_path).resolve()
    folder = plan_path.parent
    plan = json.loads(plan_path.read_bytes())
    digest = record(plan_path)["sha256"]
    ledger_path = folder / "allocations.json"
    raw = ledger_path.read_bytes()
    candidate, trailing, strict_error = None, b"", None
    try:
        candidate = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        strict_error = str(exc)
        try:
            text = raw.decode()
            candidate, end = json.JSONDecoder().raw_decode(text)
            trailing = text[end:].encode()
        except (ValueError, UnicodeDecodeError):
            trailing = raw
    rows = candidate.get("allocations", []) if isinstance(candidate, dict) else []
    bounds = node_run_upper_seconds or {}
    if any(type(rank) is not int or type(value) not in (int, float)
           or not math.isfinite(value) or value <= 0 for rank, value in bounds.items()):
        raise ValueError("Positive final node run duration bounds required; unknown is never zero")
    offsets = {"alg": {1: 0, 2: 8}, "dymos": {0: 0, 3: 8}}[plan["method"]]
    shards = []
    for shard in range(plan["nshards"]):
        rank = next(rank for rank, offset in offsets.items() if offset <= shard < offset + 8)
        evidence = {suffix: read_evidence(folder / f"shard{shard}.{suffix}.json")
                    for suffix in ("launch", "claim", "finished", "supervised")}
        launched = isinstance(evidence["launch"], dict) and "parse_error" not in evidence["launch"]
        receipt, actual = evidence["supervised"], None
        if (isinstance(receipt, dict) and receipt.get("schema") == "teacher_devset_supervised_receipt_v1"
                and receipt.get("nshards") == plan["nshards"]
                and receipt.get("cap_seconds") == plan["shard_cap_seconds"][shard]
                and receipt.get("plan_sha256") == digest
                and receipt.get("method") == plan["method"] and receipt.get("shard") == shard
                and receipt.get("allocated_gpus") == 1 and receipt.get("allocation_released") is True
                and receipt.get("owned_tree_released") is True):
            elapsed, start, end = (receipt.get(key) for key in ("elapsed_seconds", "start", "end"))
            if (all(type(value) in (int, float) and math.isfinite(value) for value in (elapsed, start, end))
                    and elapsed >= 0 and abs(end - start - elapsed) < 5):
                actual = elapsed
        upper = actual if actual is not None else bounds.get(rank)
        starts = [dict(source=suffix, start=value["start"])
                  for suffix, value in evidence.items() if isinstance(value, dict)
                  and value.get("plan_sha256") == digest and value.get("shard") == shard
                  and type(value.get("start")) in (int, float)]
        shards.append(dict(shard=shard, node_rank=rank, launch_present=launched, evidence=evidence,
                           observed_start_evidence=starts, actual_cost_unknown=actual is None,
                           ledger_candidate_rows_untrusted=[row for row in rows
                           if row.get("phase") == "generation" and row.get("shard") == shard],
                           bound_release_actual_seconds=actual, cost_upper_bound_seconds=upper,
                           upper_basis="original bound supervised receipt" if actual is not None else
                           "whole final parent duration, conservatively includes CPU-only or uncertain launches; conditional on complete owned termination" if upper is not None else
                           "UNKNOWN; no complete launch/end chain; never infer zero",
                           reserved_cap_seconds=plan["shard_cap_seconds"][shard], success_claim=False))
    return dict(schema="teacher_devset_readonly_race_audit_v1", private=True, GPU_used=False,
                originals_modified=False, recovery_written=False, plan_record=record(plan_path),
                ledger_record=record(ledger_path), strict_parse_error=strict_error,
                ledger_candidate_untrusted=candidate,
                candidate_warning="DISCOVERY ONLY; first complete object is not verified complete history",
                trailing_bytes=dict(bytes=len(trailing), sha256=hashlib.sha256(trailing).hexdigest()),
                all_phase_rows_untrusted={phase: [row for row in rows if row.get("phase") == phase]
                                          for phase in ("generation", "score", "failure")},
                smoke_prior_charged_seconds=plan["admission"]["charged_or_reserved_seconds"],
                fixed_plan_caps=dict(phase_seconds=plan["phase_seconds"], total_seconds=plan["total_seconds"],
                                     seeds=plan["seeds"], video_count=plan["video_count"]),
                inventory=[record(path) for path in sorted(folder.rglob("*")) if path.is_file()], shards=shards,
                counts=dict(launched=sum(shard["launch_present"] for shard in shards),
                            released_bound=sum(shard["bound_release_actual_seconds"] is not None for shard in shards),
                            worker_finished=sum(shard["evidence"]["finished"] is not None for shard in shards),
                            mp4=sum(path.suffix == ".mp4" for path in folder.rglob("*") if path.is_file())),
                all_shards_conservative_cost_upper_bound_seconds=sum(shard["cost_upper_bound_seconds"] for shard in shards
                    if shard["cost_upper_bound_seconds"] is not None),
                actual_cost_unknown_shards=[shard["shard"] for shard in shards if shard["actual_cost_unknown"]],
                known_launched_cost_upper_bound_seconds=sum(shard["cost_upper_bound_seconds"] for shard in shards
                    if shard["launch_present"] and shard["cost_upper_bound_seconds"] is not None),
                unknown_shards=[shard["shard"] for shard in shards if shard["cost_upper_bound_seconds"] is None],
                authorization="NONE; preserve originals; fresh main preregistration and real remaining independent caps before any retry")


def write_audit(path, value, original_folder):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_relative_to(Path(original_folder).resolve()):
        raise ValueError("Audit must be a separate non-symlink path outside the original run")
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-duration-bound", action="append", default=[], metavar="RANK:SECONDS",
                        help="Whole final node tool duration, rounded upward; not a shard receipt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bounds = {int(rank): float(seconds) for rank, seconds in
              (item.split(":", 1) for item in args.node_duration_bound)}
    value = audit(args.plan, bounds)
    if args.output:
        write_audit(args.output, value, args.plan.parent)
        print(json.dumps(dict(output=str(args.output), counts=value["counts"],
                              upper_seconds=value["known_launched_cost_upper_bound_seconds"],
                              unknown_shards=value["unknown_shards"], GPU_used=False)))
    else:
        print(json.dumps(value, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
