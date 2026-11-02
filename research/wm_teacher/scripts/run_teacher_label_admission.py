#!/usr/bin/env python3
"""Fail-closed label GPU entry and CPU-only budget/owned-child proofs."""
import argparse
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import prepare_teacher_label_admission as prep


def verify_draft(path, approved_sha):
    prep.digest(approved_sha)
    proof, draft = prep.read_evidence(path)
    prep.require(proof["sha256"] == approved_sha and draft.get("schema") == prep.SCHEMA,
                 "Approved draft hash/schema mismatch")
    rebuilt = prep.prepare(draft["manifest"]["path"], draft["media_proof"]["path"],
                           draft["model_proof"]["path"], draft["cpu_records"]["path"], draft["owned_parent"])
    prep.require(draft == rebuilt, "Draft content/source drift")
    return draft


def allocation(draft, rank):
    prep.require(type(rank) is int and rank in range(4), "Wrong rank")
    prep.require(draft.get("stage") == prep.STAGE and draft.get("stage_gpu_hours") == 24 and
                 draft.get("global_pilot_reserved_gpu_seconds") == 3600 and
                 draft.get("remaining_fixed_stage_gpu_seconds") == 82800 and
                 len(draft.get("ranks", [])) == 4, "Fixed stage/pilot reservation drift")
    for i, row in enumerate(draft["ranks"]):
        prep.require(isinstance(row, dict) and type(row.get("rank")) is int and
                     type(row.get("allocated_gpus")) is int and type(row.get("cap_seconds")) is int,
                     "Rank allocation types must be integers, not booleans")
        prep.require(row == {"rank": i, "allocated_gpus": 1, "cap_seconds": 900,
                     "pilot_indices": list(range(i * 4, i * 4 + 4)),
                     "output": str(Path(draft["owned_parent"]) / ("rank%d" % i) / "v2_labels.jsonl")},
                     "Fixed one-GPU four-case rank allocation drift")
    return draft["ranks"][rank]


def reserve_cpu(draft, approved_sha, rank, local_root):
    """CPU fixture reservation only; no live GPU ownership is asserted."""
    prep.digest(approved_sha)
    row = allocation(draft, rank)
    root = prep.path_value(local_root)
    prep.require(root.is_dir() and (root == Path("/tmp") or Path("/tmp") in root.parents or
                 root == Path("/dockerdata") or Path("/dockerdata") in root.parents),
                 "Node-local /tmp or /dockerdata only")
    path = root / ("teacher_label_cpu_" + approved_sha + "_rank%d.json" % rank)
    token = secrets.token_hex(32)
    ledger = {"schema": "teacher_label_cpu_local_reservation_v1", "rank": rank,
              "host": socket.gethostname(), "plan_sha256": approved_sha, "stage": prep.STAGE,
              "global_pilot_reserved_gpu_seconds": 3600, "local_reserved_gpu_seconds": row["cap_seconds"],
              "allocated_gpus": 1, "status": "reserved_cpu_fixture", "ownership_token": token,
              "scope": "node_local_method_owned_not_peer_mutable", "gpu_execution_allowed": False}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as sink:
        json.dump(ledger, sink, sort_keys=True)
    return path, token


def charged_seconds(receipt):
    """Missing/unknown elapsed charges cap; valid overruns are never clipped."""
    cap = 900.0
    if not isinstance(receipt, dict) or receipt.get("cost_status") != "known":
        return cap
    value = receipt.get("allocated_elapsed_seconds")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return cap
    return float(value)


def finalize_cpu(path, token, receipt):
    proof, ledger = prep.read_evidence(path)
    prep.require(ledger.get("schema") == "teacher_label_cpu_local_reservation_v1" and
                 ledger.get("ownership_token") == token and ledger.get("host") == socket.gethostname() and
                 ledger.get("status") == "reserved_cpu_fixture" and
                 type(ledger.get("rank")) is int and ledger["rank"] in range(4) and
                 ledger.get("stage") == prep.STAGE and ledger.get("local_reserved_gpu_seconds") == 900 and
                 ledger.get("global_pilot_reserved_gpu_seconds") == 3600,
                 "Not owned fixed local reservation or retry")
    # A separate exclusive finalization record preserves the original reservation.
    final = {"schema": "teacher_label_cpu_cost_v1", "reservation_sha256": proof["sha256"],
             "rank": ledger["rank"], "host": ledger["host"], "stage": prep.STAGE,
             "charged_gpu_seconds": charged_seconds(receipt), "gpu_execution_allowed": False,
             "accounting": "all_allocated_load_compile_wait_save_cleanup_failures"}
    target = Path(str(path) + ".final.json")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as sink:
        json.dump(final, sink, sort_keys=True)
    return final


def harmless_child_timeout(seconds=.1):
    """Real CPU child/tree cleanup, never arbitrary commands or GPU code."""
    prep.require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 2,
                 "Bounded CPU proof timeout required")
    import run_teacher_generation as owned
    start = time.monotonic()
    script = ("import subprocess,sys,time; "
              "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              "print(c.pid,flush=True); time.sleep(60)")
    child = subprocess.Popen([sys.executable, "-c", script], start_new_session=True,
                             stdout=subprocess.PIPE, text=True,
                             env={"PATH": os.environ.get("PATH", ""), "CUDA_VISIBLE_DEVICES": "",
                                  "PYTHONNOUSERSITE": "1"})
    owner = owned.identity(child.pid)
    tracked = []
    cleanup = None
    try:
        # Track observed descendants before timeout, including the sleeping grandchild.
        until = start + seconds
        while time.monotonic() < until:
            owned.owned_members(owner, tracked)
            time.sleep(.01)
    finally:
        cleanup = owned.cleanup_owned(child, owner, time.monotonic() + 4, tracked)
        if child.stdout is not None:
            child.stdout.close()
    elapsed = time.monotonic() - start
    return {"cost_status": "known", "allocated_elapsed_seconds": elapsed,
            "cpu_only": True, "timed_out": True, "owner": owner,
            "signalled": cleanup, "owned_tree_released": not owned.owned_members(owner, tracked),
            "gpu_release_verified": False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--approved-plan-sha256", required=True)
    ap.add_argument("--rank", required=True, type=int)
    args = ap.parse_args(argv)
    try:
        draft = verify_draft(args.plan, args.approved_plan_sha256)
        allocation(draft, args.rank)
        print(json.dumps({"status": "BLOCKED", "gpu_execution_allowed": False,
                          "blockers": draft["blockers"], "semantic_review_pass": False}))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "BLOCKED", "gpu_execution_allowed": False,
                          "error_type": type(exc).__name__}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
