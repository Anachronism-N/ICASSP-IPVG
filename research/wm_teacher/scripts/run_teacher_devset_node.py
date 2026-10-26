#!/usr/bin/env python3
"""CPU reservation and one-shot eight-GPU supervision; no extra generation budget."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import gpu_hold
import run_teacher_devset as dev

NSHARDS, CAP, TERM_SECONDS = 16, 5400, 5370
RANKS = {"alg": {1: 0, 2: 8}, "dymos": {0: 0, 3: 8}}
require, encoded, sha = dev.require, dev.encoded, dev.sha
STATE = dev.smoke.LOCAL_ROOT / "wm_logs/gpu_hold"


def source_records(root):
    return [dev.record(Path(root) / name, nonempty=True) for name in (
        "scripts/run_teacher_devset_node.py", "tests/test_teacher_devset_node.py")]


def source_digest(records):
    return sha(encoded([{k: r[k] for k in ("path", "bytes", "sha256")} for r in records]))


def node_file(plan_path, rank, suffix):
    return Path(plan_path).parent / f"node{rank}.{suffix}.json"


def ledger_file(plan_path):
    return Path(plan_path).parent / "allocations.json"


@contextlib.contextmanager
def locked(path):
    path = Path(path).absolute()
    require(path.resolve() == path, "Ledger symlink forbidden")
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def verify_inputs(method, path, digest, pf_path, pf_sha, rank, offset, approved_source=None):
    require(method in RANKS and rank in RANKS[method] and offset == RANKS[method][rank],
            "Wrong method/node rank/shard offset; fixed two nodes per method")
    plan, models = dev.verify_plan(path, digest, pf_path, pf_sha)
    require(plan["method"] == method and plan["nshards"] == NSHARDS
            and plan["shard_cap_seconds"] == [CAP] * NSHARDS and len(plan["jobs"]) == 188
            and sum(plan["shard_cap_seconds"]) == dev.PHASE_SECONDS["generation"] == 86400,
            "Requires fixed 188 jobs, 16 shards, 5400s each, independent 24 GPUh")
    pf = json.loads(Path(pf_path).read_bytes())
    require(pf["rank"] == rank and pf["hostname"] == os.uname().nodename
            and len(models) == 9, "Wrong local nine-model node preflight")
    sources = source_records(plan["root"])
    new_sha = source_digest(sources)
    require(approved_source is None or new_sha == approved_source, "New supervisor source SHA drift")
    require(dev.record(__file__, nonempty=True)["sha256"] == sources[0]["sha256"],
            "Executing supervisor bytes differ from approved source")
    return plan, sources, new_sha


def holder_pid(state_dir):
    state = json.loads((Path(state_dir) / "status.json").read_bytes())
    pid = state["pid"]
    args = dev.smoke.proc_args(pid)
    require(type(pid) is int and pid > 1 and state["running"] is True
            and state["query_ok"] is True and 0 <= time.time() - state["updated_at"] < 10
            and any(os.path.basename(a) == b"gpu_hold.py" for a in args)
            and dev.smoke.action_matches(args, "run"), "No fresh live gpu_hold run supervisor")
    return pid


def reservation_items(method):
    return [dict(phase="generation", status="reserved", allocated_gpus=1, cap_seconds=CAP,
                 shard=offset + gpu, nshards=NSHARDS, gpu_index=gpu, node_rank=rank,
                 hostname=None, supervisor_pid=None, holder_supervisor_pid=None, reservation_ready=False)
            for rank, offset in RANKS[method].items() for gpu in range(8)]


def validate_reservations(ledger, plan, digest, ready=False):
    require(ledger["schema"] == "teacher_devset_allocations_v1" and ledger["method"] == plan["method"]
            and ledger["plan_sha256"] == digest, "Wrong independent method/plan ledger")
    generations = [a for a in ledger["allocations"] if a["phase"] == "generation"]
    require(len(generations) == NSHARDS and {a["shard"] for a in generations} == set(range(NSHARDS)),
            "All sixteen generation allocations must be reserved before launch")
    for a in generations:
        require(type(a["shard"]) is int and type(a["node_rank"]) is int
                and a["node_rank"] in RANKS[plan["method"]]
                and a["shard"] == RANKS[plan["method"]][a["node_rank"]] + a["gpu_index"]
                and type(a["gpu_index"]) is int and 0 <= a["gpu_index"] < 8
                and a["nshards"] == NSHARDS and a["cap_seconds"] == CAP
                and a["allocated_gpus"] == 1 and a["status"] in ("reserved", "running", "finished"),
                "Wrong shard/host/GPU/cap mapping")
        if a["status"] == "finished":
            elapsed = a["elapsed_seconds"]
            require(dev.finite_number(elapsed) and 0 <= elapsed <= CAP
                    and dev.finite_number(a["start"]) and dev.finite_number(a["end"])
                    and abs(a["end"] - a["start"] - elapsed) < 5, "Invalid/over-cap finished allocation cost")
        if ready or a["reservation_ready"]:
            require(a["reservation_ready"] is True and isinstance(a["hostname"], str) and a["hostname"]
                    and type(a["supervisor_pid"]) is int and a["supervisor_pid"] > 1
                    and a["supervisor_pid"] == a["holder_supervisor_pid"]
                    and a["new_source_sha256"] == ledger["new_source_sha256"], "Both nodes CPU reservations must be ready")
            proof = json.loads(Path(a["node_reservation"]["path"]).read_bytes())
            dev.check_record(a["node_reservation"], nonempty=True)
            require(proof["method"] == ledger["method"] and proof["plan_sha256"] == digest
                    and proof["rank"] == a["node_rank"] and proof["hostname"] == a["hostname"]
                    and proof["holder_supervisor_pid"] == a["supervisor_pid"]
                    and proof["node_preflight"] == a["node_preflight"]
                    and proof["new_source_sha256"] == a["new_source_sha256"], "Reservation proof binding drift")
    # Use the original full phase/cost validator, including smoke/score/failure charges.
    probe = dict(generations[0], status="reserved", hostname=os.uname().nodename)
    copy = dict(ledger, allocations=[probe if a is generations[0] else a for a in ledger["allocations"]])
    dev.validate_ledger(copy, plan, digest, probe["shard"], NSHARDS, probe["gpu_index"], CAP)
    return generations


def reserve(method, path, digest, pf_path, pf_sha, rank, offset, state_dir=STATE, approved_source=None):
    plan, sources, new_sha = verify_inputs(method, path, digest, pf_path, pf_sha, rank, offset, approved_source)
    pid = holder_pid(state_dir)
    ledger_path, proof_path = ledger_file(path), node_file(path, rank, "reserve")
    with locked(ledger_path):
        require(not proof_path.exists(), "Node already reserved; do not reset or replace old evidence")
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_bytes())
            validate_reservations(ledger, plan, digest)
            require(ledger["new_source_sha256"] == new_sha, "Peer/source reservation mismatch")
        else:
            ledger = dict(schema="teacher_devset_allocations_v1", method=method, plan_sha256=digest,
                          new_source_sha256=new_sha, allocations=reservation_items(method))
        own = [a for a in ledger["allocations"] if a["phase"] == "generation" and a["node_rank"] == rank]
        require(len(own) == 8 and all(a["status"] == "reserved" and a["reservation_ready"] is False
                and a["hostname"] is None and "start" not in a for a in own), "Old owned entries; never reset")
        proof = dict(schema="teacher_devset_node_reservation_v1", method=method, plan_sha256=digest,
                     rank=rank, shard_offset=offset, hostname=os.uname().nodename,
                     node_preflight=dev.record(pf_path, nonempty=True), holder_supervisor_pid=pid,
                     state_dir=str(Path(state_dir).absolute()), source_files=sources, new_source_sha256=new_sha,
                     nshards=NSHARDS, cap_seconds=CAP, reserved_generation_GPUh=24, GPU_used=False)
        dev.exclusive(proof_path, proof)
        for a in own:
            a.update(hostname=proof["hostname"], supervisor_pid=pid, holder_supervisor_pid=pid,
                     reservation_ready=True, node_preflight=proof["node_preflight"],
                     node_reservation=dev.record(proof_path, nonempty=True), new_source_sha256=new_sha)
        validate_reservations(ledger, plan, digest)
        dev.checkpoint(ledger_path, ledger)
    return dict(reservation=str(proof_path), reservation_sha256=sha(proof_path.read_bytes()),
                new_source_sha256=new_sha, ledger=str(ledger_path), GPU_used=False, reserved_generation_GPUh=24)


def launch_preflight(method, path, digest, pf_path, pf_sha, rank, offset, state_dir, approved_source, shards=None):
    plan, sources, new_sha = verify_inputs(method, path, digest, pf_path, pf_sha, rank, offset, approved_source)
    require(approved_source is not None, "Explicit approved new supervisor source SHA required")
    pid = holder_pid(state_dir)
    with locked(ledger_file(path)):
        ledger = json.loads(ledger_file(path).read_bytes())
        items = validate_reservations(ledger, plan, digest, ready=True)
        require(ledger["new_source_sha256"] == new_sha, "Ledger source mismatch")
        own = [a for a in items if a["node_rank"] == rank and (shards is None or a["shard"] in shards)]
        for a in own:
            require(a["status"] == "reserved" and "start" not in a
                    and a["hostname"] == os.uname().nodename and a["supervisor_pid"] == pid
                    and a["node_preflight"] == dev.record(pf_path, nonempty=True), "Wrong owner/host/preflight or already started")
            proof = json.loads(Path(a["node_reservation"]["path"]).read_bytes())
            require(proof["state_dir"] == str(Path(state_dir).absolute())
                    and proof["source_files"] == sources, "Reservation state/source drift")
            for suffix in ("claim", "finished", "launch", "supervised", "ready"):
                require(not (Path(path).parent / f"shard{a['shard']}.{suffix}.json").exists(),
                        "Existing worker launch/claim/receipt; no retry or overwrite")
            for job in plan["jobs"][a["shard"]::NSHARDS]:
                media = Path(job["output"])
                if media.with_suffix(".claim.json").exists():
                    require(media.exists() and media.with_suffix(".receipt.json").exists(), "Failed old claim; no retry")
                dev.strict_reuse(job, plan, digest)
    # Queries are CPU-only; do not pause the holder or initialize CUDA here.
    uuids = {a["shard"]: dev.smoke.supervisor_scope(state_dir, pid, a["gpu_index"]) for a in own}
    require(len(set(uuids.values())) == len(own), "Duplicate physical GPU UUIDs")
    return plan, own, uuids


def command_for(plan, path, digest, pf_path, pf_sha, gpu, shard, state_dir):
    python = plan["runtime"]["python"]
    scripts = Path(plan["root"]) / "scripts"
    worker = [python, str(scripts / "run_teacher_devset.py"), "--run", "--gpu-authorized",
              "--plan", str(path), "--approved-plan-sha256", digest, "--node-preflight", str(pf_path),
              "--approved-node-preflight-sha256", pf_sha, "--ledger", str(ledger_file(path)),
              "--gpu-index", str(gpu), "--shard", str(shard), "--nshards", str(NSHARDS),
              "--gen-cap-seconds", str(CAP), "--state-dir", str(state_dir)]
    return [plan["runtime"]["tools"]["timeout"], "--signal=TERM", "--kill-after=30s", "5370s",
            python, str(scripts / "gpu_hold.py"), "--state-dir", str(state_dir),
            "--action", "exec", "--gpus", str(gpu), "--"] + worker


def session_processes(sid, exclude=()):
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(entry.name)
            if int(fields[3]) == sid and fields[0] not in ("Z", "X") and pid not in exclude:
                result[pid] = int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
    return result


def stop_session(sid, deadline, exclude=()):
    require(sid > 1 and sid != os.getsid(os.getppid()), "Cannot stop an unrelated/parent session")
    for signum, grace in ((signal.SIGTERM, 29), (signal.SIGKILL, 1)):
        for pid in session_processes(sid, exclude):
            try:
                # Recheck session membership before signaling; never touch gpu_hold run.
                if os.getsid(pid) == sid:
                    os.kill(pid, signum)
            except ProcessLookupError:
                pass
        until = min(time.monotonic() + grace, deadline)
        while session_processes(sid, exclude) and time.monotonic() < until:
            time.sleep(0.02)
    require(not session_processes(sid, exclude), "Owned process tree not released")


def gpu_released(item, state_dir):
    # Holder may already have resumed; only that verified holder can be present.
    state = json.loads((Path(state_dir) / "status.json").read_bytes())
    require(holder_pid(state_dir) == item["holder_supervisor_pid"], "Holder changed/died during cleanup")
    selected = state["gpus"][str(item["gpu_index"])]
    require(selected["gpu_uuid"] == item["gpu_uuid"], "GPU identity changed during cleanup")
    pids = gpu_hold.query_processes().get(item["gpu_uuid"], set())
    allowed = {selected["holder_pid"]} if selected["holder_pid"] else set()
    if allowed:
        args = dev.smoke.proc_args(selected["holder_pid"])
        require(any(os.path.basename(a) == b"gpu_hold.py" for a in args)
                and dev.smoke.action_matches(args, "worker"), "Unknown resumed GPU occupant")
    return pids <= allowed


def finish_allocation(path, digest, shard, owner_pid, state_dir, exit_code, timed_out, interrupted, error):
    with locked(ledger_file(path)):
        ledger = json.loads(ledger_file(path).read_bytes())
        require(ledger["plan_sha256"] == digest, "Ledger changed during cleanup")
        item = next(a for a in ledger["allocations"] if a["phase"] == "generation" and a["shard"] == shard)
        require(item.get("node_supervisor_pid") == owner_pid and "start" in item
                and item["status"] != "finished", "Cannot finalize unowned/unstarted allocation")
        released = not session_processes(owner_pid, (owner_pid,)) and gpu_released(item, state_dir)
        if not released:
            item.update(release_confirmed=False, cleanup_error=error or "Owned tree/GPU release unproven",
                        cost_status="reserved cap; allocation clock still open")
            dev.checkpoint(ledger_file(path), ledger)
            return False
        elapsed = time.monotonic() - item["allocation_start_monotonic"]
        worker_path = Path(path).parent / f"shard{shard}.finished.json"
        worker = json.loads(worker_path.read_bytes()) if worker_path.exists() else {}
        worker_bound = (worker.get("schema") == "teacher_devset_shard_receipt_v1"
                        and worker.get("plan_sha256") == digest and worker.get("method") == ledger["method"]
                        and worker.get("shard") == shard and worker.get("nshards") == NSHARDS
                        and item.get("worker_complete") is True
                        and item.get("receipt_sha256") == sha(worker_path.read_bytes()))
        success = (worker_bound and worker.get("success") is True and item.get("worker_success") is True
                   and exit_code == 0 and not timed_out and interrupted is None and error is None and elapsed <= CAP)
        receipt = dict(schema="teacher_devset_supervised_receipt_v1", method=ledger["method"],
                       plan_sha256=digest, shard=shard, nshards=NSHARDS, cap_seconds=CAP, allocated_gpus=1,
                       start=item["start"], end=time.time(), elapsed_seconds=elapsed, elapsed_GPUh=elapsed / 3600,
                       owned_tree_released=True, allocation_released=True, success=success, exit_code=exit_code,
                       timed_out=timed_out, interrupted_signal=interrupted, error=error, budget_exceeded=elapsed > CAP,
                       new_source_sha256=ledger["new_source_sha256"], quality_pass=False, sota_claim=False)
        dev.exclusive(Path(path).parent / f"shard{shard}.supervised.json", receipt)
        item.update(receipt, status="finished", release_confirmed=True, cost_status="true allocation elapsed after tree/GPU release")
        dev.checkpoint(ledger_file(path), ledger)
    return receipt


def ready_barrier(path, digest, pf_sha, rank, offset, shard, approved_source, interrupted):
    launch = json.loads(node_file(path, rank, "launch").read_bytes())
    parent = launch["node_pid"]
    require(parent == os.getppid() and launch["plan_sha256"] == digest
            and launch["new_source_sha256"] == approved_source, "Unowned node launch")
    with locked(ledger_file(path)):
        dev.exclusive(Path(path).parent / f"shard{shard}.ready.json",
                      dict(plan_sha256=digest, node_preflight_sha256=pf_sha, new_source_sha256=approved_source,
                           shard=shard, node_pid=parent, supervisor_pid=os.getpid()))
    until = time.monotonic() + 600
    while True:
        require(interrupted() is None and os.getppid() == parent
                and dev.smoke.proc_args(parent), "Parent died/user interrupted CPU ready barrier")
        ready = True
        with locked(ledger_file(path)):
            for s in range(offset, offset + 8):
                candidate = Path(path).parent / f"shard{s}.ready.json"
                if not candidate.exists():
                    ready = False
                    continue
                proof = json.loads(candidate.read_bytes())
                require(proof["plan_sha256"] == digest and proof["node_preflight_sha256"] == pf_sha
                        and proof["new_source_sha256"] == approved_source and proof["shard"] == s
                        and proof["node_pid"] == parent and dev.smoke.proc_args(proof["supervisor_pid"]),
                        "Wrong/dead worker CPU ready proof")
        if ready:
            return
        require(time.monotonic() < until, "Eight-worker CPU ready barrier timed out; no allocation")
        time.sleep(0.05)


def supervise(method, path, digest, pf_path, pf_sha, rank, offset, shard, state_dir, approved_source, gpu_authorized=False):
    require(gpu_authorized is True, "Explicit GPU authorization required")
    require(type(shard) is int and offset <= shard < offset + 8
            and os.getsid(0) == os.getpid(), "Worker supervisor must own a fresh session and local shard")
    plan, own, uuids = launch_preflight(method, path, digest, pf_path, pf_sha, rank, offset, state_dir, approved_source, [shard])
    require(len(own) == 1, "Exactly one owned allocation required")
    command = command_for(plan, path, digest, pf_path, pf_sha, own[0]["gpu_index"], shard, state_dir)
    pid, child, started, interrupted, timed_out, error = os.getpid(), None, None, None, False, None
    old = {}
    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, interrupt)
    try:
        ready_barrier(path, digest, pf_sha, rank, offset, shard, approved_source, lambda: interrupted)
        # Recheck frozen CPU inputs after waiting, before the durable allocation clock.
        require(sha(Path(path).read_bytes()) == digest and sha(Path(pf_path).read_bytes()) == pf_sha
                and source_digest(source_records(plan["root"])) == approved_source,
                "Plan/preflight/supervisor source changed at ready barrier")
        for record in plan["source_files"]:
            dev.check_record(record)
        require(dev.smoke.supervisor_scope(state_dir, own[0]["supervisor_pid"], own[0]["gpu_index"])
                == uuids[shard], "GPU scope changed at ready barrier")
        with locked(ledger_file(path)):
            ledger = json.loads(ledger_file(path).read_bytes())
            item = next(a for a in ledger["allocations"] if a["phase"] == "generation" and a["shard"] == shard)
            require(item["status"] == "reserved" and "start" not in item and interrupted is None, "Allocation already owned/interrupted")
            dev.exclusive(Path(path).parent / f"shard{shard}.launch.json",
                          dict(plan_sha256=digest, shard=shard, command=command, node_supervisor_pid=pid,
                               new_source_sha256=approved_source, gpu_uuid=uuids[shard]))
            started = time.monotonic()
            item.update(start=time.time(), allocation_start_monotonic=started, node_supervisor_pid=pid, gpu_uuid=uuids[shard])
            # Clock is durably written BEFORE gpu_hold.exec can pause. Worker owns running transition.
            dev.checkpoint(ledger_file(path), ledger)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[shard], WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True", OMP_NUM_THREADS="1")
        with (Path(path).parent / f"shard{shard}.worker.log").open("xb") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, shell=False)
            while child.poll() is None and interrupted is None:
                if time.monotonic() - started >= TERM_SECONDS:
                    timed_out = True
                    break
                time.sleep(0.05)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if started is not None:
            try:
                stop_session(pid, started + CAP, (pid,))
                if child is not None:
                    child.wait(timeout=max(0.01, started + CAP - time.monotonic()))
                    timed_out = timed_out or child.returncode in (124, 137)
            except BaseException as exc:
                error = f"Cleanup: {type(exc).__name__}: {exc}"
            try:
                result = finish_allocation(path, digest, shard, pid, state_dir,
                                           child.returncode if child is not None else None, timed_out, interrupted, error)
            except BaseException as exc:
                result = False
                error = f"Finalization: {type(exc).__name__}: {exc}"
        else:
            result = False
        for sig, handler in old.items():
            signal.signal(sig, handler)
    require(result is not False, error or "Release not confirmed; reservation remains charged, no retry")
    return result


def node_command(args, shard, python):
    return [python, str(Path(__file__).resolve()), "--worker-supervisor", "--gpu-authorized",
            "--method", args.method, "--plan", str(args.plan), "--approved-plan-sha256", args.approved_plan_sha256,
            "--node-preflight", str(args.node_preflight), "--approved-node-preflight-sha256", args.approved_node_preflight_sha256,
            "--rank", str(args.rank), "--shard-offset", str(args.shard_offset), "--shard", str(shard),
            "--state-dir", str(args.state_dir), "--approved-new-source-sha256", args.approved_new_source_sha256]


def run_node(args):
    require(args.gpu_authorized and args.approved_new_source_sha256, "Explicit GPU authorization/new source SHA required")
    plan, own, _ = launch_preflight(args.method, args.plan, args.approved_plan_sha256, args.node_preflight,
                                  args.approved_node_preflight_sha256, args.rank, args.shard_offset,
                                  args.state_dir, args.approved_new_source_sha256)
    require(len(own) == 8, "Only local eight worker supervisors may launch")
    dev.exclusive(node_file(args.plan, args.rank, "launch"),
                  dict(method=args.method, plan_sha256=args.approved_plan_sha256, node_pid=os.getpid(), owned_shards=[a["shard"] for a in own],
                       new_source_sha256=args.approved_new_source_sha256, cap_seconds=CAP, reserved_generation_GPUh=24))
    children, interrupted, error, old = [], None, None, {}
    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, interrupt)
    began = time.monotonic()
    try:
        for a in own:
            require(interrupted is None, "User interrupted node launch")
            child = subprocess.Popen(node_command(args, a["shard"], plan["runtime"]["python"]), start_new_session=True)
            children.append((a["shard"], child))
        while any(c.poll() is None for _, c in children) and interrupted is None:
            if any(c.poll() not in (None, 0) for _, c in children):
                raise RuntimeError("Owned supervisor failed; stop remaining workers, no retry")
            # CPU admission before allocation is not charged, but must not run unbounded.
            require(time.monotonic() - began < CAP + 600, "Node supervisor outer guard exhausted")
            time.sleep(0.05)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for _, c in children:
            if c.poll() is None:
                try:
                    os.kill(c.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for shard, c in children:
            try:
                ledger = json.loads(ledger_file(args.plan).read_bytes())
                item = next(a for a in ledger["allocations"] if a["phase"] == "generation" and a["shard"] == shard)
                deadline = item.get("allocation_start_monotonic", time.monotonic()) + CAP
                grace = min(time.monotonic() + 30, deadline)
                while c.poll() is None and time.monotonic() < grace:
                    time.sleep(0.02)
                stop_session(c.pid, min(deadline, time.monotonic() + 1))
                c.wait(timeout=1)
                ledger = json.loads(ledger_file(args.plan).read_bytes())
                item = next(a for a in ledger["allocations"] if a["phase"] == "generation" and a["shard"] == shard)
                if "start" in item and item["status"] != "finished":
                    require(finish_allocation(args.plan, args.approved_plan_sha256, shard, c.pid, args.state_dir,
                                              c.returncode, time.monotonic() >= deadline, interrupted, error) is not False,
                            "Owned GPU release unproven; still charged")
            except BaseException as exc:
                error = f"Cleanup: {type(exc).__name__}: {exc}"
        for sig, handler in old.items():
            signal.signal(sig, handler)
    ledger = json.loads(ledger_file(args.plan).read_bytes())
    own_final = [a for a in ledger["allocations"] if a["phase"] == "generation" and a["node_rank"] == args.rank]
    success = error is None and interrupted is None and len(children) == 8
    success = success and all(a["status"] == "finished" and a.get("success") is True for a in own_final)
    result = dict(method=args.method, rank=args.rank, success=success, interrupted_signal=interrupted, error=error,
                  owned_shards=[a["shard"] for a in own_final], allocated_gpus=8, reserved_generation_GPUh=24,
                  true_elapsed_GPUh=sum(a.get("elapsed_GPUh", 0) for a in own_final if a["status"] == "finished"),
                  unfinalized_shards=[a["shard"] for a in own_final if a["status"] != "finished"],
                  quality_pass=False, sota_claim=False)
    dev.exclusive(node_file(args.plan, args.rank, "finished"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--reserve", "--preflight", action="store_true", dest="reserve")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker-supervisor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=RANKS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    parser.add_argument("--node-preflight", type=Path, required=True)
    parser.add_argument("--approved-node-preflight-sha256", required=True)
    parser.add_argument("--approved-new-source-sha256")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--shard-offset", type=int, choices=(0, 8), required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=STATE)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.run or args.worker_supervisor:
        if not args.gpu_authorized or not args.approved_new_source_sha256:
            parser.error("Run requires explicit --gpu-authorized and --approved-new-source-sha256")
        if args.worker_supervisor:
            if args.shard is None:
                parser.error("Worker supervisor needs exactly one owned shard")
            result = supervise(args.method, args.plan, args.approved_plan_sha256, args.node_preflight,
                               args.approved_node_preflight_sha256, args.rank, args.shard_offset, args.shard,
                               args.state_dir, args.approved_new_source_sha256, True)
        else:
            if args.shard is not None:
                parser.error("Node launches its eight owned shards, not an arbitrary worker")
            result = run_node(args)
    else:
        if args.gpu_authorized or args.shard is not None:
            parser.error("CPU reserve is not GPU authorization")
        result = reserve(args.method, args.plan, args.approved_plan_sha256, args.node_preflight,
                         args.approved_node_preflight_sha256, args.rank, args.shard_offset,
                         args.state_dir, args.approved_new_source_sha256)
    print(json.dumps(result, allow_nan=False))
    if result.get("success") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
