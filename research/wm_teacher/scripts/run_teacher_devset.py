#!/usr/bin/env python3
"""Private paper-inspired ALG/DyMoS port, not a SOTA/quality claim.

Default CPU prepare; pending/blocked smoke NEVER authorizes generation. Run is
one shard UNDER caller GNU timeout/gpu_hold. Caller preregisters all shard
allowances in an independent method ledger. Score separately with the original
score_e1.cmd_vbench/cmd_gtref APIs; no CONFIGS mutation or scoring here.
"""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import run_teacher_inference_smoke as smoke
import measure_teacher_throughput as performance

ROOT = Path(__file__).resolve().parents[1]
PHASE_SECONDS = dict(smoke=3600, generation=86400, score=57600, failure=14400)
TOTAL_SECONDS = 162000
SCHEMA = "teacher_devset_plan_v1"
PARAMS, SPEC = smoke.PARAMS, dict(smoke.SPEC)
CASE_ROOT = "eval/e1_devset_20261021"
CASES_SHA256 = "0873b163284a92b207ac248cc41ff934cf73615f1f9af073eca8bfb2073e5aed"
BASE_ROOTS = {"seed0": CASE_ROOT, "seed1": "eval/e1_seed1_20261023"}
require, encoded, sha = smoke.require, smoke.encoded, smoke.sha
record, check_record = smoke.record, smoke.check_record
exclusive, checkpoint = smoke.exclusive, smoke.checkpoint


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def output_root(root, method):
    return Path(root) / f"eval/teacher_{method}_devset_20261026"


def sources(root):
    return sorted(set(smoke.source_paths(root)) | {root / "scripts" / n for n in (
        "run_teacher_devset.py", "score_e1.py", "build_e1_jobs.py", "summarize_eval.py",
        "measure_teacher_throughput.py")}
        | {root / "tests/test_teacher_devset.py", root / "tests/test_teacher_throughput.py"})


def frozen_cases(root):
    path = root / CASE_ROOT / "cases.jsonl"
    raw, manifest = path.read_bytes(), record(path, nonempty=True)
    require(manifest["sha256"] == CASES_SHA256, "Frozen case manifest SHA drift")
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    require(len(cases) == 94 and len({c["id"] for c in cases}) == 94, "Exactly frozen 94 cases required")
    require(sha(cases[0]["id"].encode()) == smoke.FIRST_CASE_SHA256, "Frozen first-case drift")
    inputs = []
    for case in cases:
        name = case["id"]
        require(isinstance(name, str) and name not in (".", "..") and name
                and all(c.isalnum() or c in "_.+-" for c in name), "Unsafe case id")
        require(isinstance(case["prompt"], str) and case["prompt"]
                and sha(case["prompt"].encode()) == case["prompt_sha256"], "Raw prompt hash drift")
        require(case.get("n_frames") == 121 and case.get("width") == 1280
                and case.get("height") == 704 and case.get("fps") == 24, "Case media schema drift")
        image = record(case["image"], nonempty=True)
        require(image["sha256"] == case["image_sha256"]
                and Path(image["path"]).resolve().is_relative_to((root / CASE_ROOT / "images").resolve()),
                "Original first-image hash/path drift")
        gt = record(case["gt"], nonempty=True)
        require(gt["sha256"] == case["video_sha256"], "GT/reference hash drift")
        inputs.append(dict(id=name, image=image, gt=gt))
    require(record(path, nonempty=True) == manifest, "Cases changed during prepare")
    return raw, cases, inputs, manifest


def smoke_config(request, method):
    require(request["schema"] == "teacher_inference_smoke_v1" and request["private"] is True
            and request["method"] == method and encoded(request["parameters"]) == encoded(PARAMS[method])
            and encoded(request["call"]) == encoded(SPEC) and request["checkpoint"] == "NONE"
            and request["active_steps"] == 10 and request["fps"] == 24 and request["video_count"] == 1
            and encoded(request["budget"]) == encoded(smoke.BUDGET), "Wrong smoke method/configuration")


def smoke_admission(root, method, nshards=1, approved_throughput_sha=None, throughput_receipt=None,
                    *, verify_local_request=True):
    attempts, charged, state, reasons, latest = [], 0.0, "ready", [], None
    for date, suffix in (("20261025", "01"), ("20261026", "02")):
        folder = root / f"eval/teacher_{method}_smoke_{date}_{suffix}"
        attempt = dict(path=str(folder), status="missing", files=[])
        if not (folder / "request.json").is_file():
            state = "pending"
            reasons.append(f"smoke{suffix} missing request/evidence")
            attempts.append(attempt)
            continue
        request = json.loads((folder / "request.json").read_bytes())
        smoke_config(request, method)
        attempt["files"].append(record(folder / "request.json", nonempty=True))
        if (folder / "launch.json").exists():
            launch = json.loads((folder / "launch.json").read_bytes())
            require(launch["request_sha256"] == sha((folder / "request.json").read_bytes())
                    and launch["budget"] == smoke.BUDGET, "Smoke launch binding mismatch")
            attempt["files"].append(record(folder / "launch.json", nonempty=True))
        if not (folder / "finished.json").exists():
            if (folder / "launch.json").exists():
                charged += smoke.BUDGET["outer_cap_seconds"]
            state = "pending"
            reasons.append(f"smoke{suffix} missing.finished/pending; not PASS")
            attempt["status"] = "pending"
            attempts.append(attempt)
            continue
        receipt = json.loads((folder / "finished.json").read_bytes())
        require(receipt["schema"] == "teacher_smoke_receipt_v1" and receipt["method"] == method
                and receipt["request_sha256"] == sha((folder / "request.json").read_bytes())
                and receipt["allocated_gpus"] == 1, "Smoke receipt binding mismatch")
        elapsed = receipt["elapsed_seconds"]
        require(finite_number(elapsed) and elapsed > 0 and finite_number(receipt["elapsed_GPUh"])
                and math.isclose(receipt["elapsed_GPUh"], elapsed / 3600, abs_tol=1e-9), "Invalid smoke cost")
        require(finite_number(receipt["start"]) and finite_number(receipt["end"])
                and abs(receipt["end"] - receipt["start"] - elapsed) < 5, "Smoke allocation timing mismatch")
        charged += elapsed
        attempt.update(status="finished", elapsed_seconds=elapsed)
        attempt["files"].append(record(folder / "finished.json", nonempty=True))
        if suffix == "02":
            latest, latest_request, evidence = receipt, request, receipt.get("evidence", {})
            if not (receipt.get("success") is True and receipt.get("mechanism_PASS") is True
                    and receipt.get("owned_tree_released") is True and receipt.get("budget_exceeded") is False
                    and receipt.get("exit_code") == 0 and receipt.get("timed_out") is False
                    and receipt.get("interrupted_signal") is None and receipt.get("error") is None
                    and evidence.get("success") is True and evidence.get("mechanism_PASS") is True
                    and evidence.get("parameters_unchanged") is True):
                state = "blocked"
                reasons.append("latest fixed smoke02 lacks complete real mechanism PASS")
            else:
                require(receipt.get("budget") == smoke.BUDGET and (folder / "worker.json").is_file(),
                        "Missing final smoke worker evidence")
                worker = json.loads((folder / "worker.json").read_bytes())
                require(worker == evidence, "Final smoke worker/receipt disagreement")
                require(evidence["parameters_before"] and evidence["parameters_before"] == evidence["parameters_after"],
                        "Smoke baseline weights changed/unproven")
                smoke.validate_stats(method, evidence["stats"]["method"], evidence["stats"]["observed"])
                # Historical 02 is immutable evidence, not an attestation of current sources.
                require(Path(request["root"]).resolve() == root, "Smoke root mismatch")
                video = smoke.validate_video(folder / "video.mp4", request["runtime"]["tools"])
                require(video == evidence["video"], "Smoke media hash/decode mismatch")
                attempt["files"] += [record(folder / "worker.json", nonempty=True), video["file"]]
        attempts.append(attempt)
    if charged > PHASE_SECONDS["smoke"]:
        state = "blocked"
        reasons.append("both smoke attempts charged/reserved exceed independent 1 GPUh")
    mechanism_pass = bool(latest and latest.get("mechanism_PASS") is True and state == "ready")
    upper = latest["elapsed_seconds"] * 188 if latest else None
    estimate, shard_estimates, timing_record, timing_sources = None, None, None, []
    extra = root / f"eval/teacher_{method}_smoke_20261026_03"
    require(not (extra / "launch.json").exists() and not (extra / "finished.json").exists(),
            "Unaccounted single-video03 allocation; batch-only admission refuses duplicate measurement")
    if state == "ready":
        if throughput_receipt is None or not Path(throughput_receipt).is_file() or not approved_throughput_sha:
            state = "blocked"
            reasons.append("missing approved real batchreceipt/PASS; cannot guess throughput")
        else:
            timing_record = record(throughput_receipt, nonempty=True)
            receipt, request, timing_sources, prior = performance.verify_measurement(
                throughput_receipt, approved_throughput_sha, method, root=root,
                verify_local_request=verify_local_request)
            require(prior == charged and request["latest_smoke02"]["sha256"]
                    == sha((root / f"eval/teacher_{method}_smoke_20261026_02/finished.json").read_bytes()),
                    "Batch prior smoke evidence/cost mismatch")
            require(request["case"] == latest_request["case"]
                    and request["model_content_sha256"] == latest_request["model_content_sha256"]
                    and request["negative_sha256"] == latest_request["negative_sha256"],
                    "Batch method/base/raw input binding mismatch")
            charged += receipt["elapsed_seconds"]
            require(charged <= PHASE_SECONDS["smoke"], "All original smoke and batch costs exceed 1 GPUh")
            load, per_videos = performance.validate_timers(receipt["evidence"],
                receipt["allocation_monotonic_start"], receipt["allocation_monotonic_end"])
            # Use slower seed and measured allocation residual, including hashes/wait/cleanup.
            per_video = max(per_videos)
            cleanup = max(35, receipt["elapsed_seconds"] - load - sum(per_videos))
            estimate = nshards * (load + cleanup) + 188 * per_video
            shard_estimates = [load + cleanup + len(range(s, 188, nshards)) * per_video for s in range(nshards)]
            if estimate > PHASE_SECONDS["generation"]:
                state = "blocked"
                reasons.append("real once-load two-seed batch estimate exceeds independent 24 GPUh")
    return dict(status=state, reasons=reasons, attempts=attempts, charged_or_reserved_seconds=charged,
                estimated_generation_seconds=estimate, estimated_shard_seconds=shard_estimates,
                conservative_total_elapsed_upper_bound_seconds=upper,
                throughput=timing_record, measurement_sources=timing_sources,
                estimate_basis="nshards*(measured load+max(35,allocation residual)) +188*max(two seed inference/save); non-guarantee",
                estimate_is_guarantee=False, mechanism_PASS=mechanism_pass)


def jobs_for(cases, root, method):
    return [dict(index=seed * 94 + i, seed=seed, case_id=c["id"],
                 output=str(output_root(root, method) / f"seed{seed}/videos/{method}/{c['id']}.mp4"))
            for seed in (0, 1) for i, c in enumerate(cases)]


def base_reuse(root, manifest):
    # Legacy videos/scores alone cannot prove generation weights/source hashes.
    return {seed: dict(root=str(root / base), config="s5_c5", seed=int(seed[-1]), allowed=False,
                      cases_byte_match=(root / base / "cases.jsonl").is_file()
                      and sha((root / base / "cases.jsonl").read_bytes()) == manifest["sha256"],
                      missing_evidence=["baseline generation source/weights/seed/config binding",
                                        "94 full-decode/media SHA bindings", "score source/runtime/input SHA bindings"],
                      restriction="No free same-protocol reuse/comparison until strict audit; no seed substitution")
            for seed, base in BASE_ROOTS.items()}


def prepare(method, node_preflight, approved_node_sha, nshards=1, gen_cap_seconds=None, root=ROOT,
            approved_throughput_sha=None, throughput_receipt=None):
    root = Path(root).resolve()
    require(method in PARAMS, "One fixed method required")
    require(type(nshards) is int and 1 <= nshards <= 188, "nshards must be 1..188")
    cap = PHASE_SECONDS["generation"] // nshards if gen_cap_seconds is None else gen_cap_seconds
    require(type(cap) is int and cap > 35 and cap * nshards <= PHASE_SECONDS["generation"],
            "All shard declared allowances must sum <=24 GPUh")
    raw, cases, inputs, manifest = frozen_cases(root)
    pf_record = record(node_preflight, nonempty=True)
    require(pf_record["sha256"] == approved_node_sha, "Approved node preflight SHA mismatch")
    pf = json.loads(Path(node_preflight).read_bytes())
    models = smoke.models_from_preflight(pf)
    admission = smoke_admission(root, method, nshards, approved_throughput_sha, throughput_receipt)
    if admission["throughput"] is not None:
        batch_request = json.loads((Path(throughput_receipt).parent / "request.json").read_bytes())
        require(batch_request["model_content_sha256"] == pf["model_content_sha256"]
                and batch_request["cases_file"] == manifest and batch_request["case"] == cases[0],
                "Current devset/base/case hash differs from authenticated batch")
    if admission["status"] == "ready" and any(t > cap for t in admission["estimated_shard_seconds"]):
        admission["status"] = "blocked"
        admission["reasons"].append("Measured per-shard load/generation/cleanup estimate exceeds declared allowance")
    runtime, negative = smoke.runtime_records(), smoke.generator_negative(root)
    value = dict(schema=SCHEMA, private=True, root=str(root), method=method, parameters=PARAMS[method],
                 call=SPEC, fps=24, active_steps=10, checkpoint="NONE", seeds=[0, 1], video_count=188,
                 phase_seconds=PHASE_SECONDS, total_seconds=TOTAL_SECONDS, cases_file=manifest,
                 cases=cases, inputs=inputs, jobs=jobs_for(cases, root, method), negative_prompt=negative,
                 negative_sha256=sha(negative.encode()), source_files=[record(p) for p in sources(root)],
                 runtime=runtime, node_preflight=pf_record, models=models,
                 model_content_sha256=pf["model_content_sha256"], output_root=str(output_root(root, method)),
                 admission=admission, base_reuse=base_reuse(root, manifest), nshards=nshards,
                 shard_cap_seconds=[cap] * nshards, GPU_used=False, quality_pass=False, sota_claim=False,
                 score_api=dict(module="score_e1", calls=["cmd_vbench", "cmd_gtref"], config=method,
                                roots=[str(output_root(root, method) / f"seed{i}") for i in (0, 1)],
                                separate_score_budget_seconds=PHASE_SECONDS["score"],
                                restriction="caller waits for all188 strict receipts; unique WORK_BASE per method/seed; no CONFIGS mutation"))
    folder = output_root(root, method)
    identity = dict(schema="teacher_devset_identity_v1", method=method, cases_sha256=manifest["sha256"], seeds=[0, 1])
    if folder.exists():
        require(folder.resolve() == folder, "Output root symlink forbidden")
        require((folder / "identity.json").is_file()
                and json.loads((folder / "identity.json").read_bytes()) == identity, "Old/unowned output root")
        require(not (folder / "plan.json").exists(), "Ready plan already exists; never overwrite")
    else:
        folder.mkdir(mode=0o700)
        exclusive(folder / "identity.json", identity)
    for seed in (0, 1):
        seed_root = folder / f"seed{seed}"
        seed_root.mkdir(mode=0o700, exist_ok=True)
        (seed_root / "videos" / method).mkdir(mode=0o700, parents=True, exist_ok=True)
        require(seed_root.resolve() == seed_root
                and (seed_root / "videos" / method).resolve() == seed_root / "videos" / method,
                "Seed/video output symlink forbidden")
        path = seed_root / "cases.jsonl"
        if path.exists():
            require(path.read_bytes() == raw, "Existing seed case metadata bytes drift")
        else:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
    name = "plan.json" if admission["status"] == "ready" else f"plan.{admission['status']}.json"
    path = folder / name
    exclusive(path, value)
    return path, sha(path.read_bytes())


def verify_plan(path, approved_sha, node_preflight, approved_node_sha):
    path = Path(path).absolute()
    require(sha(path.read_bytes()) == approved_sha, "Approved plan SHA mismatch")
    plan = json.loads(path.read_bytes())
    method = plan["method"]
    require(method in PARAMS and plan["schema"] == SCHEMA and plan["private"] is True, "Wrong plan schema/method")
    root = Path(plan["root"])
    require(path.resolve() == path and path.parent.stat().st_mode & 0o077 == 0
            and path.stat().st_mode & 0o077 == 0, "Plan must be private/non-symlink")
    require(root.resolve() == root and path == output_root(root, method) / "plan.json"
            and plan["output_root"] == str(path.parent), "Wrong method/output root or pending plan")
    require(plan["GPU_used"] is False and plan["checkpoint"] == "NONE" and plan["active_steps"] == 10
            and plan["fps"] == 24 and plan["seeds"] == [0, 1] and plan["video_count"] == 188
            and encoded(plan["call"]) == encoded(SPEC) and encoded(plan["parameters"]) == encoded(PARAMS[method])
            and plan["phase_seconds"] == PHASE_SECONDS and plan["total_seconds"] == TOTAL_SECONDS,
            "Unapproved parameters/seed/schema/budgets")
    n, caps = plan["nshards"], plan["shard_cap_seconds"]
    require(type(n) is int and 1 <= n <= 188 and len(caps) == n
            and all(type(c) is int and c > 35 for c in caps) and sum(caps) <= PHASE_SECONDS["generation"],
            "Shard allowance sum exceeds 24 GPUh")
    require({r["path"] for r in plan["source_files"]} == {str(p) for p in sources(root)}, "Source set drift")
    for r in plan["source_files"]:
        check_record(r)
    require(smoke.runtime_records() == plan["runtime"], "Runtime source/version hash drift")
    raw, cases, inputs, manifest = frozen_cases(root)
    require(manifest == plan["cases_file"] and cases == plan["cases"] and inputs == plan["inputs"]
            and plan["jobs"] == jobs_for(cases, root, method), "Case/seed/input/job schema drift")
    for seed in (0, 1):
        require((path.parent / f"seed{seed}/cases.jsonl").read_bytes() == raw, "Output case metadata byte drift")
    require(plan["base_reuse"] == base_reuse(root, manifest), "Baseline root/seed or reuse drift")
    negative = smoke.generator_negative(root)
    require(negative == plan["negative_prompt"] and sha(negative.encode()) == plan["negative_sha256"], "NEG drift")
    pf_record = record(node_preflight, nonempty=True)
    require(pf_record["sha256"] == approved_node_sha, "Approved node preflight SHA mismatch")
    pf = json.loads(Path(node_preflight).read_bytes())
    models = smoke.models_from_preflight(pf)
    require(pf["model_content_sha256"] == plan["model_content_sha256"]
            and [{k: m[k] for k in ("asset", "bytes", "sha256")} for m in models]
            == [{k: m[k] for k in ("asset", "bytes", "sha256")} for m in plan["models"]], "Baseline model content drift")
    throughput = plan["admission"].get("throughput")
    admission = smoke_admission(root, method, plan["nshards"],
                                throughput["sha256"] if throughput else None,
                                throughput["path"] if throughput else None, verify_local_request=False)
    latest_request = json.loads((root / f"eval/teacher_{method}_smoke_20261026_02/request.json").read_bytes())
    require(latest_request["model_content_sha256"] == plan["model_content_sha256"], "Smoke/base model content mismatch")
    require(admission == plan["admission"] and admission["status"] == "ready"
            and admission["mechanism_PASS"] is True
            and all(t <= c for t, c in zip(admission["estimated_shard_seconds"], caps)),
            "No current complete smoke PASS / feasibility drift")
    return plan, models


def validate_ledger(ledger, plan, digest, shard, nshards, gpu, cap):
    require(ledger["schema"] == "teacher_devset_allocations_v1" and ledger["method"] == plan["method"]
            and ledger["plan_sha256"] == digest, "Wrong method/plan allocation ledger")
    totals = {phase: 0.0 for phase in PHASE_SECONDS}
    totals["smoke"] = plan["admission"]["charged_or_reserved_seconds"]
    generations, chosen = {}, None
    for item in ledger["allocations"]:
        phase, status = item["phase"], item["status"]
        require(phase in ("generation", "score", "failure") and status in ("reserved", "running", "finished")
                and type(item["allocated_gpus"]) is int and item["allocated_gpus"] > 0
                and finite_number(item["cap_seconds"]) and item["cap_seconds"] > 0, "Invalid stage reservation")
        charged = item["cap_seconds"]
        if status == "finished":
            elapsed = item["elapsed_seconds"]
            require(finite_number(elapsed) and elapsed >= 0 and finite_number(item["start"])
                    and finite_number(item["end"]) and abs(item["end"] - item["start"] - elapsed) < 5,
                    "Invalid completed allocated cost")
            charged = max(charged, elapsed)
        totals[phase] += charged * item["allocated_gpus"]
        if phase == "generation":
            s = item["shard"]
            require(type(s) is int and 0 <= s < plan["nshards"] and s not in generations
                    and item["nshards"] == plan["nshards"] and item["allocated_gpus"] == 1
                    and item["cap_seconds"] == plan["shard_cap_seconds"][s], "Duplicate/unplanned shard reservation")
            generations[s] = item
            if s == shard:
                chosen = item
    require(all(totals[p] <= PHASE_SECONDS[p] for p in totals) and sum(totals.values()) <= TOTAL_SECONDS,
            "Independent phase or 45 GPUh budget exceeded (failures are charged)")
    require(set(generations) == set(range(plan["nshards"])), "All shards must be preregistered/reserved")
    require(type(shard) is int and type(nshards) is int and nshards == plan["nshards"]
            and chosen is not None and chosen["status"] == "reserved" and chosen["gpu_index"] == gpu
            and chosen["cap_seconds"] == cap and chosen["hostname"] == os.uname().nodename,
            "Wrong shard/cap/GPU reservation or existing owner; never steal")
    return chosen, totals


def managed_scope(item, state_dir, gpu):
    uuid = smoke.supervisor_scope(state_dir, item["supervisor_pid"], gpu, released=True)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "Supervisor must expose exactly the reserved GPU UUID")
    ancestors, pid = [], os.getppid()
    while pid > 1:
        args = smoke.proc_args(pid)
        ancestors.append(args)
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        pid = int(fields[1])
    holder = any(any(os.path.basename(a) == b"gpu_hold.py" for a in args)
                 and smoke.action_matches(args, "exec") and any(a == b"--gpus" and b == str(gpu).encode()
                 for a, b in zip(args, args[1:])) for args in ancestors)
    bounded = False
    for args in ancestors:
        if not any(os.path.basename(a) == b"timeout" for a in args):
            continue
        if b"--kill-after=30s" not in args or b"--signal=TERM" not in args:
            continue
        for a in args:
            try:
                seconds = float(a[:-1]) if a.endswith(b"s") else -1
            except ValueError:
                continue
            if 0 < seconds <= item["cap_seconds"] - 30:
                bounded = True
    require(holder and bounded, "Run requires caller GNU timeout/kill30 and single-GPU gpu_hold exec ancestry")
    return uuid


@contextlib.contextmanager
def deadline(start, cap):
    require(finite_number(start) and 0 <= time.monotonic() - start < cap - 35, "Reserved allocation clock exhausted")
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Existing timer; cannot replace supervision")
    old = signal.getsignal(signal.SIGALRM)
    def expired(_signum, _frame):
        raise TimeoutError("Declared shard wall exceeded including load/save/decode")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, cap - 35 - (time.monotonic() - start))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def strict_reuse(job, plan, digest):
    path = Path(job["output"])
    require(path.resolve() == path, "Output symlink forbidden")
    receipt_path = path.with_suffix(".receipt.json")
    if not path.exists() and not receipt_path.exists():
        return None
    require(path.is_file() and receipt_path.is_file(), "Old output lacks strict SHA/decode receipt; do not overwrite")
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt["schema"] == "teacher_devset_case_receipt_v1" and receipt["success"] is True
            and receipt["plan_sha256"] == digest and receipt["method"] == plan["method"]
            and receipt["job"] == job and receipt["parameters_unchanged"] is True, "Unbound old output; no reuse")
    smoke.validate_stats(plan["method"], receipt["stats"]["method"], receipt["stats"]["observed"])
    require(smoke.validate_video(path, plan["runtime"]["tools"]) == receipt["video"], "Old media hash/decode drift")
    return receipt


@contextlib.contextmanager
def compile_timing(torch, stats):
    original = torch.compile
    stats.update(torch_compile_setup_seconds=0.0, first_compiled_call_seconds=0.0,
                 compiled_functions=0, caveat="first-call timing includes compile AND kernel; not pure compiler time")
    def measured(*args, **kwargs):
        started = time.monotonic()
        compiled = original(*args, **kwargs)
        stats["torch_compile_setup_seconds"] += time.monotonic() - started
        stats["compiled_functions"] += 1
        first = True
        def call(*call_args, **call_kwargs):
            nonlocal first
            if not first:
                return compiled(*call_args, **call_kwargs)
            first = False
            torch.cuda.synchronize(0)
            begin = time.monotonic()
            try:
                return compiled(*call_args, **call_kwargs)
            finally:
                torch.cuda.synchronize(0)
                stats["first_compiled_call_seconds"] += time.monotonic() - begin
        return call
    torch.compile = measured
    try:
        yield
    finally:
        torch.compile = original


def generate_case(pipe, torch, generator, method_context, plan, digest, job, budget_check):
    path = Path(job["output"])
    require(not path.exists() and not path.with_suffix(".receipt.json").exists(), "Existing output; never overwrite")
    exclusive(path.with_suffix(".claim.json"), dict(plan_sha256=digest, job=job, pid=os.getpid(), hostname=os.uname().nodename))
    stats = dict(observed=dict(model_forwards=0, scheduler_steps=0, progress_steps=0, finite_checks=0), method={})
    started = time.monotonic()
    receipt = dict(schema="teacher_devset_case_receipt_v1", plan_sha256=digest, method=plan["method"], job=job,
                   success=False, failure=True, parameters_unchanged=False, stats=stats, video=None, error=None,
                   compile={}, inference_including_compile_seconds=None, save_decode_seconds=None)
    tmp = path.with_suffix(".partial.mp4")
    def progress(timesteps):
        require(len(timesteps) == 50, "Exactly 50 timesteps required")
        for timestep in timesteps:
            budget_check()
            yield timestep
            stats["observed"]["progress_steps"] += 1
        budget_check()
    try:
        budget_check()
        before = smoke.parameter_hashes(pipe.dit, torch)
        case = plan["cases"][job["index"] % 94]
        kwargs = dict(plan["call"], seed=job["seed"], prompt=case["prompt"], negative_prompt=plan["negative_prompt"],
                      input_image=generator.load_image(case["image"]), progress_bar_cmd=progress)
        inference_start = time.monotonic()
        timing = compile_timing(torch, receipt["compile"]) if plan["method"] == "dymos" else contextlib.nullcontext()
        with timing, smoke.observe(pipe, torch, stats["observed"], budget_check):
            with method_context(pipe, call_kwargs=kwargs, checkpoint="NONE", **plan["parameters"]) as method_stats:
                stats["method"] = method_stats
                video = pipe(**kwargs)
        receipt["inference_including_compile_seconds"] = time.monotonic() - inference_start
        budget_check()
        require(len(video) == 121, "Incomplete generated 121 frames")
        require(smoke.parameter_hashes(pipe.dit, torch) == before, "Baseline parameters changed; method isolation failed")
        receipt["parameters_unchanged"] = True
        smoke.validate_stats(plan["method"], stats["method"], stats["observed"])
        require(not tmp.exists(), "Existing partial output; never steal/resume")
        save_start = time.monotonic()
        generator.save_video(video, str(tmp), fps=24, quality=5)
        os.chmod(tmp, 0o600)
        budget_check()
        media = smoke.validate_video(tmp, plan["runtime"]["tools"])
        budget_check()
        # Atomic link refuses existing destinations, unlike replace.
        os.link(tmp, path)
        tmp.unlink()
        media["file"] = record(path, nonempty=True)
        budget_check()
        receipt.update(video=media, success=True, failure=False, save_decode_seconds=time.monotonic() - save_start)
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
        exclusive(path.with_suffix(".receipt.json"), receipt)
    return receipt


def _run(path, approved_sha, node_preflight, approved_node_sha, ledger_path, gpu_index, shard, nshards,
         gen_cap_seconds, state_dir, gpu_authorized=False):
    require(gpu_authorized is True, "Explicit --gpu-authorized required; prepare is not authorization")
    require(type(gpu_index) is int and 0 <= gpu_index <= 7, "Exactly one physical GPU index required")
    require(os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_WORLD_SIZE", "1") == "1",
            "No distributed/multi-GPU launch")
    plan, models = verify_plan(path, approved_sha, node_preflight, approved_node_sha)
    ledger_path = Path(ledger_path).absolute()
    require(ledger_path.is_file(), "Caller must preregister reserved allocation ledger")
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_bytes())
        item, _ = validate_ledger(ledger, plan, approved_sha, shard, nshards, gpu_index, gen_cap_seconds)
        uuid, start = managed_scope(item, state_dir, gpu_index), item["allocation_start_monotonic"]
        require(finite_number(start) and 0 <= time.monotonic() - start < gen_cap_seconds - 35
                and finite_number(item["start"]) and abs(time.time() - item["start"] - (time.monotonic() - start)) < 5,
                "Invalid/exhausted caller reservation clock")
        folder = Path(plan["output_root"])
        exclusive(folder / f"shard{shard}.claim.json", dict(plan_sha256=approved_sha, shard=shard, nshards=nshards,
                  cap_seconds=gen_cap_seconds, gpu_index=gpu_index, gpu_uuid=uuid, pid=os.getpid(), start=item["start"]))
        item.update(status="running", worker_pid=os.getpid())
        checkpoint(ledger_path, ledger)
    receipt = dict(schema="teacher_devset_shard_receipt_v1", plan_sha256=approved_sha, method=plan["method"],
                   shard=shard, nshards=nshards, allocated_gpus=1, cap_seconds=gen_cap_seconds, success=False,
                   start=item["start"], GPU_used=False, quality_pass=False, sota_claim=False, jobs=[],
                   load_seconds=None, peak_memory_bytes=None, error=None,
                   allocation_released=False, cost_status="lower_bound; caller charges until tree/GPU release")
    torch = None
    def budget_check():
        require(time.monotonic() - start < gen_cap_seconds - 35, "Declared generation cap exhausted")
    try:
        with contextlib.nullcontext():
            for key, value in dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True").items():
                os.environ[key] = value
            sys.path.insert(0, str(Path(plan["root"]) / "DiffSynth-Studio"))
            sys.path.insert(0, str(Path(plan["root"]) / "scripts"))
            import torch as torch_module
            torch = torch_module
            require(torch.cuda.device_count() == 1, "Exactly one visible CUDA device required")
            torch.cuda.set_device(0)
            receipt["GPU_used"] = True
            torch.cuda.reset_peak_memory_stats(0)
            import gen_jobs
            require(gen_jobs.NEG == plan["negative_prompt"], "Imported NEG drift")
            if plan["method"] == "alg":
                from teacher_alg import teacher_alg as method_context
            else:
                from teacher_dymos import teacher_dymos as method_context
            load_start = time.monotonic()
            pipe = smoke.build_pipe("NONE", models, gen_jobs)
            receipt["load_seconds"] = time.monotonic() - load_start
            budget_check()
            for job in plan["jobs"][shard::nshards]:
                budget_check()
                previous = strict_reuse(job, plan, approved_sha)
                if previous is not None:
                    receipt["jobs"].append(dict(job=job, reused=True,
                        receipt_sha256=sha(Path(job["output"]).with_suffix(".receipt.json").read_bytes())))
                else:
                    result = generate_case(pipe, torch, gen_jobs, method_context, plan, approved_sha, job, budget_check)
                    receipt["jobs"].append(dict(job=job, reused=False, stats=result["stats"], elapsed_seconds=result["elapsed_seconds"]))
                checkpoint(folder / f"shard{shard}.progress.json", receipt)
            budget_check()
            receipt["success"] = True
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if torch is not None and receipt["GPU_used"]:
            receipt["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
        receipt.update(end=time.time(), elapsed_seconds=time.monotonic() - start)
        receipt["elapsed_GPUh"] = receipt["elapsed_seconds"] / 3600
        receipt["budget_exceeded"] = receipt["elapsed_seconds"] >= gen_cap_seconds
        receipt["success"] = receipt["success"] and not receipt["budget_exceeded"]
        receipt["failure"] = not receipt["success"]
        exclusive(folder / f"shard{shard}.finished.json", receipt)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = json.loads(ledger_path.read_bytes())
            matching = [a for a in ledger["allocations"] if a["phase"] == "generation" and a["shard"] == shard]
            require(len(matching) == 1 and matching[0].get("worker_pid") == os.getpid()
                    and ledger["plan_sha256"] == approved_sha, "Owned ledger entry changed; do not overwrite others")
            # Only caller can finish charging after process-tree/CUDA release.
            matching[0].update(worker_complete=True, worker_success=receipt["success"],
                               worker_elapsed_seconds=receipt["elapsed_seconds"],
                               receipt_sha256=sha((folder / f"shard{shard}.finished.json").read_bytes()))
            checkpoint(ledger_path, ledger)
    return receipt


def run(path, approved_sha, node_preflight, approved_node_sha, ledger_path, gpu_index, shard, nshards,
        gen_cap_seconds, state_dir, gpu_authorized=False):
    require(gpu_authorized is True, "Explicit --gpu-authorized required; prepare is not authorization")
    require(type(gen_cap_seconds) is int and gen_cap_seconds > 35, "Explicit declared shard allowance required")
    ledger = json.loads(Path(ledger_path).read_bytes())
    matching = [a for a in ledger["allocations"] if a["phase"] == "generation" and a["shard"] == shard]
    require(ledger["schema"] == "teacher_devset_allocations_v1" and ledger["plan_sha256"] == approved_sha
            and len(matching) == 1 and matching[0]["status"] == "reserved"
            and matching[0]["cap_seconds"] == gen_cap_seconds, "Missing caller-owned reservation")
    old = {}
    def interrupted(signum, _frame):
        raise InterruptedError(f"Caller signal {signum}; fail closed")
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, interrupted)
    try:
        # Includes CPU admission, load/compile, inference, save, decode and receipts.
        with deadline(matching[0]["allocation_start_monotonic"], gen_cap_seconds):
            return _run(path, approved_sha, node_preflight, approved_node_sha, ledger_path, gpu_index,
                        shard, nshards, gen_cap_seconds, state_dir, gpu_authorized)
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--method", choices=PARAMS)
    parser.add_argument("--node-preflight", type=Path, required=True)
    parser.add_argument("--approved-node-preflight-sha256", required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--approved-throughput-sha256")
    parser.add_argument("--throughput-receipt", type=Path)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--nshards", type=int)
    parser.add_argument("--gen-cap-seconds", type=int)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--state-dir", type=Path, default=smoke.LOCAL_ROOT / "wm_logs/gpu_hold")
    args = parser.parse_args(argv)
    if not args.run:
        if not args.method or args.gpu_authorized or args.gpu_index is not None or args.shard is not None or args.ledger:
            parser.error("Default CPU prepare requires one method; no GPU/shard/ledger arguments")
        path, digest = prepare(args.method, args.node_preflight, args.approved_node_preflight_sha256,
                               args.nshards if args.nshards is not None else 1, args.gen_cap_seconds,
                               approved_throughput_sha=args.approved_throughput_sha256,
                               throughput_receipt=args.throughput_receipt)
        value = json.loads(path.read_bytes())
        print(json.dumps(dict(plan=str(path), plan_sha256=digest, admission=value["admission"], GPU_used=False,
                              base_reuse_allowed=False, quality_pass=False, sota_claim=False)))
    else:
        if not args.plan or not args.approved_plan_sha256 or args.gpu_index is None or args.shard is None \
                or args.nshards is None or args.gen_cap_seconds is None or not args.ledger or not args.gpu_authorized:
            parser.error("Run requires approved plan SHA, GPU authorization/index, shard/nshards, declared cap and reserved ledger")
        result = run(args.plan, args.approved_plan_sha256, args.node_preflight,
                     args.approved_node_preflight_sha256, args.ledger, args.gpu_index, args.shard,
                     args.nshards, args.gen_cap_seconds, args.state_dir, args.gpu_authorized)
        print(json.dumps(result, allow_nan=False))
        if not result["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
