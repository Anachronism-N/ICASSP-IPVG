#!/usr/bin/env python3
"""Fresh native NONE baseline with immutable CPU plan and managed execution.

Preparation never authorizes a GPU. Execution is delegated to the separately
approved single-GPU parent. Formal shard0 is a budget/throughput canary, not an
extra smoke or quality-selection sample. Historical reuse remains forbidden.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import run_teacher_devset as dev

smoke = dev.smoke
require, encoded, sha = smoke.require, smoke.encoded, smoke.sha
record, check_record, exclusive = smoke.record, smoke.check_record, smoke.exclusive
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "teacher_base_plan_v1"
PHASE_SECONDS = dict(generation=72000, score=18000)
TOTAL_SECONDS = 90000
NSHARDS = 16
CAP_SECONDS = 4500
CANARY_SHARD = 0
BLOCKERS = [
    "Native NONE baseline throughput has not been measured/approved; ALG/DyMoS timing is not baseline PASS",
    "Native baseline parent-owned managed single-GPU timeout/release accounting is not implemented",
]


def output_root(root):
    return Path(root) / "eval/teacher_base_fresh_20261028"


def allocations(node_ranks):
    require(isinstance(node_ranks, (list, tuple)) and len(node_ranks) == 2
            and all(type(rank) is int and 0 <= rank <= 3 for rank in node_ranks)
            and len(set(node_ranks)) == 2, "Native base requires two explicit distinct node ranks0..3")
    return [dict(shard=i * 8 + gpu, node_rank=rank, gpu_index=gpu, allocated_gpus=1, cap_seconds=CAP_SECONDS)
            for i, rank in enumerate(node_ranks) for gpu in range(8)]


def sources(root):
    return sorted(set(dev.sources(root)) | {Path(root) / p for p in (
        "scripts/run_teacher_base.py", "scripts/prepare_teacher_comparison.py",
        "tests/test_teacher_base.py", "tests/test_prepare_teacher_comparison.py",
        "scripts/run_teacher_generation.py", "tests/test_teacher_generation.py",
        "scripts/verify_teacher_history.py", "tests/test_verify_teacher_history.py")})


def model_identity(models):
    value = [{k: m[k] for k in ("asset", "bytes", "sha256")} for m in models]
    require([m["asset"] for m in value] == list(smoke.ASSETS), "Wrong native model asset set/order")
    require(all(type(m["bytes"]) is int and m["bytes"] > 0 for m in value), "Empty model asset")
    return value


def runtime_identity(value):
    return {**value, "files": [{k: v for k, v in row.items() if k != "mtime_ns"}
                               for row in value["files"]]}


def score_bundle_binding(path, approved_sha):
    """Require an externally approved content manifest, never a guessed score hash."""
    expected = record(path, nonempty=True)
    require(expected["sha256"] == approved_sha, "Approved score bundle SHA mismatch")
    import score_teacher_devset as scoring
    scoring.verify_bundle(path, approved_sha, verify_content=True)
    require(record(path, nonempty=True) == expected, "Score bundle changed during verification")
    return expected


def validate_contract(plan):
    if plan.get('schema') == 'teacher_one_recovery_plan_v1':
        import prepare_teacher_one_recovery as once
        return once.validate_plan(plan)
    require(plan.get("schema") == SCHEMA and plan.get("private") is True
            and plan.get("method") == "base" and plan.get("checkpoint") == "NONE"
            and plan.get("parameters") == {} and plan.get("active_steps") == 0
            and plan.get("method_context_installed") is False, "Native base must be NONE without any method context")
    require(encoded(plan.get("call")) == encoded(smoke.SPEC) and plan.get("fps") == 24
            and plan.get("seeds") == [0, 1] and plan.get("video_count") == 188
            and plan.get("phase_seconds") == PHASE_SECONDS and plan.get("total_seconds") == TOTAL_SECONDS,
            "Native base input/spec/independent 20+5 GPUh budget drift")
    require(plan.get("GPU_used") is False and plan.get("quality_pass") is False
            and plan.get("sota_claim") is False, "Preparation cannot claim GPU/quality/SOTA PASS")
    n, caps = plan["nshards"], plan["shard_cap_seconds"]
    require(type(n) is int and n == NSHARDS and caps == [CAP_SECONDS] * NSHARDS
            and all(type(c) is int for c in caps)
            and sum(caps) == PHASE_SECONDS["generation"], "Native generation requires fixed16x4500s=20 GPUh")
    require(plan.get("admission") == dict(status="BLOCKED", reasons=BLOCKERS,
            native_throughput_PASS=False, runnable=False, estimated_generation_seconds=None),
            "Unimplemented baseline admission cannot be made runnable")
    if "allocations" in plan:
        require(encoded(plan["allocations"]) == encoded(allocations(plan["node_ranks"])),
                "Native static two-owner allocation mapping drift")
    require(sha(encoded(model_identity(plan["models"]))) == plan["model_content_sha256"],
            "Native model content digest mismatch")


def prepare(node_preflight, approved_node_sha, score_bundle, approved_score_sha, *,
            nshards=16, gen_cap_seconds=None, node_ranks=None, root=ROOT):
    root = Path(root).resolve()
    physical_allocations = allocations(node_ranks)
    require(type(nshards) is int and nshards == NSHARDS, "Native base requires fixed16 shards")
    cap = CAP_SECONDS if gen_cap_seconds is None else gen_cap_seconds
    require(type(cap) is int and cap == CAP_SECONDS, "Native base requires fixed4500s per shard")
    folder = output_root(root)
    require(not folder.exists() and folder.resolve() == folder,
            "Fresh output root already exists or is symlinked; never overwrite or reuse old baseline")
    score_record = score_bundle_binding(score_bundle, approved_score_sha)
    raw, cases, inputs, manifest = dev.frozen_cases(root)
    pf_record = record(node_preflight, nonempty=True)
    require(pf_record["sha256"] == approved_node_sha, "Approved node preflight SHA mismatch")
    pf = json.loads(Path(node_preflight).read_bytes())
    models = smoke.models_from_preflight(pf)
    negative = smoke.generator_negative(root)
    value = dict(schema=SCHEMA, private=True, root=str(root), method="base", parameters={},
        checkpoint="NONE", method_context_installed=False, active_steps=0, call=dict(smoke.SPEC), fps=24,
        seeds=[0, 1], video_count=188, phase_seconds=dict(PHASE_SECONDS), total_seconds=TOTAL_SECONDS,
        cases_file=manifest, cases=cases, inputs=inputs, jobs=dev.jobs_for(cases, root, "base"),
        negative_prompt=negative, negative_sha256=sha(negative.encode()),
        source_files=[record(p) for p in sources(root)], runtime=smoke.runtime_records(),
        node_preflight=pf_record, models=models, model_content_sha256=pf["model_content_sha256"],
        output_root=str(folder), score_bundle=score_record, nshards=nshards, shard_cap_seconds=[cap] * nshards,
        node_ranks=list(node_ranks), allocations=physical_allocations,
        admission=dict(status="BLOCKED", reasons=list(BLOCKERS), native_throughput_PASS=False,
                       runnable=False, estimated_generation_seconds=None),
        historical_base_reuse=dev.base_reuse(root, manifest), GPU_used=False, quality_pass=False, sota_claim=False)
    value["jobs"] = [dict(j, output=str(folder / f"seed{j['seed']}/videos/base/{j['case_id']}.mp4"))
                     for j in value["jobs"]]
    validate_contract(value)
    folder.mkdir(mode=0o700)
    for seed in (0, 1):
        seed_root = folder / f"seed{seed}"
        (seed_root / "videos/base").mkdir(mode=0o700, parents=True)
        with (seed_root / "cases.jsonl").open("xb") as handle:
            os.chmod(handle.name, 0o600)
            handle.write(raw)
    path = folder / "plan.BLOCKED.json"
    exclusive(path, value)
    return path, sha(path.read_bytes())


def verify_plan(path, approved_sha, node_preflight, approved_node_sha):
    path = Path(path).absolute()
    require(path.resolve() == path and path.stat().st_mode & 0o077 == 0
            and path.parent.stat().st_mode & 0o077 == 0, "Native plan must be private/non-symlink")
    require(sha(path.read_bytes()) == approved_sha, "Approved native plan SHA mismatch")
    plan = json.loads(path.read_bytes())
    validate_contract(plan)
    root = Path(plan["root"])
    require(root.resolve() == root and path == output_root(root) / "plan.BLOCKED.json"
            and plan["output_root"] == str(path.parent), "Wrong fresh baseline root")
    require({r["path"] for r in plan["source_files"]} == {str(p) for p in sources(root)}, "Native source set drift")
    for row in plan["source_files"]:
        check_record(row)
    require(runtime_identity(smoke.runtime_records()) == runtime_identity(plan["runtime"]), "Native runtime drift")
    raw, cases, inputs, manifest = dev.frozen_cases(root)
    require((cases, inputs, manifest) == (plan["cases"], plan["inputs"], plan["cases_file"]), "Native case/input drift")
    expected_jobs = [dict(j, output=str(path.parent / f"seed{j['seed']}/videos/base/{j['case_id']}.mp4"))
                     for j in dev.jobs_for(cases, root, "base")]
    require(plan["jobs"] == expected_jobs, "Native seed/job drift")
    for seed in (0, 1):
        require((path.parent / f"seed{seed}/cases.jsonl").read_bytes() == raw, "Native seed case bytes drift")
    pf = record(node_preflight, nonempty=True)
    require(pf["sha256"] == approved_node_sha and pf == plan["node_preflight"], "Native preflight binding drift")
    models = smoke.models_from_preflight(json.loads(Path(node_preflight).read_bytes()))
    require(model_identity(models) == model_identity(plan["models"]), "Native model drift")
    negative = smoke.generator_negative(root)
    require(negative == plan["negative_prompt"] and sha(negative.encode()) == plan["negative_sha256"], "Native NEG drift")
    require(plan["historical_base_reuse"] == dev.base_reuse(root, manifest), "Historical denied reuse drift")
    check_record(plan["score_bundle"], nonempty=True)
    require(score_bundle_binding(plan["score_bundle"]["path"], plan["score_bundle"]["sha256"]) == plan["score_bundle"],
            "Native score bundle drift")
    return plan


def validate_native_stats(stats):
    require(stats.get("model_forwards") == 100 and stats.get("scheduler_steps") == 50
            and stats.get("progress_steps") == 50 and type(stats.get("finite_checks")) is int
            and stats["finite_checks"] > 0, "Incomplete native 50-step/100-forward finite observation")


def _generate_native_case(pipe, torch, generator, plan, digest, job, budget_check):
    """Internal CPU-testable component; does not authorize/release/account a GPU."""
    validate_contract(plan)
    require(digest == sha(encoded(plan)), "Native case receipt requires exact plan content SHA")
    require(job in plan["jobs"], "Unknown native job")
    path = Path(job["output"])
    require(path.resolve() == path and not path.exists() and not path.with_suffix(".receipt.json").exists(),
            "Existing or symlink native output; never overwrite")
    tmp = path.with_suffix(".partial.mp4")
    require(not tmp.exists(), "Existing native partial; never steal")
    case = plan["cases"][job["index"] % 94]
    input_binding = dict(case_id=case["id"], prompt_sha256=case["prompt_sha256"],
                         image=record(case["image"], nonempty=True), gt=record(case["gt"], nonempty=True))
    require(input_binding["image"]["sha256"] == case["image_sha256"]
            and input_binding["gt"]["sha256"] == case["video_sha256"], "Native original input content drift")
    started = time.monotonic()
    stats = dict(model_forwards=0, scheduler_steps=0, progress_steps=0, finite_checks=0)
    receipt = dict(schema="teacher_base_case_receipt_v1", plan_sha256=digest, method="base",
        checkpoint="NONE", method_context_installed=False, active_steps=0, job=job, success=False, failure=True,
        parameters_before=None, parameters_after=None, parameters_unchanged=False, stats=stats, video=None, error=None,
        input_binding=input_binding, parameters_before_seconds=None, parameters_after_seconds=None,
        score_bundle_sha256=plan["score_bundle"]["sha256"], allocation_released=False,
        cost_status="internal component; not a supervised allocated-GPU completion receipt")
    def progress(timesteps):
        require(len(timesteps) == 50, "Exactly 50 native timesteps required")
        for timestep in timesteps:
            budget_check()
            yield timestep
            stats["progress_steps"] += 1
        budget_check()
    try:
        budget_check()
        hash_start = time.monotonic()
        receipt["parameters_before"] = smoke.parameter_hashes(pipe.dit, torch)
        receipt["parameters_before_seconds"] = time.monotonic() - hash_start
        kwargs = dict(plan["call"], seed=job["seed"], prompt=case["prompt"], negative_prompt=plan["negative_prompt"],
                      input_image=generator.load_image(case["image"]), progress_bar_cmd=progress)
        inference_start = time.monotonic()
        # Observation wraps only original model/scheduler, with no ALG/DyMoS context.
        with smoke.observe(pipe, torch, stats, budget_check):
            video = pipe(**kwargs)
        receipt["inference_seconds"] = time.monotonic() - inference_start
        budget_check()
        require(len(video) == 121, "Incomplete native generated 121 frames")
        hash_start = time.monotonic()
        receipt["parameters_after"] = smoke.parameter_hashes(pipe.dit, torch)
        receipt["parameters_after_seconds"] = time.monotonic() - hash_start
        receipt["parameters_unchanged"] = receipt["parameters_before"] == receipt["parameters_after"]
        require(receipt["parameters_unchanged"], "Native baseline parameters changed")
        validate_native_stats(stats)
        save_start = time.monotonic()
        generator.save_video(video, str(tmp), fps=24, quality=5)
        os.chmod(tmp, 0o600)
        budget_check()
        media = smoke.validate_video(tmp, plan["runtime"]["tools"])
        budget_check()
        os.link(tmp, path)
        tmp.unlink()
        media["file"] = record(path, nonempty=True)
        for row in (input_binding["image"], input_binding["gt"]):
            check_record(row, nonempty=True)
        budget_check()
        receipt.update(video=media, success=True, failure=False, save_decode_seconds=time.monotonic() - save_start)
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if receipt["parameters_before"] is not None and receipt["parameters_after"] is None:
            try:
                receipt["parameters_after"] = smoke.parameter_hashes(pipe.dit, torch)
                receipt["parameters_unchanged"] = receipt["parameters_before"] == receipt["parameters_after"]
            except BaseException as exc:
                receipt.update(success=False, failure=True, error=f"Final native parameter evidence failure: {exc}")
        receipt["elapsed_seconds"] = time.monotonic() - started
        exclusive(path.with_suffix(".receipt.json"), receipt)
    return receipt


def native_canary_estimate(plan, case_receipts, *, load_seconds, allocated_elapsed_seconds):
    """Feasibility only; caller must independently verify the parent/release chain."""
    validate_contract(plan)
    require(type(load_seconds) in (int, float) and math.isfinite(load_seconds) and load_seconds > 0
            and type(allocated_elapsed_seconds) in (int, float) and math.isfinite(allocated_elapsed_seconds)
            and 0 < allocated_elapsed_seconds <= plan["shard_cap_seconds"][0], "Invalid native canary allocation/load timing")
    expected_jobs = [j for j in plan["jobs"] if j["index"] % NSHARDS == CANARY_SHARD]
    require(len(expected_jobs) == 12 and len(case_receipts) == 12
            and [r.get("job") for r in case_receipts] == expected_jobs, "Native canary is the fixed12 formal shard0 jobs")
    from prepare_teacher_comparison import validate_parameter_maps, finite_evidence
    baseline, seconds = None, []
    for receipt in case_receipts:
        finite_evidence(receipt)
        require(receipt.get("schema") == "teacher_base_case_receipt_v1" and receipt.get("method") == "base"
                and receipt.get("checkpoint") == "NONE" and receipt.get("method_context_installed") is False
                and receipt.get("active_steps") == 0 and receipt.get("success") is True
                and receipt.get("failure") is False and receipt.get("error") is None
                and receipt.get("plan_sha256") == sha(encoded(plan))
                and receipt.get("score_bundle_sha256") == plan["score_bundle"]["sha256"],
                "Canary requires successful native NONE evidence, not ALG/DyMoS timing")
        validate_native_stats(receipt["stats"])
        params = validate_parameter_maps(receipt["parameters_before"], receipt["parameters_after"])
        require(baseline is None or params == baseline, "Native canary parameter maps differ across cases")
        baseline = params
        elapsed = receipt.get("elapsed_seconds")
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0, "Missing real native per-case timing")
        seconds.append(elapsed)
    require(load_seconds + sum(seconds) <= allocated_elapsed_seconds + 1e-6,
            "Canary case/load timing exceeds entire charged parent allocation")
    import run_teacher_generation as generation
    require(hasattr(generation, "CLEANUP_SECONDS") and hasattr(generation, "WORKER_MARGIN_SECONDS"),
            "BLOCKED: managed generation cleanup/worker margin contract is missing")
    cleanup, margin = generation.CLEANUP_SECONDS, generation.WORKER_MARGIN_SECONDS
    require(type(cleanup) is int and cleanup > 0 and type(margin) is int and margin > 0,
            "Managed cleanup/worker margin constants must be frozen positive integers")
    tail = cleanup + margin
    residual = max(float(tail), allocated_elapsed_seconds - load_seconds - sum(seconds))
    per_case = max(seconds)
    body_seconds = [load_seconds + len([j for j in plan["jobs"] if j["index"] % NSHARDS == s]) * per_case
                    for s in range(NSHARDS)]
    shard_seconds = [body + residual for body in body_seconds]
    total = sum(shard_seconds)
    caps=plan["shard_cap_seconds"]
    prior=plan.get("prior_generation",{}).get("upper_seconds",0.0)
    ready = all(body < cap - tail for body,cap in zip(body_seconds,caps))
    ready = ready and all(s < cap for s,cap in zip(shard_seconds,caps)) and prior+total <= PHASE_SECONDS["generation"]
    return dict(schema="teacher_base_native_canary_admission_v1", method="base", canary_shard=CANARY_SHARD,
        canary_jobs=12, measured_allocation_seconds=allocated_elapsed_seconds, load_seconds=load_seconds,
        max_case_seconds=per_case, allocation_residual_seconds=residual, cleanup_seconds=cleanup,
        worker_margin_seconds=margin, estimated_body_seconds=body_seconds, estimated_shard_seconds=shard_seconds,
        estimated_generation_seconds=total, status="READY" if ready else "BLOCKED",
        native_throughput_PASS=ready, quality_pass=False, estimate_is_guarantee=False,
        reason="verified native formal shard0 only; body must fit4500-cleanup-worker_margin and all16 fit20 GPUh")


def verify_native_canary_admission(parent_receipt, plan):
    """Verify the supervised formal native shard0 before admitting remaining15."""
    import run_teacher_generation as generation
    import prepare_teacher_comparison as comparison
    validate_contract(plan)
    require(parent_receipt.get("method") == "base" and parent_receipt.get("shard") == CANARY_SHARD,
            "Native admission requires formal base shard0, never candidate timing")
    path = generation.shard_folder(plan, CANARY_SHARD) / "parent.json"
    parent, expected = comparison.read_evidence(path, sha(path.read_bytes()))
    require(parent == parent_receipt and generation.verify_parent_release(parent, plan) is True
            and parent.get("success") is True and parent.get("exit_code") == 0
            and parent.get("timed_out") is False and parent.get("error") is None and parent.get("interrupted") is None,
            "Native canary needs complete actual parent launch/release chain")
    worker, worker_record = comparison.read_evidence(parent["worker"]["path"], parent["worker"]["sha256"])
    require(worker.get("success") is True and worker.get("error") is None
            and worker.get("method") == "base" and worker.get("shard") == CANARY_SHARD
            and [row["job"] for row in worker["cases"]] == plan["jobs"][CANARY_SHARD::NSHARDS],
            "Native canary worker is not the fixed12 formal jobs")
    receipts = []
    for row in worker["cases"]:
        helper, _ = comparison.read_evidence(row["case_receipt"]["path"], row["case_receipt"]["sha256"])
        comparison.verify_case_receipt(row["case_receipt"]["path"], row["case_receipt"]["sha256"], plan=plan,
            plan_sha256=parent["plan_sha256"], job=row["job"], parameters_before=row["parameters_before"],
            parameters_after=row["parameters_after"], input_binding=row["input_binding"])
        receipts.append(helper)
    admission = native_canary_estimate(plan, receipts, load_seconds=worker["load_seconds"],
                                       allocated_elapsed_seconds=parent["elapsed_seconds"])
    require(record(path, nonempty=True) == expected and record(worker_record["path"], nonempty=True) == worker_record,
            "Native canary parent/worker changed during admission")
    return admission["native_throughput_PASS"]


def run(path=None, approved_sha=None, shard=None, rank=None, *, gpu_authorized=False,
        state_dir="/dockerdata/wm_logs/gpu_hold", local_root=Path("/dockerdata/wm_tmp")):
    require(gpu_authorized is True, "Explicit --gpu-authorized required in addition to approved execution sidecar")
    require(path is not None and isinstance(approved_sha, str) and len(approved_sha) == 64
            and all(c in "0123456789abcdef" for c in approved_sha)
            and type(shard) is int and 0 <= shard < NSHARDS and type(rank) is int and 0 <= rank <= 3,
            "BLOCKED: native run requires approved execution SHA, exact shard and node rank")
    import run_teacher_generation as generation
    return generation.supervise(path, approved_sha, "base", shard, rank, gpu_authorized=True,
                                state_dir=state_dir, local_root=local_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--verify", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--node-preflight", type=Path)
    parser.add_argument("--approved-node-preflight-sha256")
    parser.add_argument("--score-bundle", type=Path)
    parser.add_argument("--approved-score-bundle-sha256")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--approved-execution-sha256")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--state-dir", type=Path, default=Path("/dockerdata/wm_logs/gpu_hold"))
    parser.add_argument("--local-root", type=Path, default=Path("/dockerdata/wm_tmp"))
    parser.add_argument("--node-ranks", type=int, nargs=2)
    parser.add_argument("--nshards", type=int, default=16)
    parser.add_argument("--gen-cap-seconds", type=int)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        parser.error("BLOCKED: direct native worker forbidden; use the hash-approved generation supervisor")
    if args.run:
        if args.gpu is not None:
            parser.error("Physical GPU is fixed in the approved execution sidecar; no GPU substitution")
        path, digest = run(args.execution, args.approved_execution_sha256, args.shard, args.rank,
            gpu_authorized=args.gpu_authorized, state_dir=args.state_dir, local_root=args.local_root)
        print(json.dumps(dict(parent_receipt=str(path), parent_receipt_sha256=digest, GPU_used=True, quality_pass=False)))
        return
    if args.gpu is not None or args.gpu_authorized:
        parser.error("CPU prepare/verify refuses GPU arguments")
    if not args.node_preflight or not args.approved_node_preflight_sha256:
        parser.error("CPU preparation/verification requires explicitly approved node preflight")
    if args.verify:
        if not args.plan or not args.approved_plan_sha256:
            parser.error("Verify requires plan and approved plan SHA")
        verify_plan(args.plan, args.approved_plan_sha256, args.node_preflight, args.approved_node_preflight_sha256)
        print(json.dumps(dict(contract_verified=True, admission="BLOCKED", runnable=False, GPU_used=False)))
    else:
        if not args.score_bundle or not args.approved_score_bundle_sha256 or not args.node_ranks:
            parser.error("CPU prepare requires externally approved score bundle and explicit two --node-ranks")
        path, digest = prepare(args.node_preflight, args.approved_node_preflight_sha256,
            args.score_bundle, args.approved_score_bundle_sha256, nshards=args.nshards, gen_cap_seconds=args.gen_cap_seconds,
            node_ranks=args.node_ranks)
        print(json.dumps(dict(plan=str(path), plan_sha256=digest, admission="BLOCKED", runnable=False, GPU_used=False)))


if __name__ == "__main__":
    main()
