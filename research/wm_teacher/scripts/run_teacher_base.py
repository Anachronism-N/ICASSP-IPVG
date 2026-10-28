#!/usr/bin/env python3
"""CPU-only fresh NONE baseline preparation; not a GPU launch authorization.

Native generation is an internal component, not an executable worker. Both run
entry points fail closed until native throughput and parent-owned single-GPU
GNU-timeout/release accounting have separately passed. No historical reuse.
"""
import argparse
import json
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
BLOCKERS = [
    "Native NONE baseline throughput has not been measured/approved; ALG/DyMoS timing is not baseline PASS",
    "Native baseline parent-owned managed single-GPU timeout/release accounting is not implemented",
]


def output_root(root):
    return Path(root) / "eval/teacher_base_fresh_20261028"


def sources(root):
    return sorted(set(dev.sources(root)) | {Path(root) / p for p in (
        "scripts/run_teacher_base.py", "scripts/prepare_teacher_comparison.py",
        "tests/test_teacher_base.py", "tests/test_prepare_teacher_comparison.py")})


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
    require(type(n) is int and 1 <= n <= 188 and len(caps) == n
            and all(type(c) is int and c > 35 for c in caps)
            and sum(caps) <= PHASE_SECONDS["generation"], "Native generation reservations exceed 20 GPUh")
    require(plan.get("admission") == dict(status="BLOCKED", reasons=BLOCKERS,
            native_throughput_PASS=False, runnable=False, estimated_generation_seconds=None),
            "Unimplemented baseline admission cannot be made runnable")
    require(sha(encoded(model_identity(plan["models"]))) == plan["model_content_sha256"],
            "Native model content digest mismatch")


def prepare(node_preflight, approved_node_sha, score_bundle, approved_score_sha, *,
            nshards=16, gen_cap_seconds=None, root=ROOT):
    root = Path(root).resolve()
    require(type(nshards) is int and 1 <= nshards <= 188, "nshards must be 1..188")
    cap = PHASE_SECONDS["generation"] // nshards if gen_cap_seconds is None else gen_cap_seconds
    require(type(cap) is int and cap > 35 and cap * nshards <= PHASE_SECONDS["generation"],
            "Native generation allowances must sum <=20 GPUh")
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
            and stats["finite_checks"] >= 550, "Incomplete native 50-step/100-forward finite observation")


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
    started = time.monotonic()
    stats = dict(model_forwards=0, scheduler_steps=0, progress_steps=0, finite_checks=0)
    receipt = dict(schema="teacher_base_case_receipt_v1", plan_sha256=digest, method="base",
        checkpoint="NONE", method_context_installed=False, active_steps=0, job=job, success=False, failure=True,
        parameters_before=None, parameters_after=None, parameters_unchanged=False, stats=stats, video=None, error=None,
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
        receipt["parameters_before"] = smoke.parameter_hashes(pipe.dit, torch)
        case = plan["cases"][job["index"] % 94]
        kwargs = dict(plan["call"], seed=job["seed"], prompt=case["prompt"], negative_prompt=plan["negative_prompt"],
                      input_image=generator.load_image(case["image"]), progress_bar_cmd=progress)
        inference_start = time.monotonic()
        # Observation wraps only original model/scheduler, with no ALG/DyMoS context.
        with smoke.observe(pipe, torch, stats, budget_check):
            video = pipe(**kwargs)
        receipt["inference_seconds"] = time.monotonic() - inference_start
        budget_check()
        require(len(video) == 121, "Incomplete native generated 121 frames")
        receipt["parameters_after"] = smoke.parameter_hashes(pipe.dit, torch)
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


def run(*args, gpu_authorized=False, **kwargs):
    require(gpu_authorized is True, "Explicit --gpu-authorized required; GPU pause remains in effect")
    raise ValueError("BLOCKED: native throughput and managed parent timeout/release accounting are unimplemented; no GPU run")


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
    parser.add_argument("--nshards", type=int, default=16)
    parser.add_argument("--gen-cap-seconds", type=int)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args(argv)
    if args.run or args.worker:
        run(gpu_authorized=args.gpu_authorized)
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
        if not args.score_bundle or not args.approved_score_bundle_sha256:
            parser.error("CPU prepare requires externally approved score bundle")
        path, digest = prepare(args.node_preflight, args.approved_node_preflight_sha256,
            args.score_bundle, args.approved_score_bundle_sha256, nshards=args.nshards, gen_cap_seconds=args.gen_cap_seconds)
        print(json.dumps(dict(plan=str(path), plan_sha256=digest, admission="BLOCKED", runnable=False, GPU_used=False)))


if __name__ == "__main__":
    main()
