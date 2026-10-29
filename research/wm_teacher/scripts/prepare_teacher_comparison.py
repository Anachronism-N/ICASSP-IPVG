#!/usr/bin/env python3
"""CPU-only exact-content fresh base/ALG/DyMoS comparison preparation.

An approved commit is not a common generation/score contract. This entry point
checks frozen inputs, model bytes, current common sources and runtime content,
and refuses old plans without the same externally approved score bundle. A
validated preparation is still BLOCKED for GPU execution/quality conclusions.
"""
import argparse
import json
import math
from pathlib import Path

import run_teacher_base as base
import prepare_teacher_recovery as recovery

dev, smoke = base.dev, base.smoke
require, encoded, sha = base.require, base.encoded, base.sha
record, check_record, exclusive = base.record, base.check_record, base.exclusive
SCHEMA = "teacher_comparison_plan_v1"
METHODS = ("base", "alg", "dymos")


def plan_output(plan):
    if plan.get('schema') == 'teacher_one_recovery_plan_v1':
        import prepare_teacher_one_recovery as once
        return once.output_root(plan['root'],plan['method'])
    return base.output_root(plan['root']) if plan['method']=='base' else recovery.plan_folder(plan['root'],plan['method'])


def plan_location(plan):
    name='plan.BLOCKED.json' if plan['method']=='base' and plan.get('schema') != 'teacher_one_recovery_plan_v1' else 'plan.blocked.json'
    return plan_output(plan)/name


def plan_prior(plan):
    return plan['prior_generation']['upper_seconds'] if 'prior_generation' in plan else 0.0


def common_contract(plan):
    root = Path(plan["root"])
    common_paths = {str(p) for p in dev.sources(root)} | {str(root / name) for name in (
        "scripts/run_teacher_generation.py", "tests/test_teacher_generation.py",
        "scripts/verify_teacher_history.py", "tests/test_verify_teacher_history.py")}
    if plan.get("schema") == "teacher_one_recovery_plan_v1":
        common_paths |= {str(root/name) for name in ("scripts/prepare_teacher_one_recovery.py", "tests/test_teacher_one_recovery.py")}
    rows = plan["source_files"]
    require(len(rows) == len({r["path"] for r in rows})
            and common_paths <= {r["path"] for r in rows}, "Missing/duplicate common generation sources")
    source_content = sorted([dict(path=str(Path(r["path"]).relative_to(root)), bytes=r["bytes"], sha256=r["sha256"])
                             for r in rows if r["path"] in common_paths], key=lambda r: r["path"])
    return dict(cases_file=plan["cases_file"], cases=plan["cases"], inputs=plan["inputs"],
        call=plan["call"], fps=plan["fps"], seeds=plan["seeds"], video_count=plan["video_count"],
        checkpoint=plan["checkpoint"], negative_prompt=plan["negative_prompt"], negative_sha256=plan["negative_sha256"],
        models=base.model_identity(plan["models"]), model_content_sha256=plan["model_content_sha256"],
        runtime=base.runtime_identity(plan["runtime"]), common_sources=source_content,
        score_bundle_sha256=plan["score_bundle"]["sha256"])


def _validate_shared_plan_content(plan, method, score_binding):
    require(plan.get("method") == method and plan.get("private") is True, "Wrong approved comparison method")
    if plan.get('schema') == 'teacher_one_recovery_plan_v1':
        import prepare_teacher_one_recovery as once
        once.validate_plan(plan)
    elif method == "base":
        base.validate_contract(plan)
    else:
        require(plan.get("schema") == recovery.SCHEMA and plan.get("common_contract_frozen") is True,
                "Strict comparison requires a new frozen recovery candidate; old devset schema/root is historical only")
        recovery.validate_plan(plan)
        require(plan.get("GPU_used") is False and plan.get("quality_pass") is False
                and plan.get("sota_claim") is False, "Preparation is not GPU/quality/SOTA evidence")
        for row in (plan["source"], plan["prior_generation"]["audit"],
                    plan["prior_generation"]["old_plan"], plan["prior_generation"]["old_ledger"]):
            recovery.check_content(row)
    require(plan.get("score_bundle") == score_binding,
            "Every newly approved plan must explicitly bind same score bundle; never retrofit historical plan")
    require(plan.get("call") == smoke.SPEC and plan.get("fps") == 24 and plan.get("checkpoint") == "NONE"
            and plan.get("seeds") == [0, 1] and plan.get("video_count") == 188, "Comparison generation spec drift")
    root = Path(plan["root"])
    require(root.resolve() == root, "Comparison root must be canonical")
    expected_root = plan_output(plan)
    require(plan["output_root"] == str(expected_root), "Comparison method output root drift")
    _, cases, inputs, manifest = dev.frozen_cases(root)
    require(plan["cases_file"] == manifest and plan["cases"] == cases and plan["inputs"] == inputs,
            "Comparison frozen case/image/raw prompt/GT mismatch")
    jobs = [dict(j, output=str(expected_root / f"seed{j['seed']}/videos/{method}/{j['case_id']}.mp4"))
            for j in dev.jobs_for(cases, root, method)]
    require(plan["jobs"] == jobs, "Comparison seed/case/output jobs mismatch")
    negative = smoke.generator_negative(root)
    require(plan["negative_prompt"] == negative and plan["negative_sha256"] == sha(negative.encode()), "Comparison NEG mismatch")
    require(plan["model_content_sha256"] == sha(encoded(base.model_identity(plan["models"]))), "Comparison model digest mismatch")
    if plan.get('schema') == 'teacher_one_recovery_plan_v1':
        import prepare_teacher_one_recovery as once
        expected_sources=once.sources(root,method)
    else:
        expected_sources = base.sources(root) if method == "base" else recovery.sources(root)
    require(len(plan["source_files"]) == len(expected_sources)
            and {r["path"] for r in plan["source_files"]} == {str(p) for p in expected_sources},
            "Missing/duplicate common generation sources or unapproved source set")
    for row in plan["source_files"]:
        check_record(row)


def validate_plan_content(plan, method, score_binding):
    _validate_shared_plan_content(plan, method, score_binding)
    for row in plan["runtime"]["files"]:
        check_record(row)
    for row in plan["models"]:
        check_record({k: row[k] for k in ("path", "bytes", "mtime_ns", "sha256")}, nonempty=True)
    require(base.runtime_identity(smoke.runtime_records()) == base.runtime_identity(plan["runtime"]),
            "Comparison current runtime content drift")


def verify_three_plans(plan_paths, approved_shas, score_bundle, approved_score_sha):
    require(set(plan_paths) == set(METHODS) and set(approved_shas) == set(METHODS), "Exactly base/ALG/DyMoS plans required")
    score_binding = base.score_bundle_binding(score_bundle, approved_score_sha)
    plans, plan_records = {}, {}
    for method in METHODS:
        path = Path(plan_paths[method]).absolute()
        require(path.resolve() == path and path.stat().st_mode & 0o077 == 0, "Comparison requires private non-symlink plans")
        expected = record(path, nonempty=True)
        require(expected["sha256"] == approved_shas[method], "Approved comparison plan SHA mismatch")
        plan = json.loads(path.read_bytes())
        validate_plan_content(plan, method, score_binding)
        expected_path = plan_location(plan)
        require(path == expected_path and path.parent.stat().st_mode & 0o077 == 0,
                "Strict comparison requires immutable fresh/recovery plan location, not old candidate root")
        require(record(path, nonempty=True) == expected, "Comparison plan changed during verification")
        plans[method], plan_records[method] = plan, expected
    common = common_contract(plans["base"])
    for method in ("alg", "dymos"):
        require(common_contract(plans[method]) == common,
                "Common cases/models/runtime/generation sources/score bundle content mismatch; same commit is insufficient")
    require(all(row.get("allowed") is False for row in plans["base"]["historical_base_reuse"].values()),
            "Historical baseline allowed=False cannot become reusable")
    return plans, plan_records, common, score_binding


def prepare(plan_paths, approved_shas, score_bundle, approved_score_sha, output):
    output = Path(output).absolute()
    require(not output.exists() and output.resolve() == output
            and output.parent.stat().st_mode & 0o077 == 0, "New private comparison destination required; never overwrite")
    plans, plan_records, common, score_binding = verify_three_plans(plan_paths, approved_shas, score_bundle, approved_score_sha)
    value = dict(schema=SCHEMA, private=True, plans=plan_records, common_execution=common,
        common_execution_sha256=sha(encoded(common)), score_bundle=score_binding,
        interventions={m: dict(parameters=plans[m]["parameters"], active_steps=plans[m]["active_steps"],
                              method_context_installed=m != "base") for m in METHODS},
        phase_seconds={m: plans[m]["phase_seconds"] for m in METHODS},
        total_seconds={m: plans[m]["total_seconds"] for m in METHODS},
        prior_generation={m: plans[m]["prior_generation"] for m in ("alg", "dymos")},
        generation_charged_upper_seconds={m: sum(plans[m]["shard_cap_seconds"])
            + plan_prior(plans[m]) for m in METHODS},
        status="BLOCKED", runnable=False, GPU_used=False, quality_pass=False, sota_claim=False,
        blockers=list(base.BLOCKERS), strict_baseline_reuse_allowed=False,
        restriction="CPU common contract only; require all 3x188 real generation receipts and score content bindings before comparison")
    exclusive(output, value)
    return output, sha(output.read_bytes())


def verify(path, approved_sha):
    expected = record(path, nonempty=True)
    require(expected["sha256"] == approved_sha, "Approved comparison manifest SHA mismatch")
    value = json.loads(Path(path).read_bytes())
    require(value.get("schema") == SCHEMA and value.get("private") is True
            and value.get("status") == "BLOCKED" and value.get("runnable") is False
            and value.get("GPU_used") is False and value.get("quality_pass") is False
            and value.get("sota_claim") is False and value.get("strict_baseline_reuse_allowed") is False,
            "Comparison preparation cannot become execution/quality PASS")
    for row in value["plans"].values():
        check_record(row, nonempty=True)
    plans, plan_records, common, score_binding = verify_three_plans(
        {m: r["path"] for m, r in value["plans"].items()},
        {m: r["sha256"] for m, r in value["plans"].items()},
        value["score_bundle"]["path"], value["score_bundle"]["sha256"])
    require(value["plans"] == plan_records and value["common_execution"] == common
            and value["common_execution_sha256"] == sha(encoded(common)) and value["score_bundle"] == score_binding,
            "Comparison contract/hash drift")
    require(value["prior_generation"] == {m: plans[m]["prior_generation"] for m in ("alg", "dymos")}
            and value["generation_charged_upper_seconds"] == {m: sum(plans[m]["shard_cap_seconds"])
                + plan_prior(plans[m]) for m in METHODS},
            "Comparison historical prior plus recovery caps drift")
    require(value["blockers"] == base.BLOCKERS and value["phase_seconds"] == {m: plans[m]["phase_seconds"] for m in METHODS}
            and value["total_seconds"] == {m: plans[m]["total_seconds"] for m in METHODS}
            and value["interventions"] == {m: dict(parameters=plans[m]["parameters"], active_steps=plans[m]["active_steps"],
                method_context_installed=m != "base") for m in METHODS}, "Comparison intervention/budget drift")
    require(record(path, nonempty=True) == expected, "Comparison manifest changed during verification")
    return value


def verify_metadata(path, approved_sha):
    """Shared bytes plus frozen asset metadata, never a local asset attestation.

    Generation consumers must additionally authenticate the approved full score
    owner proof and rehash their own node models/runtime before any GPU dispatch.
    The default prepare/verify APIs retain complete local asset verification.
    """
    value, expected = read_evidence(path, approved_sha)
    require(Path(path).stat().st_mode & 0o077 == 0 and Path(path).parent.stat().st_mode & 0o077 == 0,
            "Frozen comparison view must be private")
    require(value.get("schema") == SCHEMA and value.get("private") is True
            and value.get("status") == "BLOCKED" and value.get("runnable") is False
            and value.get("GPU_used") is False and value.get("quality_pass") is False
            and value.get("sota_claim") is False and value.get("strict_baseline_reuse_allowed") is False
            and set(value["plans"]) == set(METHODS), "Frozen metadata view is not execution/quality authorization")
    bundle, bundle_record = read_evidence(value["score_bundle"]["path"], value["score_bundle"]["sha256"])
    require(bundle_record == value["score_bundle"], "Frozen score bundle JSON binding drift")
    import score_teacher_devset as scoring
    require(bundle.get("schema") == scoring.BUNDLE_SCHEMA and bundle.get("private") is True
            and bundle.get("metric_protocol") == "score_e1_all8_gt_v1"
            and bundle.get("dimensions") == scoring.e1.DIMS and bundle.get("gt_keys") == list(scoring.e1.GT_KEYS),
            "Frozen score bundle metric metadata drift")
    require(len(bundle["sources"]) == len(scoring.required_sources())
            and {r["path"] for r in bundle["sources"]} == set(scoring.required_sources()),
            "Frozen scoring shared source coverage drift")
    shared_records = [expected, bundle_record]
    for row in bundle["sources"]:
        actual = record(row["path"])
        require(all(actual[k] == row[k] for k in ("path", "bytes", "sha256")), "Frozen scoring source content drift")
        shared_records.append(actual)
    plans = {}
    for method in METHODS:
        row = value["plans"][method]
        plan, plan_record = read_evidence(row["path"], row["sha256"])
        require(plan_record == row and Path(row["path"]).stat().st_mode & 0o077 == 0, "Frozen private plan JSON binding drift")
        _validate_shared_plan_content(plan, method, bundle_record)
        expected_path = plan_location(plan)
        require(Path(row["path"]) == expected_path, "Frozen plan location drift")
        metadata_rows = plan["models"] + plan["runtime"]["files"]
        for metadata in metadata_rows:
            require(isinstance(metadata.get("path"), str) and Path(metadata["path"]).is_absolute()
                    and type(metadata.get("bytes")) is int and metadata["bytes"] > 0
                    and isinstance(metadata.get("sha256"), str) and len(metadata["sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in metadata["sha256"]),
                    "Invalid frozen model/runtime metadata record")
        require(len(plan["runtime"]["files"]) == len({r["path"] for r in plan["runtime"]["files"]}),
                "Duplicate frozen runtime metadata")
        shared_records.extend([plan_record, plan["cases_file"]] + plan["source_files"])
        for row in plan["inputs"]:
            shared_records.extend([row["image"], row["gt"]])
        plans[method] = plan
    common = common_contract(plans["base"])
    require(all(common_contract(plans[m]) == common for m in METHODS)
            and value["common_execution"] == common and value["common_execution_sha256"] == sha(encoded(common)),
            "Frozen common cases/models/runtime/source/score metadata mismatch")
    require(all(r.get("allowed") is False for r in plans["base"]["historical_base_reuse"].values())
            and value["prior_generation"] == {m: plans[m]["prior_generation"] for m in ("alg", "dymos")}
            and value["generation_charged_upper_seconds"] == {m: sum(plans[m]["shard_cap_seconds"])
                + plan_prior(plans[m]) for m in METHODS}
            and value["phase_seconds"] == {m: plans[m]["phase_seconds"] for m in METHODS}
            and value["total_seconds"] == {m: plans[m]["total_seconds"] for m in METHODS}
            and value["interventions"] == {m: dict(parameters=plans[m]["parameters"], active_steps=plans[m]["active_steps"],
                method_context_installed=m != "base") for m in METHODS}
            and value["blockers"] == base.BLOCKERS, "Frozen intervention/budget/historical reuse drift")
    for row in shared_records:
        check_record(row)
    return {**value, "frozen_contract_view": dict(shared_content_verified=True,
        actual_model_runtime_score_assets_verified=False, requires_full_score_owner_proof=True, GPU_authorized=False)}


def read_evidence(path, approved_sha):
    expected = record(path, nonempty=True)
    require(expected["sha256"] == approved_sha, "Unapproved generation evidence SHA")
    require(Path(path).absolute().resolve() == Path(path).absolute(), "Symlink generation evidence forbidden")
    def unique(pairs):
        value = {}
        for key, child in pairs:
            require(key not in value, "Duplicate generation evidence JSON key")
            value[key] = child
        return value
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    require(isinstance(value, dict), "Generation evidence root must be an object")
    finite_evidence(value)
    require(record(path, nonempty=True) == expected, "Generation evidence changed during read")
    return value, expected


def validate_parameter_maps(before, after):
    require(isinstance(before, dict) and before and before == after, "Missing or changed actual DiT parameter SHA maps")
    for name, row in before.items():
        require(isinstance(name, str) and name and isinstance(row, dict)
                and set(row) == {"shape", "dtype", "sha256"}
                and isinstance(row["shape"], list) and all(type(x) is int and x >= 0 for x in row["shape"])
                and isinstance(row["dtype"], str) and row["dtype"]
                and isinstance(row["sha256"], str) and len(row["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in row["sha256"]), "Invalid actual DiT parameter SHA record")
    return before


def finite_evidence(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_evidence(child)
    elif isinstance(value, list):
        for child in value:
            finite_evidence(child)
    elif type(value) is float:
        require(math.isfinite(value), "Nonfinite generation evidence")


def verify_case_receipt(path, approved_sha, *, plan, plan_sha256, job,
                        parameters_before, parameters_after, input_binding):
    """Recompute one native/method helper's original input, parameter and media evidence."""
    require(job in plan["jobs"] and Path(path).absolute() == Path(job["output"]).with_suffix(".receipt.json"),
            "Unexpected generation case receipt/job location")
    receipt, expected = read_evidence(path, approved_sha)
    method = plan["method"]
    schema = "teacher_base_case_receipt_v1" if method == "base" else "teacher_devset_case_receipt_v1"
    require(receipt.get("schema") == schema and receipt.get("method") == method
            and receipt.get("plan_sha256") == plan_sha256 and receipt.get("job") == job
            and receipt.get("success") is True and receipt.get("failure") is False
            and receipt.get("parameters_unchanged") is True and receipt.get("error") is None,
            "Incomplete or wrong-method generation case evidence")
    params = validate_parameter_maps(parameters_before, parameters_after)
    case = plan["cases"][job["index"] % 94]
    require(case["id"] == job["case_id"] and case["prompt_sha256"] == sha(case["prompt"].encode()),
            "Generation original case/prompt mismatch")
    expected_input = dict(case_id=case["id"], prompt_sha256=case["prompt_sha256"],
        image=record(case["image"], nonempty=True), gt=record(case["gt"], nonempty=True))
    require(input_binding == expected_input and input_binding["image"]["sha256"] == case["image_sha256"]
            and input_binding["gt"]["sha256"] == case["video_sha256"], "Generation original image/GT input binding drift")
    frozen_input = next(r for r in plan["inputs"] if r["id"] == case["id"])
    require(input_binding["image"] == frozen_input["image"] and input_binding["gt"] == frozen_input["gt"],
            "Generation source inputs differ from frozen plan")
    if method == "base":
        require(receipt.get("checkpoint") == "NONE" and receipt.get("method_context_installed") is False
                and receipt.get("active_steps") == 0 and receipt.get("score_bundle_sha256") == plan["score_bundle"]["sha256"]
                and receipt.get("input_binding") == input_binding
                and receipt.get("parameters_before") == params and receipt.get("parameters_after") == params,
                "Native case must contain actual NONE/input/parameter SHA evidence")
        base.validate_native_stats(receipt["stats"])
    else:
        smoke.validate_stats(method, receipt["stats"]["method"], receipt["stats"]["observed"])
    elapsed = receipt.get("elapsed_seconds")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0, "Invalid generation case elapsed time")
    video_path = Path(job["output"])
    before_media = record(video_path, nonempty=True)
    require(receipt["video"]["file"] == before_media, "Generation media byte SHA/metadata drift")
    video = smoke.validate_video(video_path, plan["runtime"]["tools"])
    require(video == receipt["video"] and record(video_path, nonempty=True) == before_media,
            "Generation full121f/1280x704/24fps decode/hash drift")
    for row in (input_binding["image"], input_binding["gt"]):
        check_record(row, nonempty=True)
    require(record(path, nonempty=True) == expected, "Generation case receipt changed during verification")
    return dict(job=job, receipt=expected, video=video, parameters=params, elapsed_seconds=elapsed)


def verify_generation_completion(path, approved_sha, *, common_execution_sha256, score_bundle_sha256, method):
    """Verify all188 outputs and all16 managed parent chains; never trust PASS booleans alone."""
    require(method in METHODS, "Unknown completion method")
    require(Path(path).absolute().resolve() == Path(path).absolute()
            and Path(path).stat().st_mode & 0o077 == 0, "Private non-symlink generation completion required")
    value, completion_record = read_evidence(path, approved_sha)
    require(value.get("schema") == "teacher_generation_completion_v1" and value.get("private") is True
            and value.get("method") == method and value.get("generation_complete") is True
            and value.get("GPU_used") is True and value.get("quality_pass") is False and value.get("sota_claim") is False
            and value.get("common_execution_sha256") == common_execution_sha256
            and value.get("score_bundle_sha256") == score_bundle_sha256
            and all(k in value for k in ("execution", "plan", "plan_sha256", "parent_receipts", "cases", "costs")),
            "Incomplete or wrong common generation completion; PASS booleans are not execution evidence")
    import run_teacher_generation as generation
    # The generic verifier authenticates dispatch, ancestry, owner identity, raw
    # release-query records and full allocated clocks. Recheck inputs/media here.
    verified = generation.verify_generation_completion(path, approved_sha, common_execution_sha256=common_execution_sha256,
        score_bundle_sha256=score_bundle_sha256, method=method)
    require(verified == value, "Managed completion verifier/content disagreement")
    execution = generation.verify_execution(value["execution"]["path"], value["execution"]["sha256"])
    require(execution["common_execution_sha256"] == common_execution_sha256
            and execution["score_bundle"]["sha256"] == score_bundle_sha256
            and execution["plans"][method] == value["plan"] and value["plan_sha256"] == value["plan"]["sha256"],
            "Generation execution/plan/common content binding drift")
    plan, plan_record = read_evidence(value["plan"]["path"], value["plan"]["sha256"])
    require(plan.get("method") == method and plan.get("seeds") == [0, 1] and plan.get("video_count") == 188
            and len(value["parent_receipts"]) == 16 and len(value["cases"]) == 188,
            "Generation completion requires exactly188 cases and16 parent receipts")
    evidence_records = [completion_record, plan_record, value["execution"]]
    cases, parents, workers, baseline, elapsed = [], {}, {}, None, 0.0
    case_seconds = [0.0] * 16
    for row in value["parent_receipts"]:
        parent, parent_record = read_evidence(row["path"], row["sha256"])
        shard = parent.get("shard")
        require(type(shard) is int and 0 <= shard < 16 and shard not in parents
                and parent.get("success") is True and parent.get("exit_code") == 0
                and parent.get("error") is None and parent.get("interrupted") is None and parent.get("timed_out") is False,
                "Missing, duplicate, extra or failed generation parent")
        allocation = execution["allocations"][method][shard]
        require(all(parent.get(k) == v for k, v in allocation.items()) and allocation["allocated_gpus"] == 1
                and type(parent.get("elapsed_seconds")) in (int, float)
                and 0 < parent["elapsed_seconds"] <= allocation["cap_seconds"], "Generation parent stagecap/single-GPU scope exceeded")
        worker, worker_record = read_evidence(parent["worker"]["path"], parent["worker"]["sha256"])
        require(worker.get("success") is True and worker.get("error") is None
                and worker.get("common_execution_sha256") == common_execution_sha256
                and worker.get("score_bundle_sha256") == score_bundle_sha256
                and [r["job"] for r in worker["cases"]] == plan["jobs"][shard::16], "Generation worker/shard/common coverage drift")
        cases.extend(worker["cases"])
        parents[shard], workers[shard] = parent, worker
        elapsed += parent["elapsed_seconds"]
        evidence_records.extend([parent_record, worker_record, parent["launch"], parent["supervised"]])
    require(set(parents) == set(range(16)), "All16 distinct supervised parents required")
    cases.sort(key=lambda row: row["job"]["index"])
    require(cases == value["cases"] and [row["job"] for row in cases] == plan["jobs"], "All94x2seed exact job coverage drift")
    prior = plan_prior(plan)
    require(prior + elapsed <= plan["phase_seconds"]["generation"]
            and type(value["costs"].get("generation_charged_upper_seconds")) in (int, float)
            and math.isclose(value["costs"]["generation_charged_upper_seconds"], prior + elapsed, abs_tol=1e-8),
            "Prior generation plus real parent costs exceed independent stage or mismatch completion cost")
    for row in cases:
        require("input_binding" in row, "Generation case must bind original image/GT/prompt evidence")
        proof = verify_case_receipt(row["case_receipt"]["path"], row["case_receipt"]["sha256"], plan=plan,
            plan_sha256=value["plan_sha256"], job=row["job"], parameters_before=row["parameters_before"],
            parameters_after=row["parameters_after"], input_binding=row["input_binding"])
        require(row["video"] == proof["video"] and (baseline is None or proof["parameters"] == baseline),
                "Generation media or global native baseline parameter maps differ")
        baseline = proof["parameters"]
        if method != "base":
            shard = row["job"]["index"] % 16
            claim_path = Path(row["job"]["output"]).with_suffix(".claim.json")
            require("claim" in row and row["claim"]["path"] == str(claim_path), "Candidate claim must bind its exact per-job file")
            claim, claim_record = read_evidence(claim_path, row["claim"]["sha256"])
            require(claim == dict(plan_sha256=value["plan_sha256"], job=row["job"],
                                  pid=workers[shard]["owner"]["pid"], hostname=parents[shard]["hostname"])
                    and type(claim["pid"]) is int and claim["pid"] > 1,
                    "Candidate persistent claim differs from actual worker/job/plan/hostname proof")
            evidence_records.append(claim_record)
        else:
            require("claim" not in row, "Native base cannot import a candidate intervention claim")
        case_seconds[row["job"]["index"] % 16] += proof["elapsed_seconds"]
        evidence_records.extend([proof["receipt"], proof["video"]["file"], row["input_binding"]["image"], row["input_binding"]["gt"]])
    for shard, worker in workers.items():
        load = worker.get("load_seconds")
        require(type(load) in (int, float) and math.isfinite(load) and load > 0
                and load + case_seconds[shard] <= parents[shard]["elapsed_seconds"] + 1e-6,
                "Generation measured load/case timers overlap or exceed charged parent allocation")
    for seed in (0, 1):
        video_root = Path(plan["output_root"]) / f"seed{seed}/videos/{method}"
        expected_files = {Path(j["output"]) for j in plan["jobs"] if j["seed"] == seed}
        expected_media = set(expected_files)
        expected_files |= {p.with_suffix(".receipt.json") for p in expected_media}
        if method != "base":
            expected_files |= {p.with_suffix(".claim.json") for p in expected_media}
        require(video_root.resolve() == video_root
                and {p for p in video_root.rglob("*") if p.is_file() or p.is_symlink()} == expected_files,
                "Extra, partial, missing or symlink generation output forbids completion")
    for row in evidence_records:
        check_record(row, nonempty=True)
    require(generation.verify_execution(value["execution"]["path"], value["execution"]["sha256"]) == execution,
            "Generation models/runtime/sources/execution changed during completion verification")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    for method in METHODS:
        parser.add_argument(f"--{method}-plan", type=Path)
        parser.add_argument(f"--approved-{method}-plan-sha256")
    parser.add_argument("--score-bundle", type=Path)
    parser.add_argument("--approved-score-bundle-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved-output-sha256")
    args = parser.parse_args(argv)
    if args.verify:
        if not args.output or not args.approved_output_sha256:
            parser.error("Verify requires output and approved output SHA")
        verify(args.output, args.approved_output_sha256)
        print(json.dumps(dict(contract_verified=True, status="BLOCKED", runnable=False, GPU_used=False)))
    else:
        paths = {m: getattr(args, m + "_plan") for m in METHODS}
        shas = {m: getattr(args, "approved_" + m + "_plan_sha256") for m in METHODS}
        if not all(paths.values()) or not all(shas.values()) or not args.score_bundle or not args.approved_score_bundle_sha256 or not args.output:
            parser.error("Prepare requires exactly three plans/approved SHAs, approved score bundle and new private output")
        path, digest = prepare(paths, shas, args.score_bundle, args.approved_score_bundle_sha256, args.output)
        print(json.dumps(dict(output=str(path), sha256=digest, status="BLOCKED", runnable=False, GPU_used=False)))


if __name__ == "__main__":
    main()
