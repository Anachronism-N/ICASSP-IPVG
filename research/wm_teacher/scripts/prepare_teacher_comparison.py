#!/usr/bin/env python3
"""CPU-only exact-content fresh base/ALG/DyMoS comparison preparation.

An approved commit is not a common generation/score contract. This entry point
checks frozen inputs, model bytes, current common sources and runtime content,
and refuses old plans without the same externally approved score bundle. A
validated preparation is still BLOCKED for GPU execution/quality conclusions.
"""
import argparse
import json
from pathlib import Path

import run_teacher_base as base
import prepare_teacher_recovery as recovery

dev, smoke = base.dev, base.smoke
require, encoded, sha = base.require, base.encoded, base.sha
record, check_record, exclusive = base.record, base.check_record, base.exclusive
SCHEMA = "teacher_comparison_plan_v1"
METHODS = ("base", "alg", "dymos")


def common_contract(plan):
    root = Path(plan["root"])
    common_paths = {str(p) for p in dev.sources(root)}
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


def validate_plan_content(plan, method, score_binding):
    require(plan.get("method") == method and plan.get("private") is True, "Wrong approved comparison method")
    if method == "base":
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
    expected_root = base.output_root(root) if method == "base" else recovery.plan_folder(root, method)
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
    expected_sources = base.sources(root) if method == "base" else recovery.sources(root)
    require(len(plan["source_files"]) == len(expected_sources)
            and {r["path"] for r in plan["source_files"]} == {str(p) for p in expected_sources},
            "Missing/duplicate common generation sources or unapproved source set")
    for row in plan["source_files"] + plan["runtime"]["files"]:
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
        expected_path = (base.output_root(plan["root"]) / "plan.BLOCKED.json" if method == "base"
                         else recovery.plan_folder(plan["root"], method) / "plan.blocked.json")
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
            + (0 if m == "base" else plans[m]["prior_generation"]["upper_seconds"]) for m in METHODS},
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
                + (0 if m == "base" else plans[m]["prior_generation"]["upper_seconds"]) for m in METHODS},
            "Comparison historical prior plus recovery caps drift")
    require(value["blockers"] == base.BLOCKERS and value["phase_seconds"] == {m: plans[m]["phase_seconds"] for m in METHODS}
            and value["total_seconds"] == {m: plans[m]["total_seconds"] for m in METHODS}
            and value["interventions"] == {m: dict(parameters=plans[m]["parameters"], active_steps=plans[m]["active_steps"],
                method_context_installed=m != "base") for m in METHODS}, "Comparison intervention/budget drift")
    require(record(path, nonempty=True) == expected, "Comparison manifest changed during verification")
    return value


def verify_generation_completion(path, approved_sha, *, common_execution_sha256, score_bundle_sha256, method):
    """Reserved scorer interface; no synthetic case/manifest can authorize scoring."""
    require(method in METHODS, "Unknown completion method")
    raise ValueError("BLOCKED: supervised 188-video completion verification is unimplemented; CPU plans/case receipts are not completion PASS")


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
