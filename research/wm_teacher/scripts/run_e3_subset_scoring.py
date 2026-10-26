#!/usr/bin/env python3
"""Finite C276 scoring orchestration; default CPU prepare, never generation or publication."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import score_e3_subset as s
import run_e3_subset_completion as completion

PLAN_SHA = "afd1cd967ae03088930a1637ea93f66053f2adcaeae3b213f38e1c7160e57254"
ENGINE_SHA = "5d7d5241e22a93294acb1610fb85b838791df0c4298364ce6b9e660413774203"
BUDGET = {"generation": 28, "score": 12, "total": 40}
ORIGINAL_STAGE = s.stage_jobs
ORIGINAL_GPU = s.gpu_run


def score_bundle(args):
    paths = [Path(s.__file__).resolve(), Path(__file__).resolve(), Path(completion.__file__).resolve(), args.root / "scripts/gt_ref_metrics.py",
             args.root / "scripts/summarize_eval.py", args.root / "scripts/analyze_merge_metrics.py"]
    files = {p.name: {"path": str(p), "sha256": s.sha256(p)} for p in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "e3_subset_score_bundle_v1", "files": files, "bundle_sha256": digest}


def verify_bundle(args):
    bundle = score_bundle(args)
    s.require(bundle["files"]["score_e3_subset.py"]["sha256"] == ENGINE_SHA == args.score_source_sha256,
              "Original scoring engine SHA changed")
    s.require((not args.approved_bundle_sha256 or args.approved_bundle_sha256 == bundle["bundle_sha256"]) and
              s.read_json(args.experiment / "score_source_bundle.json") == bundle, "Approved source bundle/sidecar mismatch")
    return bundle


def generation(args):
    completion.budget_contract(args.experiment, args.plan_sha256, getattr(args, "budget_amendment_sha256", None))
    if (args.experiment / (completion.RUN_ID + ".finished.json")).exists():
        try:
            return completion.validate_global(args.experiment, args.plan_sha256, getattr(args, "budget_amendment_sha256", None))
        except Exception as error:
            s.require(False, "Global completion verification failed: " + str(error))
    paths = [args.experiment / f"node_{i}.finished.json" for i in range(4)]
    paths += [args.experiment / f"node_{i}_recovery_a1.finished.json" for i in (1, 2)]
    s.exact([str(p) for p in args.experiment.glob("node_*.finished.json")], map(str, paths), "Generation receipts")
    receipts = [s.read_json(p) for p in paths]
    for index, row in enumerate(receipts):
        rank = index if index < 4 else index - 3
        s.require(type(row.get("rank")) is int and row["rank"] == rank and row.get("plan_sha256") == args.plan_sha256 and
                  row.get("owned_tree_released") is True and row.get("cap_exceeded") is False,
                  "Generation identity/release/cap failed")
        elapsed, hours = row.get("elapsed_seconds"), row.get("gpu_hours")
        s.require(s.finite(elapsed) and 0 < elapsed <= 3000 and s.finite(hours) and hours > 0 and
                  math.isclose(hours, 8 * elapsed / 3600, abs_tol=1e-9), "Missing/invalid generation cost")
        s.require(type(row.get("exit_code")) is int, "Missing generation return code")
        if index >= 4:
            s.require(row.get("run_id") == "recovery_a1", "Unapproved recovery receipt")
    for index in (0, 3, 4, 5):
        row = receipts[index]
        s.require(row["exit_code"] == 0 and not any(row.get(k) for k in ("error", "timed_out", "interrupted_signal")),
                  "Latest generation failed; no scoring or regeneration")
    s.require(all(receipts[i]["exit_code"] != 0 for i in (1, 2)), "Unexpected original retry history")
    hours = sum(row["gpu_hours"] for row in receipts)
    s.require(hours <= 28, "Generation budget exceeded, including failed original attempts")
    return hours, {str(p): s.sha256(p) for p in paths}


def completion_cost(args, receipts):
    path = args.experiment / (completion.RUN_ID + ".finished.json")
    if str(path) not in receipts:
        return None
    row = s.read_json(path)
    s.require(s.sha256(path) == receipts[str(path)], "Global completion receipt SHA changed")
    return {"receipt_sha256": receipts[str(path)], "original_six_sha256": row["original_six_sha256"],
            "actual_prior_gpu_hours": row["actual_prior_gpu_hours"],
            "actual_repair_gpu_hours": row["actual_repair_gpu_hours"],
            "total_generation_gpu_hours": row["total_generation_gpu_hours"],
            **({"budget_amendment_sha256": row["budget_amendment_sha256"]} if row.get("budget_amendment_sha256") else {})}


def ledger(args, hours, receipts):
    budget, amendment_sha = completion.budget_contract(args.experiment, args.plan_sha256, getattr(args, "budget_amendment_sha256", None))
    s.require(s.finite(hours) and 0 <= hours <= budget["generation"], "Generation cost exceeds approved budget")
    expected = {"schema": "e3_subset_score_ledger_v1", "phase": "score", "plan_sha256": args.plan_sha256,
                "score_source_sha256": args.score_source_sha256, "approved_bundle_sha256": args.approved_bundle_sha256,
                "generation_receipts_sha256": receipts,
                "budget_gpu_hours": budget, "generation_gpu_hours": hours}
    if amendment_sha is not None:
        expected["budget_amendment_sha256"] = amendment_sha
    global_cost = completion_cost(args, receipts)
    if global_cost is not None:
        expected["global_completion"] = global_cost
    if not args.gpu_ledger.exists():
        s.write_new(args.gpu_ledger, dict(expected, score_gpu_hours=0, active=None, runs=[]))
    with args.gpu_ledger.open("r") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        row = s.read_json(args.gpu_ledger)
    s.require(all(row.get(k) == v for k, v in expected.items()), "Existing ledger phase/hash/cost mismatch; never reset")
    s.require(row.get("budget_amendment_sha256") == amendment_sha, "Ledger budget amendment SHA mismatch")
    used = row.get("score_gpu_hours")
    s.require(row.get("active") is None and s.finite(used) and 0 <= used <= budget["score"] and hours + used <= budget["total"],
              "Active operation or invalid/exhausted ledger")
    runs = row.get("runs")
    s.require(isinstance(runs, list) and all(r.get("phase") == "score" and r.get("gpus") == 1 and
              s.finite(r.get("used_gpu_hours")) and r["used_gpu_hours"] >= 0 for r in runs), "Invalid score run history")
    s.require(math.isclose(used, sum(r["used_gpu_hours"] for r in runs), abs_tol=1e-8), "Unaccounted score cost")
    return row


def cached_stage(cache):
    def stage(jobs, suite, model, work, source="out", **kwargs):
        key = (suite, model, source)
        manifest = work / suite / model / "decode.json"
        if key not in cache:
            if manifest.exists():
                data = s.read_json(manifest)
                s.require(data.get("suite") == suite and data.get("model") == model and
                          data.get("all_frames_decoded") is True, "Existing decode provenance mismatch")
                cache[key] = (work / suite / model / "videos", data["outputs"])
            else:
                cache[key] = ORIGINAL_STAGE(jobs, suite, model, work, source=source, **kwargs)
        videos, outputs = cache[key]
        s.require(not videos.is_symlink() and videos.resolve().is_relative_to(work.resolve()) and not manifest.is_symlink(),
                  "Cache must remain byte-copied launcher-local, not shared symlinks")
        names = [j["id"] + "-0.mp4" if suite in ("vbench", "vbench_po", "wm") else Path(j[source]).name for j in jobs]
        s.exact(names, outputs, "Cached decode coverage")
        s.exact([p.name for p in videos.glob("*.mp4")], names, "Local staged video coverage")
        s.require(s.read_json(manifest) == {"suite": suite, "model": model, "all_frames_decoded": True,
                                           "outputs": outputs}, "Cached decode manifest changed")
        for job, name in zip(jobs, names):
            row = outputs[name]
            s.require(row.get("source") == job[source] and row.get("path") == str(videos / name) and not (videos / name).is_symlink() and
                      row.get("all_frames_decoded") is True and type(row.get("frames")) is int and row["frames"] > 1 and
                      s.finite(row.get("fps")) and row["fps"] > 0 and
                      s.sha256(job[source]) == s.sha256(row["path"]) == row.get("sha256"), "Cached source/stage changed")
            if job.get(source + "_sha256"):
                s.require(row["sha256"] == job[source + "_sha256"], "Frozen base SHA changed")
            if source == "out":
                s.require((row["frames"], row["fps"], row.get("width"), row.get("height")) == (121, 24, 1280, 704),
                          "Candidate full-decode format mismatch")
        return videos, outputs
    return stage


def result_paths(args, suite, model, action):
    names = ([model + "__" + action + "_eval_results.json"] if action in s.DIMS else
             ["physics.csv", "physics_metrics.json"] if action == "physics" else
             ["gtref.json"] if action == "gtref" else ["result.json"])
    return [s.score_path(args.experiment, suite, model, name) for name in names]


def validate_result(args, groups, info, suite, model, action, cache):
    paths = result_paths(args, suite, model, action)
    jobs = groups[suite]
    if action in s.DIMS:
        expected = s.applicable(s.filter_full_info(s.suite_info(args, suite, groups, info), [j["id"] for j in jobs]))[action]
        data = s.read_json(paths[0])
        s.exact(data, [action], "Single dimension result")
        rows = s.dimension_rows(data, action, expected)
        source = "base_out" if model in ("base", "base_po") else "out"
        _, decoded = cached_stage(cache)(jobs, suite, model, args.work_dir, source=source)
        point = s.pool_dimension(action, rows, {name: row["frames"] for name, row in decoded.items()})
        s.require(abs(point - data[action][0]) <= 1e-8, "Original aggregate/pooling mismatch")
    elif action == "gtref":
        s.validate_gtref(s.read_json(paths[0]), {j["id"] for j in jobs})
    elif action == "wmb":
        s.filter_wmb(s.read_json(paths[0]), {Path(j["out"]).stem for j in jobs})
    else:
        s.physics_values(paths[0], s.physics_scenes([Path(j["out"]).name for j in jobs]),
                         s.load_iqtable(args.bench / "physics-IQ-benchmark"), paths[1])


def boundary(args, original=ORIGINAL_GPU):
    def run(command, call_args, cwd=None, env=None):
        s.require(args.gpu_authorized and args.gpu == "0" and args.occupied_gpus == 1 and
                  args.supervisor_pid > 1 and args.supervisor_pid != os.getpid(), "Single GPU authorization/supervisor required")
        os.kill(args.supervisor_pid, 0)
        hours, receipts = generation(args)
        state = ledger(args, hours, receipts)
        budget = state["budget_gpu_hours"]
        remaining = min(budget["score"] - state["score_gpu_hours"], budget["total"] - hours - state["score_gpu_hours"])
        call_args.max_gpu_seconds = min(args.max_gpu_seconds, remaining * 3600 - 15 - 0.001)
        s.require(s.finite(call_args.max_gpu_seconds) and 0 < call_args.max_gpu_seconds <= 1800, "No bounded score reservation remains")
        verify_bundle(args)
        wrapped = [sys.executable, str(args.root / "scripts/gpu_hold.py"), "--action", "exec", "--gpus", "0", "--", *command]
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            return original(wrapped, call_args, cwd=cwd, env=dict(env or os.environ, CUDA_VISIBLE_DEVICES="0"))
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return run


def cpu_worker(context, digest, action, suite, dim, runtime):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    s.require(s.sha256(context) == digest, "CPU score context SHA changed")
    ctx = s.read_json(Path(context))
    args = argparse.Namespace(**ctx["args"], **json.loads(runtime))
    for key in ("root", "experiment", "work_dir", "bench", "vbench_repo", "gpu_ledger"):
        setattr(args, key, Path(getattr(args, key)))
    verify_bundle(args)
    cache = {(row["suite"], row["model"], row["source"]): (Path(row["videos"]), row["outputs"]) for row in ctx["stages"]}
    s.stage_jobs = cached_stage(cache)
    groups, info, plan = ctx["groups"], ctx["info"], ctx["plan"]
    model = plan["tag"] + ("_po" if suite == "vbench_po" else "")
    if action == "validate-physics":
        validate_result(args, groups, info, suite, dim, "physics", cache)
    elif action == "summary":
        report = s.summary(args, plan, groups, info)
        report["source_bundle_sha256"] = args.approved_bundle_sha256
        report["budget"] = ledger(args, *generation(args))["budget_gpu_hours"]
        if getattr(args, "budget_amendment_sha256", None) is not None:
            report["budget_amendment_sha256"] = args.budget_amendment_sha256
        public_report(report, args)
        path = args.experiment / "summary.json"
        if path.exists():
            prior = s.read_json(path)
            s.require({k: v for k, v in prior.items() if k != "created_epoch"} ==
                      {k: v for k, v in report.items() if k != "created_epoch"}, "Existing summary differs; preserve")
        else:
            s.write_new(path, report)
    else:
        args.dim = dim
        s.gpu_run = boundary(args)
        if action == "vbench":
            s.run_vbench(args, groups[suite], s.suite_info(args, suite, groups, info), suite, model)
        else:
            getattr(s, "run_" + action)(args, groups[suite], model)


def cpu_call(args, context, action, suite="", dim=""):
    runtime = {k: getattr(args, k) for k in ("gpu_authorized", "gpu", "occupied_gpus", "supervisor_pid", "max_gpu_seconds")}
    wrapper = "import sys;sys.path.insert(0,sys.argv[1]);from run_e3_subset_scoring import cpu_worker;cpu_worker(*sys.argv[2:])"
    python = args.piq_python if action in ("physics", "validate-physics", "summary") else args.vbench_python
    subprocess.run([python, "-B", "-c", wrapper, str(args.root / "scripts"), str(context), s.sha256(context),
                    action, suite, dim, json.dumps(runtime)], check=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))


def public_report(report, args=None):
    s.require(report.get("subset_counts") == s.COUNTS and report.get("bootstrap") == 2000 and report.get("seed") == 20261025 and
              report.get("not_full_acceptance") is True and report.get("not_sota") is True, "Wrong summary scope/bootstrap")
    s.exact(report.get("vbench", {}), ("vbench", "vbench_po", "wm"), "Summary suites")
    def estimate(row, points=True):
        ci = row.get("ci95") if isinstance(row, dict) else None
        s.require(isinstance(ci, list) and len(ci) == 2 and all(s.finite(v) for v in ci) and ci[0] <= ci[1] and
                  s.finite(row.get("delta")) and (not points or all(s.finite(row.get(k)) for k in ("left", "right"))),
                  "Missing/nonfinite aggregate or paired confidence interval")
    for row in report["vbench"].values():
        for key in ("left", "right", "paired", "n_by_dim"):
            s.exact(row.get(key, {}), s.DIMS, "Summary dimensions")
        s.require(all(s.finite(v) for k in ("left", "right") for v in row[k].values()) and
                  all(type(v) is int and v > 0 for v in row["n_by_dim"].values()), "Invalid VBench summary points/counts")
        estimate(row.get("aggregate_only_total"))
        for pair in row["paired"].values():
            estimate(pair, points=False)
    s.exact(report.get("gtref", {}).get("metrics", {}), s.GT_KEYS[:3], "Summary GT metrics")
    s.require(report.get("physics", {}).get("n_scenes") == 14 and report.get("wmb", {}).get("n_cases") == 70 and
              "aggregate" in report["wmb"] and len(report["wmb"].get("domains", {})) == 7, "Missing Physics/WMB summary")
    s.require(report["gtref"].get("n_cases") == 22 and report["gtref"].get("K") == 12 and
              report["physics"].get("views_per_scene") == 3 and report["wmb"].get("questions_per_case") == 8,
              "Summary count/sampler/view protocol changed")
    for row in list(report["gtref"]["metrics"].values()) + [report["physics"]]:
        estimate(row)
    for row in [report["wmb"]["aggregate"], *report["wmb"]["domains"].values()]:
        estimate(row)
        s.require(isinstance(row.get("eight_question_paired"), list) and len(row["eight_question_paired"]) == 8,
                  "Missing WMB eight-question paired intervals")
        for pair in row["eight_question_paired"]:
            estimate(pair, points=False)
    budget = report.get("budget")
    if budget == completion.AMENDED_BUDGET:
        s.require(args is not None, "Amended public budget requires verified sidecar context")
        approved, amendment_sha = completion.budget_contract(args.experiment, args.plan_sha256,
                                                            getattr(args, "budget_amendment_sha256", None))
        s.require(budget == approved and amendment_sha is not None and
                  report.get("budget_amendment_sha256") == amendment_sha, "Summary budget amendment SHA mismatch")
    else:
        s.require(budget == BUDGET and report.get("budget_amendment_sha256") is None, "Unapproved summary budget")
    bundle = report.get("source_bundle_sha256")
    s.require(isinstance(bundle, str) and len(bundle) == 64 and all(c in "0123456789abcdef" for c in bundle) and
              budget in (BUDGET, completion.AMENDED_BUDGET), "Missing summary source bundle/frozen budget")
    def pair(row, points=True):
        return {k: row[k] for k in (("left", "right", "delta", "ci95") if points else ("delta", "ci95"))}
    def world(row, count):
        s.require(type(row.get("n_cases")) is int and row["n_cases"] == count, "Missing/wrong WMB aggregate case count")
        return dict(pair(row), n_cases=row["n_cases"], eight_question_paired=[pair(p, False) for p in row["eight_question_paired"]])
    domains = report["wmb"]["domains"]
    s.require(all(isinstance(k, str) and k and "/" not in k and "\\" not in k for k in domains), "Private domain identifier")
    return {"schema": "e3_subset_public_aggregates_v1", "subset_counts": dict(s.COUNTS), "bootstrap": 2000, "seed": 20261025,
            "vbench": {suite: {"left": dict(row["left"]), "right": dict(row["right"]), "n_by_dim": dict(row["n_by_dim"]),
                               "aggregate_only_total": pair(row["aggregate_only_total"]),
                               "paired": {dim: pair(row["paired"][dim], False) for dim in s.DIMS}} for suite, row in report["vbench"].items()},
            "gtref": {"n_cases": 22, "K": 12, "metrics": {k: pair(v) for k, v in report["gtref"]["metrics"].items()}},
            "physics": dict(pair(report["physics"]), n_scenes=14, views_per_scene=3, units="percentage points"),
            "wmb": {"n_cases": 70, "questions_per_case": 8, "aggregate": world(report["wmb"]["aggregate"], 70),
                    "domains": {k: world(v, 10) for k, v in domains.items()}},
            "budget": dict(budget), "source_bundle_sha256": bundle,
            **({"budget_amendment_sha256": report["budget_amendment_sha256"]} if report.get("budget_amendment_sha256") else {}), "quality_false": True, "SOTA": False, "subset_not_full": True,
            "guards": {"complete_276_required": True, "paired_base_required": True, "generation_cap": budget["generation"], "score_cap": budget["score"]}}


def drive(args):
    s.require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"), "Start driver CPU-only, not under holder exec")
    pid = os.getppid()
    while pid > 1:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        s.require(not (any(Path(os.fsdecode(word)).name == "gpu_hold.py" for word in command) and
                  (b"exec" in command or b"--action=exec" in command)), "Do not allocate a GPU around CPU staging")
        status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
        pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    s.require(args.plan_sha256 == PLAN_SHA and s.sha256(args.experiment / "plan.json") == args.plan_sha256, "Unapproved plan")
    source = s.sha256(s.__file__)
    if args.run:
        s.require(args.gpu_authorized and args.supervisor_pid == os.getppid() and args.supervisor_pid > 1 and
                  args.occupied_gpus == 1 and args.gpu == "0", "Run needs authorized distinct live CLI parent, one GPU 0")
        os.kill(args.supervisor_pid, 0)
        s.require(args.score_source_sha256 == ENGINE_SHA == source and args.approved_bundle_sha256,
                  "Run requires fixed original engine SHA and approved bundle SHA")
    args.score_source_sha256 = args.score_source_sha256 or source
    s.require(args.score_source_sha256 == source and s.finite(args.max_gpu_seconds) and 0 < args.max_gpu_seconds <= 1800,
              "Score source/bounded action mismatch")
    bundle = verify_bundle(args)
    args.approved_bundle_sha256 = args.approved_bundle_sha256 or bundle["bundle_sha256"]
    hours, receipts = generation(args)
    plan, groups, info = s.load_context(args)
    expected = [j["out"] for rows in groups.values() for j in rows]
    tree = args.experiment / "videos"
    s.exact([str(p) for p in tree.rglob("*.mp4")], expected, "All 276 generated outputs")
    s.require(len(expected) == 276 and not any(p.is_file() and (".claim" in p.name or ".tmp" in p.name) for p in tree.rglob("*")),
              "Missing outputs or outstanding generation claims/temporary files")
    s.require(all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in expected), "Empty generation output")
    state = ledger(args, hours, receipts)
    args.work_dir = s.require_local(args.work_dir, args.root)
    cache, base_hashes = {}, {}
    stage = cached_stage(cache)
    for suite, jobs in groups.items():
        base, model = ("base_po", plan["tag"] + "_po") if suite == "vbench_po" else ("base", plan["tag"])
        provenance = s.read_json(s.score_path(args.experiment, suite, base, "provenance.json"))
        s.require(provenance.get("plan_sha256") == args.plan_sha256 and provenance.get("source_sha256"), "Base provenance missing/wrong plan")
        for path, digest in provenance["source_sha256"].items():
            s.require(s.sha256(path) == digest, "Historical base score SHA changed")
        s.require((args.work_dir / suite / base / "decode.json").is_file(), "Base full-decode provenance missing")
        stage(jobs, suite, base, args.work_dir, source="base_out")
        stage(jobs, suite, model, args.work_dir)
        for path in s.score_path(args.experiment, suite, base, "unused").parent.iterdir():
            s.require(path.is_file(), "Unexpected base score directory")
            base_hashes[str(path)] = s.sha256(path)
    ctx_args = {k: str(getattr(args, k)) if isinstance(getattr(args, k), Path) else getattr(args, k) for k in
                ("root", "experiment", "work_dir", "bench", "vbench_repo", "gpu_ledger", "plan_sha256", "score_source_sha256",
                 "vbench_python", "wmb_python", "judge", "bootstrap", "seed", "approved_bundle_sha256")}
    if getattr(args, "budget_amendment_sha256", None) is not None:
        ctx_args["budget_amendment_sha256"] = args.budget_amendment_sha256
    context = args.experiment / "score_context.json"
    ctx = {"args": ctx_args, "plan": plan, "groups": groups, "info": info, "base_sha256": base_hashes,
           "generation_gpu_hours": hours, "generation_receipts_sha256": receipts,
           "global_completion": completion_cost(args, receipts),
           "stages": [dict(suite=k[0], model=k[1], source=k[2], videos=str(v[0]), outputs=v[1]) for k, v in cache.items()]}
    if context.exists():
        s.require(s.read_json(context) == ctx, "Existing score context changed; preserve")
    else:
        s.write_new(context, ctx)
    for suite, jobs in groups.items():
        base = "base_po" if suite == "vbench_po" else "base"
        actions = list(s.DIMS) + (["gtref"] if suite == "wm" else []) if suite in ("vbench", "vbench_po", "wm") else ["physics" if suite == "physiq" else "wmb"]
        for action in actions:
            if action == "physics":
                cpu_call(args, context, "validate-physics", suite, base)
            else:
                validate_result(args, groups, info, suite, base, action, cache)
    pending = []
    for suite, jobs in groups.items():
        model = plan["tag"] + ("_po" if suite == "vbench_po" else "")
        actions = list(s.DIMS) + (["gtref"] if suite == "wm" else []) if suite in ("vbench", "vbench_po", "wm") else ["physics" if suite == "physiq" else "wmb"]
        for action in actions:
            paths = result_paths(args, suite, model, action)
            marker = paths[0].with_name(paths[0].name + ".driver.json")
            evidence = {"plan_sha256": args.plan_sha256, "score_source_sha256": source,
                        "source_bundle_sha256": args.approved_bundle_sha256,
                        "input_sha256": {r["source"]: r["sha256"] for r in cache[(suite, model, "out")][1].values()}}
            if any(p.exists() for p in paths) or marker.exists():
                s.require(all(p.is_file() for p in paths) and marker.is_file(), "Partial result/missing hash evidence; preserve")
                s.require(s.read_json(marker) == dict(evidence, result_sha256={str(p): s.sha256(p) for p in paths}), "Existing result hash mismatch")
                if action == "physics":
                    cpu_call(args, context, "validate-physics", suite, model)
                else:
                    validate_result(args, groups, info, suite, model, action, cache)
            else:
                work = args.work_dir / suite / model
                artifacts = ([work / (action + "_full_info.json")] if action in s.DIMS else
                             [work / "judge"] if action == "wmb" else
                             [work / "videos_5s", work / "real", work / "physics.csv"] if action == "physics" else [])
                s.require(not any(p.exists() for p in artifacts), "Partial scorer work exists; preserve and review")
                pending.append((suite, model, action, paths, marker, evidence))
    if not args.run:
        return {"status": "prepared", "GPU_used": False, "generation_gpu_hours": hours, "subset_not_full": True}
    s.require(not any(action in s.DIMS or action == "wmb" for _, _, action, *_ in pending) or
              state["score_gpu_hours"] < 12, "Score budget exhausted")
    for suite, model, action, paths, marker, evidence in pending:
        cpu_call(args, context, "vbench" if action in s.DIMS else action, suite, action)
        if action == "physics":
            cpu_call(args, context, "validate-physics", suite, model)
        else:
            validate_result(args, groups, info, suite, model, action, cache)
        s.write_new(marker, dict(evidence, result_sha256={str(p): s.sha256(p) for p in paths}))
    s.require(all(s.sha256(p) == digest for p, digest in base_hashes.items()), "Base scores changed during run")
    cpu_call(args, context, "summary")
    public_report(s.read_json(args.experiment / "summary.json"), args)
    final = ledger(args, hours, receipts)
    return {"status": "complete", "GPU_used": len(final["runs"]) > len(state["runs"]), "subset_not_full": True, "not_sota": True,
            "score_gpu_hours": final["score_gpu_hours"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    for name, default in (("root", s.ROOT), ("experiment", s.DEFAULT_EXPERIMENT),
                          ("work-dir", Path("/dockerdata/wm_eval/e3_subset_20261025")), ("bench", s.BENCH), ("vbench-repo", s.VBREPO)):
        parser.add_argument("--" + name, type=Path, default=default)
    parser.add_argument("--approved-plan-sha256", dest="plan_sha256")
    parser.add_argument("--score-source-sha256")
    parser.add_argument("--approved-bundle-sha256")
    parser.add_argument("--approved-budget-amendment-sha256", dest="budget_amendment_sha256")
    parser.add_argument("--gpu-ledger", type=Path)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--occupied-gpus", type=int)
    parser.add_argument("--gpu", choices=["0"])
    parser.add_argument("--max-gpu-seconds", type=float, default=1800)
    parser.add_argument("--vbench-python", default="/dockerdata/venv_vbench/bin/python")
    parser.add_argument("--piq-python", default="/dockerdata/venv_piq/bin/python")
    parser.add_argument("--wmb-python", default="/dockerdata/conda_vila/bin/python")
    parser.add_argument("--judge", default="/dockerdata/wm_eval/bench/judges/vila-ewm-qwen2-1.5b")
    args = parser.parse_args()
    if args.run and not args.plan_sha256:
        parser.error("--run requires explicit --approved-plan-sha256")
    args.plan_sha256 = args.plan_sha256 or PLAN_SHA
    args.bootstrap, args.seed = 2000, 20261025
    os.umask(0o077)
    args.root, args.experiment = args.root.resolve(), args.experiment.resolve()
    s.require(args.experiment == args.root / "eval/e3_subset_20261025", "Only the private fixed C276 tree")
    args.gpu_ledger = args.gpu_ledger or args.experiment / "score_ledger.json"
    s.require(args.gpu_ledger.resolve().is_relative_to(args.experiment), "Ledger must remain private")
    with (args.experiment / "score_driver.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = drive(args)
        except Exception as error:
            s.write_new(args.experiment / f"score_blocked_{time.time_ns()}.json",
                        {"status": "blocked", "error": str(error), "quality_false": True, "not_sota": True})
            raise
        s.write_new(args.experiment / f"score_runstate_{time.time_ns()}.json", result)
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
