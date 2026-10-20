#!/usr/bin/env python3
"""C1: 12 original FT2 captions vs 12 existing corrected P1 CFG5 controls.

One all-stage CLI, new private directory only. No training, resume, downloads,
caption optimization or Teacher promotion. The coordinator imports only stdlib;
--preflight-only hashes real assets but neither queries nor initializes CUDA.
Generation reserves four cards for <=1800s (2 allocated GPUh), scoring <=3600s
(4 allocated GPUh), including preflight, CPU work, encoding and owned cleanup.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import run_teacher_mechanism_probe as m1
import run_teacher_probe as generation
import score_teacher_probe as score

ROOT = Path(os.environ.get("TEACHER_PROBE_ROOT", str(Path(__file__).resolve().parents[1])))
P1 = Path("/srv/wm_tmp/teacher_p1_cfg_20261019_run01")
PROVENANCE = Path("/srv/wm_tmp/teacher_opt_review_20261019/p1_provenance.json")
PROVENANCE_SHA256 = "e5acbdb3694dda35d08b92ca46fad0d7eebd65def9b0255fb7277b0c7e5ead75"
CASE_ORDER, MODELS = m1.CASE_ORDER, m1.MODELS
require, sha256, read_json = m1.require, m1.sha256, score.read_json
write_json, write_bytes = m1.atomic_json, m1.atomic_bytes
CARDS = ("0", "1", "2", "3")


def budget(gpus, generation_seconds, scoring_seconds, grace):
    require(gpus == "0,1,2,3", "C1 fixes exactly GPUs 0,1,2,3")
    for value, cap in ((generation_seconds, 1800), (scoring_seconds, 3600)):
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= cap,
                "C1 finite wall budget exceeds 1800/3600 seconds")
    require(type(grace) in (int, float) and math.isfinite(grace) and 1 <= grace < min(generation_seconds, scoring_seconds) / 2,
            "Cleanup grace must fit inside both reservations")
    return {"gpus": list(CARDS), "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds,
            "cleanup_grace_seconds": grace, "generation_gpu_hours": 4 * generation_seconds / 3600,
            "scoring_gpu_hours": 4 * scoring_seconds / 3600, "experiment_cap_gpu_hours": 6}


def relative_video(root, value):
    require(isinstance(value, str) and value and "://" not in value and "\\" not in value, "Invalid full CSV path")
    path = Path(value)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in value.split("/")),
            "CSV video must be a safe ROOT-relative full path")
    # Lexical matching avoids resolving every row on the shared filesystem.
    return os.path.abspath(os.path.join(str(root), value))


def original_captions(cases, provenance, root=ROOT, deadline=None):
    require(provenance["protocol"] == "P1_CFG_ONLY_CORRECTED_AGENT_DENSE_DEVLABEL_V1", "Wrong independent P1 provenance")
    selected = provenance["selected_cases"]
    require(tuple(c["case_id"] for c in selected) == CASE_ORDER, "Provenance case order/coverage differs")
    require(tuple(c["case_id"] for c in cases) == CASE_ORDER, "P1 case order differs")
    by_case = {c["case_id"]: c for c in selected}
    csv_paths = {c["prompt_review"]["csv_path"] for c in cases}
    require(csv_paths == {str(root / "training_metadata/ft2/ft2_train.csv")}, "Original inputs must use the frozen FT2 CSV")
    targets = {str(Path(c["image"]).resolve()): c["case_id"] for c in cases}
    require(len(targets) == 6, "Duplicate input full paths")
    matches = {c: [] for c in CASE_ORDER}
    csv_path = Path(next(iter(csv_paths)))
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require({"video", "prompt"} <= set(reader.fieldnames or ()), "FT2 CSV columns differ")
        for index, row in enumerate(reader):
            m1.check_deadline(deadline)
            full = relative_video(root, row["video"])
            if full in targets:
                matches[targets[full]].append((index, reader.line_num, row))
    result = {}
    for case in cases:
        cid, review = case["case_id"], case["prompt_review"]
        require(len(matches[cid]) == 1, "GT must match exactly one FT2 row by full path: " + cid)
        index, line, row = matches[cid][0]
        p = by_case[cid]
        require(Path(relative_video(root, row["video"])).resolve() == Path(case["image"]).resolve(), "CSV matched file differs")
        require(p["csv_line"] == line and p["video"] == case["image"] and p["video_sha256"] == case["image_sha256"]
                and p["canonical_source"] == case["canonical_source"], "Independent input provenance differs")
        require(p["agent_review"] == {k: review[k] for k in p["agent_review"]}, "Independent review differs")
        original = row["prompt"]  # Parsed CSV text deliberately not stripped or rewritten.
        require(isinstance(original, str) and original and original == p["original_prompt"]
                and m1.text_hash(original) == p["original_prompt_sha256"], "Exact original caption/hash differs")
        require(case["prompt"] == p["corrected_prompt"] and m1.text_hash(case["prompt"]) == p["corrected_prompt_sha256"]
                == review["prompt_sha256"] and p["original_preserved"] is True, "Corrected frozen caption differs")
        result[cid] = {"prompt": original, "prompt_sha256": m1.text_hash(original), "csv_row": index,
                       "csv_line": line, "video_relative": row["video"], "csv_path": review["csv_path"],
                       "csv_sha256": review["csv_sha256"]}
    return result


def validate_rows(jobs, rows):
    require(len(jobs) == len(rows) == 12, "C1 requires exactly 12 rows, not fake 24")
    expected = {j["id"]: j for j in jobs}
    require(len(expected) == 12, "Duplicate jobs")
    seen, cells, sources = set(), {}, {}
    for row in rows:
        require(row["id"] in expected and row["id"] not in seen, "Duplicate/unexpected row")
        seen.add(row["id"])
        job = expected[row["id"]]
        for key in ("case_id", "canonical_source", "model", "cfg"):
            require(row[key] == job[key], "Metric binding differs: " + key)
        require(type(row["cfg"]) in (int, float) and row["cfg"] == 5, "Only actual CFG5")
        for key in score.METRICS:
            score.number(row[key], key, 0, 1 if key in ("subject_consistency", "imaging_quality") else None)
        score.normalized_imaging(row["imaging_quality"], row["imaging_quality_raw"])
        for key in ("gen_motion", "gt_motion", "motion_amount_ratio", "dur", "cpu_seconds", "gpu_seconds",
                    "peak_allocated_bytes", "peak_reserved_bytes"):
            score.number(row[key], key, 0)
        require(row["gen_motion"] > 0 and row["gt_motion"] > 0 and row["dur"] > 0, "Positive motion and duration required")
        require(math.isclose(row["motion_amount_ratio"], row["gen_motion"] / row["gt_motion"], abs_tol=1e-8)
                and math.isclose(row["motion_ratio"], abs(math.log(row["motion_amount_ratio"])), abs_tol=1e-8),
                "Original abs(log(gen/GT)) motion formula differs")
        cells.setdefault(row["case_id"], set()).add(row["model"])
        require(sources.get(row["case_id"], row["canonical_source"]) == row["canonical_source"], "Changed source")
        sources[row["case_id"]] = row["canonical_source"]
    require(seen == set(expected) and set(cells) == set(CASE_ORDER) and all(v == set(MODELS) for v in cells.values())
            and len(set(sources.values())) == 6, "Incomplete six-case two-model matrix")


def controls(p1_jobs, deadline=None):
    run = P1
    summary_path, rows_path = run / "scoring/score_summary.json", run / "scoring/score_rows.json"
    summary, manifest = read_json(summary_path), read_json(run / "scoring/score_manifest.json")
    require(summary["status"] == "completed" and summary["expected_jobs"] == summary["completed_rows"] == 24
            and not summary["failures"] and summary["teacher_passed"] is False, "P1 scoring incomplete")
    score.validate_charge(summary["allocated_gpu_workers"], summary["actual_scoring_gpu_hours"],
                          summary["reserved_scoring_gpu_hours"], summary["effective_scoring_cap_gpu_hours"], "P1 scoring")
    require(summary["allocation_includes_cpu_setup_and_final_verification"] is True
            and math.isclose(summary["actual_experiment_gpu_hours"], summary["generation_gpu_hours"] + summary["actual_scoring_gpu_hours"], abs_tol=1e-10)
            and summary["actual_experiment_gpu_hours"] <= 12, "P1 score budget differs")
    require(len(summary["metric_asset_sha256"]) == 39 and summary["metric_asset_sha256"] == manifest["metric_asset_sha256"],
            "P1 original 39 metric assets differ")
    score.verify_hashes(summary["metric_asset_sha256"], deadline)
    score.verify_hashes(summary["frozen_input_sha256"], deadline)
    require(score.metric_assets() == summary["metric_asset_sha256"], "Current metrics differ from P1 frozen original algorithm/assets")
    score.checked_hash(rows_path, summary["score_rows_sha256"], "P1 score rows", deadline)
    rows = read_json(rows_path)
    score.validate_rows(p1_jobs, rows)
    by_id = {r["id"]: r for r in rows}
    manifest_jobs = {j["id"]: j for j in manifest["jobs"]}
    require(len(manifest_jobs) == 24 and set(manifest_jobs) == {j["id"] for j in p1_jobs}, "P1 scoring jobs differ")
    for job in p1_jobs:
        bound = manifest_jobs[job["id"]]
        for key in ("id", "case_id", "canonical_source", "model", "cfg", "out", "output_sha256", "gt", "gt_sha256"):
            require(bound[key] == job[key], "P1 score input binding differs: " + key)
    selected = [j for cid in CASE_ORDER for model in MODELS for j in p1_jobs
                if (j["case_id"], j["model"], j["cfg"]) == (cid, model, 5)]
    result = []
    for job in selected:
        row = by_id[job["id"]]
        result.append({"job": job, "row": row, "row_sha256": hashlib.sha256(generation.json_bytes(row)).hexdigest(),
                       "video_sha256": job["output_sha256"]})
    validate_rows([r["job"] for r in result], [r["row"] for r in result])
    return result, summary["metric_asset_sha256"], {str(p): sha256(p, deadline) for p in
           (summary_path, rows_path, run / "scoring/score_manifest.json")}


def make_jobs(control_rows, originals, run):
    jobs = []
    for control in control_rows:
        old = control["job"]
        cid = old["case_id"]
        job = dict(old, id=old["id"] + "_original", prompt=originals[cid]["prompt"],
                   out=str(run / "generation/videos" / (old["id"] + "_original.mp4")))
        job.pop("output_sha256", None)
        job.pop("prompt_review")
        job["caption_provenance"] = dict(originals[cid], version="original",
                parsed_original_prompt_sha256=originals[cid]["prompt_sha256"],
                corrected_control_id=old["id"], corrected_control_review=old["prompt_review"])
        require(m1.text_hash(job["prompt"]) == job["caption_provenance"]["parsed_original_prompt_sha256"],
                "New original caption must bind its own exact hash")
        jobs.append(job)
        for key in set(old) - {"id", "prompt", "out", "output_sha256", "prompt_review"}:
            require(job[key] == old[key], "Only caption may change the inference condition")
        require(job["seed"] == 0 and job["cfg"] == 5 and job["fps"] == 24 and job["quality"] == 5, "Fixed inference differs")
    require(len(jobs) == 12, "Exactly 12 original-caption generations")
    return jobs


def preflight(run, deadline=None):
    run = Path(run).absolute()
    require(run == run.resolve() and run.is_relative_to(Path("/srv/wm_tmp"))
            and not run.exists() and not run.is_symlink(), "Run must be a new real local private /srv/wm_tmp directory")
    audit = m1.preflight(P1, run, m1.MODELS_DEFAULT, deadline)
    p1_jobs, generation_actual, hashes = score.validate_run(P1, decode=False)
    require(generation_actual <= 4, "P1 generation budget differs")
    score.checked_hash(PROVENANCE, PROVENANCE_SHA256, "Independent original P1 provenance", deadline)
    provenance = read_json(PROVENANCE)
    originals = original_captions(audit["cases"], provenance, ROOT, deadline)
    corrected, assets, score_hashes = controls(p1_jobs, deadline)
    hashes.update(score_hashes)
    hashes[str(PROVENANCE)] = PROVENANCE_SHA256
    own_sources = {name: m1.file_record(ROOT / "scripts" / name, deadline=deadline) for name in
                   ("run_teacher_caption_probe.py", "run_teacher_mechanism_probe.py", "score_teacher_probe.py", "gt_ref_metrics.py")}
    return {"protocol": "teacher-caption-c1-v1", "p1_audit": audit, "p1_generation_gpu_hours": generation_actual,
            "original_csv_bindings": originals, "corrected_controls": corrected, "jobs": make_jobs(corrected, originals, run),
            "p1_input_sha256": hashes, "metric_asset_sha256": assets, "own_sources": own_sources,
            "expected_new_outputs": 12, "expected_reused_outputs": 12, "case_order": list(CASE_ORDER),
            "negative_prompt_sha256": audit["negative_prompt_sha256"], "training_performed": False,
            "teacher_passed": False, "review": "pending", "promotion_allowed": False,
            "interpretation": "FT2 original inputs are not A07/FT1 training captions; caption length/wording/facts all change; no causal training benefit"}


def freeze(run, manifest, allocation, deadline):
    run.mkdir(mode=0o700)
    for name in ("code", "framework", "generation", "scoring"):
        (run / name).mkdir(mode=0o700)
    records = [("code/" + name, rec) for name, rec in manifest["own_sources"].items()]
    records += [("code/" + name, rec) for name, rec in manifest["p1_audit"]["p1_source"].items()]
    records += [("framework/" + name, rec) for name, rec in manifest["p1_audit"]["framework"].items()]
    frozen = {}
    for relative, rec in records:
        m1.check_deadline(deadline)
        data = Path(rec["path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == rec["sha256"], "Source changed during freeze")
        target = m1.private_path(run, relative)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_bytes(target, data)
        frozen[relative] = rec["sha256"]
    write_bytes(run / "jobs.frozen.jsonl", b"".join(generation.json_bytes(j) for j in manifest["jobs"]))
    frozen["jobs.frozen.jsonl"] = sha256(run / "jobs.frozen.jsonl", deadline)
    manifest = dict(manifest, allocation=allocation, snapshot_sha256=frozen)
    write_json(run / "caption_manifest.json", manifest)
    return manifest


def verify_sources(run, manifest, deadline):
    for relative, pin in manifest["snapshot_sha256"].items():
        require(sha256(m1.private_path(run, relative), deadline) == pin, "Frozen C1 runtime changed")
    for name in ("p1_files", "p1_source", "framework", "model_cache"):
        for rec in manifest["p1_audit"][name].values():
            m1.validate_record(rec, deadline)
    for rec in manifest["own_sources"].values():
        m1.validate_record(rec, deadline)
    score.verify_hashes(manifest["p1_input_sha256"], deadline)
    score.verify_hashes(manifest["metric_asset_sha256"], deadline)


def assignments(jobs):
    # P1 real worker: two GPUs per model, three jobs per GPU, original case order.
    return [{"gpu": gpu, "jobs": [j for j in jobs if j["model"] == MODELS[i // 2]][i % 2::2]}
            for i, gpu in enumerate(CARDS)]


def launch(run, kind, allocated, deadline, workers):
    directory = run / ("generation" if kind in ("gen", "verify_gen") else "scoring")
    local = []
    for index, assignment in enumerate(allocated):
        m1.check_deadline(deadline)
        path = directory / (kind + "_assignment_" + str(index) + ".json")
        write_json(path, assignment)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=assignment["gpu"], C1_PARENT_PID=str(os.getpid()),
                   TEACHER_PROBE_PARENT_PID=str(os.getpid()), TEACHER_SCORE_PARENT_PID=str(os.getpid()),
                   TEACHER_PROBE_ROOT=str(ROOT), DIFFSYNTH_SKIP_DOWNLOAD="true", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                   DIFFSYNTH_MODEL_BASE_PATH=str(m1.MODELS_DEFAULT), WM_CHECKPOINT_CACHE_DIR=str(generation.CHECKPOINT_CACHE),
                   THREADS="8", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
        env["PYTHONPATH"] = str(run / "framework") + os.pathsep + str(run / "code")
        log = (directory / (kind + "_worker_" + str(index) + ".log")).open("x")
        try:
            proc = subprocess.Popen([score.PYTHON if kind != "gen" else sys.executable,
                     str(run / "code/run_teacher_caption_probe.py"), "--out-dir", str(run),
                     "--worker", kind, "--worker-index", str(index), "--assignment-sha", sha256(path, deadline)],
                     env=env, cwd=str(run / "framework"), stdin=subprocess.DEVNULL, stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        info = {"role": kind + str(index), "index": index, "gpu": assignment["gpu"],
                "process": proc, "log": log, "started": time.perf_counter()}
        workers.append(info)
        local.append(info)
        require(all(w["process"].poll() in (None, 0) for w in local), "Worker startup failure")
    m1.wait_workers(local, deadline)
    return [{"index": w["index"], "gpu": w["gpu"], "pid": w["process"].pid,
             "returncode": w["process"].returncode} for w in local]


def worker(run, kind, index, assignment_sha):
    require(os.environ.get("C1_PARENT_PID") == str(os.getppid()), "Worker is internal to live parent")
    manifest = read_json(run / "caption_manifest.json")
    directory = run / ("generation" if kind in ("gen", "verify_gen") else "scoring")
    path = directory / (kind + "_assignment_" + str(index) + ".json")
    require(sha256(path) == assignment_sha, "Assignment changed")
    assignment = read_json(path)
    require(assignment["gpu"] == os.environ.get("CUDA_VISIBLE_DEVICES"), "Worker GPU differs")
    for relative, pin in manifest["snapshot_sha256"].items():
        require(sha256(run / relative) == pin, "Frozen worker sources changed")
    import socket
    from unittest.mock import patch
    def blocked(*args, **kwargs):
        raise RuntimeError("C1 forbids all network access")
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked):
        if kind == "gen":
            import gen_jobs
            # Keep P1 actual worker/sampler/encoder, replacing only implicit model search
            # by its byte-identical verified explicit local paths (no shared fallback).
            gen_jobs.build_pipe = lambda ckpt: m1.build_pipe(run, manifest["p1_audit"],
                                          "base" if ckpt == "NONE" else "m_full1_a07", None)
            generation.atomic_json(directory / ("assignment_" + str(index) + ".json"), assignment)
            return generation.worker(directory, index, sha256(directory / ("assignment_" + str(index) + ".json")))
        if kind in ("cpu", "gpu"):
            return score.worker(directory, kind, index, assignment_sha)
        outputs = []
        for job in assignment["jobs"]:
            pin = job.get("output_sha256")
            if pin is not None:
                score.checked_hash(job["out"], pin, "Complete decode input")
            outputs.append({"id": job["id"], "out": job["out"], "decode": score.decode_video(job["out"]),
                            "sha256": sha256(job["out"])})
        write_json(directory / (kind + "_worker_" + str(index) + ".json"),
                   {"status": "completed", "gpu": "", "outputs": outputs})
    return 0


def generation_reports(run, jobs, resources):
    require(len(resources) == 4 and all(r["returncode"] == 0 for r in resources), "Four zero-exit generation workers required")
    rows = {}
    for resource in resources:
        report = read_json(run / "generation" / ("worker_" + str(resource["index"]) + ".json"))
        require(report["status"] == "completed" and report["gpu"] == resource["gpu"], "Generation report failed")
        for row in report["jobs"]:
            require(row["id"] not in rows and row["status"] == "completed", "Partial/duplicate generation report")
            for key in ("seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
                score.number(row[key], key, 0)
            rows[row["id"]] = row
    require(set(rows) == {j["id"] for j in jobs}, "Generation must report all 12 IDs")
    require(all(rows[j["id"]]["out"] == j["out"] for j in jobs), "Generation output path differs")


def decoded_report(run, kind, jobs):
    report = read_json(run / ("generation" if kind == "verify_gen" else "scoring") / (kind + "_worker_0.json"))
    rows = report["outputs"]
    require(report["status"] == "completed" and len(rows) == len(jobs) and len({r["id"] for r in rows}) == len(jobs),
            "Full decode output coverage failed")
    by_id = {r["id"]: r for r in rows}
    for job in jobs:
        row = by_id[job["id"]]
        require(row["out"] == job["out"] and row["decode"]["frames"] == 121
                and (row["decode"]["width"], row["decode"]["height"]) == (1280, 704)
                and math.isclose(row["decode"]["fps"], 24, abs_tol=.01), "Full decode fixed video parameters differ")
        if "output_sha256" in job:
            require(row["sha256"] == job["output_sha256"], "Decoded output hash differs")
    return rows


def merge_score_reports(run, jobs, resources):
    by_kind = {}
    for kind in ("cpu", "gpu"):
        require(resources[kind] and all(r["returncode"] == 0 for r in resources[kind]), "Missing/nonzero score exits")
        rows = []
        for resource in resources[kind]:
            report = read_json(run / "scoring" / (kind + "_worker_" + str(resource["index"]) + ".json"))
            require(report["status"] == "completed" and report["kind"] == kind and report["gpu"] == resource["gpu"], "Incomplete score report")
            rows.extend(report["rows"])
        require(len(rows) == 12 and len({r["id"] for r in rows}) == 12
                and {r["id"] for r in rows} == {j["id"] for j in jobs}, "Partial/duplicate score rows")
        by_kind[kind] = {r["id"]: r for r in rows}
    merged = []
    for job in jobs:
        cpu, gpu = by_kind["cpu"][job["id"]], by_kind["gpu"][job["id"]]
        for key in ("id", "case_id", "canonical_source", "model", "cfg"):
            require(cpu[key] == gpu[key], "CPU/GPU score binding differs")
        row = dict(cpu, **{k: v for k, v in gpu.items() if k not in cpu}, cpu_seconds=cpu["seconds"], gpu_seconds=gpu["seconds"])
        merged.append(row)
    validate_rows(jobs, merged)
    return merged


def diagnostics(jobs, original, corrected):
    validate_rows(jobs, original)
    validate_rows([c["job"] for c in corrected], [c["row"] for c in corrected])
    old = {(r["case_id"], r["model"]): r for r in original}
    new = {(c["row"]["case_id"], c["row"]["model"]): c["row"] for c in corrected}
    pairs, means = [], {}
    for cid in CASE_ORDER:
        for model in MODELS:
            a, b = old[cid, model], new[cid, model]
            require(math.isclose(a["gt_motion"], b["gt_motion"], abs_tol=1e-8)
                    and math.isclose(a["dur"], b["dur"], abs_tol=1e-8), "Paired original GT motion/timing differs")
            delta = {k: b[k] - a[k] for k in score.METRICS}
            relative = b["gen_motion"] / a["gen_motion"]
            guards = {"anti_static_motion": relative >= .7, "subject_not_degraded": delta["subject_consistency"] >= 0,
                      "quality_not_degraded": delta["imaging_quality"] >= 0, "GT_mean_not_degraded": delta["dreamsim"] <= 0,
                      "GT_last_not_degraded": delta["dreamsim_last"] <= 0, "motion_error_not_degraded": delta["motion_ratio"] <= 0}
            pairs.append({"case_id": cid, "model": model, "original_id": a["id"], "corrected_id": b["id"],
                          "corrected_minus_original": delta, "original_gen_motion": a["gen_motion"],
                          "corrected_gen_motion": b["gen_motion"], "gt_motion": a["gt_motion"],
                          "corrected_over_original_motion": relative, "guards": guards,
                          "decision": "pending_semantic_review" if all(guards.values()) else "reject_or_tradeoff",
                          "visible_action_object_review": "pending"})
    for model in MODELS:
        selected = [p for p in pairs if p["model"] == model]
        means[model] = {k: math.fsum(p["corrected_minus_original"][k] for p in selected) / 6 for k in score.METRICS}
    paired_rows = [dict(r, caption_version="original", newly_generated=True) for r in original]
    paired_rows += [dict(c["row"], caption_version="corrected", newly_generated=False,
                         frozen_row_sha256=c["row_sha256"], video_sha256=c["video_sha256"]) for c in corrected]
    return {"status": "completed", "new_original_rows": 12, "reused_corrected_rows": 12, "paired_rows": paired_rows,
            "case_order": list(CASE_ORDER), "pairs": pairs, "model_mean_corrected_minus_original": means,
            "a07_change_minus_base_change": {k: means[MODELS[1]][k] - means[MODELS[0]][k] for k in score.METRICS},
            "motion_error_formula": "abs(log(gen_motion/gt_motion))", "quality_units": "raw MUSIQ / 100",
            "training_benefit": False, "causal_claim": False, "holdout": False, "teacher_passed": False,
            "promotion_allowed": False, "human_semantic_action_review": "pending",
            "interpretation": "Input-caption counterfactual only; FT2 originals are not A07 training labels; length/wording/facts confounded; six training-visible cases, one seed"}


def phase(run, name, seconds, grace, operation):
    started = time.perf_counter()
    # Reserve a conservatively charged second for ledger publication inside the cap.
    hard_deadline, deadline = started + seconds - 1, started + seconds - grace - 1
    workers, cleanup, result, failure, failure_traceback = [], [], {}, None, None
    handlers = {}
    def interrupted(signum, frame):
        raise InterruptedError("C1 " + name + " deadline/signal reached")
    for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.perf_counter()))
    try:
        m1.enable_subreaper()
        result = operation(deadline, workers)
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)
        failure_traceback = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            cleanup = m1.stop_workers(workers, hard_deadline, grace)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    elapsed = time.perf_counter() - started + 1
    complete = not failure and elapsed <= seconds and all(c["owned_group_gone"] and c["returncode"] == 0 for c in cleanup)
    result.update(status="completed" if complete else "failed", failures=[failure] if failure else ([] if complete else ["Budget/owned cleanup failure"]),
                  wall_seconds=elapsed, allocated_cards=4, allocated_gpu_hours=4 * elapsed / 3600,
                  reserved_allocated_gpu_hours=4 * seconds / 3600, hard_wall_seconds=seconds, owned_cleanup=cleanup,
                  allocation_includes_preflight_setup_CPU_wait_encoding_failure_finalverify=True,
                  teacher_passed=False, promotion_allowed=False, review="pending",
                  conservative_publication_reserve_seconds=1, failure_traceback=failure_traceback,
                  experiment_cap_gpu_hours=6)
    if name == "score":
        result["generation_gpu_hours"] = read_json(run / "generation_summary.json")["allocated_gpu_hours"]
        result["actual_experiment_gpu_hours"] = result["generation_gpu_hours"] + result["allocated_gpu_hours"]
        if result["actual_experiment_gpu_hours"] > 6:
            result["status"] = "failed"
            result["failures"].append("Total experiment exceeds 6 allocated GPUh")
    if run.is_dir():
        write_json(run / (name + "_summary.json"), result)
    return result


def execute(args, allocation):
    run = Path(args.out_dir).absolute()
    require(not run.exists() and not run.is_symlink(), "No resume/noclobber")
    manifest, generated = None, None
    os.umask(0o077)
    def generate(deadline, workers):
        nonlocal manifest, generated
        manifest = preflight(run, deadline)
        generation.assert_idle(list(CARDS))
        manifest = freeze(run, manifest, allocation, deadline)
        jobs = manifest["jobs"]
        resources = launch(run, "gen", assignments(jobs), deadline, workers)
        generation_reports(run, jobs, resources)
        all_videos = jobs + [c["job"] for c in manifest["corrected_controls"]]
        launch(run, "verify_gen", [{"gpu": "", "jobs": all_videos}], deadline, workers)
        decoded = decoded_report(run, "verify_gen", all_videos)
        generated = decoded[:12]
        verify_sources(run, manifest, deadline)
        return {"expected_new_outputs": 12, "completed_new_outputs": len(generated), "outputs": generated,
                "reused_corrected_full_decodes": decoded[12:], "workers": resources, "manifest_sha256": sha256(run / "caption_manifest.json", deadline)}
    gen = phase(run, "generation", allocation["generation_seconds"], allocation["cleanup_grace_seconds"], generate)
    if gen["status"] != "completed":
        return 1
    def scoring(deadline, workers):
        verify_sources(run, manifest, deadline)
        outputs = {r["id"]: r for r in generated}
        jobs = [dict(j, output_sha256=outputs[j["id"]]["sha256"]) for j in manifest["jobs"]]
        assets = manifest["metric_asset_sha256"]
        score.assert_idle(list(CARDS))
        write_json(run / "scoring/score_manifest.json", {"jobs": jobs, "metric_asset_sha256": assets,
                   "algorithm": "unchanged P1 CPU DreamSim/read/sample/motion and GPU DINO/MUSIQ; no metric models changed"})
        cpu = launch(run, "cpu", [{"gpu": "", "jobs": jobs, "asset_sha256": assets}], deadline, workers)
        gpu = launch(run, "gpu", [dict(a, asset_sha256=assets) for a in assignments(jobs)], deadline, workers)
        rows = merge_score_reports(run, jobs, {"cpu": cpu, "gpu": gpu})
        all_videos = jobs + [c["job"] for c in manifest["corrected_controls"]]
        launch(run, "verify_score", [{"gpu": "", "jobs": all_videos}], deadline, workers)
        decoded_report(run, "verify_score", all_videos)
        verify_sources(run, manifest, deadline)
        report = diagnostics(jobs, rows, manifest["corrected_controls"])
        write_json(run / "scoring/score_rows.json", rows)
        # This file is not the stage-completion gate: score_summary must also pass cleanup/budget.
        write_json(run / "caption_pair_diagnostics.json", report)
        return {"expected_new_rows": 12, "completed_new_rows": 12, "reused_corrected_rows": 12, "cpu_workers": cpu,
                "gpu_workers": gpu, "score_rows_sha256": sha256(run / "scoring/score_rows.json", deadline),
                "caption_pair_diagnostics_sha256": sha256(run / "caption_pair_diagnostics.json", deadline),
                "generation_gpu_hours": gen["allocated_gpu_hours"]}
    scored = phase(run, "score", allocation["scoring_seconds"], allocation["cleanup_grace_seconds"], scoring)
    return 0 if scored["status"] == "completed" and gen["allocated_gpu_hours"] + scored["allocated_gpu_hours"] <= 6 else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--generation-seconds", type=float, default=1800)
    p.add_argument("--scoring-seconds", type=float, default=3600)
    p.add_argument("--cleanup-grace-seconds", type=float, default=5)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--worker", choices=("gen", "cpu", "gpu", "verify_gen", "verify_score"), help=argparse.SUPPRESS)
    p.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    p.add_argument("--assignment-sha", help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    run = Path(args.out_dir).absolute()
    if args.worker is not None:
        require(type(args.worker_index) is int and args.worker_index >= 0 and args.assignment_sha, "Invalid internal worker")
        return worker(run, args.worker, args.worker_index, args.assignment_sha)
    allocation = budget(args.gpus, args.generation_seconds, args.scoring_seconds, args.cleanup_grace_seconds)
    if args.preflight_only:
        manifest = preflight(run)
        print(json.dumps({"status": "preflight_passed", "new_jobs": len(manifest["jobs"]), "controls": len(manifest["corrected_controls"]),
                          "model_cache_files": len(manifest["p1_audit"]["model_cache"]), "metric_assets": len(manifest["metric_asset_sha256"]),
                          "case_order": manifest["case_order"], "allocation": allocation, "GPU_queried": False,
                          "run_created": False, "teacher_passed": False}, sort_keys=True))
        return 0
    return execute(args, allocation)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("C1 refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
