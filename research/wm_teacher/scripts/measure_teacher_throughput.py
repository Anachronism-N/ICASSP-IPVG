#!/usr/bin/env python3
"""Private, bounded two-seed Teacher timing; default CPU prepare is not GPU authorization.

One NONE pipeline, fixed safe E1 first case, full seed0/1 videos. Compilation is
in inference; hashes, supervisor wait, decode and cleanup count in allocation.
No fallback, shortened generation, quality tuning or synthetic timing receipts.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_teacher_inference_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "teacher_batch_throughput_v1"
REQUEST_SCHEMA = "teacher_batch_throughput_request_v1"
HISTORY_COMMIT = "eccfdb12331ca7163e9fd29f1a66eca1febb9765"
HISTORY_REPO = Path("/dockerdata/icaspp_ipvg_publication_20261018")
OBSERVATION_HELPER_SHA = "5804b840212765443791f68f5d6316c878e8b4ebed41acab218bd078b564e15d"
BUDGET = dict(gpus=1, gpu_hours=0.75, timeout_seconds=2600, kill_seconds=30,
              internal_stop_seconds=2500, outer_cap_seconds=2660)
require, encoded, sha = smoke.require, smoke.encoded, smoke.sha
record, check_record = smoke.record, smoke.check_record
exclusive, checkpoint = smoke.exclusive, smoke.checkpoint


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_paths(root):
    return sorted(set(smoke.source_paths(root)) | {root / "scripts/measure_teacher_throughput.py",
                                                  root / "tests/test_teacher_throughput.py"})


def output_root(root, method):
    return root / f"eval/teacher_{method}_throughput_20261026_01"


def historical_helper(expected):
    manifest_path = "research/wm_teacher/PUBLICATION_MANIFEST.json"
    source_path = "research/wm_teacher/scripts/run_teacher_inference_smoke.py"
    def published(path):
        return subprocess.run(["git", "-C", str(HISTORY_REPO), "show", HISTORY_COMMIT + ":" + path],
                              check=True, capture_output=True, timeout=30).stdout
    manifest_raw, source = published(manifest_path), published(source_path)
    entries = [item for item in json.loads(manifest_raw)["files"]
               if item["path"] == "scripts/run_teacher_inference_smoke.py"]
    require(len(entries) == 1 and entries[0]["source_sha256"] == expected["sha256"]
            and entries[0]["snapshot_sha256"] == sha(source),
            "Historical smoke helper does not match immutable published source attestation")
    current = record(expected["path"])
    require(current["sha256"] in (expected["sha256"], OBSERVATION_HELPER_SHA),
            "Unapproved current smoke helper drift beyond audited observation patch")
    return dict(commit=HISTORY_COMMIT, repository=str(HISTORY_REPO), manifest=manifest_path,
                manifest_sha256=sha(manifest_raw), source_path=source_path, source_sha256=expected["sha256"],
                snapshot_sha256=sha(source))


def current_original_inputs(request):
    root = Path(request["root"]).resolve()
    sources = request["source_files"]
    require({r["path"] for r in sources} == {str(p) for p in smoke.source_paths(root)}, "Original source set drift")
    attestation = None
    for item in sources:
        if item["path"] == str(root / "scripts/run_teacher_inference_smoke.py"):
            attestation = historical_helper(item)
        else:
            check_record(item)
    check_record(request["node_preflight"], nonempty=True)
    pf = json.loads(Path(request["node_preflight"]["path"]).read_bytes())
    require(smoke.models_from_preflight(pf) == request["models"]
            and pf["model_content_sha256"] == request["model_content_sha256"], "Current model/preflight content drift")
    case, image, cases_file = smoke.first_case(root)
    require(case == request["case"] and image == request["image"] and cases_file == request["cases_file"],
            "Current original case/image/prompt drift")
    negative = smoke.generator_negative(root)
    require(negative == request["negative_prompt"] and sha(negative.encode()) == request["negative_sha256"], "Current original NEG drift")
    require(smoke.runtime_records() == request["runtime"], "Current original runtime drift")
    return attestation


def original_smokes(root, method, approved_smoke02_sha, verify_local_request=True):
    extra = root / f"eval/teacher_{method}_smoke_20261026_03"
    require(not (extra / "launch.json").exists() and not (extra / "finished.json").exists(),
            "Unaccounted single-video03 allocation; only one performance attempt allowed")
    attempts, charged, latest_request = [], 0.0, None
    for date, suffix in (("20261025", "01"), ("20261026", "02")):
        folder = root / f"eval/teacher_{method}_smoke_{date}_{suffix}"
        paths = {name: folder / (name + ".json") for name in ("request", "launch", "finished")}
        require(all(p.is_file() for p in paths.values()), "Missing original smoke request/launch/finished; blocked")
        records = {name: record(p, nonempty=True) for name, p in paths.items()}
        request, launch, receipt = (json.loads(paths[n].read_bytes()) for n in ("request", "launch", "finished"))
        require(request["schema"] == "teacher_inference_smoke_v1" and request["private"] is True
                and request["method"] == method and encoded(request["parameters"]) == encoded(smoke.PARAMS[method])
                and encoded(request["call"]) == encoded(smoke.SPEC) and request["checkpoint"] == "NONE"
                and request["fps"] == 24 and request["active_steps"] == 10 and request["video_count"] == 1
                and request["budget"] == smoke.BUDGET, "Original smoke method/configuration drift")
        require(launch["request_sha256"] == records["request"]["sha256"] and launch["budget"] == smoke.BUDGET
                and receipt["schema"] == "teacher_smoke_receipt_v1" and receipt["method"] == method
                and receipt["request_sha256"] == records["request"]["sha256"]
                and receipt["allocated_gpus"] == 1 and receipt["budget"] == smoke.BUDGET, "Original smoke binding mismatch")
        elapsed = receipt["elapsed_seconds"]
        require(finite(elapsed) and elapsed > 0 and finite(receipt["elapsed_GPUh"])
                and math.isclose(receipt["elapsed_GPUh"], elapsed / 3600, abs_tol=1e-9)
                and finite(receipt["start"]) and finite(receipt["end"])
                and abs(receipt["end"] - receipt["start"] - elapsed) < 5, "Invalid original smoke cost")
        charged += elapsed
        if suffix == "02":
            require(records["finished"]["sha256"] == approved_smoke02_sha, "Approved latest smoke02 SHA mismatch")
            evidence = receipt["evidence"]
            require(receipt.get("success") is True and receipt.get("mechanism_PASS") is True
                    and receipt.get("owned_tree_released") is True and receipt.get("budget_exceeded") is False
                    and receipt.get("exit_code") == 0 and receipt.get("timed_out") is False
                    and receipt.get("interrupted_signal") is None and receipt.get("error") is None
                    and evidence.get("success") is True and evidence.get("mechanism_PASS") is True
                    and evidence.get("parameters_unchanged") is True
                    and evidence["parameters_before"] and evidence["parameters_before"] == evidence["parameters_after"],
                    "Original smoke02 lacks complete mechanism PASS; blocked")
            records["worker"] = record(folder / "worker.json", nonempty=True)
            require(json.loads((folder / "worker.json").read_bytes()) == evidence, "Original worker/receipt mismatch")
            smoke.validate_stats(method, evidence["stats"]["method"], evidence["stats"]["observed"])
            if verify_local_request:
                require(request["hostname"] == os.uname().nodename, "Original local model/runtime node mismatch")
                current_original_inputs(request)
            require(Path(request["root"]).resolve() == root, "Original smoke root drift")
            video = smoke.validate_video(folder / "video.mp4", request["runtime"]["tools"])
            require(video == evidence["video"], "Original smoke video SHA/format drift")
            records["video"] = video["file"]
            latest_request = request
        attempts.append(dict(suffix=suffix, records=records, elapsed_seconds=elapsed))
    require(charged + BUDGET["gpu_hours"] * 3600 < 3600, "Original smoke plus reserved performance exceeds 1 GPUh")
    return attempts, charged, latest_request


def prepare(method, request_dir, approved_smoke02_sha, root=ROOT):
    require(method in smoke.PARAMS, "One fixed method required")
    root, folder = Path(root).resolve(), Path(request_dir).absolute()
    require(folder == output_root(root, method), "Only the unique fixed method performance directory is authorized")
    require(not folder.exists(), "Request directory already exists; never overwrite")
    attempts, charged, original = original_smokes(root, method, approved_smoke02_sha)
    value = dict(schema=REQUEST_SCHEMA, private=True, root=str(root), method=method,
                 parameters=smoke.PARAMS[method], call=smoke.SPEC, seeds=[0, 1], video_count=2,
                 checkpoint="NONE", fps=24, active_steps=10, budget=BUDGET,
                 case=original["case"], image=original["image"], cases_file=original["cases_file"],
                 negative_prompt=original["negative_prompt"], negative_sha256=original["negative_sha256"],
                 models=original["models"], model_content_sha256=original["model_content_sha256"],
                 node_preflight=original["node_preflight"], runtime=original["runtime"],
                 original_source_files=original["source_files"],
                 historical_helper_attestation=historical_helper(next(r for r in original["source_files"]
                    if r["path"] == str(root / "scripts/run_teacher_inference_smoke.py"))), prior_smoke_attempts=attempts,
                 latest_smoke02=attempts[-1]["records"]["finished"], prior_smoke_seconds=charged,
                 source_files=[record(p) for p in source_paths(root)],
                 request_dir=str(folder), hostname=os.uname().nodename, created_at=time.time(),
                 GPU_used=False, quality_pass=False, sota_claim=False)
    folder.mkdir(mode=0o700, parents=False)
    path = folder / "request.json"
    exclusive(path, value)
    return path, sha(path.read_bytes())


def verify_request(path, approved_sha, verify_local_request=True):
    path = Path(path).absolute()
    raw = path.read_bytes()
    require(isinstance(approved_sha, str) and len(approved_sha) == 64 and sha(raw) == approved_sha,
            "Approved performance request SHA mismatch")
    request = json.loads(raw)
    method = request["method"]
    require(request["schema"] == REQUEST_SCHEMA and request["private"] is True and method in smoke.PARAMS
            and encoded(request["parameters"]) == encoded(smoke.PARAMS[method])
            and encoded(request["call"]) == encoded(smoke.SPEC) and request["seeds"] == [0, 1]
            and request["video_count"] == 2 and request["checkpoint"] == "NONE"
            and request["fps"] == 24 and request["active_steps"] == 10
            and encoded(request["budget"]) == encoded(BUDGET), "Unapproved performance method/configuration/budget")
    require(request["request_dir"] == str(path.parent) and path.name == "request.json"
            and path.parent.stat().st_mode & 0o077 == 0 and path.stat().st_mode & 0o077 == 0,
            "Performance request must be private and immutable")
    if verify_local_request:
        require(request["hostname"] == os.uname().nodename, "Performance request is node-bound")
    root = Path(request["root"]).resolve()
    require(path.parent == output_root(root, method), "Performance request is not the unique fixed method directory")
    attempts, charged, original = original_smokes(root, method, request["latest_smoke02"]["sha256"], verify_local_request)
    require(attempts == request["prior_smoke_attempts"] and charged == request["prior_smoke_seconds"]
            and attempts[-1]["records"]["finished"] == request["latest_smoke02"], "Original evidence/cost drift")
    for key in ("case", "image", "cases_file", "negative_prompt", "negative_sha256", "models",
                "model_content_sha256", "node_preflight", "runtime"):
        require(request[key] == original[key], "Original smoke model/config/input/runtime mismatch: " + key)
    require(request["original_source_files"] == original["source_files"], "Original source binding drift")
    helper = next(r for r in original["source_files"] if r["path"] == str(root / "scripts/run_teacher_inference_smoke.py"))
    require(request["historical_helper_attestation"] == historical_helper(helper), "Historical helper attestation drift")
    require({r["path"] for r in request["source_files"]} == {str(p) for p in source_paths(root)}, "Measurement source set drift")
    for r in request["source_files"]:
        check_record(r)
    return request


def validate_timers(evidence, allocation_start, allocation_end):
    require(evidence.get("load_count") == 1 and evidence.get("clock") == "time.monotonic"
            and evidence.get("compile_accounting") == "included in inference"
            and [v["seed"] for v in evidence["videos"]] == [0, 1], "Incomplete once-load/two-seed raw timing evidence")
    segments = [evidence["timers"]["load"]]
    for video in evidence["videos"]:
        segments += [video["timers"]["inference"], video["timers"]["save_decode"]]
    previous = allocation_start
    for segment in segments:
        require(set(segment) == {"start", "end", "seconds"}
                and all(finite(segment[k]) for k in segment) and segment["seconds"] > 0
                and allocation_start <= segment["start"] < segment["end"] <= allocation_end
                and segment["start"] >= previous
                and math.isclose(segment["seconds"], segment["end"] - segment["start"], abs_tol=1e-9),
                "Invalid/out-of-order/nonrecomputable raw monotonic timing")
        previous = segment["end"]
    return evidence["timers"]["load"]["seconds"], [sum(v["timers"][k]["seconds"] for k in ("inference", "save_decode")) for v in evidence["videos"]]


def verify_measurement(path, approved_sha, method, root=ROOT, verify_local_request=True):
    path = Path(path).absolute()
    receipt_record = record(path, nonempty=True)
    require(receipt_record["sha256"] == approved_sha, "Approved performance receipt SHA mismatch")
    receipt = json.loads(path.read_bytes())
    require(path.name == "finished.json" and receipt["schema"] == SCHEMA and receipt["measured"] is True
            and receipt["method"] == method and receipt["allocated_gpus"] == 1
            and receipt["budget"] == BUDGET and receipt.get("success") is True
            and receipt.get("failure") is False and receipt.get("mechanism_PASS") is True
            and receipt.get("owned_tree_released") is True and receipt.get("budget_exceeded") is False
            and receipt.get("exit_code") == 0 and receipt.get("timed_out") is False
            and receipt.get("interrupted_signal") is None and receipt.get("error") is None,
            "Unbound/incomplete actual performance receipt; blocked")
    request = verify_request(path.parent / "request.json", receipt["request_sha256"], verify_local_request)
    require(Path(request["root"]).resolve() == Path(root).resolve() and request["method"] == method, "Performance root/method drift")
    for key in ("parameters", "call", "checkpoint", "model_content_sha256", "latest_smoke02", "prior_smoke_attempts"):
        require(receipt[key] == request[key], "Performance receipt/request binding mismatch: " + key)
    launch_path, worker_path = path.parent / "launch.json", path.parent / "worker.json"
    launch, worker = json.loads(launch_path.read_bytes()), json.loads(worker_path.read_bytes())
    require(launch["request_sha256"] == receipt["request_sha256"] and launch["budget"] == BUDGET
            and launch["gpu"] == receipt["gpu"] and launch["gpu_uuid"] == receipt["gpu_uuid"]
            and launch["supervisor_pid"] == receipt["supervisor_pid"]
            and worker == receipt["evidence"] and worker["success"] is True
            and worker["mechanism_PASS"] is True and worker["failure"] is False
            and worker["request_sha256"] == receipt["request_sha256"]
            and worker["parameters_unchanged"] is True and worker["parameters_before"]
            and worker["parameters_before"] == worker["parameters_after"], "Missing/mismatched real launch/worker/weights evidence")
    elapsed, start, end = (receipt[k] for k in ("elapsed_seconds", "allocation_monotonic_start", "allocation_monotonic_end"))
    require(all(finite(v) for v in (elapsed, start, end, receipt["elapsed_GPUh"], receipt["start"], receipt["end"]))
            and 0 < elapsed <= BUDGET["outer_cap_seconds"] and elapsed < 2700
            and math.isclose(elapsed, end - start, abs_tol=1e-9)
            and math.isclose(receipt["elapsed_GPUh"], elapsed / 3600, abs_tol=1e-9)
            and abs(receipt["end"] - receipt["start"] - elapsed) < 5
            and request["prior_smoke_seconds"] + elapsed < 3600, "Invalid actual/cumulative performance allocation cost")
    validate_timers(worker, start, end)
    sources = [record(path.parent / "request.json", nonempty=True), record(launch_path, nonempty=True), record(worker_path, nonempty=True)]
    for video in worker["videos"]:
        smoke.validate_stats(method, video["stats"]["method"], video["stats"]["observed"])
        actual = smoke.validate_video(path.parent / f"seed{video["seed"]}.mp4", request["runtime"]["tools"])
        require(actual == video["video"], "Performance video actual SHA/full-format mismatch")
        sources.append(actual["file"])
    require(receipt["measurement_sources"] == sources, "Actual measurement source SHA drift/manual summary")
    return receipt, request, sources, request["prior_smoke_seconds"]


def run_command(request, digest, gpu, supervisor_pid, state_dir, start):
    worker = [request["runtime"]["python"], str(Path(request["root"]) / "scripts/measure_teacher_throughput.py"),
              "--worker", "--request", str(Path(request["request_dir"]) / "request.json"),
              "--approved-request-sha256", digest, "--gpu", str(gpu), "--supervisor-pid", str(supervisor_pid),
              "--state-dir", str(state_dir), "--allocation-start", str(start)]
    return [request["runtime"]["tools"]["timeout"], "--signal=TERM", "--kill-after=30s", "2600s",
            request["runtime"]["python"], str(Path(request["root"]) / "scripts/gpu_hold.py"),
            "--state-dir", str(state_dir), "--action", "exec", "--gpus", str(gpu), "--"] + worker


def worker_scope(request, gpu, supervisor_pid, state_dir, start):
    launch = json.loads((Path(request["request_dir"]) / "launch.json").read_bytes())
    require(launch["gpu"] == gpu and launch["supervisor_pid"] == supervisor_pid and launch["budget"] == BUDGET
            and os.environ.get("CUDA_VISIBLE_DEVICES") == launch["gpu_uuid"]
            and os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_WORLD_SIZE", "1") == "1"
            and 0 <= time.monotonic() - start < BUDGET["internal_stop_seconds"], "Worker single-GPU/allocation scope mismatch")
    ancestors, pid = [], os.getppid()
    while pid > 1:
        ancestors.append((pid, smoke.proc_args(pid)))
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    require(any(pid == launch["parent_pid"] and b"--run" in args and b"--gpu-authorized" in args
                and any(os.path.basename(a) == b"measure_teacher_throughput.py" for a in args) for pid, args in ancestors),
            "Worker requires live explicitly authorized measurement parent")
    require(any(any(os.path.basename(a) == b"timeout" for a in args) and b"2600s" in args
                and b"--kill-after=30s" in args for _, args in ancestors), "Worker requires timeout2600/kill30 ancestry")
    require(any(any(os.path.basename(a) == b"gpu_hold.py" for a in args) and smoke.action_matches(args, "exec")
                and any(a == b"--gpus" and b == str(gpu).encode() for a, b in zip(args, args[1:])) for _, args in ancestors),
            "Worker requires single-GPU holderexec ancestry")
    smoke.supervisor_scope(state_dir, supervisor_pid, gpu, released=True)


def execute_worker(request, start, approved_sha):
    folder = Path(request["request_dir"])
    evidence = dict(success=False, failure=True, mechanism_PASS=False, quality_pass=False, sota_claim=False,
                    request_sha256=approved_sha, clock="time.monotonic", compile_accounting="included in inference",
                    load_count=0, timers={}, videos=[], parameters_before=None, parameters_after=None,
                    parameters_unchanged=False, peak_memory_bytes=None, error=None)
    torch, pipe, before = None, None, None
    def budget_check():
        require(time.monotonic() - start < BUDGET["internal_stop_seconds"], "Internal performance budget stop")
    def persist():
        evidence["elapsed_seconds"] = time.monotonic() - start
        checkpoint(folder / "progress.json", evidence)
    def timed(name, target, operation):
        segment = dict(start=time.monotonic())
        target[name] = segment
        try:
            return operation()
        finally:
            if torch is not None:
                torch.cuda.synchronize(0)
            segment["end"] = time.monotonic()
            segment["seconds"] = segment["end"] - segment["start"]
            persist()
    old = {}
    def interrupted(signum, _frame):
        raise RuntimeError(f"Worker interrupted by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, interrupted)
    try:
        persist()
        budget_check()
        import torch as torch_module
        torch = torch_module
        require(torch.cuda.device_count() == 1, "Exactly one visible CUDA device required")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        sys.path.insert(0, str(Path(request["root"]) / "scripts"))
        import gen_jobs
        require(gen_jobs.NEG == request["negative_prompt"], "Imported original NEG drift")
        def load():
            evidence["load_count"] += 1
            return smoke.build_pipe("NONE", request["models"], gen_jobs)
        pipe = timed("load", evidence["timers"], load)
        budget_check()
        before = smoke.parameter_hashes(pipe.dit, torch)
        evidence["parameters_before"] = before
        if request["method"] == "alg":
            from teacher_alg import teacher_alg as method
        else:
            from teacher_dymos import teacher_dymos as method
        for seed in request["seeds"]:
            budget_check()
            current = dict(seed=seed, stats=dict(observed=dict(model_forwards=0, scheduler_steps=0, progress_steps=0, finite_checks=0), method={}),
                           timers={}, video=None)
            evidence["videos"].append(current)
            def progress(timesteps):
                require(len(timesteps) == 50, "Exactly 50 scheduler timesteps required")
                for timestep in timesteps:
                    budget_check()
                    yield timestep
                    current["stats"]["observed"]["progress_steps"] += 1
                    persist()
                budget_check()
            kwargs = dict(request["call"], seed=seed, prompt=request["case"]["prompt"],
                          negative_prompt=request["negative_prompt"], input_image=gen_jobs.load_image(request["image"]["path"]),
                          progress_bar_cmd=progress)
            def infer():
                with smoke.observe(pipe, torch, current["stats"]["observed"], budget_check):
                    with method(pipe, call_kwargs=kwargs, checkpoint="NONE", **request["parameters"]) as stats:
                        current["stats"]["method"] = stats
                        return pipe(**kwargs)
            video = timed("inference", current["timers"], infer)
            require(len(video) == 121, "Generated video must contain full 121 frames")
            smoke.validate_stats(request["method"], current["stats"]["method"], current["stats"]["observed"])
            budget_check()
            def save():
                path = folder / f"seed{seed}.mp4"
                gen_jobs.save_video(video, str(path), fps=24, quality=5)
                os.chmod(path, 0o600)
                current["video"] = smoke.validate_video(path, request["runtime"]["tools"])
            timed("save_decode", current["timers"], save)
            del video
            budget_check()
        after = smoke.parameter_hashes(pipe.dit, torch)
        evidence["parameters_after"], evidence["parameters_unchanged"] = after, before == after
        require(before == after, "DiT parameters changed; baseline isolation failed")
        verify_request(folder / "request.json", approved_sha)
        budget_check()
        validate_timers(evidence, start, time.monotonic())
        evidence.update(success=True, failure=False, mechanism_PASS=True)
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if pipe is not None and before is not None and evidence["parameters_after"] is None:
                after = smoke.parameter_hashes(pipe.dit, torch)
                evidence["parameters_after"], evidence["parameters_unchanged"] = after, before == after
            if torch is not None:
                evidence["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
                del pipe
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
        except BaseException as exc:
            evidence.update(success=False, failure=True, mechanism_PASS=False, error=f"Final evidence/cleanup failure: {exc}")
        evidence["elapsed_seconds"] = time.monotonic() - start
        evidence["elapsed_GPUh"] = evidence["elapsed_seconds"] / 3600
        if evidence["elapsed_seconds"] >= BUDGET["internal_stop_seconds"]:
            evidence.update(success=False, failure=True, mechanism_PASS=False, error=evidence["error"] or "Internal budget exceeded")
        try:
            exclusive(folder / "worker.json", evidence)
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)
    return evidence


def run(path, approved_sha, gpu, supervisor_pid, state_dir, gpu_authorized=False):
    require(gpu_authorized is True, "Explicit --gpu-authorized required; CPU prepare is not authorization")
    request = verify_request(path, approved_sha)
    require(os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_WORLD_SIZE", "1") == "1", "No distributed/multi-GPU measurement")
    state_dir, folder = Path(state_dir), Path(request["request_dir"])
    uuid = smoke.supervisor_scope(state_dir, supervisor_pid, gpu)
    require(not any((folder / n).exists() for n in ("launch.json", "finished.json", "worker.json", "progress.json", "seed0.mp4", "seed1.mp4")),
            "Old performance run exists; never resume/overwrite")
    child, flag, started, wall_start = None, None, None, None
    error, interrupted_signal, timed_out, released = None, None, False, True
    old = {}
    def interrupted(signum, _frame):
        nonlocal interrupted_signal
        interrupted_signal = signum
    with (state_dir / f"teacher_perf.{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (state_dir / f"experiment.{gpu}.lock").open("a") as experiment:
            fcntl.flock(experiment, fcntl.LOCK_EX | fcntl.LOCK_NB)
            smoke.supervisor_scope(state_dir, supervisor_pid, gpu)
            exclusive(folder / "launch.json", dict(request_sha256=approved_sha, gpu=gpu, gpu_uuid=uuid, supervisor_pid=supervisor_pid,
                      parent_pid=os.getpid(), budget=BUDGET, complete=False))
            for signum in (signal.SIGTERM, signal.SIGINT):
                old[signum] = signal.signal(signum, interrupted)
            started, wall_start = time.monotonic(), time.time()
            try:
                flag = state_dir / f"pause.{gpu}"
                fd = os.open(flag, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
            except BaseException as exc:
                flag = None
                error = f"{type(exc).__name__}: {exc}"
        try:
            if error is None and interrupted_signal is None:
                command = run_command(request, approved_sha, gpu, supervisor_pid, state_dir, started)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, WORLD_SIZE="1", LOCAL_WORLD_SIZE="1", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True",
                           PYTHONPATH=str(Path(request["root"]) / "DiffSynth-Studio"), OMP_NUM_THREADS="1")
                with (folder / "worker.log").open("xb") as log:
                    os.chmod(folder / "worker.log", 0o600)
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
                    released = False
                    while child.poll() is None and interrupted_signal is None:
                        if time.monotonic() - started >= BUDGET["timeout_seconds"]:
                            timed_out = True
                            break
                        time.sleep(0.05)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if child is not None:
                    smoke.stop_group(child, started + BUDGET["outer_cap_seconds"])
                released = True
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            if flag is not None and released:
                try:
                    flag.unlink(missing_ok=True)
                except OSError as exc:
                    error, released = f"Pause release failed: {exc}", False
            worker, sources = {}, []
            try:
                candidate = folder / "worker.json"
                if not candidate.exists():
                    candidate = folder / "progress.json"
                if candidate.exists():
                    worker = json.loads(candidate.read_bytes())
                sources = [record(folder / "request.json", nonempty=True), record(folder / "launch.json", nonempty=True)]
                if candidate.exists():
                    sources.append(record(candidate, nonempty=True))
                for video in worker.get("videos", []):
                    if video.get("video"):
                        sources.append(record(folder / f"seed{video["seed"]}.mp4", nonempty=True))
            except (OSError, ValueError) as exc:
                error = f"Invalid actual measurement evidence: {exc}"
            ended = time.monotonic()
            elapsed = ended - started
            success = (child is not None and child.returncode == 0 and worker.get("success") is True
                       and worker.get("mechanism_PASS") is True and not timed_out and interrupted_signal is None
                       and error is None and released and elapsed <= BUDGET["outer_cap_seconds"]
                       and request["prior_smoke_seconds"] + elapsed < 3600)
            if success:
                try:
                    validate_timers(worker, started, ended)
                except (ValueError, KeyError, TypeError) as exc:
                    success, error = False, f"Invalid actual monotonic timers: {exc}"
            receipt = dict(schema=SCHEMA, measured=success, method=request["method"], parameters=request["parameters"], call=request["call"],
                           checkpoint=request["checkpoint"], model_content_sha256=request["model_content_sha256"],
                           latest_smoke02=request["latest_smoke02"], prior_smoke_attempts=request["prior_smoke_attempts"],
                           request_sha256=approved_sha, gpu=gpu, gpu_uuid=uuid, supervisor_pid=supervisor_pid, allocated_gpus=1,
                           start=wall_start, end=time.time(), allocation_monotonic_start=started, allocation_monotonic_end=ended,
                           elapsed_seconds=elapsed, elapsed_GPUh=elapsed / 3600, success=success, failure=not success, mechanism_PASS=success,
                           quality_pass=False, sota_claim=False, exit_code=child.returncode if child else None,
                           timed_out=timed_out or bool(child and child.returncode in (124, 137)), interrupted_signal=interrupted_signal,
                           error=error or worker.get("error"), evidence=worker, measurement_sources=sources,
                           owned_tree_released=released, budget=BUDGET,
                           budget_exceeded=elapsed > BUDGET["outer_cap_seconds"] or request["prior_smoke_seconds"] + elapsed >= 3600)
            try:
                exclusive(folder / "finished.json", receipt)
            finally:
                for signum, handler in old.items():
                    signal.signal(signum, handler)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=smoke.PARAMS)
    parser.add_argument("--request-dir", type=Path)
    parser.add_argument("--approved-smoke02-sha256")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--approved-request-sha256")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--state-dir", type=Path, default=smoke.LOCAL_ROOT / "wm_logs/gpu_hold")
    parser.add_argument("--allocation-start", type=float)
    args = parser.parse_args(argv)
    if not args.run and not args.worker:
        if not all((args.method, args.request_dir, args.approved_smoke02_sha256)):
            parser.error("CPU prepare requires --method, new --request-dir, --approved-smoke02-sha256")
        if args.gpu is not None or args.gpu_authorized or args.supervisor_pid is not None:
            parser.error("CPU prepare does not accept GPU launch arguments")
        path, digest = prepare(args.method, args.request_dir, args.approved_smoke02_sha256)
        print(json.dumps(dict(request=str(path), request_sha256=digest, GPU_used=False, measured=False)))
    elif args.run:
        if not args.request or not args.approved_request_sha256 or args.gpu is None or not args.supervisor_pid or not args.gpu_authorized:
            parser.error("Run requires approved request SHA, one GPU, live supervisor PID and --gpu-authorized")
        result = run(args.request, args.approved_request_sha256, args.gpu, args.supervisor_pid, args.state_dir, True)
        print(json.dumps(result, allow_nan=False))
        if not result["success"]:
            raise SystemExit(1)
    else:
        if not args.request or not args.approved_request_sha256 or args.gpu is None or not args.supervisor_pid or args.allocation_start is None:
            parser.error("Worker is only for the authorized managed outer launcher")
        request = verify_request(args.request, args.approved_request_sha256)
        worker_scope(request, args.gpu, args.supervisor_pid, args.state_dir, args.allocation_start)
        if not execute_worker(request, args.allocation_start, args.approved_request_sha256)["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
