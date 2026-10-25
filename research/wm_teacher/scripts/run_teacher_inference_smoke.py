#!/usr/bin/env python3
"""Private one-video mechanism smoke, not a quality/SOTA evaluation.

Default action is CPU prepare (no torch/generator imports). Each method requires
its own new request directory and separately approved request SHA before run.
Run is one-shot, single-GPU and bounded including load/compile/wait/failure.
"""
import argparse
import ast
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOT = Path("/dockerdata")
FIRST_CASE_SHA256 = "eba78d79578d6929c92895a74d5383d2b4c2ceb41dabb0d5c4ac2d0be790fc50"
ASSETS = tuple(f"Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-{i:05d}-of-00003.safetensors" for i in range(1, 4)) + (
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
) + tuple("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/" + n for n in (
    "special_tokens_map.json", "spiece.model", "tokenizer_config.json", "tokenizer.json"))
SPEC = dict(seed=0, sigma_shift=5.0, cfg_scale=5.0, num_inference_steps=50,
            num_frames=121, width=1280, height=704, tiled=True)
BUDGET = dict(gpus=1, gpu_hours=1.0, timeout_seconds=3450, kill_seconds=30,
              internal_stop_seconds=3350, outer_cap_seconds=3490)
PARAMS = {"alg": {"factor": 0.4}, "dymos": {"gamma": 0.6, "phi": 1}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(path, *, nonempty=False):
    path = Path(path).absolute()
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "File changed while hashing")
    require(not nonempty or after.st_size > 0, "Empty required input: " + str(path))
    return dict(path=str(path), bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=h.hexdigest())


def check_record(expected, *, nonempty=False):
    actual = record(expected["path"], nonempty=nonempty)
    require(actual == expected, "Input/source/model hash or metadata drift: " + expected["path"])
    return actual


def exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())


def checkpoint(path, value):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as handle:
        os.chmod(tmp, 0o600)
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def generator_negative(root):
    tree = ast.parse((root / "scripts/gen_jobs.py").read_text())
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "NEG" for t in n.targets)]
    require(len(assignments) == 1, "Expected original generator NEG")
    value = ast.literal_eval(assignments[0].value)
    require(isinstance(value, str) and value, "Invalid generator NEG")
    return value


def source_paths(root):
    paths = {root / "scripts" / n for n in (
        "run_teacher_inference_smoke.py", "gen_jobs.py", "teacher_alg.py", "teacher_dymos.py", "gpu_hold.py")}
    imports = set((root / "DiffSynth-Studio/diffsynth").rglob("*.py"))
    require(imports, "Missing frozen DiffSynth local sources")
    return sorted(paths | imports | {root / "docs/51_免训练Teacher多维策略有限实验预注册.md"})


def runtime_records():
    tools = {n: shutil.which(n) for n in ("ffmpeg", "ffprobe", "timeout")}
    require(all(tools.values()), "ffmpeg/ffprobe/GNU timeout required")
    modules = {
        "torch": "torch", "torchvision": "torchvision", "transformers": "transformers",
        "safetensors": "safetensors", "pillow": "PIL", "imageio": "imageio", "numpy": "numpy",
        "triton": "triton", "einops": "einops", "accelerate": "accelerate", "huggingface-hub": "huggingface_hub",
        "modelscope": "modelscope", "sentencepiece": "sentencepiece", "tokenizers": "tokenizers", "tqdm": "tqdm",
        "imageio-ffmpeg": "imageio_ffmpeg", "flash-attn": "flash_attn", "sageattention": "sageattention",
        "xformers": "xformers", "opencv-python": "cv2", "scipy": "scipy", "ftfy": "ftfy", "regex": "regex",
    }
    binary_patterns = (
        "torch/_C*.so", "torch/lib/libtorch_cuda.so", "torch/lib/libtorch_cpu.so",
        "triton/_C/libtriton.so", "safetensors/_safetensors_rust*.so",
        "numpy/_core/_multiarray_umath*.so", "numpy/core/_multiarray_umath*.so",
        "PIL/_imaging*.so", "tokenizers/tokenizers*.so", "sentencepiece/_sentencepiece*.so",
        "flash_attn_2_cuda*.so", "flash_attn_3_cuda*.so", "xformers/_C*.so",
        "imageio_ffmpeg/binaries/ffmpeg-*",
    )
    packages, paths = {}, {Path(sys.executable).resolve()} | {Path(p).resolve() for p in tools.values()}
    package_roots = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name not in modules and not name.startswith("nvidia-"):
            continue
        packages[name] = dist.version
        metadata = next((item for item in dist.files or ()
                         if str(item).endswith((".dist-info/METADATA", ".egg-info/PKG-INFO"))), None)
        require(metadata is not None, "Missing package metadata: " + name)
        metadata_path = Path(dist.locate_file(metadata)).resolve()
        require(metadata_path.is_file(), "Missing package METADATA: " + name)
        paths.add(metadata_path)
        if name in modules:
            entry = Path(dist.locate_file(modules[name] + "/__init__.py")).resolve()
            require(entry.is_file(), "Missing package entrypoint: " + name)
            paths.add(entry)
            package_roots.add(Path(dist.locate_file("")).resolve())
    require({"torch", "transformers", "safetensors", "pillow", "imageio", "numpy"} <= set(packages),
            "Missing generation runtime dependencies")
    for root in package_roots:
        for pattern in binary_patterns:
            paths.update(p.resolve() for p in root.glob(pattern) if p.is_file())
    return dict(python=str(Path(sys.executable).resolve()), tools=tools, packages=packages,
                files=[record(p) for p in sorted(paths)],
                hash_scope="package entrypoints, METADATA, Python/tool executables and selected inference binaries",
                caveat="Native/shared libraries and transitive package files are not exhaustively hashed; "
                       "runtime records do not attest the entire installed environment.")

def models_from_preflight(preflight):
    require(preflight["schema"] == "e3_subset_node_preflight_v1"
            and preflight["GPU_used"] is False
            and preflight["hostname"] == os.uname().nodename, "Wrong verified local node preflight")
    models = preflight["models"]
    require([m["asset"] for m in models] == list(ASSETS), "Exactly nine verified local assets required")
    for model in models:
        path = LOCAL_ROOT / "models" / model["asset"]
        require(str(path) == model["path"] and path.resolve().is_relative_to(LOCAL_ROOT.resolve()),
                "Asset must be local, not shared or remote")
        require(model["bytes"] > 0, "Empty required model asset")
        check_record({k: model[k] for k in ("path", "bytes", "mtime_ns", "sha256")}, nonempty=True)
    expected_shards = {LOCAL_ROOT / "models" / a for a in ASSETS[:3]}
    require(set((LOCAL_ROOT / "models/Wan-AI/Wan2.2-TI2V-5B").glob("diffusion_pytorch_model*.safetensors"))
            == expected_shards, "Unexpected DiT shard set")
    content_sha = sha(encoded([{k: m[k] for k in ("asset", "bytes", "sha256")} for m in models]))
    require(content_sha == preflight["model_content_sha256"], "Preflight model content digest mismatch")
    return models


def first_case(root):
    path = root / "eval/e1_devset_20261021/cases.jsonl"
    cases_record = record(path, nonempty=True)
    cases = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    require(len(cases) == 94 and len({c["id"] for c in cases}) == 94, "Expected frozen 94-case E1 development set")
    case = cases[0]
    require(sha(case["id"].encode()) == FIRST_CASE_SHA256, "Only the fixed first safe E1 case is supported")
    require(isinstance(case["prompt"], str) and case["prompt"], "Empty/invalid original prompt")
    require(sha(case["prompt"].encode()) == case["prompt_sha256"], "Original prompt hash mismatch")
    image = record(case["image"], nonempty=True)
    require(image["sha256"] == case["image_sha256"], "Original image hash mismatch")
    return case, image, cases_record


def prepare(method, request_dir, node_preflight, approved_node_preflight_sha256, root=ROOT):
    require(method in PARAMS, "Choose one independent method")
    root, request_dir = Path(root).resolve(), Path(request_dir).absolute()
    require(not request_dir.exists(), "Request directory already exists; never overwrite")
    pf_record = record(node_preflight, nonempty=True)
    require(pf_record["sha256"] == approved_node_preflight_sha256, "Approved node preflight SHA mismatch")
    pf = json.loads(Path(node_preflight).read_bytes())
    models = models_from_preflight(pf)
    case, image, cases_record = first_case(root)
    negative = generator_negative(root)
    runtime = runtime_records()
    value = dict(schema="teacher_inference_smoke_v1", private=True, root=str(root), method=method,
                 parameters=PARAMS[method], case=case, image=image, cases_file=cases_record,
                 negative_prompt=negative, negative_sha256=sha(negative.encode()), call=SPEC, fps=24,
                 checkpoint="NONE", active_steps=10, video_count=1, budget=BUDGET,
                 node_preflight=pf_record, model_content_sha256=pf["model_content_sha256"], models=models,
                 source_files=[record(p) for p in source_paths(root)], runtime=runtime,
                 request_dir=str(request_dir), created_at=time.time(), hostname=os.uname().nodename,
                 GPU_used=False, quality_pass=False, sota_claim=False)
    request_dir.mkdir(mode=0o700, parents=False)
    path = request_dir / "request.json"
    exclusive(path, value)
    return path, sha(path.read_bytes())


def verify_request(path, approved_sha):
    path = Path(path).absolute()
    data = path.read_bytes()
    require(isinstance(approved_sha, str) and len(approved_sha) == 64 and sha(data) == approved_sha,
            "Approved request SHA256 mismatch")
    request = json.loads(data)
    require(request["schema"] == "teacher_inference_smoke_v1" and request["private"] is True,
            "Unexpected smoke request")
    require(request["method"] in PARAMS and encoded(request["parameters"]) == encoded(PARAMS[request["method"]])
            and encoded(request["call"]) == encoded(SPEC) and request["fps"] == 24 and request["checkpoint"] == "NONE"
            and request["active_steps"] == 10 and request["video_count"] == 1
            and encoded(request["budget"]) == encoded(BUDGET), "Unapproved method/configuration/budget")
    require(request["hostname"] == os.uname().nodename and request["request_dir"] == str(path.parent)
            and path.name == "request.json" and path.parent.stat().st_mode & 0o077 == 0
            and path.stat().st_mode & 0o077 == 0, "Request must be private and node-bound")
    root = Path(request["root"])
    sources = request["source_files"]
    require({r["path"] for r in sources} == {str(p) for p in source_paths(root)}, "Source/import set drift")
    for r in sources:
        check_record(r)
    require(runtime_records() == request["runtime"], "Runtime version/entrypoint/METADATA/binary drift")
    require(request["runtime"]["python"] == str(Path(sys.executable).resolve()), "Runtime Python drift")
    check_record(request["node_preflight"], nonempty=True)
    pf = json.loads(Path(request["node_preflight"]["path"]).read_bytes())
    require(models_from_preflight(pf) == request["models"]
            and pf["model_content_sha256"] == request["model_content_sha256"], "Model/preflight drift")
    check_record(request["cases_file"], nonempty=True)
    case, image, _ = first_case(root)
    require(case == request["case"] and image == request["image"], "Original case/image drift")
    negative = generator_negative(root)
    require(negative == request["negative_prompt"] and sha(negative.encode()) == request["negative_sha256"], "NEG drift")
    return request


def proc_args(pid):
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except OSError:
        return []


def action_matches(args, action):
    return b"--action=" + action.encode() in args or any(a == b"--action" and b == action.encode() for a, b in zip(args, args[1:]))


def supervisor_scope(state_dir, supervisor_pid, gpu, released=False):
    require(type(gpu) is int and 0 <= gpu <= 7, "Exactly one physical GPU index 0..7 required")
    require(type(supervisor_pid) is int and supervisor_pid > 1, "Explicit live supervisor PID required")
    args = proc_args(supervisor_pid)
    require(any(os.path.basename(a) == b"gpu_hold.py" for a in args) and action_matches(args, "run"), "Supervisor PID is not a live gpu_hold run")
    state = json.loads((Path(state_dir) / "status.json").read_bytes())
    require(state["pid"] == supervisor_pid and state["running"] is True and state["query_ok"] is True
            and 0 <= time.time() - state["updated_at"] < 10, "Stale/dead supervisor state")
    selected = state["gpus"].get(str(gpu))
    require(selected and not selected["external_pids"], "Selected GPU has an existing workload; do not touch it")
    if released:
        require(selected["paused"] and selected["holder_pid"] is None, "Holder not released before CUDA initialization")
    else:
        require(not selected["paused"] and not (Path(state_dir) / f"pause.{gpu}").exists(), "Selected GPU is already paused/owned")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    pids = {int(p) for u, p in csv.reader(io.StringIO(result.stdout)) if u.strip() == selected["gpu_uuid"]}
    allowed = set() if released else ({selected["holder_pid"]} if selected["holder_pid"] else set())
    require(pids <= allowed, "Live GPU process query shows an unrelated workload")
    if allowed:
        holder_args = proc_args(selected["holder_pid"])
        require(any(os.path.basename(a) == b"gpu_hold.py" for a in holder_args) and action_matches(holder_args, "worker"), "Unknown GPU occupant")
    return selected["gpu_uuid"]


def run_command(request, request_sha, gpu, supervisor_pid, state_dir, start):
    worker = [request["runtime"]["python"], str(Path(request["root"]) / "scripts/run_teacher_inference_smoke.py"),
              "--worker", "--request", str(Path(request["request_dir"]) / "request.json"),
              "--approved-request-sha256", request_sha, "--gpu", str(gpu), "--supervisor-pid", str(supervisor_pid),
              "--state-dir", str(state_dir), "--allocation-start", str(start)]
    return [request["runtime"]["tools"]["timeout"], "--signal=TERM", "--kill-after=30s", "3450s",
            request["runtime"]["python"], str(Path(request["root"]) / "scripts/gpu_hold.py"),
            "--state-dir", str(state_dir), "--action", "exec", "--gpus", str(gpu), "--"] + worker


def group_alive(pgid):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop_group(child, deadline):
    for signum, allowance in ((signal.SIGTERM, 30), (signal.SIGKILL, 1)):
        if not group_alive(child.pid):
            break
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        until = min(time.monotonic() + allowance, deadline)
        while group_alive(child.pid) and time.monotonic() < until:
            time.sleep(0.02)
    child.wait(timeout=max(0.01, deadline - time.monotonic()))
    require(not group_alive(child.pid), "Owned tree not released; keep holder paused")


def run(path, approved_sha, gpu, supervisor_pid, state_dir, gpu_authorized=False):
    require(gpu_authorized is True, "Explicit --gpu-authorized required; CPU prepare is not GPU authorization")
    request = verify_request(path, approved_sha)
    require(os.environ.get("WORLD_SIZE", "1") == "1" and os.environ.get("LOCAL_WORLD_SIZE", "1") == "1", "No distributed/multi-GPU smoke")
    state_dir = Path(state_dir)
    uuid = supervisor_scope(state_dir, supervisor_pid, gpu)
    folder = Path(request["request_dir"])
    require(not any((folder / n).exists() for n in ("launch.json", "finished.json", "worker.json", "video.mp4", "progress.json")), "Old run exists; never resume/overwrite")
    child, flag, started, wall_start = None, None, None, None
    error, interrupted, timed_out, released = None, None, False, True
    old = {}
    def interrupted_handler(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    with (state_dir / f"smoke.{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (state_dir / f"experiment.{gpu}.lock").open("a") as experiment:
            fcntl.flock(experiment, fcntl.LOCK_EX | fcntl.LOCK_NB)
            supervisor_scope(state_dir, supervisor_pid, gpu)
            exclusive(folder / "launch.json", dict(request_sha256=approved_sha, gpu=gpu, gpu_uuid=uuid,
                      supervisor_pid=supervisor_pid, parent_pid=os.getpid(), budget=BUDGET, complete=False))
            for signum in (signal.SIGTERM, signal.SIGINT):
                old[signum] = signal.signal(signum, interrupted_handler)
            started, wall_start = time.monotonic(), time.time()
            try:
                flag = state_dir / f"pause.{gpu}"
                fd = os.open(flag, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
            except BaseException as exc:
                flag = None
                error = f"{type(exc).__name__}: {exc}"
        try:
            if error is None and interrupted is None:
                command = run_command(request, approved_sha, gpu, supervisor_pid, state_dir, started)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, WORLD_SIZE="1", LOCAL_WORLD_SIZE="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True",
                           PYTHONPATH=str(Path(request["root"]) / "DiffSynth-Studio"), OMP_NUM_THREADS="1")
                with (folder / "worker.log").open("xb") as log:
                    os.chmod(folder / "worker.log", 0o600)
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, shell=False, start_new_session=True)
                    released = False
                    while child.poll() is None and interrupted is None:
                        if time.monotonic() - started >= BUDGET["timeout_seconds"]:
                            timed_out = True
                            break
                        time.sleep(0.05)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                if child is not None:
                    stop_group(child, started + BUDGET["outer_cap_seconds"])
                released = True
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            if flag is not None and released:
                try:
                    flag.unlink(missing_ok=True)
                except OSError as exc:
                    error = f"Pause release failed: {exc}"
                    released = False
            elapsed = time.monotonic() - started
            worker = {}
            try:
                candidate = folder / "worker.json"
                if not candidate.exists():
                    candidate = folder / "progress.json"
                if candidate.exists():
                    worker = json.loads(candidate.read_bytes())
            except (OSError, ValueError) as exc:
                error = f"Invalid worker evidence: {exc}"
            success = (child is not None and child.returncode == 0 and worker.get("success") is True
                       and worker.get("mechanism_PASS") is True and not timed_out and interrupted is None
                       and error is None and released and elapsed <= BUDGET["outer_cap_seconds"])
            receipt = dict(schema="teacher_smoke_receipt_v1", method=request["method"], request_sha256=approved_sha,
                           gpu=gpu, gpu_uuid=uuid, allocated_gpus=1, start=wall_start, end=time.time(), elapsed_seconds=elapsed,
                           elapsed_GPUh=elapsed / 3600, success=success, failure=not success, mechanism_PASS=success,
                           quality_pass=False, sota_claim=False, exit_code=child.returncode if child else None,
                           timed_out=timed_out or bool(child and child.returncode in (124, 137)), interrupted_signal=interrupted,
                           error=error or worker.get("error"), peak_memory_bytes=worker.get("peak_memory_bytes"),
                           stats=worker.get("stats", {}), evidence=worker, owned_tree_released=released,
                           budget=BUDGET, budget_exceeded=elapsed > BUDGET["outer_cap_seconds"])
            try:
                exclusive(folder / "finished.json", receipt)
            finally:
                for signum, handler in old.items():
                    signal.signal(signum, handler)
    return receipt


def build_pipe(checkpoint_name, models, generator):
    require(checkpoint_name == "NONE", "Smoke must build a new baseline pipe with NONE")
    original = generator.ModelConfig
    paths = {m["asset"]: m["path"] for m in models}
    mapping = {
        ("Wan-AI/Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth"): paths[ASSETS[3]],
        ("Wan-AI/Wan2.2-TI2V-5B", "diffusion_pytorch_model*.safetensors"): [paths[a] for a in ASSETS[:3]],
        ("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"): paths[ASSETS[4]],
        ("Wan-AI/Wan2.1-T2V-1.3B", "google/umt5-xxl/"): str(Path(paths[ASSETS[5]]).parent),
    }
    def local_config(*, model_id, origin_file_pattern):
        require((model_id, origin_file_pattern) in mapping, "Unapproved model asset lookup")
        return original(path=mapping[model_id, origin_file_pattern], skip_download=True)
    generator.ModelConfig = local_config
    try:
        return generator.build_pipe("NONE")
    finally:
        generator.ModelConfig = original


def parameter_hashes(dit, torch):
    result = {}
    for name, parameter in dit.named_parameters():
        require(name not in result, "Duplicate DiT parameter name")
        raw = parameter.detach().contiguous().view(torch.uint8).reshape(-1)
        h = hashlib.sha256()
        for chunk in raw.split(8 * 1024 * 1024):
            h.update(chunk.cpu().numpy().tobytes())
        result[name] = dict(shape=list(parameter.shape), dtype=str(parameter.dtype), sha256=h.hexdigest())
    require(result, "No authenticated DiT parameters")
    return result


@contextlib.contextmanager
def observe(pipe, torch, stats, budget_check):
    original_model, original_step = pipe.model_fn, pipe.scheduler.step
    model_owned, step_owned = "model_fn" in vars(pipe), "step" in vars(pipe.scheduler)
    def finite(value, name):
        require(isinstance(value, torch.Tensor) and value.is_floating_point()
                and torch.isfinite(value).all().item(), "Nonfinite/invalid " + name)
        stats["finite_checks"] += 1
    def model(*args, **kwargs):
        budget_check()
        finite(kwargs.get("latents"), "model latents")
        finite(kwargs.get("first_frame_latents"), "reference latent")
        finite(kwargs.get("context"), "text context")
        prediction = original_model(*args, **kwargs)
        finite(prediction, "model prediction")
        stats["model_forwards"] += 1
        return prediction
    def step(prediction, timestep, sample, *args, **kwargs):
        budget_check()
        finite(prediction, "CFG prediction")
        finite(sample, "scheduler input latent")
        result = original_step(prediction, timestep, sample, *args, **kwargs)
        finite(result, "scheduler output latent")
        stats["scheduler_steps"] += 1
        require(stats["scheduler_steps"] <= 50, "Too many scheduler steps")
        return result
    pipe.model_fn, pipe.scheduler.step = model, step
    try:
        yield
    finally:
        if model_owned:
            pipe.model_fn = original_model
        else:
            vars(pipe).pop("model_fn", None)
        if step_owned:
            pipe.scheduler.step = original_step
        else:
            vars(pipe.scheduler).pop("step", None)


def validate_stats(method, method_stats, observed):
    require(method_stats.get("completed_steps") == 50 and method_stats.get("active_steps") == 10
            and method_stats.get("bypassed") is False and observed["scheduler_steps"] == 50
            and observed["progress_steps"] == 50 and observed["finite_checks"] > 0, "Incomplete 50-step/10-active mechanism evidence")
    forwards = 110 if method == "alg" else 100
    require(method_stats.get("model_forwards") == forwards and observed["model_forwards"] == forwards, "Model forward count mismatch")
    if method == "alg":
        require(isinstance(method_stats.get("first_frame_sha256"), str) and len(method_stats["first_frame_sha256"]) == 64, "Missing original ALG latent hash")
    else:
        require(method_stats.get("biased_attention_calls") == 300 and method_stats.get("original_attention_calls") == 2700
                and method_stats.get("reference_tokens") == 880 and method_stats.get("sequence_tokens") == 27280,
                "DyMoS fused attention/geometry statistics mismatch")


def validate_video(path, tools):
    subprocess.run([tools["ffmpeg"], "-nostdin", "-v", "error", "-xerror", "-i", str(path),
                    "-map", "0:v:0", "-f", "null", "-"], check=True, capture_output=True, timeout=120)
    result = subprocess.run([tools["ffprobe"], "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True, timeout=120)
    streams = json.loads(result.stdout)["streams"]
    require(len(streams) == 1 and streams[0]["codec_type"] == "video", "Unexpected MP4 streams")
    stream = streams[0]
    require(stream["width"] == 1280 and stream["height"] == 704 and int(stream["nb_read_frames"]) == 121,
            "Actual decoded video must be 121f/1280x704")
    for key in ("avg_frame_rate", "r_frame_rate"):
        numerator, denominator = map(int, stream[key].split("/"))
        require(denominator > 0 and numerator == 24 * denominator, "Actual MP4 fps must be 24")
    return dict(file=record(path, nonempty=True), decoded_frames=121, width=1280, height=704, fps=24, full_decode_xerror=True)


def worker_scope(request, gpu, supervisor_pid, state_dir, start):
    launch = json.loads((Path(request["request_dir"]) / "launch.json").read_bytes())
    require(launch["gpu"] == gpu and launch["supervisor_pid"] == supervisor_pid
            and launch["budget"] == BUDGET and os.environ.get("CUDA_VISIBLE_DEVICES") == launch["gpu_uuid"], "Worker GPU/launch scope mismatch")
    require(0 <= time.monotonic() - start < BUDGET["internal_stop_seconds"], "Worker allocation clock out of budget")
    ancestors, pid = [], os.getppid()
    while pid > 1:
        args = proc_args(pid)
        ancestors.append((pid, args))
        try:
            fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
    require(any(pid == launch["parent_pid"] and b"--run" in args for pid, args in ancestors), "Worker requires live authorized outer supervisor")
    require(any(any(os.path.basename(a) == b"timeout" for a in args) and b"3450s" in args and b"--kill-after=30s" in args for _, args in ancestors), "Worker requires managed timeout3450/kill30 ancestry")
    require(any(any(os.path.basename(a) == b"gpu_hold.py" for a in args) and action_matches(args, "exec")
                and any(a == b"--gpus" and b == str(gpu).encode() for a, b in zip(args, args[1:])) for _, args in ancestors), "Worker requires single-GPU holderexec ancestry")
    supervisor_scope(state_dir, supervisor_pid, gpu, released=True)


def execute_worker(request, start, approved_sha):
    folder = Path(request["request_dir"])
    evidence = dict(success=False, failure=True, mechanism_PASS=False, quality_pass=False, sota_claim=False,
                    stats=dict(observed=dict(model_forwards=0, scheduler_steps=0, progress_steps=0, finite_checks=0), method={}),
                    peak_memory_bytes=None, parameters_before=None, parameters_after=None, parameters_unchanged=False,
                    video=None, error=None)
    torch, pipe, before = None, None, None
    def budget_check():
        require(time.monotonic() - start < BUDGET["internal_stop_seconds"], "Internal 3350s budget stop")
    def persist():
        evidence["elapsed_seconds"] = time.monotonic() - start
        evidence["elapsed_GPUh"] = evidence["elapsed_seconds"] / 3600
        if torch is not None:
            evidence["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
        checkpoint(folder / "progress.json", evidence)
    def progress(timesteps):
        require(len(timesteps) == 50, "Exactly 50 scheduler timesteps required")
        for timestep in timesteps:
            budget_check()
            yield timestep
            evidence["stats"]["observed"]["progress_steps"] += 1
            persist()
        budget_check()
    old = {}
    def interrupted_handler(signum, _frame):
        raise RuntimeError(f"Worker interrupted by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, interrupted_handler)
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
        pipe = build_pipe("NONE", request["models"], gen_jobs)
        budget_check()
        before = parameter_hashes(pipe.dit, torch)
        evidence["parameters_before"] = before
        kwargs = dict(request["call"], prompt=request["case"]["prompt"], negative_prompt=request["negative_prompt"],
                      input_image=gen_jobs.load_image(request["image"]["path"]), progress_bar_cmd=progress)
        if request["method"] == "alg":
            from teacher_alg import teacher_alg
            method = teacher_alg
        else:
            from teacher_dymos import teacher_dymos
            method = teacher_dymos
        with observe(pipe, torch, evidence["stats"]["observed"], budget_check):
            with method(pipe, call_kwargs=kwargs, checkpoint="NONE", **request["parameters"]) as stats:
                evidence["stats"]["method"] = stats
                video = pipe(**kwargs)
        budget_check()
        require(len(video) == 121, "Generated frame count differs from full specification")
        after = parameter_hashes(pipe.dit, torch)
        evidence["parameters_after"], evidence["parameters_unchanged"] = after, before == after
        require(before == after, "DiT parameters changed; baseline isolation failed")
        validate_stats(request["method"], evidence["stats"]["method"], evidence["stats"]["observed"])
        gen_jobs.save_video(video, str(folder / "video.mp4"), fps=24, quality=5)
        os.chmod(folder / "video.mp4", 0o600)
        evidence["video"] = validate_video(folder / "video.mp4", request["runtime"]["tools"])
        budget_check()
        verify_request(folder / "request.json", approved_sha)
        budget_check()
        evidence.update(success=True, failure=False, mechanism_PASS=True)
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if pipe is not None and before is not None and evidence["parameters_after"] is None:
                after = parameter_hashes(pipe.dit, torch)
                evidence["parameters_after"], evidence["parameters_unchanged"] = after, before == after
            if torch is not None:
                evidence["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(0))
        except BaseException as exc:
            evidence.update(success=False, failure=True, mechanism_PASS=False, error=f"Final evidence failure: {exc}")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=PARAMS)
    parser.add_argument("--request-dir", type=Path)
    parser.add_argument("--node-preflight", type=Path)
    parser.add_argument("--approved-node-preflight-sha256")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--approved-request-sha256")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--state-dir", type=Path, default=LOCAL_ROOT / "wm_logs/gpu_hold")
    parser.add_argument("--allocation-start", type=float)
    args = parser.parse_args(argv)
    if not args.run and not args.worker:
        if not all((args.method, args.request_dir, args.node_preflight, args.approved_node_preflight_sha256)):
            parser.error("CPU prepare requires --method, new --request-dir, --node-preflight and its approved SHA256")
        if args.gpu is not None or args.gpu_authorized or args.supervisor_pid is not None:
            parser.error("CPU prepare does not accept GPU launch arguments")
        path, digest = prepare(args.method, args.request_dir, args.node_preflight, args.approved_node_preflight_sha256)
        print(json.dumps(dict(request=str(path), request_sha256=digest, GPU_used=False, success=False, mechanism_PASS=False)))
    elif args.run:
        if not args.request or not args.approved_request_sha256 or args.gpu is None or not args.supervisor_pid:
            parser.error("Run requires request/approved SHA256/one GPU/live supervisor PID/explicit GPU authorization")
        result = run(args.request, args.approved_request_sha256, args.gpu, args.supervisor_pid, args.state_dir, args.gpu_authorized)
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
