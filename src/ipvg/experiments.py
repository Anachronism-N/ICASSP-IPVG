"""Finite, auditable subprocess experiments, with a real-video synthetic backend."""
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .common import fingerprint, load_json, positive, resolve_path, safe_name, sha256_file, utc_now, write_json
from .data import read_manifest, split_report


def _git_head(root):
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _source_hash(root):
    files = {}
    for folder in ("src", "scripts"):
        for path in sorted((Path(root) / folder).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".sh", ".sbatch"):
                files[str(path.relative_to(root)).replace("\\", "/")] = sha256_file(path)
    return fingerprint(files)


def _config(config_path, root):
    path = resolve_path(config_path, root)
    config = load_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("experiment schema_version must be 1")
    safe_name(config.get("run_id"), "run_id")
    from .evaluate import safe_sample_id
    safe_sample_id(config["run_id"])
    if config.get("task", "inference") not in ("inference", "training"):
        raise ValueError("task must be inference or training")
    if config.get("purpose", "development") not in ("development", "final_generation"):
        raise ValueError("purpose must be development or final_generation")
    if config.get("purpose") == "final_generation" and config.get("task", "inference") != "inference":
        raise ValueError("final_generation is inference only; it cannot bypass training split isolation")
    if not isinstance(config.get("hypothesis"), str) or not config["hypothesis"].strip():
        raise ValueError("A nonempty hypothesis is required")
    if not isinstance(config.get("parameters", {}), dict):
        raise ValueError("parameters must be an object")
    resource = config.get("resources", {})
    for key in ("gpus", "timeout_s", "max_gpu_hours"):
        if key not in resource:
            raise ValueError("resources.%s is required" % key)
        positive(resource[key], key, allow_zero=key != "timeout_s", integer=key == "gpus")
    worst = resource["gpus"] * resource["timeout_s"] / 3600
    if worst > resource["max_gpu_hours"] + 1e-9:
        raise ValueError("timeout_s * gpus exceeds declared max_gpu_hours")
    backend = config.get("backend", {})
    if backend.get("kind") not in ("synthetic", "command"):
        raise ValueError("backend.kind must be synthetic or command")
    if backend["kind"] == "command":
        command = backend.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(x, str) or not x for x in command):
            raise ValueError("backend.command must be a nonempty argv list")
    if backend["kind"] == "synthetic" and (resource["gpus"] != 0 or config.get("task", "inference") != "inference"):
        raise ValueError("synthetic backend is CPU inference only")
    artifacts = config.get("expected_artifacts", [])
    if not isinstance(artifacts, list) or any(
        not isinstance(p, str) or not p or p.startswith("/") or Path(p).is_absolute() or "\\" in p or ":" in p
        or "\x00" in p or ".." in p.split("/") for p in artifacts
    ):
        raise ValueError("expected_artifacts must be relative output patterns")
    if config.get("task") == "training" and not artifacts:
        raise ValueError("training requires expected_artifacts, e.g. ['*.safetensors']")
    config["task"] = config.get("task", "inference")
    return config


def plan_experiment(config_path, *, root="."):
    root = Path(root).resolve()
    config = _config(config_path, root)
    manifest_path = resolve_path(config["manifest"], root)
    samples = read_manifest(manifest_path)
    if not samples:
        raise ValueError("manifest is empty")
    from .evaluate import safe_sample_id
    ids = [safe_sample_id(s["sample_id"]) for s in samples]
    if len({sample_id.casefold() for sample_id in ids}) != len(ids):
        raise ValueError("duplicate sample_id")
    if not split_report(samples)["leakage"]["safe"]:
        raise ValueError("manifest leaks identities or source videos across splits; repair the split before running")
    requested_split = config.get("split")
    if requested_split:
        if requested_split not in ("train", "dev"):
            raise ValueError("experiment split may be train or dev; holdout requires a separate evaluation workflow")
        samples = [sample for sample in samples if sample.get("split") == requested_split]
        if not samples:
            raise ValueError("no samples in configured split")
    if config.get("purpose", "development") != "final_generation" and any(s.get("split") in ("holdout", "test") for s in samples):
        raise ValueError("development experiments cannot use holdout/test records")
    if config["task"] == "training" and any(s.get("split") not in (None, "train") for s in samples):
        raise ValueError("training cannot consume dev or holdout records; select the train split")
    run_dir = resolve_path(config.get("runs_dir", "runs"), root) / config["run_id"]
    source_hash = _source_hash(root)
    digest = fingerprint({"config": config, "manifest_sha256": sha256_file(manifest_path), "source_sha256": source_hash})
    return {"schema_version": 1, "config": config, "fingerprint": digest, "sample_count": len(samples),
            "manifest_sha256": sha256_file(manifest_path), "manifest_path": str(manifest_path),
            "source_sha256": source_hash,
            "run_dir": str(run_dir), "outputs_dir": str(run_dir / "outputs"),
            "max_backend_gpu_hours": config["resources"]["gpus"] * config["resources"]["timeout_s"] / 3600,
            "executes_commands": False, "synthetic": config["backend"]["kind"] == "synthetic"}


def run_process(command, *, cwd, timeout_s, stdout_path, stderr_path):
    """Terminate this child process tree on timeout; never invoke a shell."""
    if timeout_s <= 0:
        raise TimeoutError("Experiment backend time budget exhausted")
    with Path(stdout_path).open("ab") as stdout, Path(stderr_path).open("ab") as stderr:
        kwargs = {"cwd": str(cwd), "stdout": stdout, "stderr": stderr,
                  "stdin": subprocess.DEVNULL, "shell": False}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        try:
            code = process.wait(timeout=timeout_s)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait(timeout=5)
            if os.name == "posix":
                # The leader may exit on SIGTERM while a worker ignores it.
                # Always stop remaining process-group members before returning.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise TimeoutError("Backend timed out or was interrupted")
        if os.name == "posix":
            # A successful leader must not leave background GPU workers running
            # outside the experiment's time allocation.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if code != 0:
            raise RuntimeError("Backend exited with code %s; inspect local stderr.log" % code)


def _request_samples(samples, root):
    rows = copy.deepcopy(samples)
    for row in rows:
        row["reference_images"] = [str(resolve_path(p, root)) for p in row.get("reference_images", [])]
        for field in ("video_path", "control_video_path"):
            if row.get(field):
                row[field] = str(resolve_path(row[field], root))
    return rows


def _artifact_hashes(outputs_dir, patterns):
    """Require every declared artifact pattern on both completion and resume."""
    outputs_dir = Path(outputs_dir)
    artifacts = {}
    for pattern in patterns:
        matching = [p for p in outputs_dir.glob(pattern) if p.is_file() and p.stat().st_size > 0]
        if not matching:
            raise ValueError("Expected nonempty artifact missing: %s" % pattern)
        for path in matching:
            if not path.resolve().is_relative_to(outputs_dir.resolve()):
                raise ValueError("Artifact escaped output directory")
            artifacts[str(path.relative_to(outputs_dir)).replace("\\", "/")] = sha256_file(path)
    return artifacts


def run_experiment(config_path, *, root=".", resume=False):
    root = Path(root).resolve()
    plan = plan_experiment(config_path, root=root)
    config = plan["config"]
    run_dir = Path(plan["run_dir"])
    result_path = run_dir / "result.json"
    if run_dir.exists():
        if resume and result_path.exists():
            previous = load_json(result_path)
            if previous.get("status") == "completed" and previous.get("fingerprint") == plan["fingerprint"]:
                frozen_manifest = run_dir / "manifest.jsonl"
                if (not frozen_manifest.is_file()
                        or sha256_file(frozen_manifest) != previous.get("selected_manifest_sha256")):
                    raise ValueError("Existing frozen manifest changed or missing; choose a new run_id")
                if load_json(run_dir / "config.json") != config:
                    raise ValueError("Existing frozen config changed; choose a new run_id")
                if config["task"] == "inference":
                    from .evaluate import evaluate_outputs
                    checked = evaluate_outputs(run_dir / "manifest.jsonl", plan["outputs_dir"], fail_on_extra=True)
                    hashes = {p.name: sha256_file(p) for p in sorted(Path(plan["outputs_dir"]).glob("*.mp4"))}
                    if not checked["ok"] or hashes != previous.get("output_sha256"):
                        raise ValueError("Existing outputs failed validation; choose a new run_id")
                artifacts = _artifact_hashes(plan["outputs_dir"], config.get("expected_artifacts", []))
                if artifacts != previous.get("artifact_sha256", {}):
                    raise ValueError("Existing training artifact hashes changed or missing")
                return previous
        raise ValueError("run_id already exists; use --resume for an unchanged completed run or choose a new run_id")
    run_dir.mkdir(parents=True)
    outputs_dir = Path(plan["outputs_dir"])
    outputs_dir.mkdir()
    config_copy = run_dir / "config.json"
    write_json(config_copy, config)
    samples = read_manifest(plan["manifest_path"])
    if config.get("split"):
        samples = [row for row in samples if row.get("split") == config["split"]]
    # Freeze the exact evaluated subset, not the original full manifest.
    selected_manifest = run_dir / "manifest.jsonl"
    selected_manifest.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples), encoding="utf-8")
    request = {"schema_version": 1, "run_id": config["run_id"], "task": config["task"],
               "samples": _request_samples(samples, root), "parameters": config.get("parameters", {}),
               "resources": config["resources"],
               "output_dir": str(outputs_dir), "repo_root": str(root), "run_dir": str(run_dir)}
    request_path = run_dir / "request.json"
    write_json(request_path, request)
    result = {"schema_version": 1, "run_id": config["run_id"], "status": "running", "started_at": utc_now(),
              "fingerprint": plan["fingerprint"], "manifest_sha256": plan["manifest_sha256"],
              "selected_manifest_sha256": sha256_file(selected_manifest), "code_commit": _git_head(root),
              "source_sha256": plan["source_sha256"],
              "model_revisions": config.get("model_revisions", {}), "hypothesis": config["hypothesis"],
              "synthetic": plan["synthetic"], "task": config["task"], "resources": config["resources"],
              "outputs_dir": str(outputs_dir), "estimated_backend_gpu_hours": 0}
    write_json(result_path, result)
    start = time.monotonic()
    try:
        limit = config["resources"]["timeout_s"]
        stdout, stderr = run_dir / "stdout.log", run_dir / "stderr.log"
        if plan["synthetic"]:
            import shutil
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("synthetic smoke requires ffmpeg; install it before running")
            params = config.get("parameters", {})
            width, height, fps = params.get("width", 64), params.get("height", 64), params.get("fps", 8)
            duration = params.get("duration_s", 1)
            for name, value in [("width", width), ("height", height), ("fps", fps), ("duration_s", duration)]:
                positive(value, name, integer=name in ("width", "height"))
            for row in samples:
                run_process([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                             "testsrc2=size=%dx%d:rate=%s" % (width, height, fps), "-t", str(duration),
                             "-an", "-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p",
                             str(outputs_dir / (row["sample_id"] + ".mp4"))], cwd=root,
                            timeout_s=limit - (time.monotonic() - start), stdout_path=stdout, stderr_path=stderr)
        else:
            replacements = {"{request}": str(request_path), "{output_dir}": str(outputs_dir),
                            "{repo_root}": str(root), "{run_dir}": str(run_dir), "{python}": sys.executable}
            command = []
            for argument in config["backend"]["command"]:
                for key, value in replacements.items():
                    argument = argument.replace(key, value)
                command.append(argument)
            run_process(command, cwd=root, timeout_s=limit, stdout_path=stdout, stderr_path=stderr)
        result["backend_elapsed_seconds"] = time.monotonic() - start
        result["estimated_backend_gpu_hours"] = result["backend_elapsed_seconds"] * config["resources"]["gpus"] / 3600
        if config["task"] == "inference":
            from .evaluate import evaluate_outputs
            validation = evaluate_outputs(selected_manifest, outputs_dir, run_dir / "validation.json", fail_on_extra=True)
            if not validation["ok"]:
                raise RuntimeError("Output validation failed; inspect validation.json")
        result["artifact_sha256"] = _artifact_hashes(outputs_dir, config.get("expected_artifacts", []))
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["backend_elapsed_seconds"] = time.monotonic() - start
        result["estimated_backend_gpu_hours"] = result["backend_elapsed_seconds"] * config["resources"]["gpus"] / 3600
    result["finished_at"] = utc_now()
    result["gpu_hours_note"] = "Allocated GPUs times backend wall time; not NVML utilization or a scheduler billing measurement"
    result["output_sha256"] = {p.name: sha256_file(p) for p in sorted(outputs_dir.glob("*.mp4"))}
    write_json(result_path, result)
    return result


def run_campaign(config_path, *, root=".", resume=False):
    root = Path(root).resolve()
    campaign = load_json(resolve_path(config_path, root))
    if campaign.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")
    safe_name(campaign.get("campaign_id"), "campaign_id")
    experiments = campaign.get("experiments")
    if not isinstance(experiments, list) or not experiments or not all(isinstance(x, str) for x in experiments):
        raise ValueError("campaign.experiments must be a nonempty list of config paths")
    positive(campaign.get("max_trials"), "max_trials", integer=True)
    positive(campaign.get("max_gpu_hours"), "max_gpu_hours", allow_zero=True)
    if len(experiments) > campaign["max_trials"]:
        raise ValueError("campaign exceeds max_trials")
    plans = [plan_experiment(path, root=root) for path in experiments]
    if any(p["config"].get("purpose") == "final_generation" for p in plans):
        raise ValueError("campaign search does not run final_generation; launch the frozen primary experiment explicitly")
    if len({p["config"]["run_id"] for p in plans}) != len(plans):
        raise ValueError("campaign run_ids must be unique")
    # Conservative admission: reserve every run's full timeout before launching anything.
    if sum(p["max_backend_gpu_hours"] for p in plans) > campaign["max_gpu_hours"] + 1e-9:
        raise ValueError("campaign maximum timeout allocation exceeds total GPU-hour budget")
    report_path = resolve_path(campaign.get("report", "runs/campaign_%s.json" % campaign["campaign_id"]), root)
    protected = {resolve_path(config_path, root)} | {resolve_path(path, root) for path in experiments}
    protected.update(Path(plan["manifest_path"]) for plan in plans)
    if report_path in protected or any(report_path.is_relative_to(Path(plan["run_dir"])) for plan in plans):
        raise ValueError("campaign report must not overwrite configs, manifests, or experiment run files")
    results = []
    for path, plan in zip(experiments, plans):
        try:
            result = run_experiment(path, root=root, resume=resume)
        except Exception as exc:
            result = {"run_id": plan["config"]["run_id"], "status": "failed", "error": str(exc),
                      "estimated_backend_gpu_hours": plan["max_backend_gpu_hours"],
                      "gpu_hours_note": "Execution did not return a reliable usage result; conservatively charged the reserved timeout allocation"}
        results.append(result)
        if result["status"] != "completed" and campaign.get("stop_on_failure", True):
            break
    summary = {"campaign_id": campaign["campaign_id"], "results": results,
               "status": "completed" if len(results) == len(experiments) and all(r["status"] == "completed" for r in results) else "failed",
               "estimated_backend_gpu_hours": sum(r["estimated_backend_gpu_hours"] for r in results),
               "automation_scope": "Finite configured jobs; no built-in autonomous LLM hypothesis generation or paid API calls"}
    write_json(report_path, summary)
    return summary


def build_report(runs_dir, output_path):
    rows = [load_json(p) for p in sorted(Path(runs_dir).glob("*/result.json"))]
    lines = ["# IPVG 实验台账", "", "GPU小时为声明卡数乘后端墙钟时间；synthetic仅验证工程流程。", "",
             "| Run | 状态 | 类型 | GPU小时 | 假设 |", "|---|---|---|---:|---|"]
    for row in rows:
        hypothesis = str(row.get("hypothesis", "")).replace("|", "\\|").replace("\n", " ")
        lines.append("| %s | %s | %s | %.4f | %s |" % (row["run_id"], row["status"],
                     "synthetic" if row.get("synthetic") else row.get("task", "inference"),
                     row.get("estimated_backend_gpu_hours", 0), hypothesis))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"experiments": len(rows), "report": str(output_path)}
