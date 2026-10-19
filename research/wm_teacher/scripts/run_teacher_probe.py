#!/usr/bin/env python3
"""Bounded, offline 6-case x 2-model x 2-CFG generation; never train or score.

Jobs are a local JSONL (24 rows, <=1 MiB). Each row requires id, case_id,
canonical_source, model, ckpt, image, prompt, seed, cfg, shift, steps, out,
and prompt_review {csv_path, csv_sha256, prompt_sha256, reviewed_by, reviewed_at}.
Additional original/new caption and source audit fields are retained when present;
independent provenance records may bind the original CSV row. The reviewed prompt
is an agent-authored dense internal-dev label, not human Teacher approval.
The original CSV is preserved, not rewritten; the recorded agent review
must bind the >=80-word prompt by SHA256. Out paths are relative to --out-dir
or absolute inside it. No resume, downloads, VLM calls, or work stealing.

Example (only after independent development-source/prompt review approval):
  python scripts/run_teacher_probe.py --jobs reviewed.jsonl --out-dir NEW_RUN
Defaults reserve 4 GPUh generation plus 8 GPUh scoring within a 12 GPUh plan.
--max-gpu-hours defaults to 8, but generation is always hard-capped at 4.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(os.environ.get("TEACHER_PROBE_ROOT", str(Path(__file__).resolve().parents[1])))
CHECKPOINT_CACHE = Path("/dockerdata/models/wm_checkpoints")
MERGE = ROOT / "models/train/merge/full1_a07.safetensors"
MERGE_SHA256 = "057d9a1e682e2f237594f5ef99f994f07699188c97b4196016fd371ab40402a6"
GENERATION_CAP = 4.0
SCORING_RESERVE = 8.0
EXPERIMENT_CAP = 12.0
MAX_MANIFEST_BYTES = 1024 * 1024
W, H, FRAMES, FPS, QUALITY = 1280, 704, 121, 24, 5
ARMS = {("base", 3), ("base", 5), ("m_full1_a07", 3), ("m_full1_a07", 5)}
ACTIVE_SCRIPTS = {"gen_jobs.py", "run_teacher_probe.py", "score_pool.py",
                  "vbench_eval.py", "evaluation.py", "wmb_merge_score.py",
                  "eval_val_loss.py", "val_merge_score.py", "score_merge_cpu.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def inside(root, path):
    require(isinstance(path, str) and path and "://" not in path, "Invalid output path")
    target = (Path(path) if Path(path).is_absolute() else root / path).resolve()
    require(target != root and target.is_relative_to(root), "Output escapes --out-dir")
    return target


def local_file(path, label):
    require(isinstance(path, str) and path and "://" not in path, label + " must be local")
    target = Path(path).resolve()
    require(target.is_file() and target.stat().st_size > 0, label + " must be a nonempty real file")
    return target


def budget(gpus, timeout, maximum, expected):
    require(expected == 24, "This first probe requires --expected-jobs 24")
    require(re.fullmatch(r"[0-9]+(?:,[0-9]+)*", gpus or ""), "GPU IDs must be comma-separated integers")
    cards = gpus.split(",")
    require(len(cards) == len(set(cards)) and all(str(int(g)) == g for g in cards), "Duplicate/noncanonical GPU IDs")
    require(len(cards) <= expected, "More GPUs than jobs")
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
                for x in (timeout, maximum)), "Budgets must be finite and positive")
    reserved = len(cards) * timeout / 3600
    effective = min(maximum, GENERATION_CAP, EXPERIMENT_CAP - SCORING_RESERVE)
    require(reserved <= effective, "GPU count * timeout exceeds generation GPU-hour budget (hard cap 4)")
    return cards, reserved, effective


def validate_jobs(path, out_dir, expected=24):
    require(not out_dir.exists() and not out_dir.is_symlink(), "--out-dir must be a new directory; resume is unsupported")
    path = local_file(str(path), "Jobs JSONL")
    require(path.suffix.lower() == ".jsonl" and path.stat().st_size <= MAX_MANIFEST_BYTES,
            "Jobs must be a small local JSONL (<=1 MiB)")
    original = path.read_bytes()
    rows = [json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()]
    require(expected == 24 and len(rows) == expected, "Expected exactly 24 jobs")
    required = {"id", "case_id", "canonical_source", "model", "ckpt", "image", "prompt",
                "seed", "cfg", "shift", "steps", "out", "prompt_review"}
    optional = {"ckpt_sha256", "cfg_zero_star", "cfg_zero_steps", "width", "height", "num_frames", "fps", "quality"}
    ids, outputs, cases, hashes, csvs, csv_rows = set(), set(), defaultdict(list), {}, {}, {}
    normalized = []
    for row in rows:
        require(isinstance(row, dict) and required <= row.keys() and row.keys() <= required | optional,
                "Missing or unsupported job fields")
        row = dict(row)
        for key in ("id", "case_id"):
            require(isinstance(row[key], str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", row[key]), "Invalid " + key)
        require(row["id"] not in ids, "Duplicate job IDs")
        ids.add(row["id"])
        require(isinstance(row["canonical_source"], str) and row["canonical_source"].strip(), "Missing canonical_source")
        require(isinstance(row["prompt"], str) and len(row["prompt"].split()) >= 80, "Prompt must contain >=80 words")
        require(type(row["seed"]) is int and 0 <= row["seed"] <= 2**32 - 1, "Seed out of range")
        require(type(row["cfg"]) in (int, float) and row["cfg"] in (3, 5), "Only CFG 3/5 are permitted")
        require(type(row["shift"]) in (int, float) and row["shift"] == 5 and type(row["steps"]) is int and row["steps"] == 50,
                "Require shift=5, steps=50")
        require(row.get("cfg_zero_star", False) is False and type(row.get("cfg_zero_steps", 0)) is int
                and row.get("cfg_zero_steps", 0) == 0, "CFG-Zero is disabled in this round")
        for key, fixed in (("width", W), ("height", H), ("num_frames", FRAMES), ("fps", FPS), ("quality", QUALITY)):
            require(type(row.get(key, fixed)) is int and row.get(key, fixed) == fixed, "Wrong fixed video parameter: " + key)
            row[key] = fixed
        require(row["model"] in {"base", "m_full1_a07"}, "Unsupported model")
        if row["model"] == "base":
            require(row["ckpt"] == "NONE" and row.get("ckpt_sha256", "NONE") == "NONE", "base requires ckpt NONE")
            row["ckpt_sha256"] = "NONE"
        else:
            ckpt = local_file(row["ckpt"], "Checkpoint")
            require(ckpt == MERGE.resolve(), "m_full1_a07 requires the fixed merge checkpoint")
            cache = CHECKPOINT_CACHE / (hashlib.sha256(str(ckpt).encode()).hexdigest() + ".safetensors")
            require(cache.is_file() and cache.stat().st_size == ckpt.stat().st_size,
                    "Verified local checkpoint cache is required; do not mmap shared weights")
            if ckpt not in hashes:
                hashes[ckpt] = sha256(cache)
            require(hashes[ckpt] == MERGE_SHA256 and row.get("ckpt_sha256", MERGE_SHA256) == MERGE_SHA256,
                    "Fixed merge checkpoint SHA256 mismatch")
            row["ckpt"], row["ckpt_sha256"] = str(ckpt), MERGE_SHA256
        image = local_file(row["image"], "Input image/video")
        if image not in hashes:
            hashes[image] = sha256(image)
        row["image"], row["image_sha256"] = str(image), hashes[image]
        target = inside(out_dir, row["out"])
        require(target.suffix.lower() == ".mp4" and target not in outputs, "Duplicate or non-MP4 output")
        require(target.relative_to(out_dir).parts[0] == "videos", "Outputs must be under --out-dir/videos/")
        outputs.add(target)
        row["out"] = str(target)
        review = row["prompt_review"]
        review_fields = {"csv_path", "csv_sha256", "prompt_sha256", "reviewed_by", "reviewed_at"}
        require(isinstance(review, dict) and review_fields <= review.keys(), "Missing external prompt review record")
        require(all(isinstance(review[k], str) and review[k].strip() for k in review_fields - {"csv_row"}),
                "Invalid prompt review record")
        if "label_protocol" in review:
            require(review["label_protocol"] == "agent_authored_conservative_dense_internal_dev_v1", "Wrong label protocol")
        if "reviewer_type" in review:
            require(review["reviewer_type"] == "agent", "Agent label review is not human Teacher approval")
        if "use_scope" in review:
            require(review["use_scope"] == "training_visible_internal_diagnostic_only", "Wrong internal diagnostic scope")
        require(review["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest(), "Reviewed prompt SHA256 mismatch")
        require(datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00")).tzinfo is not None,
                "reviewed_at must be an ISO timestamp with timezone")
        csv_path = local_file(review["csv_path"], "Original CSV")
        require(csv_path.suffix.lower() == ".csv", "Original prompt source must be CSV")
        if csv_path not in hashes:
            hashes[csv_path] = sha256(csv_path)
        require(hashes[csv_path] == review["csv_sha256"], "Original CSV SHA256 mismatch")
        if "csv_row" in review:
            require(type(review["csv_row"]) is int and review["csv_row"] >= 0, "Invalid zero-based CSV row")
            if csv_path not in csv_rows:
                with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                    csv_rows[csv_path] = list(csv.DictReader(handle))
            require(review["csv_row"] < len(csv_rows[csv_path]), "Original CSV row is missing")
            original_row = csv_rows[csv_path][review["csv_row"]]
            require(original_row.get(review["video_column"]) == review["original_video"], "Original CSV video identity mismatch")
            original_prompt = original_row.get(review["prompt_column"])
            require(isinstance(original_prompt, str) and hashlib.sha256(original_prompt.encode()).hexdigest()
                    == review["original_prompt_sha256"], "Original CSV prompt SHA256 mismatch")
        csvs[str(csv_path)] = hashes[csv_path]
        row["prompt_review"] = dict(review, csv_path=str(csv_path),
                                    preserved_csv=str(out_dir / "original_csv" / (hashes[csv_path] + ".csv")))
        row["cfg_zero_star"], row["cfg_zero_steps"] = False, 0
        cases[row["case_id"]].append(row)
        normalized.append(row)
    require(len(cases) == 6, "Require six cases")
    sources, images = set(), set()
    for case_rows in cases.values():
        require(len(case_rows) == 4 and {(r["model"], r["cfg"]) for r in case_rows} == ARMS,
                "Each case must have all four model/CFG arms exactly once")
        first = case_rows[0]
        for key in ("canonical_source", "image", "image_sha256", "prompt", "seed", "shift", "steps", "prompt_review"):
            require(all(r[key] == first[key] for r in case_rows), "Changed " + key + " within case")
        require(first["canonical_source"] not in sources, "Cases must use independent canonical_source values")
        require(first["image_sha256"] not in images, "Cases must use distinct input image/video content")
        sources.add(first["canonical_source"])
        images.add(first["image_sha256"])
    return normalized, original, csvs


def assert_idle(gpus):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15, check=True)
    memory = {}
    for line in result.stdout.splitlines():
        index, used = (p.strip() for p in line.split(","))
        memory[index] = int(used)
    require(all(g in memory and memory[g] < 100 for g in gpus), "Selected GPU is missing or uses >=100 MiB; do not stop other jobs")
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            cmd = (Path("/proc") / pid / "cmdline").read_bytes().split(b"\0")
            if not any(os.path.basename(part.decode(errors="replace")) in ACTIVE_SCRIPTS for part in cmd):
                continue
            entries = (Path("/proc") / pid / "environ").read_bytes().split(b"\0")
            visible = next((e.split(b"=", 1)[1].decode().split(",") for e in entries
                            if e.startswith(b"CUDA_VISIBLE_DEVICES=")), None)
            require(visible is not None and not set(gpus).intersection(visible),
                    "Active generation/scoring process on selected (or unspecified) GPU: PID " + pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise ValueError("Cannot verify GPU process ownership: PID " + pid) from error


def decode_video(path):
    import imageio.v2 as imageio
    require(Path(path).is_file() and Path(path).stat().st_size > 0, "Missing/empty MP4")
    reader = imageio.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        require(math.isclose(float(metadata.get("fps", 0)), FPS, abs_tol=0.01), "Wrong decoded FPS")
        count = 0
        for frame in reader:
            require(frame.shape[:2] == (H, W), "Wrong decoded video dimensions")
            count += 1
        require(count == FRAMES, "Wrong fully decoded frame count")
    finally:
        reader.close()
    return {"frames": count, "width": W, "height": H, "fps": FPS, "bytes": Path(path).stat().st_size}


def worker(out_dir, index, assignment_sha):
    require(os.environ.get("TEACHER_PROBE_PARENT_PID") == str(os.getppid()), "--worker is internal to the parent runner")
    path = out_dir / ("assignment_" + str(index) + ".json")
    require(sha256(path) == assignment_sha, "Worker assignment changed")
    assignment = json.loads(path.read_text())
    require(assignment["gpu"] == os.environ.get("CUDA_VISIBLE_DEVICES"), "Worker GPU differs from fixed assignment")
    report_path = out_dir / ("worker_" + str(index) + ".json")
    started = time.perf_counter()
    report = {"status": "running", "gpu": assignment["gpu"], "pid": os.getpid(), "loads": [], "jobs": []}
    pipe, torch = None, None
    failed = False
    current = None
    try:
        import torch
        from gen_jobs import build_pipe, load_image, NEG
        from diffsynth.utils.data import save_video
        groups = defaultdict(list)
        for job in assignment["jobs"]:
            groups[job["ckpt"]].append(job)
        for ckpt, jobs in groups.items():
            current = {"ckpt": ckpt, "status": "running"}
            report["loads"].append(current)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                pipe = build_pipe(ckpt)
                torch.cuda.synchronize()
                current["status"] = "completed"
            finally:
                current.update(seconds=time.perf_counter() - t0,
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
            for job in jobs:
                current = {"id": job["id"], "status": "running", "out": job["out"]}
                report["jobs"].append(current)
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                try:
                    require(sha256(job["image"]) == job["image_sha256"], "Input changed after manifest freeze")
                    target = inside(out_dir, job["out"])
                    require(not target.exists(), "Assigned output already exists; no skip or overwrite")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.stem + "." + uuid.uuid4().hex + ".tmp.mp4")
                    current["temporary_output"] = str(tmp)
                    video = pipe(prompt=job["prompt"], negative_prompt=NEG, seed=job["seed"], tiled=True,
                                 height=H, width=W, num_frames=FRAMES, input_image=load_image(job["image"]),
                                 cfg_scale=job["cfg"], sigma_shift=5, num_inference_steps=50,
                                 cfg_zero_star=False, cfg_zero_steps=0)
                    torch.cuda.synchronize()
                    save_video(video, str(tmp), fps=FPS, quality=QUALITY)
                    current["decode"] = decode_video(tmp)
                    require(not target.exists(), "Output collision before rename")
                    os.rename(tmp, target)
                    current["status"] = "completed"
                    del video
                finally:
                    current.update(seconds=time.perf_counter() - t0,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    atomic_json(report_path, report)
            del pipe
            pipe = None
            torch.cuda.empty_cache()
    except BaseException as error:
        failed = True
        if current is not None:
            current["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if pipe is not None:
            del pipe
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        report["status"] = "failed" if failed else "completed"
        report["worker_seconds"] = time.perf_counter() - started
        atomic_json(report_path, report)
    return 1 if failed else 0


def stop_workers(workers, grace):
    # Every session is created here for one owned worker and its encoder children.
    # Signal only that exact PID/PGID; never pkill, unrelated processes, or other GPUs.
    stopping = []
    for info in workers:
        if info["process"].poll() is None:
            stopping.append(info)
            try:
                os.killpg(info["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    until = time.perf_counter() + grace
    for info in workers:
        proc = info["process"]
        try:
            proc.wait(timeout=max(0, until - time.perf_counter()))
        except subprocess.TimeoutExpired:
            pass
    for info in stopping:
        # The Python leader may already have exited but an FFmpeg child may not.
        try:
            os.killpg(info["process"].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for info in workers:
        info["process"].wait()
        if "ended" not in info:
            info["ended"] = time.perf_counter()


def freeze_framework(out_dir):
    source = ROOT / "DiffSynth-Studio" / "diffsynth"
    require(source.is_dir(), "Local DiffSynth runtime package is missing")
    destination = out_dir / "framework"
    destination.mkdir()
    hashes = {}
    for current, directories, files in os.walk(source):
        directories[:] = sorted(d for d in directories if d not in {"__pycache__", ".git"})
        for name in sorted(files):
            if name.endswith(".pyc"):
                continue
            path = Path(current) / name
            relative = Path("diffsynth") / path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            atomic_bytes(target, data)
            hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    require("diffsynth/__init__.py" in hashes, "Frozen DiffSynth package is incomplete")
    return hashes


def launch_workers(out_dir, assignments, timeout):
    workers, failure = [], None
    grace = min(2.0, timeout / 10)
    previous_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError("Parent received termination signal")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, assignment in enumerate(assignments):
            path = out_dir / ("assignment_" + str(index) + ".json")
            atomic_json(path, assignment)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=assignment["gpu"],
                       TEACHER_PROBE_PARENT_PID=str(os.getpid()),
                       DIFFSYNTH_MODEL_BASE_PATH="/dockerdata/models", DIFFSYNTH_SKIP_DOWNLOAD="true",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       WM_CHECKPOINT_CACHE_DIR=str(CHECKPOINT_CACHE), TEACHER_PROBE_ROOT=str(ROOT))
            env["PYTHONPATH"] = str(out_dir / "framework") + os.pathsep + env.get("PYTHONPATH", "")
            log = (out_dir / ("worker_" + str(index) + ".log")).open("x")
            started = time.perf_counter()
            try:
                proc = subprocess.Popen([sys.executable, str(out_dir / "code" / "run_teacher_probe.py"),
                                         "--jobs", str(out_dir / "jobs.frozen.jsonl"), "--out-dir", str(out_dir),
                                         "--worker", str(index), "--assignment-sha", sha256(path)],
                                        env=env, cwd=str(out_dir / "framework"),
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            except BaseException:
                log.close()
                raise
            workers.append({"index": index, "gpu": assignment["gpu"], "process": proc,
                            "started": started, "log": log})
            # Launch time consumes the same reservation; do not start another worker after an early failure.
            for info in workers:
                if info["process"].poll() not in (None, 0):
                    raise RuntimeError("Worker failed during startup")
        while True:
            running = False
            now = time.perf_counter()
            for info in workers:
                rc = info["process"].poll()
                if rc is not None:
                    info.setdefault("ended", now)
                    if rc:
                        raise RuntimeError("Worker " + str(info["index"]) + " failed with exit " + str(rc))
                else:
                    running = True
                    # Leave shutdown grace inside the reserved timeout, not in addition to it.
                    if now - info["started"] >= max(0, timeout - grace - min(0.1, timeout / 10)):
                        raise TimeoutError("Worker " + str(info["index"]) + " reached reserved timeout")
            if not running:
                break
            time.sleep(min(0.02, timeout / 20))
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            stop_workers(workers, grace)
            for info in workers:
                info["log"].close()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
    result = [{"index": i["index"], "gpu": i["gpu"], "pid": i["process"].pid,
               "returncode": i["process"].returncode, "seconds": i["ended"] - i["started"],
               "gpu_hours": (i["ended"] - i["started"]) / 3600} for i in workers]
    return result, failure


def finish(out_dir, jobs, resources, error, reserved, effective, manifest_sha):
    failures, completed = ([error] if error else []), []
    charge = math.fsum(r["gpu_hours"] for r in resources)
    if len(resources) == 0 or any(r["returncode"] != 0 for r in resources):
        failures.append("Missing or nonzero worker exits")
    if charge > effective or charge > reserved:
        failures.append("Actual GPU charge exceeds generation reservation/budget")
    reports = {}
    for resource in resources:
        try:
            report = json.loads((out_dir / ("worker_" + str(resource["index"]) + ".json")).read_text())
            require(report["status"] == "completed" and report["gpu"] == resource["gpu"], "Incomplete worker report")
            for row in report["jobs"]:
                require(row["id"] not in reports and row["status"] == "completed", "Duplicate/failed worker output record")
                require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                            for k in ("seconds", "peak_allocated_bytes", "peak_reserved_bytes")), "Invalid job resource metrics")
                reports[row["id"]] = row
        except (OSError, ValueError, KeyError, TypeError) as e:
            failures.append("Worker report: " + str(e))
    if not failures:
        try:
            require(set(reports) == {j["id"] for j in jobs}, "Incomplete assigned output IDs")
            for job in jobs:
                require(reports[job["id"]]["out"] == job["out"], "Worker output assignment mismatch")
                # Re-decode in the parent before the final completed status, never just count paths.
                completed.append({"id": job["id"], "out": job["out"], "decode": decode_video(job["out"]),
                                  "sha256": sha256(job["out"])})
        except Exception as e:
            failures.append("Final complete-decode validation: " + str(e))
    summary = {"status": "failed" if failures else "completed", "expected_jobs": 24,
               "completed_outputs": len(completed), "outputs": completed, "workers": resources,
               "gpu_hours": charge, "reserved_generation_gpu_hours": reserved,
               "effective_generation_cap_gpu_hours": effective, "scoring_reserve_gpu_hours": SCORING_RESERVE,
               "experiment_plan_cap_gpu_hours": EXPERIMENT_CAP, "manifest_sha256": manifest_sha,
               "failures": failures, "training_performed": False, "scoring_performed": False}
    atomic_json(out_dir / "probe_summary.json", summary)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--max-gpu-hours", type=float, default=8)
    parser.add_argument("--expected-jobs", type=int, default=24)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--assignment-sha", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).absolute()
    if args.worker is not None:
        require(type(args.worker) is int and args.worker >= 0 and args.assignment_sha, "Invalid internal worker")
        return worker(out_dir.resolve(), args.worker, args.assignment_sha)
    gpus, reserved, effective = budget(args.gpus, args.timeout_seconds, args.max_gpu_hours, args.expected_jobs)
    require(not out_dir.exists() and not out_dir.is_symlink(), "--out-dir must be new; no resume/overwrite")
    out_dir = out_dir.resolve()
    jobs, original, csvs = validate_jobs(args.jobs, out_dir, args.expected_jobs)
    assert_idle(gpus)
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        code_dir = out_dir / "code"
        code_dir.mkdir()
        source_sha = {}
        for name in ("run_teacher_probe.py", "gen_jobs.py"):
            data = (ROOT / "scripts" / name).read_bytes()
            atomic_bytes(code_dir / name, data)
            source_sha[name] = hashlib.sha256(data).hexdigest()
        framework_sha = freeze_framework(out_dir)
        atomic_bytes(out_dir / "jobs.input.jsonl", original)
        frozen = b"".join(json_bytes(j) for j in jobs)
        manifest_sha = hashlib.sha256(frozen).hexdigest()
        atomic_bytes(out_dir / "jobs.frozen.jsonl", frozen)
        (out_dir / "original_csv").mkdir()
        for source, digest in csvs.items():
            data = Path(source).read_bytes()
            require(hashlib.sha256(data).hexdigest() == digest, "Original CSV changed before preservation")
            atomic_bytes(out_dir / "original_csv" / (digest + ".csv"), data)
        assignments = [{"gpu": g, "jobs": []} for g in gpus]
        groups = defaultdict(list)
        for job in jobs:
            groups[job["ckpt"]].append(job)
        for group in groups.values():
            for i, job in enumerate(group):
                assignments[i % len(gpus)]["jobs"].append(job)
        atomic_json(out_dir / "probe_manifest.json", {
            "protocol": "internal-dev-teacher-cfg-v1", "jobs_input_sha256": hashlib.sha256(original).hexdigest(),
            "jobs_frozen_sha256": manifest_sha, "jobs": jobs, "gpus": gpus,
            "reserved_generation_gpu_hours": reserved, "effective_generation_cap_gpu_hours": effective,
            "timeout_seconds": args.timeout_seconds, "requested_max_gpu_hours": args.max_gpu_hours,
            "scoring_reserve_gpu_hours": SCORING_RESERVE, "experiment_plan_cap_gpu_hours": EXPERIMENT_CAP,
            "checkpoint_sha256": {"m_full1_a07": MERGE_SHA256}, "zero_disabled": True,
            "source_sha256": source_sha, "framework_root": str(out_dir / "framework"),
            "framework_source_sha256": framework_sha,
            "original_csv_sha256": csvs, "external_review_only": True,
            "label_protocol": "agent_authored_conservative_dense_internal_dev_v1",
            "use_scope": "training_visible_internal_diagnostic_only", "teacher_human_review": "pending"})
        resources, error = launch_workers(out_dir, assignments, args.timeout_seconds)
        return finish(out_dir, jobs, resources, error, reserved, effective, manifest_sha)
    except BaseException as error:
        atomic_json(out_dir / "probe_summary.json", {"status": "failed", "expected_jobs": 24,
                    "completed_outputs": 0, "failures": [type(error).__name__ + ": " + str(error)]})
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("Probe refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
