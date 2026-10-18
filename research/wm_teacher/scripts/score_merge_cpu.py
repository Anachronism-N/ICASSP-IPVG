#!/usr/bin/env python3
"""Wait for fixed merge outputs, then validate/decode, GTref108 and original Physics-IQ.

Run on the launcher with its restored local scoring environments. Existing incomplete
results (including failed work directories) require manual review; nothing is deleted.
Locks coordinate launchers using this script, not older unlocked scoring commands.
"""
import argparse
import concurrent.futures
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path("/dockerdata/wm_eval/bench")
VBench_PYTHON = "/dockerdata/venv_vbench/bin/python"
PHYSIQ_PYTHON = "/dockerdata/venv_piq/bin/python"
COUNTS = {"wm_bench": 108, "vbench_i2v": 355, "wmb": 350, "physiq": 198}
GT_KEYS = ("dreamsim", "dreamsim_last", "motion_ratio", "dur")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    try:
        with path.open() as handle:
            return json.load(handle, object_pairs_hook=unique_keys)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unreadable/partial JSON; preserve and review {path}: {error}") from error


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_jobs(root, model):
    require(re.fullmatch(r"[A-Za-z0-9_-]+", model), f"Invalid model name: {model!r}")
    with (root / "eval/jobs_merge_all.jsonl").open() as handle:
        jobs = [job for line in handle if (job := json.loads(line))["model"] == model]
    groups = {suite: [] for suite in COUNTS}
    for job in jobs:
        path = Path(job["out"])
        parts = path.relative_to(root / "eval").parts
        require(parts[0] in groups, f"Unknown suite for {path}")
        groups[parts[0]].append(job)
    require(len(jobs) == 1011 and len({j["out"] for j in jobs}) == 1011,
            f"{model}: expected 1011 unique fixed jobs, got {len(jobs)}")
    require(all(len(groups[suite]) == count for suite, count in COUNTS.items()),
            f"{model}: incorrect fixed suite counts")
    with (root / "eval/wm_bench/cases.jsonl").open() as handle:
        cases = [json.loads(line)["id"] for line in handle]
    require(len(cases) == len(set(cases)) == 108, "WM manifest must contain 108 unique original cases")
    require({j["id"] for j in groups["wm_bench"]} == set(cases),
            f"{model}: WM jobs differ from original 108 cases")
    require({j["out"] for j in groups["wm_bench"]} ==
            {str(root / "eval/wm_bench/videos" / model / (case + ".mp4")) for case in cases},
            f"{model}: WM output paths differ from fixed cases")
    return jobs, groups, set(cases)


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def wait_for_outputs(jobs, poll_seconds=60, sleep=time.sleep):
    while True:
        missing = sum(not nonempty(Path(job["out"])) for job in jobs)
        if not missing:
            return
        print(f"WAIT {jobs[0]['model']}: {missing}/1011 missing/empty outputs", flush=True)
        sleep(poll_seconds)


@contextmanager
def suite_lock(root, model, suite):
    directory = root / "eval/diag"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".score_merge_cpu.{model}.{suite}.lock"
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_work(directory, prefix):
    leftovers = sorted(directory.glob(prefix + "*"))
    require(not leftovers, f"Unreviewed partial work; preserve and review: {leftovers}")


def publish(source, destination):
    require(not destination.exists(), f"Refusing to overwrite unreviewed result: {destination}")
    with source.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(source, destination)


def check_video(path, frames=121, height=704):
    import av

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            require(stream.average_rate == 24 and (stream.width, stream.height) == (1280, height),
                    f"Unexpected video format: {path}")
            count = 0
            for frame in container.decode(video=0):
                require((frame.width, frame.height) == (1280, height), f"Unexpected frame size: {path}")
                count += 1
            require(count == frames, f"Expected {frames} decoded frames, got {count}: {path}")
    except Exception as error:
        raise RuntimeError(f"Video decode failed; preserve and review {path}: {error}") from error
    return str(path)


def validate_decode(path, model, jobs):
    report = read_json(path)
    require(isinstance(report, dict), f"Invalid decode report; preserve and review {path}")
    expected = {"model": model, "count": 1011, "expected": 1011, "all_frames_decoded": True,
                "frames": 121, "fps": 24, "width": 1280, "height": 704,
                "outputs": [job["out"] for job in jobs]}
    require(all(report.get(key) == value for key, value in expected.items()),
            f"Incomplete/mismatched decode report; preserve and review {path}")
    require(report.get("all_frames_decoded") is True, f"Invalid decode flag: {path}")


def decode_model(root, model, jobs, decode=check_video):
    path = root / "eval/diag" / f"decode_{model}_20261018.json"
    with suite_lock(root, model, "decode"):
        check_work(path.parent, f".{path.name}.cpu-")
        if path.exists():
            validate_decode(path, model, jobs)
            print(f"SKIP_VALID_DECODE {model}", flush=True)
            return
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            outputs = list(pool.map(decode, (Path(job["out"]) for job in jobs)))
        report = {"model": model, "count": 1011, "expected": 1011, "all_frames_decoded": True,
                  "frames": 121, "fps": 24, "width": 1280, "height": 704,
                  "at": time.strftime("%F %T %Z"), "outputs": outputs}
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=f".{path.name}.cpu-", suffix=".tmp", delete=False) as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        validate_decode(temporary, model, jobs)
        publish(temporary, path)
        print(f"MODEL_ALL1011_FULL_DECODE_PASS {model}", flush=True)


def validate_gtref(path, cases):
    result = read_json(path)
    require(isinstance(result, dict) and len(result) == 108 and set(result) == cases,
            f"Incomplete/wrong GTref108 cases; preserve and review {path}")
    require(all(isinstance(row, dict) and all(finite(row.get(key)) for key in GT_KEYS)
                for row in result.values()),
            f"Missing/non-finite GTref values; preserve and review {path}")


def score_gtref(root, model, cases, run=subprocess.run):
    directory = root / "eval/wm_bench"
    path = directory / f"gtref_{model}.json"
    with suite_lock(root, model, "gtref"):
        prefix = f".{model}.gtref.work."
        check_work(directory, prefix)
        if path.exists():
            validate_gtref(path, cases)
            print(f"SKIP_VALID_GTREF108 {model}", flush=True)
            return
        work = Path(tempfile.mkdtemp(dir=directory, prefix=prefix))
        (work / "cases.jsonl").symlink_to(directory / "cases.jsonl")
        (work / "videos").symlink_to(directory / "videos", target_is_directory=True)
        # Override only output/assets roots; use the existing scorer and DreamSim cache unchanged.
        wrapper = ("import runpy,sys; ns=runpy.run_path(sys.argv[1]); "
                   "ns['main'].__globals__.update(ROOT=sys.argv[2],EV=sys.argv[3]); "
                   "sys.argv=[sys.argv[1],sys.argv[4]]; ns['main']()")
        logs = root / "dataset/download_logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", THREADS="16")
        try:
            with (logs / f"score_gtref_{model}_20261018.log").open("a") as handle:
                run([VBench_PYTHON, "-u", "-c", wrapper, str(root / "scripts/gt_ref_metrics.py"),
                     str(root), str(work), model], env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
            validate_gtref(work / path.name, cases)
            publish(work / path.name, path)
        except Exception as error:
            raise RuntimeError(f"GTref failed; preserved work for review: {work}") from error
        (work / "cases.jsonl").unlink()
        (work / "videos").unlink()
        work.rmdir()
        print(f"MODEL_GTREF108_PASS {model}", flush=True)


def physics_inputs(root, model, jobs, bench):
    paths = [Path(job["out"]) for job in jobs]
    parents = {path.parent for path in paths}
    require(len(paths) == len(set(paths)) == 198 and len(parents) == 1,
            f"{model}: Physics requires exactly 198 unique inputs in one folder")
    source = paths[0].parent
    require(source.parent == root / "eval/physiq/videos" and
            re.fullmatch(re.escape(model) + r"-op-run_\d+", source.name),
            f"Unexpected Physics input folder: {source}")
    with (bench / "physics-IQ-benchmark/descriptions/descriptions_original.csv").open(newline="") as handle:
        descriptions = list(csv.DictReader(handle))
    expected = {row["generated_video_name"] for row in descriptions if "take-2" not in row["scenario"]}
    scenarios = {name.rsplit("_", 1)[-1] for name in expected}
    require(len(expected) == 198 and len(scenarios) == 66,
            "Original Physics descriptions must cover 198 inputs/66 scenarios")
    require({path.name for path in paths} == expected and
            {path.name for path in source.glob("*.mp4")} == expected and all(map(nonempty, paths)),
            f"{model}: incomplete/mismatched Physics198 input coverage; preserve and review {source}")
    return source, paths, scenarios


def physics_result_paths(directory, name):
    base = directory / "physics-IQ-benchmark/results"
    return base / (name + "_metrics.json"), base / (name + ".csv")


def validate_physics(metrics_path, csv_path, scenarios):
    metrics = read_json(metrics_path)
    try:
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        actual = [row["scenario"] for row in rows]
    except (OSError, KeyError, csv.Error) as error:
        raise RuntimeError(f"Unreadable/partial Physics CSV; preserve and review {csv_path}") from error
    require(len(actual) == len(set(actual)) == 66 and set(actual) == scenarios,
            f"Incomplete/wrong Physics66 scenarios; preserve and review {csv_path}")
    require(isinstance(metrics, dict) and finite(metrics.get("final_score_orig")),
            f"Missing/non-finite original Physics score; preserve and review {metrics_path}")


def prepare_physics(source, paths, destination, run=subprocess.run, decode=check_video):
    import imageio_ffmpeg

    destination.mkdir(parents=True, exist_ok=True)
    require({p.name for p in destination.glob("*.mp4")} <= {p.name for p in paths},
            f"Unexpected/partial Physics videos; preserve and review {destination}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    def pad(path):
        output = destination / path.name
        check_work(destination, f".{path.stem}.cpu-")
        if output.exists():
            decode(output, frames=120, height=720)
            return
        with tempfile.NamedTemporaryFile(dir=destination, prefix=f".{path.stem}.cpu-",
                                         suffix=".tmp.mp4", delete=False) as handle:
            temporary = Path(handle.name)
        # -y applies only to this process's unique empty temporary, never an existing result.
        run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-frames:v", "120", "-an",
             "-vf", "pad=1280:720:0:8,fillborders=top=8:bottom=8:mode=smear",
             "-c:v", "libx264", "-threads", "2", "-crf", "12", "-pix_fmt", "yuv420p", str(temporary)],
            check=True)
        decode(temporary, frames=120, height=720)
        publish(temporary, output)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(pad, paths))


def score_physics(root, model, jobs, bench=BENCH, run=subprocess.run, decode=check_video):
    with suite_lock(root, model, "physiq"):
        source, paths, scenarios = physics_inputs(root, model, jobs, bench)
        directory = root / "eval/physiq/results"
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f".{model}.physiq.work."
        check_work(directory, prefix)
        metrics, table = physics_result_paths(directory, source.name)
        if metrics.exists() or table.exists():
            validate_physics(metrics, table, scenarios)
            print(f"SKIP_VALID_PHYSIQ198_66 {model}", flush=True)
            return
        destination = root / "eval/physiq/videos_5s" / source.name
        prepare_physics(source, paths, destination, run=run, decode=decode)
        work = Path(tempfile.mkdtemp(dir=directory, prefix=prefix))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PATH="/opt/conda/envs/torch-base/bin:" + os.environ["PATH"])
        logs = root / "dataset/download_logs"
        logs.mkdir(parents=True, exist_ok=True)
        try:
            with (logs / "score_physiq.log").open("a") as handle:
                run([PHYSIQ_PYTHON, "physiq/run_physics_iq.py", "--input_folders", str(destination),
                     "--output_folder", str(work), "--descriptions_file",
                     str(bench / "physics-IQ-benchmark/descriptions/descriptions_original.csv"),
                     "--original_physics_iq", "--benchmark_base_folder", str(bench / "physics-IQ-benchmark-data"),
                     "--n_process", "16"], cwd=bench / "physics-IQ-benchmark", env=env,
                    stdout=handle, stderr=subprocess.STDOUT, check=True)
            temporary_metrics, temporary_table = physics_result_paths(work, source.name)
            validate_physics(temporary_metrics, temporary_table, scenarios)
            metrics.parent.mkdir(parents=True, exist_ok=True)
            publish(temporary_table, table)
            publish(temporary_metrics, metrics)
        except Exception as error:
            raise RuntimeError(f"Physics scorer failed; preserved work for review: {work}") from error
        # Keep ancillary scorer plots in the reviewed work directory, not shared result paths.
        work.rename(work.with_name(work.name.replace(prefix, f".{model}.physiq.complete.", 1)))
        print(f"MODEL_PHYSIQ198_66_PASS {model}", flush=True)


def evaluate_model(root, model, poll_seconds=60, sleep=time.sleep, run=subprocess.run,
                   decode=check_video, bench=BENCH):
    jobs, groups, cases = load_jobs(root, model)
    wait_for_outputs(jobs, poll_seconds, sleep=sleep)
    decode_model(root, model, jobs, decode=decode)
    score_gtref(root, model, cases, run=run)
    score_physics(root, model, groups["physiq"], bench=bench, run=run, decode=decode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="m_full1_a03,m_full1_a07", help="Comma-separated model order")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    models = args.models.split(",")
    require(models and len(models) == len(set(models)), "Provide unique nonempty models")
    require(args.poll_seconds > 0 and math.isfinite(args.poll_seconds), "Polling interval must be positive and finite")
    for model in models:
        evaluate_model(args.root.resolve(), model, args.poll_seconds)
    print("CPU_EVAL_AND_DECODE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
