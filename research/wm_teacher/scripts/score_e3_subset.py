#!/usr/bin/env python3
"""Strict fixed-C diagnostic scoring, not full-benchmark acceptance or SOTA.

No action starts by default. GPU actions are one VBench dimension or WMB only,
require explicit authorization, and must run under the experiment supervisor.
All video/GT staging is byte copied to launcher-local storage with streaming SHA256;
no shared-filesystem mmap, changed metrics, new prompts, or base re-scoring.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from score_merge_cpu import require, finite, read_json, check_video, prepare_physics
from wmb_merge_score import validate_result as validate_wmb
from summarize_eval import DIMS, total
from analyze_merge_metrics import original_score, load_iqtable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = ROOT / "eval/e3_subset_20261025"
VBREPO = Path("/dockerdata/wm_eval/VBench")
BENCH = Path("/dockerdata/wm_eval/bench")
COUNTS = {"vbench": 71, "vbench_po": 71, "wm": 22, "physiq": 42, "wmb": 70}
VIEWS = {"perspective-left", "perspective-center", "perspective-right"}
GT_KEYS = ("dreamsim", "dreamsim_last", "motion_ratio", "dur")
CAVEAT = ("Fixed C subset diagnostic only; neither full-benchmark acceptance nor SOTA. "
          "Paired case/complete three-view scene percentile bootstrap; no training, "
          "generation-seed, source-cluster or judge variability; comparisons unadjusted. "
          "VBench Total is an aggregate of dimension-specific applicable sets, never a per-case metric.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact(actual, expected, label):
    names = list(actual)
    require(len(names) == len(set(names)) and set(names) == set(expected),
            f"{label}: duplicate/missing/extra IDs; missing={sorted(set(expected) - set(names))}, "
            f"extra={sorted(set(names) - set(expected))}")


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"Blank JSONL row {path}:{number}")
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    require(key not in result, f"Duplicate JSON key {path}:{number}: {key}")
                    result[key] = value
                return result
            row = json.loads(line, object_pairs_hook=unique)
            require(isinstance(row, dict), f"Non-object row {path}:{number}")
            rows.append(row)
    return rows


def write_new(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"Preserve existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name + ".", delete=False) as h:
        temporary = Path(h.name)
        json.dump(value, h, indent=2, allow_nan=False)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.link(temporary, path)
    temporary.unlink()


def verified_copy(source, destination, expected_sha=None):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and source.stat().st_size > 0, f"Missing/empty source: {source}")
    before = sha256(source)
    if expected_sha is not None:
        require(before == expected_sha, f"Source SHA256 mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.is_symlink(), f"Local stage must not be a symlink: {destination}")
    if destination.exists():
        require(sha256(destination) == before, f"Local stage SHA256 mismatch: {destination}")
    else:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name + ".", delete=False) as h:
            temporary = Path(h.name)
        shutil.copyfile(source, temporary)
        require(sha256(temporary) == before and sha256(source) == before,
                f"Source changed or failed byte-copy: {source}; preserve {temporary}")
        os.link(temporary, destination)
        temporary.unlink()
    require(sha256(source) == before, f"Source changed during staging: {source}")
    return before


def physics_scenes(names):
    groups = {}
    for name in names:
        parts = Path(name).name.split("_")
        require(len(parts) == 3 and parts[1] in VIEWS and parts[-1].endswith(".mp4"),
                f"Invalid original Physics filename: {name}")
        groups.setdefault(parts[-1], []).append(parts[1])
    for scene, views in groups.items():
        exact(views, VIEWS, f"Physics complete three-view scene {scene}")
    return sorted(groups)


def filter_full_info(info, ids):
    require(isinstance(info, list), "VBench full_info must be a list")
    names = [row["prompt_en"] for row in info]
    require(len(names) == len(set(names)), "Duplicate official full_info prompt IDs")
    require(set(ids) <= set(names), "Selected IDs missing from official full_info")
    return [dict(row) for row in info if row["prompt_en"] in set(ids)]


def applicable(info):
    return {dim: {row["prompt_en"] + "-0.mp4" for row in info if dim in row["dimension"]} for dim in DIMS}


def dimension_rows(result, dim, expected, full_expected=None):
    require(isinstance(result, dict) and dim in result, f"Missing VBench dimension {dim}")
    pair = result[dim]
    require(isinstance(pair, list) and len(pair) == 2 and finite(pair[0]) and isinstance(pair[1], list),
            f"Invalid/nonfinite VBench aggregate {dim}")
    rows = pair[1]
    names = [Path(row["video_path"]).name for row in rows]
    exact(names, full_expected if full_expected is not None else expected, dim)
    require(all(finite(row.get("video_results")) or
                dim == "dynamic_degree" and isinstance(row.get("video_results"), bool) for row in rows),
            f"Missing/nonfinite VBench per-video {dim}")
    selected = {Path(row["video_path"]).name: dict(row) for row in rows if Path(row["video_path"]).name in expected}
    exact(selected, expected, dim + " selected")
    return selected


def pool_dimension(dim, rows, frame_counts=None, ids=None):
    chosen = [rows[key] for key in (ids if ids is not None else sorted(rows))]
    require(bool(chosen), f"Empty applicable set for {dim}")
    scores = [float(row["video_results"]) for row in chosen]
    require(all(math.isfinite(v) for v in scores), f"Nonfinite resample {dim}")
    if dim == "background_consistency":
        require(all(finite(row.get("video_sim")) and finite(row.get("cnt_per_video")) and
                    row["cnt_per_video"] > 0 for row in chosen), "Background pooling requires original video_sim/count")
        require(all(math.isclose(row["video_sim"] / row["cnt_per_video"], row["video_results"],
                                 rel_tol=1e-8, abs_tol=1e-8) for row in chosen), "Background numerator/count mismatch")
        return sum(row["video_sim"] for row in chosen) / sum(row["cnt_per_video"] for row in chosen)
    if dim == "subject_consistency":
        require(frame_counts is not None, "Subject pooling needs full-decoded frame counts")
        weights = [frame_counts[Path(row["video_path"]).name] - 1 for row in chosen]
        require(all(isinstance(w, int) and w > 0 for w in weights), "Invalid subject frame counts")
        return sum(v * w for v, w in zip(scores, weights)) / sum(weights)
    return sum(scores) / len(scores) / (100 if dim == "imaging_quality" else 1)


def filter_wmb(data, expected, full_expected=None):
    # validate_result is unchanged; it checks all eight responses and exact aggregate provenance.
    require(isinstance(data, dict) and isinstance(data.get("preds"), dict), "Malformed WMB predictions")
    exact(data["preds"], full_expected if full_expected is not None else expected, "WMB source")
    pools = {"instruction": 1, "physical_laws": 5, "common_sense": 2}
    source_accs = {key: [] for key in pools}
    for row in data["preds"].values():
        for key, count in pools.items():
            preds = row.get(key)
            require(isinstance(preds, list) and len(preds) == count and
                    all(isinstance(p, str) and p.strip() for p in preds), f"WMB missing {key} questions")
            for pred in preds:
                if key == "instruction":
                    try:
                        value = float(pred.split(":")[-1].strip(" ."))
                    except ValueError:
                        value = 0.0
                    require(finite(value) and 0 <= value <= 3, "WMB instruction out of range/nonfinite")
                else:
                    value = "no" in pred.lower()
                source_accs[key].append(value)
    require(data.get("accs") == source_accs, "WMB stored accs differ from original reward parsing")
    selected = {key: data["preds"][key] for key in data["preds"] if key in expected}
    exact(selected, expected, "WMB selected")
    result = {"model_name": data.get("model_name"), "preds": selected, "accs": {key: [] for key in pools}}
    names = list(data["preds"])
    for key, width in pools.items():
        for index, name in enumerate(names):
            if name in expected:
                result["accs"][key].extend(source_accs[key][index * width:(index + 1) * width])
    return result


def wmb_vectors(data, ids):
    filtered = filter_wmb(data, set(ids))
    order = {name: i for i, name in enumerate(filtered["preds"])}
    return [sum((filtered["accs"][key][order[name] * width:(order[name] + 1) * width]
                 for key, width in (("instruction", 1), ("physical_laws", 5), ("common_sense", 2))), [])
            for name in ids]


def validate_physics_rows(rows, expected, iq_class):
    exact([row.get("scenario") for row in rows], expected, "Physics CSV")
    lists = iq_class.get_list_columns()
    scalars = [f"{key}_{view}" for key in iq_class.get_scalar_keys() for view in iq_class.views]
    lengths = {key: set() for key in lists}
    for row in rows:
        require(set(lists + scalars) <= set(row), "Physics missing original metric/view columns")
        for key in lists:
            try:
                values = json.loads(row[key]) if isinstance(row[key], str) else row[key]
            except (ValueError, TypeError) as error:
                raise RuntimeError(f"Malformed Physics list {key}") from error
            require(isinstance(values, list) and values and all(finite(v) for v in values),
                    f"Missing/nonfinite Physics list {key}")
            lengths[key].add(len(values))
        for key in scalars:
            try:
                value = float(row[key])
            except (ValueError, TypeError) as error:
                raise RuntimeError(f"Malformed Physics scalar {key}") from error
            require(math.isfinite(value), f"Nonfinite Physics scalar {key}")
    require(all(len(v) == 1 for v in lengths.values()), "Physics inconsistent per-view frame-list lengths")


def physics_values(csv_path, expected, iq_class, metrics_path=None, full_expected=None):
    import numpy as np
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), "Duplicate Physics CSV header")
        rows = list(reader)
    validate_physics_rows(rows, full_expected if full_expected is not None else expected, iq_class)
    source = iq_class.from_csv(str(csv_path))
    source_point = float(source.compute_final_score_orig())
    if metrics_path is not None:
        stored = read_json(Path(metrics_path))
        require(finite(stored.get("final_score_orig")) and
                abs(source_point - stored["final_score_orig"]) <= 1e-8, "Physics original source metric mismatch")
    frame = source.df.set_index("scenario").loc[list(expected)].reset_index()
    selected = iq_class(frame)
    point = float(selected.compute_final_score_orig())
    columns = [selected.spatiotemporal_iou_key, selected.variance_spatiotemporal_iou_key,
               selected.spatial_iou_key, selected.variance_spatial_key, selected.weighted_spatial_iou_key,
               selected.variance_weighted_spatial_key, selected.mse_key, selected.variance_mse_key]
    values = selected.df[columns].to_numpy(dtype=float)
    require(values.shape == (len(expected), 8) and np.isfinite(values).all(), "Physics aggregate shape/nonfinite")
    require(abs(float(original_score(values.mean(axis=0))) - point) <= 1e-8, "Physics original formula mismatch")
    return values, frame, point


def interval(point, draws):
    import numpy as np
    require(finite(float(point)) and np.isfinite(draws).all(), "Nonfinite bootstrap")
    return {"delta": float(point), "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}


def paired_vbench(left, right, counts_left, counts_right, universe, bootstrap=2000, seed=20261025):
    import numpy as np
    ids = sorted(universe)
    require(bootstrap > 0 and len(ids) > 0, "Invalid bootstrap settings")
    for dim in DIMS:
        exact(left[dim], right[dim], f"Paired applicable {dim}")
        require(set(left[dim]) <= set(ids) and left[dim], f"Missing applicable {dim}")
    point_left = {dim: pool_dimension(dim, left[dim], counts_left) for dim in DIMS}
    point_right = {dim: pool_dimension(dim, right[dim], counts_right) for dim in DIMS}
    rng = np.random.default_rng(seed)
    dimension_draws = {dim: [] for dim in DIMS}
    total_draws = []
    attempts = 0
    while len(total_draws) < bootstrap:
        attempts += 1
        require(attempts <= 100 * bootstrap, "Applicable subsets too sparse for valid same-case bootstrap")
        sampled = rng.choice(ids, size=len(ids), replace=True).tolist()
        selected = {dim: [case for case in sampled if case in left[dim]] for dim in DIMS}
        if any(not rows for rows in selected.values()):
            continue  # condition on every original dimension being defined; disclosed in report.
        lv = {dim: pool_dimension(dim, left[dim], counts_left, selected[dim]) for dim in DIMS}
        rv = {dim: pool_dimension(dim, right[dim], counts_right, selected[dim]) for dim in DIMS}
        for dim in DIMS:
            dimension_draws[dim].append(lv[dim] - rv[dim])
        total_draws.append(total(lv)[0] - total(rv)[0])
    return {"n_by_dim": {dim: len(left[dim]) for dim in DIMS}, "left": point_left, "right": point_right,
            "aggregate_only_total": {"left": total(point_left)[0], "right": total(point_right)[0],
                                     **interval(total(point_left)[0] - total(point_right)[0], total_draws)},
            "paired": {dim: interval(point_left[dim] - point_right[dim], dimension_draws[dim]) for dim in DIMS},
            "bootstrap_unit": "same case sample across every dimension, filtered to its official applicable IDs",
            "rejected_empty_dimension_draws": attempts - bootstrap}


def decode_local(path, generated=True):
    if any(importlib.util.find_spec(name) is None for name in ("av", "decord")):
        python = "/dockerdata/venv_vbench/bin/python"
        require(Path(sys.executable).absolute() != Path(python), "Original VBench decoder environment is incomplete")
        wrapper = ("import json,sys; sys.path.insert(0,sys.argv[1]); from score_e3_subset import decode_local; "
                   "print(json.dumps(decode_local(sys.argv[2],generated=sys.argv[3]=='true'),allow_nan=False))")
        result = subprocess.run([python, "-B", "-c", wrapper, str(Path(__file__).resolve().parent), str(path),
                                 "true" if generated else "false"],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
                                check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        require(isinstance(report, dict), "Original CPU decoder returned a non-object report")
        return report
    import av
    require(Path(path).is_file() and Path(path).stat().st_size > 0, f"Missing/empty video {path}")
    before = sha256(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        require(finite(fps) and fps > 0, f"Invalid fps {path}")
        width, height, count = stream.width, stream.height, 0
        for frame in container.decode(video=0):
            require((frame.width, frame.height) == (width, height), f"Changed frame dimensions {path}")
            count += 1
    require(count > 1 and sha256(path) == before, f"Incomplete decode/source changed {path}")
    from decord import VideoReader
    require(len(VideoReader(str(path), num_threads=1)) == count,
            f"Official VBench Decord/PyAV full frame counts differ: {path}")
    if generated:
        require(count == 121 and fps == 24 and (width, height) == (1280, 704),
                f"Expected generated 121f@24 1280x704, decoded {count}f@{fps} {width}x{height}: {path}")
    return {"sha256": before, "all_frames_decoded": True, "frames": count, "fps": fps,
            "width": width, "height": height, "path": str(path)}


def score_path(experiment, suite, model, name):
    return Path(experiment) / "scores" / suite / model / name


def require_local(directory, root):
    directory = Path(directory).resolve()
    require(directory.is_absolute() and not directory.is_relative_to(root.resolve()) and
            str(directory).startswith(("/dockerdata/", "/tmp/", "/dev/shm/")),
            "Scoring work must use launcher-local /dockerdata, /tmp or /dev/shm, not shared root")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stage_jobs(jobs, suite, model, work, source="out", decoder=decode_local):
    directory = work / suite / model
    videos = directory / "videos"
    names = [job["id"] + "-0.mp4" if suite in ("vbench", "vbench_po") or suite == "wm"
             else Path(job[source]).name for job in jobs]
    exact(names, set(names), "Stage filenames")
    require(not videos.exists() or {p.name for p in videos.glob("*.mp4")} <= set(names),
            f"Unexpected staged videos: {videos}")
    outputs = {}
    for job, name in zip(jobs, names):
        destination = videos / name
        digest = verified_copy(job[source], destination, job.get(source + "_sha256"))
        report = decoder(destination, generated=source == "out")
        require(report.get("sha256") == digest and report.get("all_frames_decoded") is True and
                isinstance(report.get("frames"), int) and report["frames"] > 1 and
                finite(report.get("fps")) and report["fps"] > 0,
                f"Decoder hash/completion/format mismatch: {destination}")
        if source == "out":
            require((report["frames"], report["fps"], report.get("width"), report.get("height")) == (121, 24, 1280, 704),
                    f"Generated output has wrong full-decoded format: {destination}")
        outputs[name] = {**report, "source": job[source]}
    exact(outputs, names, "Full decode report")
    manifest = directory / "decode.json"
    result = {"suite": suite, "model": model, "all_frames_decoded": True, "outputs": outputs}
    if manifest.exists():
        require(read_json(manifest) == result, f"Changed staged decode manifest: {manifest}")
    else:
        write_new(manifest, result)
    return videos, outputs


def ledger_budget(ledger, args):
    import run_e3_subset_completion as completion
    try:
        budget, amendment_sha = completion.budget_contract(
            Path(getattr(args, "experiment", Path(args.gpu_ledger).parent)),
            getattr(args, "plan_sha256", None), getattr(args, "budget_amendment_sha256", None))
    except Exception as error:
        require(False, "Budget amendment verification failed: " + str(error))
    require(ledger.get("budget_gpu_hours") == budget and
            ledger.get("budget_amendment_sha256") == amendment_sha,
            "Ledger must enforce approved gen28/score12/40 or exact authorized gen30/score12/42 amendment")
    if amendment_sha is not None:
        require(ledger.get("phase") == "score" and ledger.get("plan_sha256") == args.plan_sha256 and
                ledger.get("score_source_sha256") == getattr(args, "score_source_sha256", None) == sha256(__file__),
                "Amended ledger phase/plan/source identity mismatch")
    return budget


def gpu_run(command, args, cwd=None, env=None):
    require(args.gpu_authorized and args.supervisor_pid and args.gpu_ledger,
            "GPU scoring requires --gpu-authorized, live --supervisor-pid, and shared --gpu-ledger")
    require(args.supervisor_pid > 1 and args.supervisor_pid != os.getpid(), "Supervisor must be a distinct live process")
    os.kill(args.supervisor_pid, 0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(len(visible) == 1 and visible[0].strip() not in ("", "-1"),
            "Expose exactly one scoring GPU; no automatic GPU batch")
    require(type(args.occupied_gpus) is int and args.occupied_gpus > 0,
            "Explicit --occupied-gpus must count all allocated/reserved GPUs, including hidden idle devices")
    require(finite(args.max_gpu_seconds) and 0 < args.max_gpu_seconds <= 12 * 3600,
            "--max-gpu-seconds must explicitly bound a single supervised score action")
    ledger_path = Path(args.gpu_ledger)
    require(ledger_path.is_file(), "Supervisor must initialize the shared GPU ledger")
    with ledger_path.open("r+") as h:
        fcntl.flock(h, fcntl.LOCK_EX)
        ledger = json.load(h)
        budget = ledger_budget(ledger, args)
        gen, scoring = ledger.get("generation_gpu_hours"), ledger.get("score_gpu_hours")
        require(finite(gen) and finite(scoring) and 0 <= gen <= budget["generation"] and 0 <= scoring <= budget["score"],
                "Missing/invalid recorded GPU usage")
        require(not ledger.get("active"), "Another occupied/reserved GPU operation is active")
        runs = ledger.get("runs")
        require(isinstance(runs, list) and all(r.get("phase") == "score" and
                type(r.get("gpus")) is int and r["gpus"] > 0 and finite(r.get("used_gpu_hours")) and
                r["used_gpu_hours"] >= 0 for r in runs) and
                math.isclose(scoring, sum(r["used_gpu_hours"] for r in runs), abs_tol=1e-8),
                "Invalid/unaccounted score cost history")
        reserve = args.occupied_gpus * (args.max_gpu_seconds + 15) / 3600
        require(scoring + reserve <= budget["score"] and gen + scoring + reserve <= budget["total"], "Insufficient remaining GPU budget")
        token = f"score-{os.getpid()}-{time.time_ns()}"
        ledger["active"] = {"token": token, "phase": "score", "gpus": args.occupied_gpus, "reserved_gpu_hours": reserve,
                            "pid": os.getpid(), "supervisor_pid": args.supervisor_pid, "command": command,
                            "start_epoch": time.time()}
        # Reserve before allocating the GPU. A killed scorer remains charged at its full reservation.
        ledger["score_gpu_hours"] += reserve
        h.seek(0); json.dump(ledger, h, indent=2, allow_nan=False); h.truncate(); h.flush(); os.fsync(h.fileno())
    started, succeeded = time.monotonic(), False
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env)
        try:
            while process.poll() is None:
                os.kill(args.supervisor_pid, 0)
                require(time.monotonic() - started < args.max_gpu_seconds, "Supervised GPU score wall limit exceeded")
                time.sleep(min(1, args.max_gpu_seconds))
            require(process.returncode == 0, f"Original GPU scorer exited {process.returncode}")
            succeeded = True
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    finally:
        used = args.occupied_gpus * (time.monotonic() - started) / 3600
        with ledger_path.open("r+") as h:
            fcntl.flock(h, fcntl.LOCK_EX)
            ledger = json.load(h)
            require(ledger.get("active", {}).get("token") == token, "Supervisor ledger changed; preserve reservation")
            ledger["score_gpu_hours"] -= reserve - used
            ledger.setdefault("runs", []).append({**ledger["active"], "used_gpu_hours": used,
                                                  "success": succeeded, "end_epoch": time.time()})
            ledger["active"] = None
            h.seek(0); json.dump(ledger, h, indent=2, allow_nan=False); h.truncate(); h.flush(); os.fsync(h.fileno())
            require(ledger_budget(ledger, args) == budget, "GPU budget contract changed during scoring")
            require(ledger["score_gpu_hours"] <= budget["score"] and
                    ledger["generation_gpu_hours"] + ledger["score_gpu_hours"] <= budget["total"],
                    "GPU budget exceeded; actual full occupancy retained in ledger")


def run_vbench(args, jobs, info, suite, model):
    videos, outputs = stage_jobs(jobs, suite, model, args.work_dir)
    selected = filter_full_info(info, [job["id"] for job in jobs])
    if suite == "wm":
        from PIL import Image
        import av
        for row, job in zip(sorted(selected, key=lambda row: row["prompt_en"]), sorted(jobs, key=lambda row: row["id"])):
            require(row["prompt_en"] == job["id"], "WM full_info/job mismatch")
            gt = args.work_dir / "wm_gt" / (job["id"] + ".mp4")
            verified_copy(job["gt"], gt, job.get("gt_sha256"))
            image = videos.parent / "images" / (job["id"] + ".png")
            image.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(gt)) as container:
                first = Image.fromarray(next(container.decode(video=0)).to_ndarray(format="rgb24"))
            if image.exists():
                with Image.open(image) as existing:
                    require(existing.size == first.size and existing.convert("RGB").tobytes() == first.tobytes(),
                            f"Existing WM conditioning frame differs from GT: {image}")
            else:
                first.save(image)
            row["custom_image_path"] = str(image)
    fi = videos.parent / (args.dim + "_full_info.json")
    write_new(fi, selected)
    output = score_path(args.experiment, suite, model, "unused").parent
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / (model + "__" + args.dim + "_eval_results.json")
    require(not result_path.exists(), f"Preserve existing result {result_path}")
    wrapper = ("import sys,torch; sys.path.insert(0,sys.argv[1]); from vbench2_beta_i2v import VBenchI2V; "
               "v=VBenchI2V(torch.device('cuda'),sys.argv[2],sys.argv[3]); "
               "v.evaluate(videos_path=sys.argv[4],name=sys.argv[5],dimension_list=[sys.argv[6]],"
               "resolution='16-9',mode='vbench_standard')")
    gpu_run([args.vbench_python, "-u", "-c", wrapper, str(args.vbench_repo), str(fi), str(output),
             str(videos), model + "__" + args.dim, args.dim], args, cwd=args.vbench_repo)
    rows = dimension_rows(read_json(result_path), args.dim, applicable(selected)[args.dim])
    counts = {name: row["frames"] for name, row in outputs.items()}
    aggregate = pool_dimension(args.dim, rows, counts)
    require(abs(aggregate - read_json(result_path)[args.dim][0]) <= 1e-8, "Official VBench point/pooling mismatch")


def run_wmb(args, jobs, model):
    videos, _ = stage_jobs(jobs, "wmb", model, args.work_dir)
    official = read_json(args.bench / "WorldModelBench/worldmodelbench.json")
    expected = {Path(job["out"]).stem for job in jobs}
    exact([Path(item["first_frame"]).stem for item in official],
          {Path(item["first_frame"]).stem for item in official}, "Official WMB benchmark")
    selected = [item for item in official if Path(item["first_frame"]).stem in expected]
    exact([Path(item["first_frame"]).stem for item in selected], expected, "WMB question pool subset")
    work = videos.parent / "judge"
    work.mkdir()
    write_new(work / "worldmodelbench.json", selected)
    save = work / "result"
    env = dict(os.environ, PYTHONPATH=str(args.bench / "VILA") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    gpu_run([args.wmb_python, str(args.bench / "WorldModelBench/evaluation.py"), "--judge", args.judge,
             "--video_dir", str(videos), "--model_name", model, "--save_name", str(save)], args, cwd=work, env=env)
    result = validate_wmb(save.with_suffix(".json"), expected)
    write_new(score_path(args.experiment, "wmb", model, "result.json"), filter_wmb(result, expected))


def run_gtref(args, jobs, model):
    import numpy as np
    import torch
    sys.path.insert(0, str(args.vbench_repo))
    module = importlib.import_module("gt_ref_metrics")
    require(module.K == 12, "Original GT sampler K must remain 12")
    videos, _ = stage_jobs(jobs, "wm", model, args.work_dir)
    from dreamsim import dreamsim
    os.chdir(args.vbench_repo)
    network, preprocess = dreamsim(pretrained=True, device="cpu", cache_dir="./models")
    result = {}
    for job in jobs:
        gt = args.work_dir / "wm_gt" / (job["id"] + ".mp4")
        verified_copy(job["gt"], gt, job.get("gt_sha256"))
        gf, gfps = module.read(str(gt))
        vf, vfps = module.read(str(videos / (job["id"] + "-0.mp4")))
        duration = min(len(gf) / gfps, len(vf) / vfps) - 1e-3
        a = [frame.resize((256, 144)) for frame in module.sample(gf, gfps, duration)]
        b = [frame.resize((256, 144)) for frame in module.sample(vf, vfps, duration)]
        with torch.no_grad():
            distances = network(torch.cat([preprocess(frame) for frame in a]),
                                torch.cat([preprocess(frame) for frame in b])).flatten().tolist()
        require(len(distances) == 12 and all(finite(value) for value in distances), "Invalid original DreamSim12")
        result[job["id"]] = {"dataset": job.get("dataset", "unknown"), "dreamsim": float(np.mean(distances)),
                             "dreamsim_last": distances[-1], "motion_ratio": abs(math.log(module.motion(b) / module.motion(a))),
                             "dur": duration}
    validate_gtref(result, {job["id"] for job in jobs})
    write_new(score_path(args.experiment, "wm", model, "gtref.json"), result)


def validate_gtref(result, expected, full_expected=None):
    require(isinstance(result, dict), "GTref must be a per-case dict")
    exact(result, full_expected if full_expected is not None else expected, "GTref exact coverage")
    require(all(isinstance(row, dict) and all(finite(row.get(key)) for key in GT_KEYS) for row in result.values()),
            "Missing/nonfinite GTref per-case values")
    selected = {name: result[name] for name in sorted(expected) if name in result}
    exact(selected, expected, "GTref selected")
    return selected


def run_physics(args, jobs, model):
    sys.path.insert(0, str(args.bench / "physics-IQ-benchmark"))
    from physiq.binary_mask_generator import generate_binary_masks
    from physiq.calculate_and_write_metrics_to_csv import process_videos
    scenes = physics_scenes([Path(job["out"]).name for job in jobs])
    videos, _ = stage_jobs(jobs, "physiq", model, args.work_dir)
    padded = videos.parent / "videos_5s"
    def decode_padded(path, frames=120, height=720):
        row = decode_local(path, generated=False)
        require((row["frames"], row["fps"], row["width"], row["height"]) == (frames, 24, 1280, height),
                f"Wrong original Physics padded full-decode format: {path}")
        return str(path)
    prepare_physics(videos, list(videos.glob("*.mp4")), padded, decode=decode_padded)
    gt_source = args.bench / "physics-IQ-benchmark-data/physics-IQ-benchmark/split-videos/testing/24FPS"
    gt_names = [path for path in gt_source.glob("*.mp4") if path.name.split("_")[-1] in scenes]
    require(len(gt_names) == 6 * len(scenes), "Physics original GT needs both takes, all three views")
    gt, real_masks, generated_masks = videos.parent / "real", videos.parent / "real_masks", videos.parent / "generated_masks"
    gt.mkdir()
    for path in gt_names:
        verified_copy(path, gt / path.name)
    # The original high-level runner requires 198; invoke its unchanged scoring primitives,
    # with only complete selected scenes in real_folder. No bypass of full-benchmark coverage.
    generate_binary_masks(str(gt), str(real_masks), is_real=True)
    generate_binary_masks(str(padded), str(generated_masks), is_real=False)
    table = videos.parent / "physics.csv"
    require(not table.exists(), f"Preserve existing Physics work {table}")
    process_videos(real_folder=str(gt), generated_folder=str(padded), binary_real_folder=str(real_masks),
                   binary_generated_folder=str(generated_masks), csv_file_path=str(table), fps=24,
                   video_time_selection="first", n_processes=0)
    iq_class = load_iqtable(args.bench / "physics-IQ-benchmark")
    _, _, point = physics_values(table, scenes, iq_class)
    destination = score_path(args.experiment, "physiq", model, "physics.csv")
    verified_copy(table, destination)
    write_new(destination.with_name("physics_metrics.json"), {"final_score_orig": point, "scene_ids": scenes,
                                                             "protocol": "original 120f@24 pad1280x720 smear crf12"})


def load_context(args):
    from prepare_e3_subset import verify_plan
    plan, verified_jobs, verified_cases = verify_plan(args.experiment / "plan.json", expected_plan_sha256=args.plan_sha256)
    require(plan["selection"]["counts"] == COUNTS, "Fixed C276 suite counts changed")
    require({key: plan["budget"][key] for key in ("total_gpu_hours", "generation_gpu_hours", "score_gpu_hours")} ==
            {"total_gpu_hours": 40, "generation_gpu_hours": 28, "score_gpu_hours": 12}, "Frozen GPU budget changed")
    jobs, cases = load_jsonl(plan["jobs_path"]), load_jsonl(plan["cases_path"])
    require(jobs == verified_jobs and cases == verified_cases, "Manifest changed after verify_plan")
    exact([j["out"] for j in jobs], [c["out"] for c in cases], "Jobs/cases outputs")
    by_out = {c["out"]: c for c in cases}
    groups = {suite: [] for suite in COUNTS}
    records = {r["path"]: r for r in plan["frozen_files"]}
    for job in jobs:
        case = by_out[job["out"]]
        require(all(case.get(key) == value for key, value in job.items() if key != "model"), "Jobs/cases parameter drift")
        require(job["model"] == plan["tag"] or
                case["suite"] == "vbench_po" and job["model"] == plan["tag"] + "_po",
                "Wrong fixed C generator arm")
        require(case["suite"] in groups, "Unknown case suite")
        require(Path(case["out"]).is_relative_to(args.experiment / "videos"), "Candidate outside private video tree")
        require(case["video"] == case["out"] and case["base_video"] in records and case["image"] in records,
                "Unfrozen source or changed video identity")
        require(case["model"] == plan["tag"] + ("_po" if case["suite"] == "vbench_po" else ""), "Wrong fixed C arm")
        require(case["ckpt"] == plan["checkpoint"]["path"], "Wrong fixed C checkpoint")
        for key in ("id", "model"):
            require(isinstance(case[key], str) and case[key] and "/" not in case[key] and "\x00" not in case[key],
                    f"Unsafe case {key}")
        row = dict(case, base_out=case["base_video"], base_out_sha256=records[case["base_video"]]["sha256"])
        if "gt" in row:
            require(row["gt"] in records, "WM GT not frozen")
            row["gt_sha256"] = records[row["gt"]]["sha256"]
        groups[case["suite"]].append(row)
    for suite, expected in COUNTS.items():
        require(len(groups[suite]) == expected, f"Wrong subset size {suite}")
        exact([c["id"] for c in groups[suite]], {c["id"] for c in groups[suite]}, suite)
    exact([c["id"] for c in groups["vbench"]], [c["id"] for c in groups["vbench_po"]], "Raw/po selected IDs")
    scenes = physics_scenes([Path(c["out"]).name for c in groups["physiq"]])
    require(len(scenes) == 14, "Physics requires 14 complete scenes")
    domains = {c["domain"] for c in groups["wmb"]}
    require(len(domains) == 7 and all(sum(c["domain"] == domain for c in groups["wmb"]) == 10
                                   for domain in domains), "WMB requires 10 fixed cases in each of seven domains")
    official_path = args.vbench_repo / "vbench2_beta_i2v/vbench2_i2v_full_info.json"
    require(str(official_path) in records and sha256(official_path) == records[str(official_path)]["sha256"],
            "VBench full_info outside frozen plan or wrong SHA256")
    info = read_json(official_path)
    raw_info = filter_full_info(info, [c["id"] for c in groups["vbench"]])
    expected = applicable(raw_info)
    require(all(expected[dim] for dim in DIMS), "Fixed subset leaves an official dimension empty")
    for c in groups["vbench"] + groups["vbench_po"]:
        official = next(row for row in raw_info if row["prompt_en"] == c["id"])
        require(set(c["dims"]) & set(DIMS) == set(official["dimension"]) & set(DIMS), "Case dimension applicability drift")
    return plan, groups, info


def suite_info(args, suite, groups, official):
    if suite in ("vbench", "vbench_po"):
        return official
    require(suite == "wm", "VBench scoring supports only raw/po/WM")
    # Exactly the original prepare_wm full_info contract; these are conditioning metadata,
    # not invented per-case scores. Image paths are overwritten with staged GT first frames.
    return [{"prompt_en": row["id"], "dimension": DIMS, "custom_image_path": row["gt"]}
            for row in groups["wm"]]


def extract_base(args, groups, official):
    require(args.suite in COUNTS, "Select one suite for historical base extraction")
    suite, jobs = args.suite, groups[args.suite]
    model = "base_po" if suite == "vbench_po" else "base"
    sources = {}
    _, decoded = stage_jobs(jobs, suite, model, args.work_dir, source="base_out")
    if suite in ("vbench", "vbench_po", "wm"):
        counts = {name: row["frames"] for name, row in decoded.items()}
        info = suite_info(args, suite, groups, official)
        selected = applicable(filter_full_info(info, [row["id"] for row in jobs]))
        if suite == "wm":
            all_cases = load_jsonl(args.root / "eval/wm_bench/cases.jsonl")
            full = {dim: {row["id"] + "-0.mp4" for row in all_cases} for dim in DIMS}
            directory = args.root / "eval/wm_bench"
        else:
            full = applicable(official)
            directory = args.root / "eval/vbench_i2v"
        for dim in DIMS:
            source = directory / "scores" / (model + "__" + dim + "_eval_results.json")
            before = sha256(source)
            rows = dimension_rows(read_json(source), dim, selected[dim], full[dim])
            point = pool_dimension(dim, rows, counts)
            result = {dim: [point, [rows[name] for name in sorted(rows)]]}
            require(sha256(source) == before, "Historical source score changed during extraction")
            sources[str(source)] = before
            write_new(score_path(args.experiment, suite, model, model + "__" + dim + "_eval_results.json"), result)
        if suite == "wm":
            source = directory / "gtref_base.json"
            before = sha256(source)
            rows = validate_gtref(read_json(source), {row["id"] for row in jobs}, {row["id"] for row in all_cases})
            require(sha256(source) == before, "Historical GTref changed during extraction")
            sources[str(source)] = before
            write_new(score_path(args.experiment, suite, model, "gtref.json"), rows)
    elif suite == "wmb":
        source = args.root / "eval/wmb/results/base.json"
        before = sha256(source)
        expected = {Path(row["base_out"]).stem for row in jobs}
        full = {Path(row["first_frame"]).stem for row in read_json(args.bench / "WorldModelBench/worldmodelbench.json")}
        result = filter_wmb(read_json(source), expected, full)
        require(sha256(source) == before, "Historical WMB changed during extraction")
        sources[str(source)] = before
        write_new(score_path(args.experiment, suite, model, "result.json"), result)
    else:
        iq_class = load_iqtable(args.bench / "physics-IQ-benchmark")
        prefix = args.root / "eval/physiq/results/physics-IQ-benchmark/results/base-op-run_01"
        table, metrics = prefix.with_suffix(".csv"), prefix.with_name(prefix.name + "_metrics.json")
        with (args.bench / "physics-IQ-benchmark/descriptions/descriptions_original.csv").open(newline="") as h:
            descriptions = list(csv.DictReader(h))
        full_names = [r["generated_video_name"] for r in descriptions if "take-2" not in r["scenario"]]
        full_scenes = physics_scenes(full_names)
        scenes = physics_scenes([Path(row["base_out"]).name for row in jobs])
        for path in [table, metrics]:
            sources[str(path)] = sha256(path)
        _, _, point = physics_values(table, scenes, iq_class, metrics, full_scenes)
        with table.open(newline="") as h:
            reader = csv.DictReader(h)
            fields, rows = reader.fieldnames, list(reader)
        selected = [row for row in rows if row["scenario"] in scenes]
        validate_physics_rows(selected, scenes, iq_class)
        destination = score_path(args.experiment, suite, model, "physics.csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=fields)
            writer.writeheader(); writer.writerows(selected)
        write_new(destination.with_name("physics_metrics.json"), {"final_score_orig": point})
        require(all(sha256(Path(path)) == digest for path, digest in sources.items()), "Historical Physics source changed")
    write_new(score_path(args.experiment, suite, model, "provenance.json"),
              {"kind": "exact existing historical per-case filtering and original reaggregation; no base rescoring",
               "source_sha256": sources, "plan_sha256": sha256(args.experiment / "plan.json")})


def summary(args, plan, groups, official):
    import numpy as np
    require(args.bootstrap > 0 and args.seed >= 0, "Invalid bootstrap configuration")
    report = {"schema": "e3_subset_scores_v1", "caveat": CAVEAT, "checkpoint": plan["checkpoint"],
              "subset_counts": COUNTS, "bootstrap": args.bootstrap, "seed": args.seed,
              "plan_sha256": sha256(args.experiment / "plan.json"), "created_epoch": time.time(),
              "not_full_acceptance": True, "not_sota": True, "vbench": {}, "source_sha256": {}}
    sources = report["source_sha256"]
    def read_score(path):
        sources[str(path)] = sha256(path)
        return read_json(path)
    all_decoded = {}
    for suite, jobs in groups.items():
        model = plan["tag"] + ("_po" if suite == "vbench_po" else "")
        base = "base_po" if suite == "vbench_po" else "base"
        expected_names = {row["id"] + "-0.mp4" if suite in ("vbench", "vbench_po", "wm")
                          else Path(row["out"]).name for row in jobs}
        for arm in (model, base):
            decoded = read_score(args.work_dir / suite / arm / "decode.json")
            exact(decoded["outputs"], expected_names, f"Summary all-suite decode coverage {suite}/{arm}")
            require(decoded.get("suite") == suite and decoded.get("model") == arm and
                    decoded.get("all_frames_decoded") is True, "Summary decode identity/completion mismatch")
            for row in decoded["outputs"].values():
                require(row.get("all_frames_decoded") is True and isinstance(row.get("frames"), int) and row["frames"] > 1,
                        "Invalid full-decode row")
                if arm == model:
                    require((row["frames"], row["fps"], row["width"], row["height"]) == (121, 24, 1280, 704),
                            "Candidate full-decoded format changed")
                require(sha256(row["path"]) == row["sha256"] and sha256(row["source"]) == row["sha256"],
                        "Summary stage/source video SHA256 mismatch")
                sources[row["path"]] = row["sha256"]
                sources[row["source"]] = row["sha256"]
            all_decoded[(suite, arm)] = decoded
    for suite in ("vbench", "vbench_po", "wm"):
        model = plan["tag"] + ("_po" if suite == "vbench_po" else "")
        base = "base_po" if suite == "vbench_po" else "base"
        expected = applicable(filter_full_info(suite_info(args, suite, groups, official), [row["id"] for row in groups[suite]]))
        metrics, counts = {}, {}
        for arm in (model, base):
            provenance = score_path(args.experiment, suite, base, "provenance.json")
            read_score(provenance)
            decoded = all_decoded[(suite, arm)]
            counts[arm] = {name: row["frames"] for name, row in decoded["outputs"].items()}
            metrics[arm] = {}
            for dim in DIMS:
                path = score_path(args.experiment, suite, arm, arm + "__" + dim + "_eval_results.json")
                data = read_score(path)
                metrics[arm][dim] = dimension_rows(data, dim, expected[dim])
                require(abs(pool_dimension(dim, metrics[arm][dim], counts[arm]) - data[dim][0]) <= 1e-8,
                        f"Summary original VBench point mismatch: {suite}/{arm}/{dim}")
        report["vbench"][suite] = paired_vbench(metrics[model], metrics[base], counts[model], counts[base],
                                                {row["id"] + "-0.mp4" for row in groups[suite]}, args.bootstrap, args.seed)
    ids = sorted(row["id"] for row in groups["wm"])
    model = plan["tag"]
    gt = {arm: validate_gtref(read_score(score_path(args.experiment, "wm", arm, "gtref.json")), ids)
          for arm in (model, "base")}
    draws = np.random.default_rng(args.seed).integers(0, len(ids), size=(args.bootstrap, len(ids)))
    report["gtref"] = {"n_cases": len(ids), "K": 12, "case_ids": ids, "metrics": {}}
    for metric in GT_KEYS[:3]:
        left = np.asarray([gt[model][name][metric] for name in ids])
        right = np.asarray([gt["base"][name][metric] for name in ids])
        report["gtref"]["metrics"][metric] = {"left": float(left.mean()), "right": float(right.mean()),
                                               **interval((left-right).mean(), (left-right)[draws].mean(axis=1))}
    iq_class = load_iqtable(args.bench / "physics-IQ-benchmark")
    scenes = physics_scenes([Path(row["out"]).name for row in groups["physiq"]])
    piq = {}
    indexes = np.random.default_rng(args.seed).integers(0, len(scenes), size=(args.bootstrap, len(scenes)))
    for arm in (model, "base"):
        table = score_path(args.experiment, "physiq", arm, "physics.csv")
        metrics = table.with_name("physics_metrics.json")
        sources[str(table)], sources[str(metrics)] = sha256(table), sha256(metrics)
        values, frame, point = physics_values(table, scenes, iq_class, metrics)
        samples = original_score(values[indexes].mean(axis=1))
        for i in range(min(3, args.bootstrap)):
            require(abs(float(iq_class(frame.iloc[indexes[i]].reset_index(drop=True)).compute_final_score_orig()) -
                        float(samples[i])) <= 1e-8, "Physics scene-cluster bootstrap original IQTable mismatch")
        piq[arm] = (100 * point, 100 * samples)
    report["physics"] = {"n_scenes": len(scenes), "scene_ids": scenes, "views_per_scene": 3,
                         "left": piq[model][0], "right": piq["base"][0], "units": "percentage points",
                         **interval(piq[model][0]-piq["base"][0], piq[model][1]-piq["base"][1])}
    ids = sorted(Path(row["out"]).stem for row in groups["wmb"])
    world = {arm: np.asarray(wmb_vectors(read_score(score_path(args.experiment, "wmb", arm, "result.json")), ids), dtype=float)
             for arm in (model, "base")}
    domain_by_id = {Path(row["out"]).stem: row["domain"] for row in groups["wmb"]}
    report["wmb"] = {"n_cases": len(ids), "questions_per_case": 8, "case_ids": ids, "domains": {}}
    for domain in ["all"] + sorted(set(domain_by_id.values())):
        selected = [i for i, name in enumerate(ids) if domain == "all" or domain_by_id[name] == domain]
        l, r = world[model][selected], world["base"][selected]
        draws = np.random.default_rng(args.seed).integers(0, len(selected), size=(args.bootstrap, len(selected)))
        delta = l - r
        row = {"n_cases": len(selected), "left": float(l.mean(axis=0).sum()), "right": float(r.mean(axis=0).sum()),
               **interval(delta.mean(axis=0).sum(), delta[draws].mean(axis=1).sum(axis=1)),
               "eight_question_paired": [interval(delta[:, i].mean(), delta[draws, i].mean(axis=1)) for i in range(8)]}
        if domain == "all":
            report["wmb"]["aggregate"] = row
        else:
            report["wmb"]["domains"][domain] = row
    for suite in COUNTS:
        base = "base_po" if suite == "vbench_po" else "base"
        provenance = read_score(score_path(args.experiment, suite, base, "provenance.json"))
        require(provenance["plan_sha256"] == report["plan_sha256"], "Historical score extracted for another plan")
        for source, digest in provenance["source_sha256"].items():
            require(sha256(source) == digest, "Historical score source changed since extraction")
            sources[source] = digest
    require(all(sha256(path) == digest for path, digest in sources.items()), "Summary source changed during analysis")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "stage", "extract-base", "vbench", "wmb", "gtref", "physics", "summary"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--plan-sha256", help="Externally pinned plan SHA256; otherwise reads frozen plan.sha256")
    parser.add_argument("--suite", choices=list(COUNTS))
    parser.add_argument("--source", choices=["out", "base_out"], default="out")
    parser.add_argument("--dim", choices=DIMS)
    parser.add_argument("--work-dir", type=Path, default=Path("/dockerdata/wm_eval/e3_subset_20261025"))
    parser.add_argument("--vbench-repo", type=Path, default=VBREPO)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--vbench-python", default="/dockerdata/venv_vbench/bin/python")
    parser.add_argument("--wmb-python", default="/dockerdata/conda_vila/bin/python")
    parser.add_argument("--judge", default="/dockerdata/wm_eval/bench/judges/vila-ewm-qwen2-1.5b")
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--gpu-ledger", type=Path)
    parser.add_argument("--approved-budget-amendment-sha256", dest="budget_amendment_sha256")
    parser.add_argument("--supervisor-pid", type=int)
    parser.add_argument("--max-gpu-seconds", type=float)
    parser.add_argument("--occupied-gpus", type=int, help="All allocated/reserved GPUs, not only CUDA-visible devices")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action not in ("vbench", "wmb"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.umask(0o077)
    args.root, args.experiment = args.root.resolve(), args.experiment.resolve()
    require(args.experiment == args.root / "eval/e3_subset_20261025", "Use only private fixed C276 experiment tree")
    plan, groups, official = load_context(args)
    if args.action == "validate":
        print("FIXED_C276_PLAN_COUNTS_HASHES_PASS; diagnostic only; no GPU allocation")
        return
    args.work_dir = require_local(args.work_dir, args.root)
    if args.action == "summary":
        destination = args.out or args.experiment / "summary.json"
        require(destination.resolve().is_relative_to(args.experiment), "Summary must remain in private experiment")
        write_new(destination, summary(args, plan, groups, official))
    else:
        suite = args.suite
        require(suite is not None, "Choose exactly one --suite; no automatic suite or GPU batches")
        model = plan["tag"] + ("_po" if suite == "vbench_po" else "")
        jobs = groups[suite]
        if args.action == "stage":
            stage_jobs(jobs, suite, ("base_po" if suite == "vbench_po" else "base") if args.source == "base_out" else model,
                       args.work_dir, source=args.source)
        elif args.action == "extract-base":
            extract_base(args, groups, official)
        elif args.action == "vbench":
            require(suite in ("vbench", "vbench_po", "wm") and args.dim, "Select exactly one applicable VBench dimension")
            run_vbench(args, jobs, suite_info(args, suite, groups, official), suite, model)
        elif args.action == "wmb":
            require(suite == "wmb", "WMB action requires --suite wmb")
            run_wmb(args, jobs, model)
        elif args.action == "gtref":
            require(suite == "wm", "GTref action requires --suite wm")
            run_gtref(args, jobs, model)
        elif args.action == "physics":
            require(suite == "physiq", "Physics action requires --suite physiq")
            run_physics(args, jobs, model)
    print(f"SUBSET_{args.action.upper()}_COMPLETE; not full acceptance or SOTA", flush=True)


if __name__ == "__main__":
    main()
