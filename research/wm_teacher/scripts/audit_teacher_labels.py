#!/usr/bin/env python3
"""Private CPU-only frozen-window decoder precheck; never a semantic label review."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import random
import signal
import sys
import time
import uuid

ROOT = Path("/srv/wm_dataset")
MANIFEST_REL = "eval/diag/teacher_label_audit_L1_20261019.json"
CSV_REL = "training_metadata/ft2/ft2_train.csv"
MANIFEST_SHA = "267a9ff4c1f318c25758e2ffd97d70e49b8ec2cfe314668501cfa8d7f41391ab"
CSV_SHA = "2fd1615c3c538d9d9462ad166b78129f8c349174bffd5ce1085888d5f001a0a6"
POPULATION = 32365
DATASETS = 17
SAMPLE_PER_DATASET = 20
SEED = 20261019
DESIGN = "dataset_stratified_equal_probability_window_sampling_without_replacement"


class DeadlineExceeded(Exception):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("invalid ROOT-relative path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != relative:
        raise ValueError("unsafe ROOT-relative path")
    resolved = (Path(root) / relative).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("path escapes ROOT through symlink")
    return resolved


def dataset_name(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) < 4 or parts[:2] != ("dataset", "wan22_5s") or not parts[2].endswith("_5s"):
        raise ValueError("unexpected original CSV video layout")
    return parts[2][:-3]


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def validate_inputs(root):
    root = Path(root).resolve()
    manifest_path = safe_path(root, MANIFEST_REL)
    csv_path = safe_path(root, CSV_REL)
    if sha256(manifest_path) != MANIFEST_SHA or sha256(csv_path) != CSV_SHA:
        raise ValueError("frozen input SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["design"] != DESIGN or manifest["rng_seed"] != SEED
            or manifest["csv_sha256"] != CSV_SHA or manifest["population_windows"] != POPULATION
            or manifest["semantic_review"] != "pending" or manifest["media_uploaded"] is not False
            or manifest["parent_source_independence"] != "NOT_CLAIMED; possible duplicates/UNKNOWN"):
        raise ValueError("frozen design or non-semantic state mismatch")
    rows = []
    groups = defaultdict(list)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["video", "prompt"]:
            raise ValueError("unexpected original CSV columns")
        while True:
            physical_line = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            if None in row or None in row.values():
                raise ValueError("malformed original CSV record")
            index = len(rows)
            rows.append((row, physical_line))
            groups[dataset_name(row["video"])].append(index)
    if len(rows) != POPULATION or len(groups) != DATASETS or set(groups) != set(manifest["sources"]):
        raise ValueError("frozen population mismatch")
    expected = []
    rng = random.Random(SEED)
    for dataset in sorted(groups):
        population = len(groups[dataset])
        size = min(SAMPLE_PER_DATASET, population)
        source = manifest["sources"][dataset]
        probability = size / population
        if (source["population_windows"] != population or source["sample_windows"] != size
                or source["inclusion_probability"] != probability):
            raise ValueError("source population or inclusion probability mismatch")
        expected.extend((dataset, index) for index in sorted(rng.sample(sorted(groups[dataset]), size)))
    cases = manifest["cases"]
    if (manifest["sample_windows"] != len(expected) or len(cases) != len(expected)
            or [(c["dataset"], c["csv_record_index0"]) for c in cases] != expected):
        raise ValueError("fixed RNG sample/order does not reproduce")
    identifiers = set()
    for case in cases:
        index = case["csv_record_index0"]
        row, physical_line = rows[index]
        prompt_sha = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        if (case["audit_id"] != f'{case["dataset"]}_{index}' or case["audit_id"] in identifiers
                or case["csv_physical_line"] != physical_line
                or case["video"] != row["video"] or case["prompt"] != row["prompt"]
                or case["prompt_sha256"] != prompt_sha
                or case["inclusion_probability"] != manifest["sources"][case["dataset"]]["inclusion_probability"]
                or case["semantic_review"] != "pending" or case["decode_review"] != "pending"
                or case["canonical_parent"] != "UNRESOLVED"):
            raise ValueError("frozen case provenance mismatch")
        safe_path(root, case["video"])
        identifiers.add(case["audit_id"])
    if sha256(manifest_path) != MANIFEST_SHA or sha256(csv_path) != CSV_SHA:
        raise ValueError("inputs changed during validation")
    return manifest


def initial_result(case):
    return {key: case[key] for key in ("audit_id", "dataset", "csv_record_index0", "csv_physical_line",
                                      "video", "prompt_sha256", "inclusion_probability")} | {
        "semantic_review": "pending", "semantic_label": "NOT_REVIEWED",
        "canonical_parent": "UNKNOWN", "source_identity_status": "UNKNOWN",
        "parent_source_independence": "NOT_CLAIMED", "replacement": False,
        "decode_status": "not_started", "contact_status": "not_started", "media_uploaded": False,
    }


def contact_sheet(path, case_id, selected):
    from PIL import Image, ImageDraw
    width, height = 320, 240
    sheet = Image.new("RGB", (width * len(selected), height + 52), "#202020")
    draw = ImageDraw.Draw(sheet)
    records = []
    for column, (requested, index, timestamp, image) in enumerate(selected):
        thumbnail = image.copy()
        thumbnail.thumbnail((width, height))
        sheet.paste(thumbnail, (column * width + (width - thumbnail.width) // 2, 0))
        actual = "NA" if timestamp is None else f"{timestamp:.3f}s"
        draw.text((column * width + 4, height + 2), f"{requested}: frame {index}, {actual}", fill="white")
        draw.text((column * width + 4, height + 22), case_id, fill="white")
        records.append({"requested": requested, "decoded_index0": index, "relative_time_seconds": timestamp})
    temporary = Path(path).with_name(Path(path).name + ".tmp." + uuid.uuid4().hex)
    sheet.save(temporary, format="JPEG", quality=85)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return {"sha256": sha256(path), "frames": records,
            "navigation_only": True, "not_evidence_of_absence": True}


def decode_case(root, case, contact_path):
    import av
    result = initial_result(case)
    started = time.monotonic()
    path = safe_path(root, case["video"])
    count = 0
    warnings = []
    pending_times = [0.0, 0.5, 1.5, 3.0]
    selected = []
    first_pts = None
    last_time = None
    last_frame = None
    dimensions = Counter()
    try:
        if not path.exists():
            result.update(decode_status="missing", contact_status="unavailable")
            return result
        if not path.is_file():
            result.update(decode_status="not_regular_file", contact_status="unavailable")
            return result
        before = path.stat()
        result["video_sha256"] = sha256(path)
        with av.open(str(path), options={"err_detect": "explode"}) as container:
            videos = list(container.streams.video)
            if not videos:
                raise ValueError("no video stream")
            stream = videos[0]
            stream.codec_context.thread_count = 1
            stream.codec_context.options = {"err_detect": "explode"}
            fps = float(stream.average_rate) if stream.average_rate is not None else None
            if fps is not None and (not math.isfinite(fps) or fps <= 0):
                fps = None
            stream_duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
            container_duration = container.duration / av.time_base if container.duration is not None else None
            result["metadata"] = {"decoder": "PyAV", "pyav_version": av.__version__,
                                  "ffmpeg_library_versions": {k: list(v) for k, v in av.library_versions.items()},
                                  "strict_err_detect": "explode", "decoder_threads": 1,
                                  "video_streams": len(videos), "codec": stream.codec_context.name,
                                  "declared_frames": stream.frames, "average_fps": fps,
                                  "time_base": str(stream.time_base), "stream_duration_seconds": stream_duration,
                                  "container_duration_seconds": container_duration,
                                  "width": stream.width, "height": stream.height}
            if len(videos) != 1:
                warnings.append("multiple_video_streams_only_first_decoded")
            for frame in container.decode(stream):
                dimensions[(frame.width, frame.height)] += 1
                if frame.pts is not None and frame.time_base is not None:
                    current = float(frame.pts * frame.time_base)
                    if first_pts is None:
                        first_pts = current
                    timestamp = current - first_pts
                    if last_time is not None and timestamp < last_time:
                        warnings.append("nonmonotonic_pts")
                else:
                    timestamp = count / fps if fps is not None else None
                    warnings.append("missing_pts_navigation_uses_fps" if fps is not None else "missing_pts_and_fps")
                take = []
                while pending_times and timestamp is not None and timestamp >= pending_times[0] - 1e-6:
                    take.append(pending_times.pop(0))
                if take:
                    image = frame.to_image()
                    image.thumbnail((320, 240))
                    for requested in take:
                        selected.append((f"{requested:g}s", count, timestamp, image.copy()))
                last_frame = frame
                last_time = timestamp
                count += 1
            result["decoded_frames"] = count
            if count == 0:
                raise ValueError("zero decoded frames")
            declared = result["metadata"]["declared_frames"]
            if declared and count != declared:
                warnings.append("decoded_frame_count_differs_from_declared")
            if count != 121:
                warnings.append("frame_count_not_121_observation_not_rejection")
            if fps is not None and abs(fps - 24.0) > 0.01:
                warnings.append("average_fps_not_24_observation_not_rejection")
            if stream_duration is not None and (stream_duration < 4.9 or stream_duration > 5.2):
                warnings.append("duration_outside_5s_reference_observation_not_rejection")
            if len(dimensions) > 1:
                warnings.append("decoded_dimensions_change")
            result["decoded_dimensions"] = [{"width": key[0], "height": key[1], "frames": value}
                                            for key, value in sorted(dimensions.items())]
            result["first_pts_seconds"] = first_pts
            result["last_relative_pts_seconds"] = last_time
            result["navigation_times_unavailable"] = pending_times
            if pending_times:
                warnings.append("requested_navigation_times_beyond_decoded_extent")
            image = last_frame.to_image()
            image.thumbnail((320, 240))
            selected.append(("last", count - 1, last_time, image))
        after = path.stat()
        result["video_sha256_after"] = sha256(path)
        result["file_unchanged"] = (before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
                                    and result["video_sha256"] == result["video_sha256_after"])
        result["file_size_bytes"] = before.st_size
        result["decode_status"] = "decoded_all_frames"
        if not result["file_unchanged"]:
            result["decode_status"] = "file_changed"
            warnings.append("original_file_changed_during_decode")
        result["full_decode_matches_declared_count"] = not declared or count == declared
        if declared and count != declared:
            result["decode_status"] = "decoded_count_mismatch"
        try:
            result["contact"] = contact_sheet(contact_path, case["audit_id"], selected)
            result["contact"]["relative_path"] = "contacts/" + Path(contact_path).name
            result["contact_status"] = "created"
        except Exception as error:
            result["contact_status"] = "failed"
            result["contact_error_type"] = type(error).__name__
            result["contact_error_detail_private"] = str(error)[:2000]
    except Exception as error:
        result["decode_status"] = "decode_failed"
        result["contact_status"] = "unavailable"
        result["error_type"] = type(error).__name__
        result["error_detail_private"] = str(error)[:2000]
        result["decoded_frames"] = count
    finally:
        result["metadata_observations"] = sorted(set(warnings))
        result["wall_seconds"] = time.monotonic() - started
    return result


def worker(task):
    os.umask(0o077)
    root, case, output = task
    try:
        result = decode_case(root, case, Path(output) / "contacts" / (case["audit_id"] + ".jpg"))
    except Exception as error:
        result = initial_result(case)
        result.update(decode_status="worker_failed", contact_status="unavailable", error_type=type(error).__name__)
    atomic_json(Path(output) / "cases" / (case["audit_id"] + ".json"), result)
    return result


def build_report(manifest, results, run, started, workers, wall_limit, termination, inputs_unchanged):
    coverage = {}
    by_dataset = defaultdict(list)
    for result in results:
        by_dataset[result["dataset"]].append(result)
    for dataset, source in sorted(manifest["sources"].items()):
        rows = by_dataset[dataset]
        coverage[dataset] = {"population_windows": source["population_windows"],
                             "sample_windows": source["sample_windows"], "results": len(rows),
                             "decode_status": dict(Counter(row["decode_status"] for row in rows)),
                             "contacts_created": sum(row["contact_status"] == "created" for row in rows),
                             "canonical_parent_unknown": sum(row["canonical_parent"] == "UNKNOWN" for row in rows),
                             "semantic_review_pending": len(rows)}
    success = inputs_unchanged and all(r["decode_status"] == "decoded_all_frames" and r["contact_status"] == "created" for r in results)
    return {"protocol": "frozen_L1_cpu_provenance_full_decode_only_v1", "run_id": Path(run).name,
            "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": MANIFEST_SHA, "csv_sha256": CSV_SHA,
            "runner_sha256": sha256(Path(__file__)), "design": manifest["design"], "rng_seed": manifest["rng_seed"],
            "frozen_sample_reproduced": True, "inputs_unchanged_after_decode": inputs_unchanged,
            "cpu_only": True, "workers": workers, "decoder_threads_per_worker": 1,
            "wall_limit_seconds": wall_limit, "termination": termination,
            "sample_windows": len(manifest["cases"]), "result_count": len(results),
            "decode_status": dict(Counter(r["decode_status"] for r in results)), "per_dataset_coverage": coverage,
            "all_frozen_cases_accounted_for": len(results) == len(manifest["cases"]),
            "decode_and_contact_complete": success, "semantic_review": "pending", "semantic_labels_assigned": 0,
            "semantic_accuracy": None, "parent_source_independence": manifest["parent_source_independence"],
            "source_identity_status": "UNKNOWN", "replacements": 0, "media_uploaded": False,
            "reference_121_frames_is_not_a_universal_acceptance_requirement": True,
            "five_frame_contacts_are_navigation_only": True,
            "dataset_weighted_semantic_error_rate": None, "cases": results}


def budget(workers, seconds):
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be 1..8")
    if not math.isfinite(seconds) or not 15 <= seconds <= 3600:
        raise ValueError("CPU wall limit must be 15..3600 seconds")


def alarm_handler(signum, frame):
    raise DeadlineExceeded()


def run_audit(root, output, workers=8, wall_limit=3600):
    budget(workers, wall_limit)
    root = Path(root).resolve()
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output = output.absolute()
    private_root = safe_path(root, "eval/diag")
    if not output.resolve().is_relative_to(private_root) or output.resolve() == private_root:
        raise ValueError("new run must be strictly inside ROOT/eval/diag")
    if output.exists() or output.is_symlink():
        raise ValueError("run directory already exists; no overwrites")
    started_clock = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    previous_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, wall_limit - 10)
    pool = None
    manifest = None
    results = {}
    termination = "completed"
    runner_error = None
    try:
        manifest = validate_inputs(root)
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        (output / "contacts").mkdir(mode=0o700)
        (output / "cases").mkdir(mode=0o700)
        atomic_json(output / "run_start.json", {"started_utc": started_utc, "workers": workers,
                    "wall_limit_seconds": wall_limit, "manifest_sha256": MANIFEST_SHA, "csv_sha256": CSV_SHA,
                    "frozen_sample_reproduced": True, "semantic_review": "pending", "media_uploaded": False,
                    "runner_sha256": sha256(Path(__file__))})
        first = [min((c for c in manifest["cases"] if c["dataset"] == dataset), key=lambda c: c["csv_record_index0"])
                 for dataset in sorted(manifest["sources"])]
        first_ids = {c["audit_id"] for c in first}
        rest = [c for c in manifest["cases"] if c["audit_id"] not in first_ids]
        pool = multiprocessing.get_context("spawn").Pool(workers)
        for phase, cases in (("first_csv_window_per_dataset", first), ("remaining_frozen_windows", rest)):
            iterator = pool.imap_unordered(worker, [(str(root), c, str(output)) for c in cases], chunksize=1)
            for result in iterator:
                results[result["audit_id"]] = result
            if phase == "first_csv_window_per_dataset":
                atomic_json(output / "first_17_contacts.json", {"selection": "minimum_frozen_csv_record_index0_per_dataset",
                    "not_selected_by_model_outputs": True, "navigation_only": True, "semantic_review": "pending",
                    "cases": [results[c["audit_id"]] for c in first]})
                made = sum(results[c["audit_id"]]["contact_status"] == "created" for c in first)
                print(f"first_contacts_ready={len(first)} contacts_created={made}", flush=True)
        pool.close()
        pool.join()
        pool = None
    except DeadlineExceeded:
        termination = "wall_deadline"
    except KeyboardInterrupt:
        termination = "interrupted"
    except Exception as error:
        if manifest is None or not output.exists():
            raise
        termination = "runner_failed"
        runner_error = {"type": type(error).__name__, "detail_private": str(error)[:2000]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if pool is not None:
            pool.terminate()
            pool.join()
        signal.signal(signal.SIGALRM, previous_alarm)
    if manifest is None:
        raise DeadlineExceeded("wall deadline expired before frozen inputs were validated")
    for case in manifest["cases"]:
        identifier = case["audit_id"]
        saved = output / "cases" / (identifier + ".json")
        if identifier not in results and saved.exists():
            results[identifier] = json.loads(saved.read_text(encoding="utf-8"))
        if identifier not in results:
            result = initial_result(case)
            result.update(decode_status="not_completed_" + termination, contact_status="unavailable")
            atomic_json(saved, result)
            results[identifier] = result
    ordered = [results[c["audit_id"]] for c in manifest["cases"]]
    unchanged = sha256(safe_path(root, MANIFEST_REL)) == MANIFEST_SHA and sha256(safe_path(root, CSV_REL)) == CSV_SHA
    report = build_report(manifest, ordered, output, started_utc, workers, wall_limit, termination, unchanged)
    report["wall_seconds"] = time.monotonic() - started_clock
    report["runner_error"] = runner_error
    report_sha = atomic_json(output / "report.json", report)
    atomic_json(output / "result_hashes.json", {"report_sha256": report_sha,
        "case_json_sha256": {c["audit_id"]: sha256(output / "cases" / (c["audit_id"] + ".json")) for c in manifest["cases"]},
        "first_17_contacts_json_sha256": sha256(output / "first_17_contacts.json") if (output / "first_17_contacts.json").exists() else None})
    print(json.dumps({"run_id": output.name, "result_count": len(ordered), "decode_status": report["decode_status"],
                      "termination": termination, "semantic_review": "pending", "report_sha256": report_sha}), flush=True)
    return 0 if report["decode_and_contact_complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    args = parser.parse_args(argv)
    os.umask(0o077)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[name] = ""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        return run_audit(args.root, args.out_dir, args.workers, args.max_wall_seconds)
    except (ValueError, DeadlineExceeded) as error:
        print(f"audit_precheck_failed={type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
