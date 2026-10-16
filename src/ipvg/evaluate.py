"""Validate generated media and summarize explicitly local proxy scores.

No official IPVG scoring weights or unpublished output specification are assumed.
Proxy JSONL uses ``sample_id``, ``scores`` (a mapping of names to finite
numbers), and optionally ``failure_classes`` (a list of strings).
"""

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]
_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)


def safe_sample_id(value: Any) -> str:
    """Return a filename-safe identifier, rejecting traversal on any platform."""
    if not isinstance(value, str) or not _SAMPLE_ID.fullmatch(value):
        raise ValueError("sample_id must be 1–128 ASCII letters/digits/._- and start with a letter or digit")
    if value.endswith(".") or _RESERVED.fullmatch(value):
        raise ValueError("sample_id is not portable as a filename: {!r}".format(value))
    return value


def load_manifest_records(manifest_path: PathLike) -> List[Dict[str, Any]]:
    """Read JSONL with unique safe sample IDs; domain validation stays upstream."""
    records = []
    seen = set()
    with Path(manifest_path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("manifest line {} is invalid JSON: {}".format(line_number, exc.msg)) from exc
            if not isinstance(record, dict):
                raise ValueError("manifest line {} must be an object".format(line_number))
            sample_id = safe_sample_id(record.get("sample_id"))
            if sample_id.casefold() in seen:
                raise ValueError("duplicate sample_id: {}".format(sample_id))
            seen.add(sample_id.casefold())
            records.append(record)
    if not records:
        raise ValueError("manifest contains no samples")
    return records


def sha256_file(path: PathLike) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value: Any, name: str, integer: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a positive {}".format(name, "integer" if integer else "number"))
    if not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
        raise ValueError("{} must be a positive {}".format(name, "integer" if integer else "number"))
    return float(value)


def _number(value: Any) -> Optional[float]:
    try:
        result = float(Fraction(str(value)))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def probe_video(path: PathLike, *, ffprobe: str = "ffprobe", timeout: float = 120.0) -> Dict[str, Any]:
    """Decode/count the first video stream using ffprobe, without a shell.

    ``-count_frames`` requests frame decoding rather than trusting container
    metadata alone. A decoder error on stderr is treated as a failed video.
    """
    _positive(timeout, "probe timeout")
    command = [
        str(ffprobe), "-v", "error", "-show_error", "-select_streams", "v:0",
        "-count_frames", "-show_streams", "-show_format", "-of", "json", str(Path(path)),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return {"ok": False, "failure_class": "probe_unavailable", "error": "ffprobe executable was not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "failure_class": "probe_timeout", "error": "ffprobe exceeded its timeout"}
    except OSError as exc:
        return {"ok": False, "failure_class": "probe_unavailable", "error": str(exc)}
    if completed.returncode != 0 or completed.stderr.strip():
        return {"ok": False, "failure_class": "decode_error", "error": completed.stderr.strip()[:2000] or "ffprobe failed"}
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        stream = streams[0] if streams else None
        if payload.get("error") or not isinstance(stream, dict) or stream.get("codec_type") != "video":
            raise ValueError("no readable video stream")
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
        frames = int(stream.get("nb_read_frames", 0))
        fps = _number(stream.get("avg_frame_rate")) or _number(stream.get("r_frame_rate"))
        duration = _number(stream.get("duration")) or _number(payload.get("format", {}).get("duration"))
        if duration is None and fps:
            duration = frames / fps
        if width <= 0 or height <= 0 or frames <= 0 or fps is None or duration is None:
            raise ValueError("missing positive dimensions, decoded frame count, fps or duration")
    except (ValueError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        return {"ok": False, "failure_class": "invalid_video_metadata", "error": str(exc)}
    return {"ok": True, "width": width, "height": height, "fps": fps,
            "duration_s": duration, "decoded_frames": frames, "codec": stream.get("codec_name")}


def _proxy_scores(path: Optional[PathLike], sample_ids: set) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    imported = {}
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("proxy score line {} is invalid JSON".format(line_number)) from exc
            if not isinstance(row, dict):
                raise ValueError("proxy score line {} must be an object".format(line_number))
            sample_id = safe_sample_id(row.get("sample_id"))
            if sample_id not in sample_ids:
                raise ValueError("proxy score references unknown sample_id: {}".format(sample_id))
            if sample_id in imported:
                raise ValueError("duplicate proxy score sample_id: {}".format(sample_id))
            if row.get("score_kind", "local_proxy") != "local_proxy":
                raise ValueError("imported scores must be labelled local_proxy, never official")
            scores = row.get("scores", {})
            if not isinstance(scores, dict):
                raise ValueError("proxy scores must be a mapping of metric names to finite numbers")
            for name, value in scores.items():
                if not isinstance(name, str) or not name.strip() or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("proxy scores must contain named finite numeric values")
            failures = row.get("failure_classes", [])
            if not isinstance(failures, list) or any(not isinstance(label, str) or not label.strip() for label in failures):
                raise ValueError("proxy failure_classes must be a list of nonempty strings")
            imported[sample_id] = {"scores": scores, "failure_classes": sorted(set(failures))}
    return imported


def evaluate_outputs(
    manifest_path: PathLike,
    outputs_dir: PathLike,
    report_path: Optional[PathLike] = None,
    *,
    ffprobe: str = "ffprobe",
    proxy_scores_path: Optional[PathLike] = None,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_fps: Optional[float] = None,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = 0.25,
    fps_tolerance: float = 0.01,
    probe_timeout: float = 120.0,
    fail_on_extra: bool = False,
) -> Dict[str, Any]:
    """Check every ``<sample_id>.mp4`` and summarize optional local scores.

    Expected output specifications are explicit caller input, not official
    defaults. ``ok`` reports media completeness/validation only; proxy scores
    do not establish official acceptance or quality. Missing scores are never
    replaced with invented zeroes or combined using invented weights.
    """
    for name, value in (("expected_width", expected_width), ("expected_height", expected_height)):
        _positive(value, name, integer=True)
    for name, value in (("expected_fps", expected_fps), ("expected_duration", expected_duration), ("probe_timeout", probe_timeout)):
        _positive(value, name)
    for name, value in (("duration_tolerance", duration_tolerance), ("fps_tolerance", fps_tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("{} must be a finite nonnegative number".format(name))
    records = load_manifest_records(manifest_path)
    root = Path(outputs_dir).resolve()
    if not root.is_dir():
        raise ValueError("outputs_dir is not a directory: {}".format(root))
    sample_ids = {row["sample_id"] for row in records}
    proxies = _proxy_scores(proxy_scores_path, sample_ids)
    expected_names = {sample_id + ".mp4" for sample_id in sample_ids}
    extras = sorted(path.name for path in root.iterdir() if path.suffix.lower() == ".mp4" and path.name not in expected_names)
    samples = []
    failures = Counter()
    proxy_failures = Counter()
    numeric = {}
    valid_count = 0
    present_count = 0
    for row in records:
        sample_id = row["sample_id"]
        video_path = root / (sample_id + ".mp4")
        result = {"sample_id": sample_id, "filename": video_path.name, "ok": False, "failure_classes": []}
        if video_path.is_symlink():
            result["failure_classes"].append("unsafe_output_path")
        elif not video_path.exists():
            result["failure_classes"].append("missing_output")
        elif not video_path.is_file() or video_path.resolve().parent != root:
            result["failure_classes"].append("unsafe_output_path")
        else:
            present_count += 1
            metadata = probe_video(video_path, ffprobe=ffprobe, timeout=probe_timeout)
            if not metadata["ok"]:
                result["failure_classes"].append(metadata["failure_class"])
                result["error"] = metadata.get("error", "")
            else:
                result["media"] = {key: value for key, value in metadata.items() if key != "ok"}
                if expected_width is not None and metadata["width"] != expected_width:
                    result["failure_classes"].append("width_mismatch")
                if expected_height is not None and metadata["height"] != expected_height:
                    result["failure_classes"].append("height_mismatch")
                if expected_fps is not None and abs(metadata["fps"] - expected_fps) > fps_tolerance:
                    result["failure_classes"].append("fps_mismatch")
                if expected_duration is not None and abs(metadata["duration_s"] - expected_duration) > duration_tolerance:
                    result["failure_classes"].append("duration_mismatch")
                try:
                    result["sha256"] = sha256_file(video_path)
                    result["bytes"] = video_path.stat().st_size
                except OSError as exc:
                    result["failure_classes"].append("output_read_error")
                    result["error"] = str(exc)
                result["ok"] = not result["failure_classes"]
                valid_count += int(result["ok"])
        failures.update(result["failure_classes"])
        if sample_id in proxies:
            proxy = proxies[sample_id]
            result["local_proxy"] = proxy
            proxy_failures.update(proxy["failure_classes"])
            for metric, value in proxy["scores"].items():
                numeric.setdefault(metric, []).append(value)
        samples.append(result)
    metrics = {metric: {"count": len(values), "coverage": len(values) / len(records),
                        "mean": math.fsum(value / len(values) for value in values), "min": min(values), "max": max(values)}
               for metric, values in sorted(numeric.items())}
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_kind": "local_proxy",
        "official_score": None,
        "notice": "Local media checks and imported proxy scores only; no official IPVG score or publication/submission acceptance is inferred.",
        "ok": valid_count == len(records) and not (fail_on_extra and extras),
        "manifest_sha256": sha256_file(manifest_path),
        "expected_spec": {"width": expected_width, "height": expected_height, "fps": expected_fps,
                          "duration_s": expected_duration, "duration_tolerance_s": duration_tolerance, "fps_tolerance": fps_tolerance},
        "summary": {"expected_samples": len(records), "present_outputs": present_count,
                    "valid_outputs": valid_count, "output_coverage": present_count / len(records),
                    "valid_output_coverage": valid_count / len(records), "proxy_samples": len(proxies),
                    "proxy_coverage": len(proxies) / len(records), "failure_classes": dict(sorted(failures.items())),
                    "proxy_failure_classes": dict(sorted(proxy_failures.items())), "metrics": metrics,
                    "unexpected_videos": extras},
        "samples": samples,
    }
    if report_path is not None:
        target = Path(report_path)
        if target.suffix.lower() != ".json":
            raise ValueError("report_path must end in .json")
        protected_inputs = {Path(manifest_path).resolve()}
        if proxy_scores_path is not None:
            protected_inputs.add(Path(proxy_scores_path).resolve())
        if target.resolve() in protected_inputs:
            raise ValueError("report_path must not overwrite an input manifest or proxy score file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
