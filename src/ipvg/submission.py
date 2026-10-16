"""Build an allowlisted local draft ZIP; this module never submits anything."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .evaluate import PathLike, evaluate_outputs, load_manifest_records


def create_draft_submission(
    manifest_path: PathLike,
    outputs_dir: PathLike,
    archive_path: PathLike,
    *,
    ffprobe: str = "ffprobe",
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
    expected_fps: Optional[float] = None,
    expected_duration: Optional[float] = None,
    duration_tolerance: float = 0.25,
    fps_tolerance: float = 0.01,
    probe_timeout: float = 120.0,
) -> Dict[str, Any]:
    """Validate exactly one video per sample, then create a local draft only.

    ZIP entries are allowlisted to generated videos and a newly constructed,
    minimal hash manifest. Reference images, source data, prompts, full input
    manifests, score files and arbitrary directory contents are never copied.
    Existing archives are not overwritten. Missing official output rules are
    left unknown unless the caller provides explicit expected specifications.
    """
    target = Path(archive_path)
    if target.suffix.lower() != ".zip":
        raise ValueError("archive_path must end in .zip")
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite an existing draft: {}".format(target))
    report = evaluate_outputs(
        manifest_path, outputs_dir, ffprobe=ffprobe,
        expected_width=expected_width, expected_height=expected_height,
        expected_fps=expected_fps, expected_duration=expected_duration,
        duration_tolerance=duration_tolerance, fps_tolerance=fps_tolerance,
        probe_timeout=probe_timeout, fail_on_extra=True,
    )
    if not report["ok"]:
        raise ValueError("draft validation failed: {}".format(json.dumps(report["summary"], ensure_ascii=False)))
    source_records = load_manifest_records(manifest_path)
    reference_paths = set()
    manifest_parent = Path(manifest_path).resolve().parent
    for row in source_records:
        references = row.get("reference_images", [])
        if not isinstance(references, list) or any(not isinstance(item, str) for item in references):
            raise ValueError("reference_images must be a list of paths")
        for reference in references:
            path = Path(reference)
            if not path.is_absolute():
                path = manifest_parent / path
            reference_paths.add(path.resolve())
    root = Path(outputs_dir).resolve()
    for sample in report["samples"]:
        if (root / sample["filename"]).resolve() in reference_paths:
            raise ValueError("refusing to package a reference input as generated video")
    draft = {
        "schema_version": 1,
        "status": "local_draft_not_submitted",
        "official_submission_format_confirmed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": report["manifest_sha256"],
        "expected_spec": report["expected_spec"],
        "notice": "Local draft layout only. Confirm the organizer's final format and output rules before any manual submission.",
        "samples": [],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        # Exclusive creation prevents overwriting a file created by another job.
        with target.open("xb") as output_stream:
            created = True
            with zipfile.ZipFile(output_stream, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for sample in report["samples"]:
                    path = root / sample["filename"]
                    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
                        raise ValueError("output path changed after validation: {}".format(sample["sample_id"]))
                    archive_name = "videos/" + sample["filename"]
                    digest = hashlib.sha256()
                    byte_count = 0
                    with path.open("rb") as source, archive.open(archive_name, mode="w", force_zip64=True) as destination:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(chunk)
                            byte_count += len(chunk)
                            destination.write(chunk)
                    if digest.hexdigest() != sample["sha256"] or byte_count != sample["bytes"]:
                        raise ValueError("output content changed after validation: {}".format(sample["sample_id"]))
                    draft["samples"].append({"sample_id": sample["sample_id"], "path": archive_name,
                                             "sha256": digest.hexdigest(), "bytes": byte_count,
                                             "media": sample["media"]})
                archive.writestr("draft_manifest.json", json.dumps(draft, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return {"status": "local_draft_not_submitted", "archive_path": str(target.resolve()),
            "sample_count": len(draft["samples"]), "draft_manifest": draft}
