"""Crop only frozen CREMA-D dev conditioning images; evaluation keeps original anchors."""

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ipvg.common import sha256_file, write_json
from ipvg.data import read_manifest, validate_manifest, write_manifest


def square_crop(image_bgr, bbox, margin=0.3):
    """Return an unscaled square copy and integer, exclusive xyxy source-pixel box.

    Expand each side by ``margin`` times the corresponding detected face extent.
    Shift the square inside the image; clamp its side to the shorter image edge
    when necessary. Never pad, resize, align, or modify the original image.
    """
    if isinstance(margin, (bool, np.bool_)) or not isinstance(margin, (int, float, np.number)):
        raise ValueError("margin must be a finite number between 0 and 1")
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("margin must be a finite number between 0 and 1")
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must be a nonempty HxWx3 BGR array")
    height, width = image_bgr.shape[:2]
    if min(height, width) < 1:
        raise ValueError("image must be a nonempty HxWx3 BGR array")
    try:
        box = np.asarray(bbox, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox must contain four finite xyxy coordinates") from exc
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bbox must contain four finite xyxy coordinates")
    x1, y1, x2, y2 = box.tolist()
    face_width, face_height = x2 - x1, y2 - y1
    if face_width <= 0 or face_height <= 0:
        raise ValueError("bbox must have positive width and height")
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        raise ValueError("bbox does not intersect the image")
    extent = max(face_width, face_height) * (1 + 2 * float(margin))
    if not math.isfinite(extent):
        raise ValueError("bbox extent must be finite")
    expanded_left = math.floor(x1 - float(margin) * face_width)
    expanded_top = math.floor(y1 - float(margin) * face_height)
    expanded_right = math.ceil(x2 + float(margin) * face_width)
    expanded_bottom = math.ceil(y2 + float(margin) * face_height)
    side = min(max(expanded_right - expanded_left, expanded_bottom - expanded_top), height, width)
    left = min(max((expanded_left + expanded_right - side) // 2, 0), width - side)
    top = min(max((expanded_top + expanded_bottom - side) // 2, 0), height - side)
    crop_box = (left, top, left + side, top + side)
    return image_bgr[top:top + side, left:left + side].copy(), crop_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.3)
    args = parser.parse_args()
    if not math.isfinite(args.margin) or not 0 <= args.margin <= 1:
        raise ValueError("margin must be a finite number between 0 and 1")
    if not args.manifest.is_absolute():
        raise ValueError("manifest must be the original full-path 128-row manifest")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError("Output already exists; preserve it and choose a new output directory")
    manifest = args.manifest.resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError("Output parent directory must already exist")
    if output == manifest.parent or manifest.parent in output.parents or model_dir in output.parents:
        raise ValueError("Output must not be inside original data or model directories")
    manifest_hash = sha256_file(manifest)
    rows = read_manifest(manifest, check_paths=False)
    split_report = validate_manifest(rows, check_paths=False)
    if (len(rows) != 128 or split_report["counts"] !=
            {"train": 32, "dev": 64, "holdout": 32, "unassigned": 0}):
        raise ValueError("Expected the frozen CREMA-D train32/dev64/holdout32 manifest")
    if not split_report["leakage"]["safe"]:
        raise ValueError("Original manifest has identity/source split leakage")
    if any(any(key in row for key in ("original_reference_images", "reference_crop", "evaluator_anchor_manifest"))
           for row in rows):
        raise ValueError("Input must be the original manifest, not previously cropped conditioning")
    dev = sorted((row for row in rows if row["split"] == "dev"), key=lambda row: row["sample_id"])
    actors = Counter(row["identity_id"] for row in dev)
    if len(actors) != 32 or any(count != 2 for count in actors.values()):
        raise ValueError("Expected exactly two frozen dev rows for each of 32 actors")
    actor_references = {}
    for row in dev:
        references = row["reference_images"]
        if len(references) != 1 or not Path(references[0]).is_absolute():
            raise ValueError("Each dev actor requires one original full-path reference image")
        original = Path(references[0]).resolve()
        if output == original.parent or original.parent in output.parents:
            raise ValueError("Output must not be inside an original reference directory")
        reference_key = (original, row.get("reference_sha256"))
        previous = actor_references.setdefault(row["identity_id"], reference_key)
        if previous != reference_key:
            raise ValueError("Dev rows for an actor must share the same frozen reference and checksum")
    if len({path for path, _ in actor_references.values()}) != 32:
        raise ValueError("Dev actors must have distinct original reference images")
    model_files = [model_dir / name for name in ("det_10g.onnx", "w600k_r50.onnx")]
    if not all(path.is_file() for path in model_files):
        raise FileNotFoundError("Local det_10g.onnx and w600k_r50.onnx are required; no downloads are allowed")
    model_hashes = {path.name: sha256_file(path) for path in model_files}

    import cv2
    from identity_eval import OfflineFaceEncoder

    output.mkdir(exist_ok=False)
    (output / "references").mkdir(exist_ok=False)
    crops, prepared = {}, []
    current_actor, current_reference = None, None
    provenance = {
        "evaluator_anchor_manifest": str(manifest),
        "evaluator_anchor_manifest_sha256": manifest_hash,
        "original_records": len(rows),
        "consumed_split": "dev",
        "conditioning_only": True,
        "margin": args.margin,
        "model_dir": str(model_dir),
        "model_sha256": model_hashes,
        "notice": "Local reference preparation only; no identity quality score or official evaluation",
    }
    try:
        encoder = OfflineFaceEncoder(model_dir)
        for row in dev:
            current_actor = row["identity_id"]
            original, expected_hash = actor_references[current_actor]
            current_reference = str(original)
            if current_actor not in crops:
                original_hash = sha256_file(original)
                if original_hash != expected_hash:
                    raise ValueError("Original reference checksum differs from the frozen manifest")
                image_bgr = cv2.imread(str(original), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise ValueError("invalid_reference_image")
                face, reason = encoder.face(image_bgr)
                if face is None or reason is not None:
                    raise ValueError(reason or "invalid_reference_face")
                crop, box = square_crop(image_bgr, face["bbox"], args.margin)
                if sha256_file(original) != original_hash:
                    raise ValueError("Original reference changed while being read")
                destination = output / "references" / (original_hash + ".png")
                if destination.exists():
                    raise FileExistsError("Crop path already exists; do not overwrite media")
                if not cv2.imwrite(str(destination), crop):
                    raise RuntimeError("Reference crop write failed")
                metadata = {
                    "bbox_xyxy": list(box),
                    "detector_bbox_xyxy": np.asarray(face["bbox"], dtype=float).tolist(),
                    "coordinate_system": "original_image_pixels_xyxy_exclusive",
                    "original_shape": list(image_bgr.shape),
                    "crop_shape": list(crop.shape),
                    "margin": args.margin,
                    "original_sha256": original_hash,
                    "sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                }
                crops[current_actor] = (str(destination), metadata)
            destination, metadata = crops[current_actor]
            prepared.append(dict(row, reference_images=[destination],
                                 original_reference_images=list(row["reference_images"]),
                                 reference_crop=dict(metadata),
                                 evaluator_anchor_manifest=str(manifest),
                                 evaluator_anchor_manifest_sha256=manifest_hash))
        if sha256_file(manifest) != manifest_hash:
            raise ValueError("Original anchor manifest changed during preparation")
        validate_manifest(prepared, check_paths=False)
        outputs = {
            "manifest.jsonl": prepared,
            "pilot.jsonl": prepared[:8],
            "dev_main.jsonl": prepared[8:36],
            "dev_worker.jsonl": prepared[36:],
        }
        if any((output / name).exists() for name in outputs):
            raise FileExistsError("Prepared manifests already exist; do not overwrite frozen data")
        manifests = {}
        for name, records in outputs.items():
            path = output / name
            write_manifest(path, records)
            manifests[name] = {"path": str(path), "records": len(records),
                               "sha256": sha256_file(path), "bytes": path.stat().st_size}
        report = dict(provenance, status="passed", records=len(prepared),
                      unique_reference_crops=len(crops), manifests=manifests, failures=[])
        write_json(output / "preparation_report.json", report)
    except Exception as exc:
        failures = [{"identity_id": current_actor, "reference": current_reference,
                     "error_class": type(exc).__name__, "reason": str(exc)}]
        write_json(output / "failures.json", failures)
        write_json(output / "preparation_report.json",
                   dict(provenance, status="failed", prepared_records=len(prepared),
                        unique_reference_crops=len(crops), failures=failures))
        raise
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
