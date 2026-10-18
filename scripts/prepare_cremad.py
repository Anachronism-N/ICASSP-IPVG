"""Fetch a capped, pinned CREMA-D subset for local research; never a full LFS checkout."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen
from ipvg.common import load_json, sha256_file, write_json
from ipvg.data import validate_manifest, write_manifest
from ipvg.evaluate import probe_video

SOURCE = "https://github.com/CheyneyComputerScience/CREMA-D"
REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"
NAME = re.compile(r"(?P<actor>\d{4})_(?P<utterance>[A-Z]{3})_(?P<emotion>NEU|HAP)_XX\.flv\Z")


def parse_pointer(text):
    match = re.fullmatch(r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?", text)
    if not match or int(match[2]) <= 0:
        raise ValueError("Invalid Git LFS pointer")
    return {"sha256": match[1], "size": int(match[2])}


def select_clips(names, actors=64):
    if not 1 <= actors <= 64:
        raise ValueError("actor count must be between 1 and 64")
    groups = {}
    for name in sorted(names):
        match = NAME.fullmatch(name)
        if match:
            groups.setdefault(match["actor"], []).append((name, match["utterance"], match["emotion"]))
    selected = []
    for actor in sorted(groups, key=lambda a: hashlib.sha256(("cremad-42:" + a).encode()).hexdigest()):
        entries = groups[actor]
        neutral = [e for e in entries if e[2] == "NEU"]
        if not neutral:
            continue
        reference = neutral[0]
        targets = [next((e for e in entries if e[2] == emotion and e[1] != reference[1]), None)
                   for emotion in ("NEU", "HAP")]
        if all(targets) and len({reference[0], *(t[0] for t in targets)}) == 3:
            selected.append({"actor": actor, "reference": reference[0], "targets": [t[0] for t in targets]})
        if len(selected) == actors:
            break
    if len(selected) != actors:
        raise ValueError("Insufficient actors with independent reference and targets")
    return selected


def save_plan(path, plan):
    if path.exists():
        if load_json(path) != plan:
            raise FileExistsError("Download plan differs from the frozen selection; choose a new output directory")
        return
    write_json(path, plan)


def fetch_file(url, destination, expected, remaining):
    if expected["size"] > remaining:
        raise ValueError("Selected downloads exceed the cumulative byte cap")
    if destination.exists():
        if destination.stat().st_size != expected["size"] or sha256_file(destination) != expected["sha256"]:
            raise ValueError("Existing media differs from its immutable LFS pointer")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("Retain the previous partial download and choose a fresh output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as response, partial.open("xb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            size += len(chunk)
            if size > expected["size"] or size > remaining:
                raise ValueError("Response exceeds the pinned file size or byte cap")
            handle.write(chunk)
            digest.update(chunk)
    if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError("Downloaded media checksum or size mismatch")
    partial.rename(destination)


def reference_frame(path, output):
    import cv2
    capture = cv2.VideoCapture(str(path))
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    if detector.empty():
        raise RuntimeError("Local face visibility detector unavailable")
    best = None
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for fraction in (.25, .5, .75):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(total * fraction))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces):
                area = max(int(w * h) for _, _, w, h in faces)
                if best is None or area > best[0]:
                    best = (area, frame, fraction)
    finally:
        capture.release()
    if best is None:
        raise ValueError("No visible face detected in the independent reference clip")
    if output.exists():
        raise FileExistsError("Reference output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), best[1]):
        raise RuntimeError("Reference image write failed")
    return {"frame_fraction": best[2], "face_visibility_detector": "opencv_haar_frontalface", "face_area_pixels": best[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actors", type=int, default=64)
    parser.add_argument("--max-bytes", type=int, default=1000000000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_bytes <= 1000000000:
        raise ValueError("Download cap must be between 1 and 1,000,000,000 bytes")
    head = subprocess.run(["git", "-C", str(args.metadata), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if head != REVISION:
        raise ValueError("CREMA-D checkout is not the fixed revision")
    dirty = subprocess.run(["git", "-C", str(args.metadata), "status", "--porcelain"], check=True, capture_output=True, text=True)
    if dirty.stdout:
        raise ValueError("CREMA-D metadata checkout must be unchanged")
    selected = select_clips([p.name for p in (args.metadata / "VideoFlash").glob("*.flv")], args.actors)
    files = {}
    for entry in selected:
        for name in [entry["reference"], *entry["targets"]]:
            if not NAME.fullmatch(name):
                raise ValueError("Unsafe media filename")
            files[name] = parse_pointer((args.metadata / "VideoFlash" / name).read_text())
    total = sum(item["size"] for item in files.values())
    if total > args.max_bytes:
        raise ValueError("Pinned selection exceeds download cap")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"source": SOURCE, "revision": REVISION, "actor_count": len(selected), "file_count": len(files),
            "total_bytes": total, "max_bytes": args.max_bytes, "selected": selected, "files": files,
            "license": "ODbL-1.0 database / DbCL-1.0 contents", "scope": "Local noncommercial research only; no public identity generation or redistribution"}
    save_plan(args.output / "download_plan.json", plan)
    if not args.execute:
        print(json.dumps({k: plan[k] for k in ("revision", "actor_count", "file_count", "total_bytes")}, indent=2))
        return
    manifest = args.output / "manifest.jsonl"
    if manifest.exists():
        raise FileExistsError("Prepared manifest already exists; do not overwrite frozen data")
    for filename in ("README.md", "LICENSE.txt"):
        destination = args.output / filename
        if not destination.exists():
            destination.write_bytes((args.metadata / filename).read_bytes())
    used, media, errors = 0, {}, []
    for name, expected in files.items():
        path = args.output / "raw" / name
        try:
            url = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/" + REVISION + "/VideoFlash/" + name
            fetch_file(url, path, expected, args.max_bytes - used)
            used += expected["size"]
            probe = probe_video(path)
            if not probe["ok"] or probe["duration_s"] < .75:
                raise ValueError("Invalid or too-short video: " + str(probe))
            media[name] = {"sha256": expected["sha256"], "bytes": expected["size"], "probe": probe}
        except Exception as exc:
            errors.append({"file": name, "error": str(exc)})
            write_json(args.output / "failures.json", errors)
            raise
    rows = []
    for index, entry in enumerate(selected):
        split = "train" if index < 16 else "dev" if index < 48 else "holdout"
        reference = args.output / "references" / (entry["actor"] + ".png")
        try:
            visibility = reference_frame(args.output / "raw" / entry["reference"], reference)
        except Exception as exc:
            errors.append({"actor_id": entry["actor"], "reference": entry["reference"], "error": str(exc)})
            write_json(args.output / "failures.json", errors)
            raise
        for target in entry["targets"]:
            if target == entry["reference"] or target.split("_")[1] == entry["reference"].split("_")[1]:
                raise ValueError("Reference must come from a different utterance/source clip")
            emotion = NAME.fullmatch(target)["emotion"]
            prompt = ("A person faces the camera and speaks naturally with a neutral expression. The head makes small natural movements."
                      if emotion == "NEU" else "A person faces the camera and speaks with a gentle smile and a happy expression. The head makes small natural movements.")
            rows.append({"sample_id": "cremad_" + target[:-4], "identity_id": "cremad_actor_" + entry["actor"],
                         "source_video_id": "cremad_" + target[:-4], "reference_source_video_id": "cremad_" + entry["reference"][:-4],
                         "reference_images": [str(reference)], "video_path": str(args.output / "raw" / target),
                         "prompt": prompt, "prompt_source": "Conservative template from acted emotion label, not dense caption or transcript",
                         "track": 1, "split": split, "timeline": [], "source": SOURCE, "source_revision": REVISION,
                         "license": plan["license"], "media_sha256": media[target]["sha256"],
                         "reference_sha256": sha256_file(reference), "reference_visibility": visibility})
    report = validate_manifest(rows, check_paths=True)
    if not report["leakage"]["safe"]:
        raise ValueError("Actor/source split leakage")
    write_manifest(manifest, rows)
    dev = sorted([r for r in rows if r["split"] == "dev"], key=lambda r: r["sample_id"])
    write_manifest(args.output / "pilot.jsonl", dev[:8])
    write_manifest(args.output / "dev_main.jsonl", dev[8:36])
    write_manifest(args.output / "dev_worker.jsonl", dev[36:])
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    if size > 10000000000:
        raise ValueError("Raw and derived data exceed the storage cap")
    write_json(args.output / "preparation_report.json", {"status": "passed", "split": report, "files": media,
               "manifest_sha256": sha256_file(manifest), "downloaded_bytes": used, "storage_bytes": size,
               "reference_target_independent": True, "failures": errors,
               "notice": "Auxiliary talking-head-domain data; not official IPVG evaluation and no identity quality score"})
    print(json.dumps({"status": "passed", "records": len(rows), "dev_pairs": len(dev), "downloaded_bytes": used, "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
