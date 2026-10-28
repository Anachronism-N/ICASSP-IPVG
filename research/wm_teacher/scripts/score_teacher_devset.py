#!/usr/bin/env python3
"""CPU-only frozen Teacher scoring contract; real scoring is deliberately closed.

Explicit supplier records are not proof of a scoring runtime or GPU authorization.
No command loads a metric model, invokes VBench, or modifies old plans/ledgers.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import score_e1 as e1

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("base", "alg", "dymos")
SEEDS = ("seed0", "seed1")
BUNDLE_SCHEMA = "teacher_devset_score_bundle_v1"
PLAN_SCHEMA = "teacher_devset_scoring_plan_v1"
RECEIPT_SCHEMA = "teacher_devset_score_receipt_v1"
FROZEN_CASES_SHA256 = "0873b163284a92b207ac248cc41ff934cf73615f1f9af073eca8bfb2073e5aed"
RUNTIME_PACKAGES = {"python", "torch", "vbench", "dreamsim", "av", "decord", "PIL", "numpy"}
DEPENDENCIES = RUNTIME_PACKAGES - {"python"}
SCORE_CAP_SECONDS = {"base": 18000, "alg": 57600, "dymos": 57600}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise RuntimeError("Non-finite JSON constant: " + value)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle, object_pairs_hook=unique, parse_constant=reject_constant)


def read_cases(path):
    with Path(path).open() as handle:
        rows = [json.loads(line, object_pairs_hook=unique, parse_constant=reject_constant) for line in handle]
    require(len(rows) == 94 and all(isinstance(c, dict) for c in rows), "Exactly 94 case objects required")
    ids = [c.get("id") for c in rows]
    require(all(isinstance(i, str) and i not in ("", ".", "..") and
                all(x.isalnum() or x in "_.+-" for x in i) for i in ids), "Unsafe/missing case ID")
    require(len(set(ids)) == 94, "Duplicate case ID")
    require(all(isinstance(c.get("dataset"), str) and c["dataset"] for c in rows), "Missing dataset")
    json.dumps(rows, allow_nan=False)
    return rows


def file_record(path):
    path = Path(path).absolute()
    require(path.resolve() == path and path.is_file() and path.stat().st_size > 0,
            "Missing/empty/symlink file: " + str(path))
    before = path.stat()
    result = {"path": str(path), "bytes": before.st_size, "sha256": sha256(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "File changed while hashing")
    return result


def check_record(record, content=True):
    require(isinstance(record, dict) and set(record) == {"path", "bytes", "sha256"}, "Invalid file record schema")
    path = Path(record["path"])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), "Missing/symlink recorded file")
    require(type(record["bytes"]) is int and record["bytes"] > 0 and is_sha(record["sha256"]), "Invalid file size/SHA")
    require(path.stat().st_size == record["bytes"], "Recorded file size drift")
    if content:
        require(file_record(path) == record, "Recorded file SHA drift: " + str(path))
    return path


def exact(values, expected, label):
    values = list(values)
    require(len(values) == len(set(values)) and set(values) == set(expected), label + ": missing/duplicate/extra coverage")


def write_new(path, value):
    path = Path(path).absolute()
    require(path.resolve() == path and not path.exists(), "Preserve existing/symlink output")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink()
    return file_record(path)


def required_sources():
    return [str(ROOT / "scripts" / name) for name in (
        "score_teacher_devset.py", "summarize_teacher_devset.py", "score_e1.py",
        "build_e1_jobs.py", "summarize_eval.py", "gt_ref_metrics.py", "vbench_eval.py")]


def coverage(roots, records, roles, label):
    require(isinstance(roots, dict) and set(roots) == set(roles), label + ": explicit role roots required")
    expected = set()
    for role, name in roots.items():
        root = Path(name)
        require(root.is_absolute() and root.resolve() == root and root.is_dir(), label + ": invalid root " + role)
        found = set()
        for path in root.rglob("*"):
            require(not path.is_symlink(), label + ": symlink dependency forbidden")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                found.add(str(path))
        require(found, label + ": empty coverage for " + role)
        expected.update(found)
    require(isinstance(records, list), label + ": records must be a list")
    exact([r["path"] for r in records], expected, label)


def validate_bundle(bundle, verify_content=True):
    require(bundle.get("schema") == BUNDLE_SCHEMA and bundle.get("private") is True and
            bundle.get("metric_protocol") == "score_e1_all8_gt_v1" and bundle.get("dimensions") == e1.DIMS and
            bundle.get("gt_keys") == list(e1.GT_KEYS), "Frozen metric schema drift")
    exact([r["path"] for r in bundle["sources"]], required_sources(), "Scoring source set")
    coverage(bundle["dependency_roots"], bundle["dependencies"], DEPENDENCIES, "Runtime dependencies")
    coverage(bundle["model_roots"], bundle["models"], {"vbench", "dreamsim"}, "Metric model files")
    runtime = bundle["runtime"]
    require(isinstance(runtime.get("versions"), dict) and set(runtime["versions"]) == RUNTIME_PACKAGES and
            all(isinstance(v, str) and v.strip() for v in runtime["versions"].values()), "Missing explicit runtime versions")
    require(isinstance(runtime.get("tools"), dict) and set(runtime["tools"]) == {"python", "ffmpeg", "ffprobe"},
            "Missing runtime executable paths")
    paths = [r["path"] for r in runtime["files"]]
    require(len(paths) == len(set(paths)) and set(runtime["tools"].values()) <= set(paths), "Unbound runtime executable files")
    records = bundle["sources"] + bundle["dependencies"] + bundle["models"] + runtime["files"]
    by_path = {}
    for record in records:
        require(record["path"] not in by_path or by_path[record["path"]] == record, "Conflicting overlapping record")
        by_path[record["path"]] = record
        check_record(record, content=False)
    if verify_content:
        for record in by_path.values():
            check_record(record)
    return bundle


def verify_bundle(path, approved_sha, verify_content=True):
    require(is_sha(approved_sha) and sha256(path) == approved_sha, "Unapproved scoring bundle SHA")
    bundle = validate_bundle(read_json(path), verify_content)
    require(sha256(path) == approved_sha, "Score bundle changed during verification")
    return bundle


def freeze_bundle(records_path, output):
    bundle = read_json(records_path)
    validate_bundle(bundle, verify_content=False)
    # Suppliers must supply actual records. Schema-only freezing is explicitly not content verification.
    return write_new(output, bundle)


def verified_copy(source_record, destination):
    source = check_record(source_record)
    destination = Path(destination).absolute()
    require(destination.resolve() == destination, "Local stage symlink forbidden")
    require(str(destination).startswith(("/tmp/", "/dockerdata/", "/dev/shm/")), "Stage must be node-local")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha256(destination) == source_record["sha256"], "Local-stage SHA drift (even if same size)")
    else:
        fd, temporary = tempfile.mkstemp(prefix="." + destination.name, dir=destination.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            require(sha256(temporary) == source_record["sha256"], "Byte-copy SHA mismatch")
            check_record(source_record)
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink()
    check_record(source_record)
    return file_record(destination)


def verify_completion(record, method, common_sha, bundle_sha):
    check_record(record)
    # The base/comparison driver independently re-verifies all underlying generation evidence.
    from prepare_teacher_comparison import verify_generation_completion
    return verify_generation_completion(record["path"], record["sha256"],
                                        common_execution_sha256=common_sha,
                                        score_bundle_sha256=bundle_sha, method=method)


def validate_plan(plan, verify_generation=True, verify_bundle_content=True):
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("private") is True and
            plan.get("score_cap_seconds") == SCORE_CAP_SECONDS and plan.get("seeds") == list(SEEDS) and
            is_sha(plan.get("common_execution_sha256")), "Scoring plan protocol/stage budget drift")
    check_record(plan["bundle"], content=False)
    verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"], verify_bundle_content)
    require(plan["cases"]["sha256"] == FROZEN_CASES_SHA256, "Frozen original cases byte SHA drift")
    check_record(plan["cases"])
    cases = read_cases(plan["cases"]["path"])
    require(set(plan["groups"]) == set(METHODS), "All three methods must be frozen before scoring")
    for method, group in plan["groups"].items():
        require(set(group["seeds"]) == set(SEEDS), "Exactly two seeds required")
        check_record(group["generation_completion"])
        if verify_generation:
            verify_completion(group["generation_completion"], method, plan["common_execution_sha256"], plan["bundle"]["sha256"])
        for seed, entry in group["seeds"].items():
            require(entry["config"] == method and Path(entry["root"]).is_absolute(), "Method/root identity drift")
            metadata = file_record(Path(entry["root"]) / "cases.jsonl")
            require(metadata["sha256"] == plan["cases"]["sha256"] and
                    read_cases(metadata["path"]) == cases, "Per-seed metadata drift")
            exact([r["case_id"] for r in entry["media"]], [c["id"] for c in cases], "Generated media")
            for row in entry["media"]:
                expected = str(Path(entry["root"]) / "videos" / method / (row["case_id"] + ".mp4"))
                require(row["file"]["path"] == expected, "Generated media path/seed mismatch")
                check_record(row["file"])
    return cases


def prepare_plan(contract_path, output):
    plan = read_json(contract_path)
    # A frozen CPU contract can be pending generation, never a completion/score authorization.
    validate_plan(plan, verify_generation=False, verify_bundle_content=False)
    return write_new(output, plan)


def cache_binding(plan, method, seed):
    cases = read_cases(plan["cases"]["path"])
    media_rows = plan["groups"][method]["seeds"][seed]["media"]
    exact([r["case_id"] for r in media_rows], [c["id"] for c in cases], "Cache generated media")
    media = {}
    for row in media_rows:
        check_record(row["file"])
        media[row["case_id"]] = row["file"]["sha256"]
    inputs = {}
    for case in cases:
        gt, image = file_record(case["gt"]), file_record(case["image"])
        require(gt["sha256"] == case["video_sha256"] and image["sha256"] == case["image_sha256"], "GT/image SHA drift")
        inputs[case["id"]] = {"generated_sha256": media[case["id"]], "gt_sha256": gt["sha256"],
                              "image_sha256": image["sha256"]}
    return {"schema": "teacher_devset_metric_cache_binding_v1", "method": method, "seed": seed,
            "bundle_sha256": plan["bundle"]["sha256"], "common_execution_sha256": plan["common_execution_sha256"],
            "cases_sha256": plan["cases"]["sha256"], "inputs": inputs}


def verify_cache(plan, method, seed, binding_path):
    require(read_json(binding_path) == cache_binding(plan, method, seed), "Old ID-only cache or cache input/source SHA drift")
    return True


def verify_scoring_execution(receipt, plan):
    raise RuntimeError("Scoring execution evidence verifier is not integrated: parent release, fixed stage budget, "
                       "and occupied GPU scope must be independently verified; receipt booleans are not authorization")


def verify_score_receipt(plan, plan_sha, method, seed, receipt_path, verify_execution=True):
    receipt = read_json(receipt_path)
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("success") is True and
            receipt.get("plan_sha256") == plan_sha and receipt.get("method") == method and receipt.get("seed") == seed and
            receipt.get("bundle_sha256") == plan["bundle"]["sha256"] and
            receipt.get("common_execution_sha256") == plan["common_execution_sha256"], "Unbound scoring receipt")
    require(receipt.get("parent_release_verified") is True and receipt.get("budget_verified") is True and
            receipt.get("scope_verified") is True and receipt.get("owned_tree_released") is True and
            receipt.get("stage_cap_seconds") == SCORE_CAP_SECONDS[method] and
            type(receipt.get("allocated_gpu_seconds")) in (int, float) and
            math.isfinite(receipt["allocated_gpu_seconds"]) and 0 <= receipt["allocated_gpu_seconds"] <= SCORE_CAP_SECONDS[method],
            "Missing scoring release/scope/budget evidence")
    if verify_execution:
        verify_scoring_execution(receipt, plan)
    entry = plan["groups"][method]["seeds"][seed]
    require(receipt.get("source_media") == entry["media"], "Score source media SHA binding mismatch")
    exact([r["case_id"] for r in receipt["stage_media"]], [r["case_id"] for r in entry["media"]], "Stage media")
    source = {r["case_id"]: r["file"] for r in entry["media"]}
    for row in receipt["stage_media"]:
        check_record(source[row["case_id"]])
        check_record(row["file"])
        require(row["file"]["sha256"] == source[row["case_id"]]["sha256"] and
                Path(row["file"]["path"]).name == row["case_id"] + "-0.mp4" and
                str(row["file"]["path"]).startswith(("/tmp/", "/dockerdata/", "/dev/shm/")), "Stage SHA/path identity drift")
    check_record(receipt["cache_binding"])
    verify_cache(plan, method, seed, receipt["cache_binding"]["path"])
    scores = receipt["score_files"]
    require(set(scores) == set(e1.DIMS + ["gtref"]), "Exactly eight dimensions and GT receipt required")
    for key, record in scores.items():
        expected = Path(entry["root"]) / ("gtref/" + method + ".json" if key == "gtref" else
                                         "scores/" + method + "__" + key + "_eval_results.json")
        require(record["path"] == str(expected), "Score file path drift")
        check_record(record)
        read_json(record["path"])  # Reject duplicate JSON keys before the original loader reads the file.
    cases = read_cases(plan["cases"]["path"])
    gt = read_json(scores["gtref"]["path"])
    by_id = {c["id"]: c for c in cases}
    require(all(row.get("dataset") == by_id[i]["dataset"] for i, row in gt.items() if i in by_id), "GT dataset metadata drift")
    rows = e1.load_config_rows(method, cases, entry["root"])
    return rows, receipt


def real_score(*args, **kwargs):
    raise RuntimeError("Real scoring is closed: independent parent release, node-local stage ledger, occupied GPU scope, "
                       "and complete scoring-runtime/model content verification are not integrated; CLI flags are not authorization")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze-bundle")
    p.add_argument("--records", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("prepare")
    p.add_argument("--contract", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("verify-bundle")
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--approved-sha256", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--approved-sha256", required=True)
    sub.add_parser("run")
    args = parser.parse_args(argv)
    if args.command == "run":
        real_score()
    elif args.command == "freeze-bundle":
        freeze_bundle(args.records, args.out)
    elif args.command == "prepare":
        prepare_plan(args.contract, args.out)
    elif args.command == "verify-bundle":
        verify_bundle(args.bundle, args.approved_sha256)
    else:
        require(sha256(args.plan) == args.approved_sha256, "Unapproved scoring plan SHA")
        validate_plan(read_json(args.plan))


if __name__ == "__main__":
    main()
