"""Synthetic CPU fixtures only: no real video, metric model, CUDA, or ledger."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_e1 as e1
import score_teacher_devset as score


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return score.file_record(path)


def small_file(path, data=b"synthetic"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return score.file_record(path)


def make_bundle(tmp):
    dependency_roots, dependencies = {}, []
    for name in sorted(score.DEPENDENCIES):
        root = tmp / "dependencies" / name
        dependencies.append(small_file(root / "module.py", b"pass"))
        dependency_roots[name] = str(root)
    model_roots, models = {}, []
    for name in ("vbench", "dreamsim"):
        root = tmp / "models" / name
        models.append(small_file(root / "weights.bin", b"fixture-not-model"))
        model_roots[name] = str(root)
    runtime_files = [small_file(tmp / "runtime" / name, b"fixture-not-executable") for name in ("python", "ffmpeg", "ffprobe")]
    return {"schema": score.BUNDLE_SCHEMA, "private": True, "metric_protocol": "score_e1_all8_gt_v1",
            "dimensions": e1.DIMS, "gt_keys": list(e1.GT_KEYS),
            "sources": [score.file_record(path) for path in score.required_sources()],
            "dependency_roots": dependency_roots, "dependencies": dependencies,
            "model_roots": model_roots, "models": models,
            "runtime": {"versions": {name: "synthetic-fixture" for name in score.RUNTIME_PACKAGES},
                        "files": runtime_files, "tools": {Path(r["path"]).name: r["path"] for r in runtime_files}}}


def make_plan(tmp):
    cases = []
    for index in range(94):
        name = "case" + str(index).zfill(3)
        gt = small_file(tmp / "inputs" / (name + ".mp4"))
        image = small_file(tmp / "inputs" / (name + ".png"))
        known = index < 88
        cases.append({"id": name, "dataset": "synthetic" if known else "egoexo4d",
                      "parent_resolved": known, "parent_keys": ["parent" + str(index)],
                      "gt": gt["path"], "image": image["path"], "video_sha256": gt["sha256"],
                      "image_sha256": image["sha256"], "prompt": "synthetic action",
                      "prompt_sha256": score.digest("synthetic action"), "n_frames": 121, "fps": 24,
                      "width": 1280, "height": 704})
    raw = "".join(json.dumps(c) + chr(10) for c in cases)
    path = tmp / "cases.jsonl"
    path.write_text(raw)
    bundle = dump(tmp / "bundle.json", make_bundle(tmp))
    plan = {"schema": score.PLAN_SCHEMA, "private": True, "score_cap_seconds": score.SCORE_CAP_SECONDS,
            "seeds": list(score.SEEDS), "common_execution_sha256": "1" * 64, "bundle": bundle,
            "cases": score.file_record(path), "groups": {}}
    for method in score.METHODS:
        completion = dump(tmp / method / "generation.complete.json", {"synthetic": True})
        group = {"generation_completion": completion, "seeds": {}}
        for seed in score.SEEDS:
            root = tmp / method / seed
            root.mkdir(parents=True)
            (root / "cases.jsonl").write_text(raw)
            group["seeds"][seed] = {"root": str(root), "config": method,
                                    "media": [{"case_id": c["id"], "file": small_file(root / "videos" / method / (c["id"] + ".mp4"))}
                                              for c in cases]}
        plan["groups"][method] = group
    return plan


def make_scores(plan, plan_sha, method, seed, dynamic=0.5, started_epoch=20):
    entry = plan["groups"][method]["seeds"][seed]
    root = Path(entry["root"])
    cases = score.read_cases(plan["cases"]["path"])
    files = {}
    for dim in e1.DIMS:
        value = dynamic if dim == "dynamic_degree" else 50.0 if dim == "imaging_quality" else 0.5
        aggregate = value / 100 if dim == "imaging_quality" else value
        result = {dim: [aggregate, [{"video_path": c["id"] + "-0.mp4", "video_results": value} for c in cases]]}
        files[dim] = dump(root / "scores" / (method + "__" + dim + "_eval_results.json"), result)
    files["gtref"] = dump(root / "gtref" / (method + ".json"),
                         {c["id"]: {"dataset": c["dataset"], "dreamsim": 0.1, "dreamsim_last": 0.1, "motion_ratio": 0.1, "dur": 5}
                          for c in cases})
    binding = dump(root / "cache.binding.json", score.cache_binding(plan, method, seed))
    stage = [{"case_id": c["id"], "file": small_file(root / "stage" / (c["id"] + "-0.mp4"))} for c in cases]
    receipt = {"schema": score.RECEIPT_SCHEMA, "success": True, "plan_sha256": plan_sha,
               "method": method, "seed": seed, "bundle_sha256": plan["bundle"]["sha256"],
               "common_execution_sha256": plan["common_execution_sha256"], "parent_release_verified": True,
               "budget_verified": True, "scope_verified": True, "owned_tree_released": True,
               "stage_cap_seconds": score.SCORE_CAP_SECONDS[method], "allocated_gpu_seconds": 1,
               "source_media": entry["media"], "stage_media": stage, "cache_binding": binding,
               "score_files": files, "started_epoch": started_epoch}
    path = root / "teacher_score.receipt.json"
    dump(path, receipt)
    return path


@pytest.fixture
def plan(tmp_path, monkeypatch):
    value = make_plan(tmp_path)
    monkeypatch.setattr(score, "FROZEN_CASES_SHA256", value["cases"]["sha256"])
    return value


def test_bundle_content_and_complete_coverage(tmp_path):
    bundle = make_bundle(tmp_path)
    path = tmp_path / "bundle.json"
    record = dump(path, bundle)
    assert score.verify_bundle(path, record["sha256"]) == bundle


@pytest.mark.parametrize("field", ["sources", "dependencies", "models"])
def test_bundle_missing_coverage_rejected(tmp_path, field):
    bundle = make_bundle(tmp_path)
    bundle[field].pop()
    with pytest.raises(RuntimeError, match="coverage|source set"):
        score.validate_bundle(bundle)


@pytest.mark.parametrize("field", ["sources", "dependencies", "models"])
def test_bundle_duplicate_coverage_rejected(tmp_path, field):
    bundle = make_bundle(tmp_path)
    bundle[field].append(bundle[field][0])
    with pytest.raises(RuntimeError, match="coverage|source set"):
        score.validate_bundle(bundle)


def test_bundle_same_size_dependency_sha_drift(tmp_path):
    bundle = make_bundle(tmp_path)
    Path(bundle["dependencies"][0]["path"]).write_bytes(b"fail")
    with pytest.raises(RuntimeError, match="SHA drift"):
        score.validate_bundle(bundle)


def test_bundle_model_unknown_file_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    small_file(Path(bundle["model_roots"]["vbench"]) / "unrecorded.bin")
    with pytest.raises(RuntimeError, match="coverage"):
        score.validate_bundle(bundle)


def test_bundle_no_arbitrary_model_size_cap(tmp_path, monkeypatch):
    bundle = make_bundle(tmp_path)
    bundle["models"][0]["bytes"] = 128 * 1024 * 1024
    calls = []
    # Symbolic size only: no real large model is opened or hashed in this CPU test.
    monkeypatch.setattr(score, "check_record", lambda record, content=True: calls.append((record, content)))
    assert score.validate_bundle(bundle) == bundle
    assert any(record["bytes"] == 128 * 1024 * 1024 and content for record, content in calls)


def test_sha256_streams_bounded_chunks(monkeypatch):
    data = b"synthetic" * (300000)
    reads = []

    class Input(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda path, mode: Input(data))
    assert score.sha256("synthetic-not-real-model") == hashlib.sha256(data).hexdigest()
    assert len(reads) > 2 and set(reads) == {1024 * 1024}


def test_bundle_symlink_dependency_rejected(tmp_path):
    bundle = make_bundle(tmp_path)
    source = Path(bundle["dependencies"][0]["path"])
    (source.parent / "alias.py").symlink_to(source)
    with pytest.raises(RuntimeError, match="symlink"):
        score.validate_bundle(bundle)


def test_freeze_is_not_content_verification(tmp_path):
    bundle = make_bundle(tmp_path)
    Path(bundle["models"][0]["path"]).write_bytes(b"changed-not-model")
    source = tmp_path / "records.json"
    dump(source, bundle)
    target = tmp_path / "frozen.json"
    score.freeze_bundle(source, target)
    with pytest.raises(RuntimeError, match="SHA drift"):
        score.verify_bundle(target, score.sha256(target))


def test_preserve_existing_frozen_bundle(tmp_path):
    source = tmp_path / "records.json"
    dump(source, make_bundle(tmp_path))
    target = tmp_path / "bundle.json"
    score.freeze_bundle(source, target)
    with pytest.raises(RuntimeError, match="Preserve"):
        score.freeze_bundle(source, target)


def test_plan_all_three_and_exact_two_seed(plan):
    assert len(score.validate_plan(plan, verify_generation=False)) == 94
    plan["groups"]["alg"]["seeds"]["seed2"] = plan["groups"]["alg"]["seeds"]["seed0"]
    with pytest.raises(RuntimeError, match="two seeds"):
        score.validate_plan(plan, verify_generation=False)


def test_plan_seed_metadata_drift(plan):
    path = Path(plan["groups"]["base"]["seeds"]["seed1"]["root"]) / "cases.jsonl"
    path.write_text(path.read_text().replace("synthetic action", "other action"))
    with pytest.raises(RuntimeError, match="metadata drift"):
        score.validate_plan(plan, verify_generation=False)


def test_plan_duplicate_media(plan):
    media = plan["groups"]["dymos"]["seeds"]["seed1"]["media"]
    media[-1] = media[0]
    with pytest.raises(RuntimeError, match="coverage"):
        score.validate_plan(plan, verify_generation=False)


def test_plan_source_sha_not_same_size(plan):
    row = plan["groups"]["base"]["seeds"]["seed0"]["media"][0]
    Path(row["file"]["path"]).write_bytes(b"different")
    with pytest.raises(RuntimeError, match="SHA drift"):
        score.validate_plan(plan, verify_generation=False)


def test_no_stage_budget_reallocation(plan):
    plan["score_cap_seconds"] = {**score.SCORE_CAP_SECONDS, "alg": 72000}
    with pytest.raises(RuntimeError, match="stage budget"):
        score.validate_plan(plan, verify_generation=False)


def test_case_duplicate_keys_and_nonfinite_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(RuntimeError, match="Duplicate"):
        score.read_json(path)
    path.write_text('{"x":NaN}')
    with pytest.raises(RuntimeError, match="Non-finite"):
        score.read_json(path)


def test_verified_local_copy_same_size_cache_drift(tmp_path):
    record = small_file(tmp_path / "source.bin", b"AAAA")
    destination = tmp_path / "local" / "copy.bin"
    score.verified_copy(record, destination)
    destination.write_bytes(b"BBBB")
    with pytest.raises(RuntimeError, match="Local-stage SHA"):
        score.verified_copy(record, destination)


def test_old_id_only_gt_cache_refused(plan, tmp_path):
    path = tmp_path / "old.binding.json"
    dump(path, {"inputs": {c["id"]: {} for c in score.read_cases(plan["cases"]["path"])}})
    with pytest.raises(RuntimeError, match="ID-only cache"):
        score.verify_cache(plan, "base", "seed0", path)


def test_gt_sha_bound_even_same_size(plan, tmp_path):
    path = tmp_path / "binding.json"
    dump(path, score.cache_binding(plan, "base", "seed0"))
    gt = score.read_cases(plan["cases"]["path"])[0]["gt"]
    Path(gt).write_bytes(b"different")
    with pytest.raises(RuntimeError, match="GT/image SHA"):
        score.verify_cache(plan, "base", "seed0", path)


def test_generated_same_size_restored_mtime_cache_refused_without_rewrite(plan, tmp_path):
    path = tmp_path / "binding.json"
    dump(path, score.cache_binding(plan, "base", "seed0"))
    original_binding = path.read_bytes()
    media = plan["groups"]["base"]["seeds"]["seed0"]["media"][0]["file"]
    source = Path(media["path"])
    original_stat = source.stat()
    source.write_bytes(b"different")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == media["bytes"]
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(RuntimeError, match="SHA drift"):
        score.verify_cache(plan, "base", "seed0", path)
    assert path.read_bytes() == original_binding


def test_score_receipt_all8_original_loader(plan):
    path = make_scores(plan, "2" * 64, "alg", "seed0", dynamic=0.6)
    rows, _ = score.verify_score_receipt(plan, "2" * 64, "alg", "seed0", path, verify_execution=False)
    assert len(rows) == 94
    assert rows["case000"]["imaging_quality"] == 0.5
    assert e1.mean_stats(rows)["dynamic_degree"] == pytest.approx(0.6)


@pytest.mark.parametrize("kind", ["aggregate", "duplicate", "nonfinite", "gt_dataset", "stage_sha", "cache_sha", "media_sha", "budget", "scope"])
def test_strict_score_receipt_rejects_drift(plan, kind):
    path = make_scores(plan, "2" * 64, "base", "seed0")
    receipt = score.read_json(path)
    if kind in ("aggregate", "duplicate", "nonfinite"):
        dim = e1.DIMS[0]
        record = receipt["score_files"][dim]
        value = score.read_json(record["path"])
        if kind == "aggregate":
            value[dim][0] = 0.123
        elif kind == "duplicate":
            value[dim][1][-1] = value[dim][1][0]
        else:
            value[dim][1][0]["video_results"] = "nan"
        receipt["score_files"][dim] = dump(Path(record["path"]), value)
    elif kind == "gt_dataset":
        record = receipt["score_files"]["gtref"]
        value = score.read_json(record["path"])
        value["case000"]["dataset"] = "wrong"
        receipt["score_files"]["gtref"] = dump(Path(record["path"]), value)
    elif kind == "stage_sha":
        Path(receipt["stage_media"][0]["file"]["path"]).write_bytes(b"different")
    elif kind == "cache_sha":
        record = receipt["cache_binding"]
        value = score.read_json(record["path"])
        value["bundle_sha256"] = "3" * 64
        receipt["cache_binding"] = dump(Path(record["path"]), value)
    elif kind == "media_sha":
        receipt["source_media"] = copy.deepcopy(receipt["source_media"])
        receipt["source_media"][0]["file"]["sha256"] = "3" * 64
    elif kind == "budget":
        receipt["stage_cap_seconds"] = 19000
    else:
        receipt["scope_verified"] = False
    dump(path, receipt)
    with pytest.raises(RuntimeError):
        score.verify_score_receipt(plan, "2" * 64, "base", "seed0", path, verify_execution=False)


def test_booleans_do_not_prove_real_scoring_execution(plan):
    path = make_scores(plan, "2" * 64, "base", "seed0")
    with pytest.raises(RuntimeError, match="evidence verifier is not integrated"):
        score.verify_score_receipt(plan, "2" * 64, "base", "seed0", path)


def test_real_run_failclosed_no_cli_authorization():
    with pytest.raises(RuntimeError, match="Real scoring is closed"):
        score.real_score(gpu_authorized=True)
    with pytest.raises(RuntimeError, match="Real scoring is closed"):
        score.main(["run"])


def test_frozen_original_case_sha_not_replaceable(plan, monkeypatch):
    monkeypatch.setattr(score, "FROZEN_CASES_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="Frozen original cases byte SHA"):
        score.validate_plan(plan, verify_generation=False)


def test_generation_completion_adapter_currently_blocked(plan):
    with pytest.raises(ValueError, match="completion verification is unimplemented"):
        score.validate_plan(plan)


def test_configs_not_mutated():
    assert "base" not in e1.CONFIGS and "alg" not in e1.CONFIGS and "dymos" not in e1.CONFIGS
