"""CPU-only synthetic metadata; no models, scoring inputs, or media decoding."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_teacher_label_admission as prep


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return prep.bound(path)[0]


@contextmanager
def changed(path, raw):
    previous = path.read_bytes()
    path.write_bytes(raw)
    try:
        yield
    finally:
        path.write_bytes(previous)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("label_metadata")
    videos = root / "videos"
    videos.mkdir()
    source = videos / "source.bin"
    source.write_bytes(b"CPU")
    rows, media = [], []
    for i in range(1856):
        video = videos / ("clip_%04d.bin" % i)
        os.link(source, video)
        row = {"sample_id": "fixture_%04d" % i, "video": str(video),
               "captions": {"orig": "A still view of trees.", "fixed": "Trees remain still."}}
        if i % 10 == 0:
            row["captions"]["fixed"] = "A weapon is present."
        rows.append(row)
        media.append({"sample_id": row["sample_id"], "video": row["video"],
                      "bytes": 3, "sha256": prep.sha(b"CPU")})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    media_path = root / "media.json"
    media_proof = put(media_path, {"schema": "teacher_label_media_binding_v1",
                                  "manifest_sha256": prep.bound(manifest)[0]["sha256"], "items": media})
    model_root = root / "model"
    model_root.mkdir()
    (model_root / "config.json").write_bytes(b"{}")
    model_path = root / "model.json"
    model_proof = put(model_path, {"schema": "teacher_label_model_file_proof_v1",
                                  "name": "Qwen2.5-VL-7B-Instruct", "revision": "fixture-only",
                                  "root": str(model_root), "files": [{"relative_path": "config.json",
                                  "bytes": 2, "sha256": prep.sha(b"{}")} ]})
    runtime = root / "runtime.txt"
    runtime.write_bytes(b"cpu runtime fixture")
    environment = root / "environment.txt"
    environment.write_bytes(b"cpu environment fixture")
    approval = root / "approval.json"
    put(approval, {"scope": "cpu_metadata_only", "fixture_only": True})
    cpu_path = root / "cpu_records.json"
    cpu = {"schema": "teacher_label_cpu_records_v1", "scope": "cpu_metadata_only",
           "source": [prep.bound(SCRIPTS / name)[0] for name in ("recaption_e3.py",
                      "prepare_teacher_label_admission.py", "run_teacher_label_admission.py")],
           "runtime": [prep.bound(runtime)[0]], "environment": [prep.bound(environment)[0]],
           "model_proof_sha256": model_proof["sha256"], "media_binding_sha256": media_proof["sha256"],
           "approval_record": prep.bound(approval)[0]}
    put(cpu_path, cpu)
    owned = root / "owned"
    owned.mkdir(mode=0o700)
    return {"args": (str(manifest), str(media_path), str(model_path), str(cpu_path), str(owned)),
            "root": root, "rows": rows, "cpu": cpu, "runtime": runtime, "owned": owned}


def test_deterministic_scoreblind_binding(evidence):
    a = prep.prepare(*evidence["args"])
    b = prep.prepare(*evidence["args"])
    assert a == b and len(a["pilot"]) == 16 and a["random_seed"] == 20261101
    assert a["metadata_eligible_rows"] == 1670
    assert a["global_pilot_reserved_gpu_seconds"] == 3600
    assert a["remaining_fixed_stage_gpu_seconds"] == 82800
    assert a["stage_gpu_hours"] == 24
    assert {r["rank"] for r in a["ranks"]} == set(range(4))
    assert len({r["sample_id"] for r in a["pilot"]}) == 16
    assert all(int(r["sample_id"].rsplit("_", 1)[1]) % 10 for r in a["pilot"])
    assert all("captions" not in r and r["original_prompt_sha256"] ==
               prep.sha(b"A still view of trees.") for r in a["pilot"])
    assert a["gpu_execution_allowed"] is False and a["semantic_review_pass"] is False
    assert a["pilot_is_semantic_gold"] is False and a["media_content_verified"] is False
    assert a["model_content_verified"] is False


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}',
                                 b'{"x":NaN}', b'{"x":1e999}', b'{"x":"\\u0000"}'])
def test_strict_metadata_rejection(raw):
    with pytest.raises(ValueError):
        prep.strict_json(raw)


@pytest.mark.parametrize("value", ["relative", "/tmp/../etc/passwd", "/tmp/\0bad"])
def test_unsafe_paths(value):
    with pytest.raises(ValueError):
        prep.path_value(value, exists=False)


def test_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"CPU")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        prep.bound(link)


def test_same_size_source_content_drift(evidence):
    runtime = evidence["runtime"]
    with changed(runtime, b"X" * runtime.stat().st_size):
        with pytest.raises(ValueError, match="Content proof drift"):
            prep.prepare(*evidence["args"])


@pytest.mark.parametrize("kind", ["duplicate_id", "duplicate_video", "score", "nul", "missing"])
def test_manifest_rejection(evidence, kind):
    rows = copy.deepcopy(evidence["rows"])
    if kind == "duplicate_id":
        rows[1]["sample_id"] = rows[0]["sample_id"]
    elif kind == "duplicate_video":
        rows[1]["video"] = rows[0]["video"]
    elif kind == "score":
        rows[0]["score"] = 1
    elif kind == "nul":
        rows[0]["sample_id"] = "bad\0id"
    else:
        rows.pop()
    path = Path(evidence["args"][0])
    with changed(path, "".join(json.dumps(r) + "\n" for r in rows).encode()):
        with pytest.raises(ValueError):
            prep.manifest_rows(path)


@pytest.mark.parametrize("relative", ["../config.json", "/tmp/config.json", "bad\0file"])
def test_model_path_traversal_rejected(evidence, relative):
    _, model = prep.read_evidence(evidence["args"][2])
    model["files"][0]["relative_path"] = relative
    with pytest.raises(ValueError):
        prep.validate_model(model)


def test_source_hash_and_cpu_scope_rejected(evidence):
    cpu_path = Path(evidence["args"][3])
    for mutation in ("source_sha", "gpu_scope", "approval_bool"):
        cpu = copy.deepcopy(evidence["cpu"])
        if mutation == "source_sha":
            cpu["source"][0]["sha256"] = "0" * 64
        elif mutation == "gpu_scope":
            cpu["scope"] = "fullassets_gpu_approved"
        else:
            cpu["approved"] = True
        with changed(cpu_path, json.dumps(cpu).encode()):
            with pytest.raises(ValueError):
                prep.prepare(*evidence["args"])


def test_duplicate_model_file_rejected(evidence):
    _, model = prep.read_evidence(evidence["args"][2])
    model["files"].append(copy.deepcopy(model["files"][0]))
    with pytest.raises(ValueError, match="Duplicate model"):
        prep.validate_model(model)


def test_existing_rank_output_refused(evidence):
    rank = evidence["owned"] / "rank0"
    rank.mkdir()
    try:
        with pytest.raises(ValueError, match="Existing rank paths"):
            prep.prepare(*evidence["args"])
    finally:
        rank.rmdir()


def valid_facts():
    return {"scene": "Trees.", "rendered_or_screen": False, "entities": [],
            "hands": {"visible": False, "held_object": None}, "actions_by_time": [],
            "start_end_difference": "no visible change"}


def test_schema_never_means_semantic_gold():
    record = {"prompt_version": prep.PROMPT_VERSION, "facts_json": valid_facts(), "caption": "Trees remain still."}
    assert prep.validate_caption_record(record)["semantic_review_pass"] is False
    record["prompt_version"] = "e3-grounded-recaption-v1"
    with pytest.raises(ValueError):
        prep.validate_caption_record(record)
    record["prompt_version"] = prep.PROMPT_VERSION
    record["facts_json"]["rendered_or_screen"] = 1
    with pytest.raises(ValueError):
        prep.validate_caption_record(record)


def test_default_cli_cpu_no_torch():
    code = "import sys; sys.path.insert(0, sys.argv[1]); import prepare_teacher_label_admission,run_teacher_label_admission; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code, str(SCRIPTS)], capture_output=True)
    assert result.returncode == 0, result.stderr
    for name in ("prepare_teacher_label_admission.py", "run_teacher_label_admission.py"):
        result = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], capture_output=True)
        assert result.returncode == 0
