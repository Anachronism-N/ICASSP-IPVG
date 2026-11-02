import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from test_prepare_teacher_label_admission import evidence, put
import prepare_teacher_label_admission as prep
import run_teacher_label_admission as run


@pytest.mark.parametrize("rank", [-1, 4, True, False, "0", None])
def test_wrong_rank_refused(evidence, rank):
    with pytest.raises(ValueError, match="Wrong rank"):
        run.allocation(prep.prepare(*evidence["args"]), rank)


@pytest.mark.parametrize("field,value", [("stage", "multinodevalidation6"),
                                        ("stage_gpu_hours", 32),
                                        ("global_pilot_reserved_gpu_seconds", 900),
                                        ("remaining_fixed_stage_gpu_seconds", 86400)])
def test_no_stage_borrowing(evidence, field, value):
    draft = prep.prepare(*evidence["args"])
    draft[field] = value
    with pytest.raises(ValueError):
        run.allocation(draft, 0)


@pytest.mark.parametrize("field,value", [("cap_seconds", 901), ("allocated_gpus", 8),
                                        ("rank", False), ("allocated_gpus", True),
                                        ("pilot_indices", [0, 1, 2]),
                                        ("output", "/tmp/not-owned.jsonl")])
def test_fixed_rank_scope(evidence, field, value):
    draft = prep.prepare(*evidence["args"])
    draft["ranks"][0][field] = value
    with pytest.raises(ValueError):
        run.allocation(draft, 0)


@pytest.mark.parametrize("receipt,expected", [(None, 900), ({}, 900),
    ({"cost_status": "unknown", "allocated_elapsed_seconds": 1}, 900),
    ({"cost_status": "known", "allocated_elapsed_seconds": None}, 900),
    ({"cost_status": "known", "allocated_elapsed_seconds": -1}, 900),
    ({"cost_status": "known", "allocated_elapsed_seconds": True}, 900),
    ({"cost_status": "known", "allocated_elapsed_seconds": float("nan")}, 900),
    ({"cost_status": "known", "allocated_elapsed_seconds": float("inf")}, 900),
    ({"cost_status": "known", "allocated_elapsed_seconds": 1200}, 1200),
    ({"cost_status": "known", "allocated_elapsed_seconds": 123.5}, 123.5)])
def test_conservative_cost_and_unclipped_overrun(receipt, expected):
    assert run.charged_seconds(receipt) == expected


def test_node_local_owned_reservation_retry_and_peer_refusal(evidence, tmp_path):
    draft = prep.prepare(*evidence["args"])
    digest = prep.sha(b"synthetic CPU fixture")
    p0, t0 = run.reserve_cpu(draft, digest, 0, str(tmp_path))
    p1, t1 = run.reserve_cpu(draft, digest, 1, str(tmp_path))
    before = p1.read_bytes()
    with pytest.raises(FileExistsError):
        run.reserve_cpu(draft, digest, 0, str(tmp_path))
    with pytest.raises(ValueError):
        run.finalize_cpu(p0, t1, None)
    cost = run.finalize_cpu(p0, t0, None)
    assert cost["charged_gpu_seconds"] == 900 and cost["gpu_execution_allowed"] is False
    assert p1.read_bytes() == before
    with pytest.raises(FileExistsError):
        run.finalize_cpu(p0, t0, {"cost_status": "known", "allocated_elapsed_seconds": 0})
    assert json.loads(p0.read_text())["global_pilot_reserved_gpu_seconds"] == 3600


def test_shared_ledger_path_refused(evidence):
    with pytest.raises(ValueError, match="Node-local"):
        run.reserve_cpu(prep.prepare(*evidence["args"]), prep.sha(b"fixture"), 0,
                        "/")


def test_approved_sha_and_draft_tampering_refused(evidence, tmp_path):
    path = tmp_path / "plan.json"
    draft = prep.prepare(*evidence["args"])
    proof = put(path, draft)
    assert run.verify_draft(str(path), proof["sha256"]) == draft
    with pytest.raises(ValueError):
        run.verify_draft(str(path), "0" * 64)
    draft["semantic_review_pass"] = True
    proof = put(path, draft)
    with pytest.raises(ValueError, match="Draft content/source drift"):
        run.verify_draft(str(path), proof["sha256"])


def test_real_run_entry_always_blocked(evidence, tmp_path, capsys):
    proof = put(tmp_path / "plan.json", prep.prepare(*evidence["args"]))
    assert run.main(["--plan", proof["path"], "--approved-plan-sha256", proof["sha256"], "--rank", "0"]) == 2
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["gpu_execution_allowed"] is False and result["status"] == "BLOCKED"
    assert "production_worker_not_implemented" in result["blockers"]
    assert "fixture_" not in output and "Trees" not in output
    with pytest.raises(SystemExit):
        run.main(["--plan", proof["path"], "--approved-plan-sha256", proof["sha256"],
                  "--rank", "0", "--gpu-authorized"])


def test_real_cpu_timeout_tree_cleanup_and_external_process_untouched():
    import run_teacher_generation as owned
    external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        other = owned.identity(external.pid)
        fake = copy.deepcopy(other)
        fake["start_ticks"] += 1
        assert owned.same_process(fake) is False
        with pytest.raises(ValueError):
            owned.session_members(fake)
        receipt = run.harmless_child_timeout(.35)
        assert receipt["cpu_only"] is True and receipt["gpu_release_verified"] is False
        assert receipt["timed_out"] is True and receipt["owned_tree_released"] is True
        assert receipt["allocated_elapsed_seconds"] >= .35
        signalled = receipt["signalled"]
        assert len({row["identity"]["pid"] for row in signalled}) >= 2
        assert all(row["signal"] == signal.SIGTERM for row in signalled)
        assert external.pid not in {row["identity"]["pid"] for row in signalled}
        assert not owned.same_process(receipt["owner"]) and owned.same_process(other)
    finally:
        external.terminate()
        external.wait(timeout=4)


@pytest.mark.parametrize("seconds", [0, -1, 3, True, float("nan")])
def test_cpu_timeout_cap_refused(seconds):
    with pytest.raises(ValueError):
        run.harmless_child_timeout(seconds)
