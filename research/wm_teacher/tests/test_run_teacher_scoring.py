"""Synthetic CPU-only runtime/scope/ledger attacks and real timeout/proc tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_teacher_scoring as runner
import score_teacher_devset as score
import score_e1 as e1
from test_score_teacher_devset import dump, make_plan, small_file, make_bundle, stable_source_inventory


def schedule():
    return [{"method": m, "seed": s, "dimension": d, "cap_seconds": 200}
            for m in score.METHODS for s in score.SEEDS for d in e1.DIMS]


@pytest.fixture
def plan(tmp_path, monkeypatch):
    value = make_plan(tmp_path)
    value["score_allocations"] = schedule()
    value["score_owner"] = {"hostname": os.uname().nodename, "ledger_path": str(tmp_path / "score.ledger.json")}
    monkeypatch.setattr(score, "FROZEN_CASES_SHA256", value["cases"]["sha256"])
    return value


def ledger(plan):
    return {"schema": runner.SCHEMA, "plan_sha256": "2" * 64, "common_execution_sha256": plan["common_execution_sha256"],
            "bundle_sha256": plan["bundle"]["sha256"], "caps": score.SCORE_CAP_SECONDS, "hostname": os.uname().nodename,
            "allocations": [{**a, "token": str(i), "status": "reserved", "allocated_gpus": 1} for i, a in enumerate(schedule())]}


def test_no_model_or_cuda_import_at_module_load():
    script = "import sys;sys.path.insert(0,'scripts');import run_teacher_scoring;assert not any(n in sys.modules for n in ('torch','dreamsim','vbench2_beta_i2v'))"
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=score.ROOT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True)
    assert result.returncode == 0, result.stderr


def test_exact_fixed48_schedule_and_no_stage_borrow():
    assert runner.validate_schedule(schedule())
    value = schedule(); value[0]["cap_seconds"] = 20000
    with pytest.raises(RuntimeError, match="exceed method cap"):
        runner.validate_schedule(value)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "boolcap", "extra", "seed2"])
def test_schedule_attack_rejected(kind):
    value = schedule()
    if kind == "missing": value.pop()
    elif kind == "duplicate": value[-1] = value[0]
    elif kind == "boolcap": value[0]["cap_seconds"] = True
    elif kind == "extra": value[0]["generation_cap"] = 1
    else: value[0]["seed"] = "seed2"
    with pytest.raises(RuntimeError): runner.validate_schedule(value)


def test_immutable_owner_prevents_second_host_or_ledger(plan, tmp_path):
    runner.owner_matches(plan, plan["score_owner"]["ledger_path"])
    with pytest.raises(RuntimeError, match="immutable plan"):
        runner.owner_matches(plan, tmp_path / "other.ledger.json")
    plan["score_owner"]["hostname"] = "other-node"
    with pytest.raises(RuntimeError, match="immutable plan"):
        runner.owner_matches(plan, tmp_path / "score.ledger.json")


def test_local_only_ledger_rejects_shared(plan):
    with pytest.raises(RuntimeError, match="Node-local"):
        runner.local_path(Path("/shared_teacher_fixture/eval/score.ledger.json"))


def test_reserved_or_lost_parent_retains_full_cap(plan):
    value = ledger(plan)
    value["allocations"][0]["status"] = "running"
    totals = runner.validate_ledger(value, plan, "2" * 64)
    assert totals == {m: 3200 for m in score.METHODS}
    value["allocations"][0].update(status="finished", parent_receipt=None)
    with pytest.raises(RuntimeError, match="retain cap"):
        runner.validate_ledger(value, plan, "2" * 64)


@pytest.mark.parametrize("field,new", [("caps", {"base": 99999, "alg": 57600, "dymos": 57600}), ("hostname", "another-host"), ("plan_sha256", "3" * 64)])
def test_ledger_identity_attack(plan, field, new):
    value = ledger(plan); value[field] = new
    with pytest.raises(RuntimeError, match="identity/fixed budget"):
        runner.validate_ledger(value, plan, "2" * 64)


def test_ledger_duplicate_action_and_cap_change_rejected(plan):
    value = ledger(plan); value["allocations"][-1] = value["allocations"][0]
    with pytest.raises(RuntimeError): runner.validate_ledger(value, plan, "2" * 64)
    value = ledger(plan); value["allocations"][0]["cap_seconds"] += 1
    with pytest.raises(RuntimeError, match="cap drift"):
        runner.validate_ledger(value, plan, "2" * 64)


def test_fcntl_durable_transaction_no_mutation_on_failed_check(plan, tmp_path):
    path = tmp_path / "score.ledger.json"
    dump(path, ledger(plan)); before = path.read_bytes()
    with pytest.raises(RuntimeError):
        with runner.ledger_locked(path) as value:
            value["caps"] = {}
            raise RuntimeError("abort")
    assert path.read_bytes() == before
    with runner.ledger_locked(path) as value:
        value["cpu_fixture_counter"] = 1
    assert score.read_json(path)["cpu_fixture_counter"] == 1


def test_fcntl_cpu_concurrent_updates(plan, tmp_path):
    path = tmp_path / "score.ledger.json"
    dump(path, {"counter": 0})
    command = "import sys;sys.path.insert(0,sys.argv[1]);import run_teacher_scoring as r;" + chr(10) + "for _ in range(20):" + chr(10) + " with r.ledger_locked(sys.argv[2]) as v: v['counter']+=1"
    children = [subprocess.Popen([sys.executable, "-B", "-c", command, str(score.ROOT / "scripts"), str(path)],
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")) for _ in range(4)]
    assert all(p.wait(timeout=30) == 0 for p in children)
    assert score.read_json(path)["counter"] == 80


def test_real_cpu_proc_identity_start_ticks_and_session():
    value = runner.proc_identity(os.getpid())
    assert value["pid"] == os.getpid() and value["ppid"] == os.getppid()
    assert value["start_ticks"] > 0 and value["sid"] == os.getsid(0)


def test_real_cpu_timeout_kill30_and_owned_cleanup(tmp_path):
    command = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30s", "0.2s", sys.executable, "-B", "-c", "import time;time.sleep(20)"]
    child = subprocess.Popen(command, start_new_session=True)
    identity = runner.proc_identity(child.pid)
    assert identity["sid"] == child.pid
    assert child.wait(timeout=5) == 124
    runner.stop_owned(child, time.monotonic() + 1)
    assert runner.session_processes(child.pid) == []


def test_live_missing_parent_timeout_scope_refused():
    with pytest.raises(RuntimeError, match="approved parent"):
        runner.bounded_ancestry(100, 0, {"pid": 999999999, "start_ticks": 1})


def test_execution_bundle_requires_actual_runtime_not_supplier_schema(tmp_path):
    bundle = make_bundle(tmp_path)
    with pytest.raises(RuntimeError, match="Actual scoring execution runtime"):
        runner.validate_execution_bundle(bundle)


def test_original_dimension_formula_and_imaging_scale(plan, tmp_path):
    ids = [c["id"] for c in score.read_cases(plan["cases"]["path"])]
    dim = "imaging_quality"
    path = tmp_path / "dimension.json"
    value = {dim: [0.5, [{"video_path": i + "-0.mp4", "video_results": 50} for i in ids]]}
    dump(path, value)
    assert runner.validate_dimension(path, dim, plan) == value
    value[dim][0] = 50; dump(path, value)
    with pytest.raises(RuntimeError, match="mean/aggregate"):
        runner.validate_dimension(path, dim, plan)


@pytest.mark.parametrize("kind", ["duplicate", "nan", "missing"])
def test_dimension_coverage_attack(plan, tmp_path, kind):
    dim = e1.DIMS[0]; ids = [c["id"] for c in score.read_cases(plan["cases"]["path"])]
    rows = [{"video_path": i + "-0.mp4", "video_results": 0.5} for i in ids]
    if kind == "duplicate": rows[-1] = rows[0]
    elif kind == "missing": rows.pop()
    else: rows[0]["video_results"] = "nan"
    path = tmp_path / "dimension.json"; dump(path, {dim: [0.5, rows]})
    with pytest.raises(RuntimeError): runner.validate_dimension(path, dim, plan)


def query_fixture():
    identity = {"pid": 222, "ppid": 1, "sid": 222, "pgid": 222, "start_ticks": 10,
                "cmdline": ["python", "gpu_hold.py", "--action", "run"]}
    return {"exit_code": 0, "gpu_uuid": "GPU-synthetic", "gpu_index": 0, "queried_epoch": 100,
            "stdout": "", "observed_pids": [], "allowed_holder_pids": [], "holder_identity": identity,
            "resumed_holder_identity": None,
            "status": {"pid": 222, "running": True, "monitor_only": True, "query_ok": True, "updated_at": 99,
                       "gpus": {"0": {"gpu_uuid": "GPU-synthetic", "external_pids": [], "holder_pid": None, "paused": True}}}}


def test_raw_original_query_receipt_not_boolean():
    value = query_fixture()
    assert runner.verify_query(value, "GPU-synthetic")
    value["stdout"] = "GPU-synthetic, 12345\n"
    with pytest.raises(RuntimeError, match="Raw GPU query"):
        runner.verify_query(value, "GPU-synthetic")


@pytest.mark.parametrize("kind", ["stale", "external", "holder_identity", "allowed", "query_exit"])
def test_query_scope_attack(kind):
    value = query_fixture()
    if kind == "stale": value["status"]["updated_at"] = 80
    elif kind == "external": value["status"]["gpus"]["0"]["external_pids"] = [777]
    elif kind == "holder_identity": value["holder_identity"]["cmdline"] = ["unrelated"]
    elif kind == "allowed": value["allowed_holder_pids"] = [777]
    else: value["exit_code"] = 1
    with pytest.raises(RuntimeError): runner.verify_query(value, "GPU-synthetic")


def test_final_scoring_booleans_without_real_chain_rejected(plan):
    receipt = {"success": True, "parent_release_verified": True, "budget_verified": True,
               "scope_verified": True, "owned_tree_released": True}
    with pytest.raises(RuntimeError, match="Independent scoring evidence missing"):
        runner.verify_execution(receipt, plan)


def test_worker_unknown_action_fails_before_import_or_cuda(tmp_path):
    path = tmp_path / "action.json"
    dump(path, {"schema": "wrong", "hostname": os.uname().nodename})
    with pytest.raises(RuntimeError, match="schema"):
        runner.worker(path, score.sha256(path))


def test_plan_without_three_true_completions_cannot_assemble(plan, tmp_path, monkeypatch):
    comparison = {"score_bundle": plan["bundle"]}
    import prepare_teacher_comparison
    monkeypatch.setattr(prepare_teacher_comparison, "verify", lambda *args: comparison)
    with pytest.raises(RuntimeError, match="All three real generation completions"):
        score.assemble_plan(tmp_path / "comparison", "2" * 64, {}, {}, [], tmp_path / "out.json")


def parent_fixture(plan, tmp_path):
    bundle = score.read_json(plan["bundle"]["path"])
    bundle["runtime"]["observation"] = {"synthetic_runtime": True}
    bundle["runtime"]["library_files"] = []
    tools = {name: small_file(tmp_path / "tools" / name)["path"] for name in ("python", "ffmpeg", "ffprobe", "timeout", "nvidia_smi")}
    bundle["execution"] = {"python": tools["python"], "vbench_repo": str(tmp_path), "tools": tools}
    plan["bundle"] = dump(Path(plan["bundle"]["path"]), bundle)
    plan_record = dump(tmp_path / "scoreplan.json", plan)
    parent_id = {"pid": 1000, "ppid": 999, "sid": 999, "pgid": 999, "start_ticks": 100, "cmdline": ["synthetic-parent"]}
    stage_record = dump(tmp_path / "stage.json", {"synthetic": True, "work": str(tmp_path)})
    action_path = tmp_path / "action.json"
    action = {"schema": runner.ACTION_SCHEMA, "hostname": os.uname().nodename, "plan": plan_record, "stage": stage_record,
              "token": "synthetic-token", "method": "base", "seed": "seed0", "dimension": e1.DIMS[0], "cap_seconds": 200,
              "gpu_index": 0, "gpu_uuid": "GPU-synthetic", "holder_pid": 222, "state_dir": str(tmp_path),
              "parent_identity": parent_id, "allocation_start_monotonic": 200, "allocation_start_epoch": 100,
              "worker_receipt_path": str(tmp_path / "worker.json")}
    parent_id["cmdline"] = ["python", str(Path(runner.__file__).resolve()), "run", "--plan", plan_record["path"],
                            "--approved-plan-sha256", plan_record["sha256"], "--method", "base", "--seed", "seed0",
                            "--dimension", e1.DIMS[0], "--gpu", "0", "--holder-pid", "222", "--stage", stage_record["path"],
                            "--ledger", plan["score_owner"]["ledger_path"], "--state-dir", str(tmp_path)]
    action_record = dump(action_path, action)
    command = runner.command_for({**action, **{"path": action_record["path"], "sha256": action_record["sha256"]}}, bundle["execution"])
    child_id = {"pid": 1001, "ppid": 1000, "sid": 1001, "pgid": 1001, "start_ticks": 101, "cmdline": command}
    query_before = query_fixture()
    launch = {"action": action_record, "parent_identity": parent_id, "child_identity": child_id, "command": command,
              "before_gpu_query": query_before, "observed_monotonic": 200.5, "observed_epoch": 100.5}
    launch_record = dump(tmp_path / "launch.json", launch)
    dim = e1.DIMS[0]
    rows = [{"video_path": c["id"] + "-0.mp4", "video_results": 0.5} for c in score.read_cases(plan["cases"]["path"])]
    score_file = dump(tmp_path / "scores" / ("base__" + dim + "_eval_results.json"), {dim: [0.5, rows]})
    query_worker = copy.deepcopy(query_before); query_worker["queried_epoch"] = 101; query_worker["status"]["updated_at"] = 100
    worker_id = {"pid": 1003, "ppid": 1002, "sid": 1001, "pgid": 1001, "start_ticks": 103, "cmdline": command[command.index("--") + 1:]}
    holder_exec = {"pid": 1002, "ppid": 1001, "sid": 1001, "pgid": 1001, "start_ticks": 102, "cmdline": command[4:]}
    worker = {"schema": "teacher_score_worker_receipt_v1", "success": True, "action_sha256": action_record["sha256"],
              "token": action["token"], "bundle_sha256": plan["bundle"]["sha256"], "cuda_visible": "GPU-synthetic",
              "worker_identity": worker_id, "ancestry": [holder_exec, child_id, parent_id], "before_gpu_query": query_worker,
              "score_file": score_file, "asset_reads": [bundle["models"][0]], "method": "base", "seed": "seed0", "dimension": dim,
              "start_monotonic": 201, "model_begin_monotonic": 202, "end_monotonic": 203, "elapsed_seconds": 2,
              "start_epoch": 101, "end_epoch": 103}
    worker_record = dump(tmp_path / "worker.json", worker)
    query_after = copy.deepcopy(query_before); query_after["queried_epoch"] = 105; query_after["status"]["updated_at"] = 104
    proof = {"schema": runner.PARENT_SCHEMA, "action": action_record, "launch": launch_record, "worker": worker_record,
             "allocated_gpus": 1, "method": "base", "seed": "seed0", "dimension": dim, "token": action["token"],
             "parent_identity": parent_id, "child_identity": child_id, "elapsed_seconds": 5, "cap_seconds": 200,
             "allocation_start_monotonic": 200, "allocation_end_monotonic": 205, "start_epoch": 100, "end_epoch": 105,
             "session_remaining_pids": [], "after_gpu_query": query_after, "success": True, "exit_code": 0,
             "timed_out": False, "interrupted_signal": None, "error": None,
             "runtime_observation": bundle["runtime"]["observation"], "bundle_content_sha256": score.digest(bundle)}
    release_command = [tools["nvidia_smi"], "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    for q in (proof["after_gpu_query"],):
        q.update(command=release_command, start_monotonic=204, end_monotonic=204.1)
    first_release = copy.deepcopy(proof["after_gpu_query"])
    first_release.update(start_monotonic=203.5, end_monotonic=203.6, queried_epoch=104)
    proof["release_queries"] = [first_release, proof["after_gpu_query"]]
    proof["child_owner"] = {"pid": 1001, "session": 1001, "start_ticks": 101}
    proof["tracked_owned_processes"] = []
    proof["cleanup_signals"] = []
    record = dump(tmp_path / "supervised.json", proof)
    return plan_record, record, proof


def test_complete_mock_only_parent_chain_and_original_dimension(plan, tmp_path):
    plan_record, record, proof = parent_fixture(plan, tmp_path)
    assert runner.verify_parent_release(record["path"], record["sha256"], plan, plan_record["sha256"]) == proof


@pytest.mark.parametrize("kind", ["elapsed", "wallclock", "session", "pid", "release_query", "runtime", "missing_worker", "timeout", "bundle", "command"])
def test_parent_release_chain_attacks(plan, tmp_path, kind):
    plan_record, record, proof = parent_fixture(plan, tmp_path)
    proof = copy.deepcopy(proof)
    if kind == "elapsed": proof["elapsed_seconds"] = 0
    elif kind == "wallclock": proof["end_epoch"] = 110
    elif kind == "session": proof["session_remaining_pids"] = [123]
    elif kind == "pid": proof["child_identity"]["ppid"] = 777
    elif kind == "release_query": proof["after_gpu_query"]["stdout"] = "GPU-synthetic, 123\n"
    elif kind == "runtime": proof["runtime_observation"] = {"fake": True}
    elif kind == "missing_worker": proof["worker"] = None
    elif kind == "timeout": proof["timed_out"] = True
    elif kind == "bundle": proof["bundle_content_sha256"] = "4" * 64
    else: proof["child_identity"]["cmdline"] = ["unbounded-python"]
    record = dump(Path(record["path"]), proof)
    with pytest.raises(RuntimeError):
        runner.verify_parent_release(record["path"], record["sha256"], plan, plan_record["sha256"])


def test_released_failure_has_measured_cost_not_zero(plan, tmp_path):
    plan_record, record, proof = parent_fixture(plan, tmp_path)
    proof.update(success=False, exit_code=1, worker=None, error="synthetic failure")
    record = dump(Path(record["path"]), proof)
    result = runner.verify_parent_release(record["path"], record["sha256"], plan, plan_record["sha256"], require_success=False)
    assert result["elapsed_seconds"] == 5
    with pytest.raises(RuntimeError, match="failed/missing worker"):
        runner.verify_parent_release(record["path"], record["sha256"], plan, plan_record["sha256"])


def test_gt_duration_original_min_seconds_minus1e3(plan):
    cases = score.read_cases(plan["cases"]["path"])
    stage = {"decode": [{"case_id": c["id"], "generated": {"frames": 121, "fps": 24}, "gt": {"frames": 125, "fps": 25}} for c in cases]}
    values = {c["id"]: {"dataset": c["dataset"], "dreamsim": 0.1, "dreamsim_last": 0.1, "motion_ratio": 0.1, "dur": 4.999} for c in cases}
    assert runner.validate_gt_duration(values, stage, cases)
    values["case000"]["dur"] = 5
    with pytest.raises(RuntimeError, match="shared duration changed"):
        runner.validate_gt_duration(values, stage, cases)


def real_bundle_fixture(tmp_path, monkeypatch):
    supplied = make_bundle(tmp_path)
    tools = {name: small_file(tmp_path / "runtime" / name)["path"] for name in ("python", "ffmpeg", "ffprobe", "timeout", "nvidia_smi")}
    lib = tmp_path / "stdlib"; small_file(lib / "module.py", b"synthetic standard library")
    observation = {"versions": supplied["runtime"]["versions"], "dependency_roots": supplied["dependency_roots"],
                   "loaded_modules": {k: str(Path(v) / "module.py") for k, v in supplied["dependency_roots"].items()},
                   "library_roots": {"stdlib": str(lib)}, "python_invocation": tools["python"],
                   "python_prefix": str(tmp_path), "base_prefix": str(tmp_path)}
    monkeypatch.setattr(runner, "runtime_probe", lambda *args, **kwargs: copy.deepcopy(observation))
    config = {"python": tools["python"], "vbench_repo": supplied["dependency_roots"]["vbench"],
              "model_roots": supplied["model_roots"], "tools": {k: v for k, v in tools.items() if k != "python"}}
    source = dump(tmp_path / "runtime.config.json", config)
    record = runner.collect_bundle(source["path"], tmp_path / "actual.bundle.json")
    return score.read_json(record["path"]), record


def test_collect_actual_bundle_interface_with_mock_only_probe(tmp_path, monkeypatch):
    bundle, record = real_bundle_fixture(tmp_path, monkeypatch)
    assert score.verify_bundle(record["path"], record["sha256"]) == bundle
    assert bundle["execution"]["gt_K"] == 12
    assert runner.validate_execution_bundle(bundle) == bundle["execution"]


def test_actual_full_library_inventory_and_tool_sha_drift(tmp_path, monkeypatch):
    bundle, _ = real_bundle_fixture(tmp_path, monkeypatch)
    small_file(Path(bundle["runtime"]["library_roots"]["stdlib"]) / "new.py")
    with pytest.raises(RuntimeError, match="coverage"):
        runner.validate_execution_bundle(bundle)
    bundle, _ = real_bundle_fixture(tmp_path / "second", monkeypatch)
    binary = Path(bundle["execution"]["tools"]["timeout"]); binary.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="SHA drift"):
        score.validate_bundle(bundle)


def test_actual_prefix_version_probe_drift_refused(tmp_path, monkeypatch):
    bundle, _ = real_bundle_fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(bundle["runtime"]["observation"])
    changed["python_prefix"] = "/different-prefix"
    monkeypatch.setattr(runner, "runtime_probe", lambda *args, **kwargs: changed)
    with pytest.raises(RuntimeError, match="versions/prefix drift"):
        runner.validate_execution_bundle(bundle)


def test_real_cpu_asset_audit_refuses_unbound_weights_and_network(tmp_path):
    bound = small_file(tmp_path / "bound.pt", b"synthetic-not-a-model")
    unknown = small_file(tmp_path / "unknown.pt", b"synthetic-not-a-model")
    bundle = {"models": [bound], "dependencies": [], "runtime": {"files": [], "library_files": []}}
    record = dump(tmp_path / "audit.bundle.json", bundle)
    command = r'''
import json,sys,socket
sys.path.insert(0,sys.argv[1]);import run_teacher_scoring as r
b=r.score.read_json(sys.argv[2]); reads=r.install_asset_audit(b)
assert open(sys.argv[3],"rb").read()==b"synthetic-not-a-model"
for operation in (lambda:open(sys.argv[4],"rb"),lambda:socket.socket().connect(("127.0.0.1",1))):
 try:operation()
 except RuntimeError:pass
 else:raise AssertionError("Audit should reject before any actual network or model load")
assert list(reads)==[sys.argv[3]]
'''
    result = subprocess.run([sys.executable, "-B", "-c", command, str(score.ROOT / "scripts"), record["path"], bound["path"], unknown["path"]],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["wrong_plan", "wrong_sha", "seed", "duplicate"])
def test_completion188_media_exact_binding(plan, tmp_path, kind):
    group = plan["groups"]["base"]
    generation_plan = dump(tmp_path / "generation.plan.json", {"synthetic": True})
    group["generation_plan"] = generation_plan
    rows = [{"job": {"seed": int(seed[-1]), "case_id": row["case_id"]}, "video": {"file": row["file"]}}
            for seed, entry in group["seeds"].items() for row in entry["media"]]
    completed = {"plan": generation_plan, "cases": copy.deepcopy(rows)}
    assert score.verify_completion_media(completed, group)
    if kind == "wrong_plan": completed["plan"] = {**generation_plan, "sha256": "4" * 64}
    elif kind == "wrong_sha": completed["cases"][0]["video"]["file"]["sha256"] = "4" * 64
    elif kind == "seed": completed["cases"][0]["job"]["seed"] = 2
    else: completed["cases"][-1] = completed["cases"][0]
    with pytest.raises(RuntimeError): score.verify_completion_media(completed, group)


def test_assemble_scoring_plan_mock_only_true_completion_interface(plan, tmp_path, monkeypatch):
    import prepare_teacher_comparison
    comparison_plans, completions, verified = {}, {}, {}
    for method in score.METHODS:
        comparison_plans[method] = dump(tmp_path / method / "generation.plan.json", {"output_root": str(tmp_path / method)})
        completions[method] = plan["groups"][method]["generation_completion"]
        verified[method] = {"plan": comparison_plans[method], "cases": [
            {"job": {"seed": int(seed[-1]), "case_id": row["case_id"]}, "video": {"file": row["file"]}}
            for seed, entry in plan["groups"][method]["seeds"].items() for row in entry["media"]]}
    comparison = {"common_execution_sha256": plan["common_execution_sha256"], "score_bundle": plan["bundle"], "plans": comparison_plans}
    record = dump(tmp_path / "comparison.json", comparison)
    monkeypatch.setattr(prepare_teacher_comparison, "verify", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(score, "verify_completion", lambda row, method, *args: verified[method])
    monkeypatch.setattr(runner, "validate_execution_bundle", lambda *args, **kwargs: True)
    output = tmp_path / "assembled.json"
    result = score.assemble_plan(record["path"], record["sha256"], completions, plan["score_owner"], schedule(), output)
    value = score.read_json(result["path"])
    assert value["score_owner"] == plan["score_owner"] and value["score_allocations"] == schedule()
    assert value["comparison"] == record and value["groups"]["base"]["generation_plan"] == comparison_plans["base"]
    assert score.validate_plan(value)


def test_real_score_proxy_dispatches_exact_controlled_entry_not_custom_parent(monkeypatch):
    calls = []
    monkeypatch.setattr(score.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    score.real_score(["--", "run", "--plan", "/tmp/approved-private-plan.json"])
    assert calls == [([sys.executable, "-B", str(score.ROOT / "scripts/run_teacher_scoring.py"),
                       "run", "--plan", "/tmp/approved-private-plan.json"], {"check": True})]


def test_atomic_ledger_partial_temp_write_preserves_state_and_lock_inode(tmp_path, monkeypatch):
    import prepare_teacher_recovery as recovery
    path = tmp_path / "scoring.ledger.json"
    dump(path, {"counter": 1})
    with runner.ledger_locked(path) as value:
        assert value == {"counter": 1}
    before = path.read_bytes()
    lock = path.with_name(path.name + ".lock")
    lock_inode = lock.stat().st_ino
    original = recovery.os.fdopen

    class BrokenWrite:
        def __init__(self, handle): self.handle = handle
        def __enter__(self): self.handle.__enter__(); return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def write(self, payload):
            self.handle.write(b"{PARTIAL")
            self.handle.flush()
            raise OSError("injected partial temporary-state write")
        def __getattr__(self, name): return getattr(self.handle, name)

    with monkeypatch.context() as patch:
        patch.setattr(recovery.os, "fdopen", lambda *args, **kwargs: BrokenWrite(original(*args, **kwargs)))
        with pytest.raises(OSError, match="partial temporary"):
            with runner.ledger_locked(path) as value:
                value["counter"] = 2
    assert path.read_bytes() == before
    assert score.read_json(path) == {"counter": 1}
    assert lock.stat().st_ino == lock_inode
    with runner.ledger_locked(path) as value: value["counter"] = 3
    assert score.read_json(path) == {"counter": 3} and lock.stat().st_ino == lock_inode


@pytest.mark.parametrize("kind", ["zero_elapsed", "zero_start", "query_zero_duration", "repeated_same_query"])
def test_parent_release_strict_positive_sequential_clocks(plan, tmp_path, kind):
    plan_record, record, proof = parent_fixture(plan, tmp_path)
    if kind == "zero_elapsed":
        proof.update(elapsed_seconds=0, end_epoch=proof["start_epoch"],
                     allocation_end_monotonic=proof["allocation_start_monotonic"])
    elif kind == "zero_start":
        proof["start_epoch"] = 0
    elif kind == "query_zero_duration":
        for query in proof["release_queries"]: query["end_monotonic"] = query["start_monotonic"]
    else:
        proof["release_queries"] = [copy.deepcopy(proof["after_gpu_query"]), proof["after_gpu_query"]]
    record = dump(Path(record["path"]), proof)
    with pytest.raises(RuntimeError):
        runner.verify_parent_release(record["path"], record["sha256"], plan, plan_record["sha256"], require_success=False)


def test_failed_ledger_known_overcap_uses_observed_upper(plan):
    value = ledger(plan)
    chosen = value["allocations"][0]
    chosen.update(status="running", elapsed_observed_seconds=235, charged_upper_seconds=235, release_verified=False)
    totals = runner.validate_ledger(value, plan, "2" * 64)
    assert totals["base"] == 3200 + 35
    assert chosen["status"] == "running" and chosen["charged_upper_seconds"] > chosen["cap_seconds"]


def test_real_cpu_native_safetensors_known_same_value_unknown_refused(tmp_path):
    command = r'''
import json,sys,io
sys.path.insert(0,sys.argv[1])
import numpy as np, torch, safetensors, safetensors.numpy as sn, safetensors.torch as st
import run_teacher_scoring as r
known=sys.argv[2]+"/known.safetensors";unknown=sys.argv[2]+"/unknown.safetensors"
sn.save_file({"v":np.array([0.0],dtype=np.float32)},known)
sn.save_file({"v":np.array([1.0],dtype=np.float32)},unknown)
b={"sources":[],"models":[r.score.file_record(known)],"dependencies":[],"runtime":{"files":[],"library_files":[]}}
reads=r.install_asset_audit(b)
for call in (lambda:safetensors.safe_open(unknown,framework="np"),lambda:st.safe_open(unknown,framework="pt"),lambda:st.load_file(unknown)):
 try:call()
 except RuntimeError:pass
 else:raise AssertionError("unfrozen native asset accepted")
with safetensors.safe_open(known,framework="np") as f: assert f.get_tensor("v").tolist()==[0.0]
assert st.load_file(known)["v"].tolist()==[0.0]
assert known in reads
print("native known value unchanged; unknown native files rejected before read")
'''
    result = subprocess.run([sys.executable, "-B", "-c", command, str(score.ROOT / "scripts"), str(tmp_path)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_real_cpu_native_named_bytesio_cannot_borrow_frozen_filename(tmp_path):
    command = r'''
import sys,io
sys.path.insert(0,sys.argv[1]);import torch, safetensors, safetensors.torch
import run_teacher_scoring as r
path=sys.argv[2]+"/known.pt"
torch.save(torch.tensor([0.0]),path)
unknown=io.BytesIO();torch.save(torch.tensor([99.0]),unknown);unknown.seek(0);unknown.name=path
b={"sources":[],"models":[r.score.file_record(path)],"dependencies":[],"runtime":{"files":[],"library_files":[]}}
r.install_asset_audit(b)
try:torch.load(unknown,weights_only=True)
except RuntimeError:pass
else:raise AssertionError("unknown BytesIO borrowed frozen filename and loaded foreign bytes")
assert torch.load(path,weights_only=True).tolist()==[0.0]
'''
    result = subprocess.run([sys.executable, "-B", "-c", command, str(score.ROOT / "scripts"), str(tmp_path)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_real_sigalrm_posthash_has_durable_failure_and_retains_observed_upper(plan, tmp_path, monkeypatch):
    import summarize_teacher_devset as summary
    import run_teacher_generation as generation
    plan_record = dump(tmp_path / "timer.plan.json", plan)
    work = tmp_path / "timer-work"; work.mkdir()
    review_record = dump(work / "review.json", {"created_epoch": 1})
    stage = {"plan_sha256": plan_record["sha256"], "work": str(work), "review_protocol": review_record,
             "cache_binding": {"synthetic": True}}
    stage_record = dump(work / "stage.json", stage)
    gt_file = small_file(work / "gtref.json")
    dump(work / "gtref.receipt.json", {"plan_sha256": plan_record["sha256"], "bundle_sha256": plan["bundle"]["sha256"],
                                     "cache_binding": stage["cache_binding"], "score_file": gt_file})
    value = ledger(plan); value["plan_sha256"] = plan_record["sha256"]
    value["allocations"][0]["cap_seconds"] = 0.6
    ledger_path = Path(plan["score_owner"]["ledger_path"]); dump(ledger_path, value)
    query = query_fixture();query["status"]["gpus"]["0"]["paused"] = False
    bundle = {"execution": {"python": sys.executable}, "runtime": {"observation": {}}}
    calls = []
    def posthash(*args, **kwargs):
        calls.append("hash")
        if len(calls)>1: time.sleep(1.0)
        return bundle
    monkeypatch.setattr(score, "validate_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(score, "verify_bundle", posthash)
    monkeypatch.setattr(runner, "verify_stage", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "validate_ledger", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "verify_parent_command", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "gpu_query", lambda *args, **kwargs: copy.deepcopy(query))
    monkeypatch.setattr(summary, "validate_review_protocol", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "CLEANUP_SECONDS", 0.02)
    # Actual subprocess/timeout/signal execution; only GPU/content evidence is mocked.
    monkeypatch.setattr(runner, "command_for", lambda *args, **kwargs:
                        ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30s", "0.03s", sys.executable, "-B", "-c", "import time;time.sleep(.01)"])
    with pytest.raises(RuntimeError, match="hard cap exhausted"):
        runner.run_action(plan_record["path"], plan_record["sha256"], "base", "seed0", e1.DIMS[0],
                          stage_record["path"], ledger_path, tmp_path, 0, 222)
    folder = work / "actions" / e1.DIMS[0]
    failure_path = folder / "supervised.failure.json"
    failure = score.read_json(failure_path)
    assert len(calls) == 2 and failure["success"] is False and failure["actual_allocated_seconds"] is None
    assert failure["release_verified"] is False and failure["charged_upper_seconds"] >= max(.15, failure["elapsed_observed_seconds"])
    assert failure["elapsed_observed_seconds"] >= .1 and failure["allocation_end_monotonic"] > failure["allocation_start_monotonic"]
    updated = score.read_json(ledger_path)["allocations"][0]
    assert updated["status"] == "running" and updated["failure_receipt"]["sha256"] == score.sha256(failure_path)
    assert updated["charged_upper_seconds"] == failure["charged_upper_seconds"]


def test_official_local_cache_route_is_frozen_and_precedes_import(tmp_path, monkeypatch):
    import inspect
    bundle, _ = real_bundle_fixture(tmp_path, monkeypatch)
    assert bundle["execution"]["vbench_local"] is True
    assert bundle["execution"]["cache_dir"] == bundle["model_roots"]["vbench"]
    source = inspect.getsource(runner.worker)
    assert source.index('os.environ["VBENCH_CACHE_DIR"]') < source.index('from vbench2_beta_i2v import VBenchI2V')
    assert 'mode="vbench_standard", local=True' in source
    assert 'dimension_list=[action["dimension"]]' in source
    changed = copy.deepcopy(bundle)
    changed["execution"]["cache_dir"] = "/tmp/unapproved-metric-cache"
    with pytest.raises(RuntimeError):
        runner.validate_execution_bundle(changed, observe=False, verify_content=False)


def test_actual_regular_filehandle_reads_bound_native_tensor(tmp_path):
    command = r'''
import sys
sys.path.insert(0,sys.argv[1]);import torch,safetensors,safetensors.torch
import run_teacher_scoring as r
p=sys.argv[2]+"/known.pt";torch.save(torch.tensor([0.0]),p)
b={"sources":[],"models":[r.score.file_record(p)],"dependencies":[],"runtime":{"files":[],"library_files":[]}}
reads=r.install_asset_audit(b)
with open(p,"rb") as handle: assert torch.load(handle,weights_only=True).tolist()==[0.0]
assert p in reads
'''
    result = subprocess.run([sys.executable, "-B", "-c", command, str(score.ROOT / "scripts"), str(tmp_path)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_collect_and_verify_legitimate_empty_python_source_scopes(tmp_path, monkeypatch):
    bundle, _ = real_bundle_fixture(tmp_path, monkeypatch)
    root = Path(bundle["dependency_roots"]["numpy"])
    (root / "__init__.py").write_bytes(b"")
    library = Path(bundle["runtime"]["library_roots"]["stdlib"])
    (library / "namespace.py").write_bytes(b"")
    dino = Path(bundle["model_roots"]["vbench"])
    (dino / "__init__.py").write_bytes(b"")
    cfg = {"python": bundle["execution"]["python"], "vbench_repo": bundle["execution"]["vbench_repo"],
           "model_roots": bundle["model_roots"], "tools": {k: v for k, v in bundle["execution"]["tools"].items() if k != "python"}}
    config = dump(tmp_path / "empty-source.config.json", cfg)
    record = runner.collect_bundle(config["path"], tmp_path / "empty-source.bundle.json")
    complete = score.verify_bundle(record["path"], record["sha256"])
    zero = [r for key in ("dependencies", "models") for r in complete[key] if r["bytes"] == 0]
    zero += [r for r in complete["runtime"]["library_files"] if r["bytes"] == 0]
    assert len(zero) == 3 and all(Path(r["path"]).suffix == ".py" for r in zero)


def test_actual_cpu_asset_audit_allows_empty_python_code_not_empty_weight(tmp_path):
    command = r'''
import sys
sys.path.insert(0,sys.argv[1]);import torch,safetensors,safetensors.torch
import run_teacher_scoring as r
from pathlib import Path
source=Path(sys.argv[2])/"__init__.py";source.write_bytes(b"")
weight=Path(sys.argv[2])/"empty.pt";weight.write_bytes(b"")
b={"sources":[r.score.file_record(source,allow_empty=True)],"models":[r.score.file_record(weight,allow_empty=True)],
   "dependencies":[],"runtime":{"files":[],"library_files":[]}}
r.install_asset_audit(b)
assert source.read_bytes()==b""
try: torch.load(weight,weights_only=True)
except RuntimeError: pass
else: raise AssertionError("Empty model weight accepted as source")
'''
    result = subprocess.run([sys.executable, "-B", "-c", command, str(score.ROOT / "scripts"), str(tmp_path)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"), capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
