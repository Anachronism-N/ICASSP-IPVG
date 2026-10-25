#!/usr/bin/env python3
"""CPU preflight and bounded, one-shot per-node launch for the reviewed E3 subset."""
import argparse
from collections import Counter
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path("/dockerdata/models")
STATE = Path("/dockerdata/wm_logs/gpu_hold")
LOCAL_ROOT = Path("/dockerdata")
TAG = "e3C_subset_20261025"
COUNTS = {"vbench": 71, "vbench_po": 71, "wm": 22, "physiq": 42, "wmb": 70}
TERM_SECONDS = 2940
WALL_SECONDS = 3000
GPUS = "0,1,2,3,4,5,6,7"
ASSETS = tuple(
    f"Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-{i:05d}-of-00003.safetensors"
    for i in range(1, 4)
) + (
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
) + tuple("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/" + name for name in (
    "special_tokens_map.json", "spiece.model", "tokenizer_config.json", "tokenizer.json"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exclusive(path, value):
    with Path(path).open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())


def record(path):
    path = Path(path).absolute()
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "File changed during content hashing")
    return {"path": str(path), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": h.hexdigest()}


def check_record(expected):
    actual = record(expected["path"])
    require(all(actual[k] == expected[k] for k in ("path", "bytes", "sha256")),
            "Frozen content SHA256/size mismatch: " + expected["path"])
    return actual


def node_path(plan_path, rank, suffix):
    require(type(rank) is int and 0 <= rank < 4, "Node rank must be 0..3")
    return Path(plan_path).absolute().parent / f"node_{rank}{suffix}"


def verify_inputs(plan_path, approved_sha):
    data = Path(plan_path).read_bytes()
    require(len(approved_sha) == 64 and sha(data) == approved_sha, "Approved plan SHA256 mismatch")
    plan = json.loads(data)
    require(plan["schema"] == "e3_subset_v1" and plan["private"] is True and plan["tag"] == TAG,
            "Unexpected subset plan")
    require(plan["selection"]["counts"] == COUNTS, "Unapproved subset counts")
    b = plan["budget"]
    require(b["per_node_wall_seconds"] == 3000 and b["generation_nodes"] == 4
            and b["gpus_per_node"] == 8 and b["total_generation_gpus"] == 32
            and b["generation_gpu_hours"] == 28 and b["score_gpu_hours"] == 12
            and b["total_gpu_hours"] == 40 and b["training_authorized"] is False
            and b["full_1366_authorized"] is False, "Unapproved GPU budget")
    manifests = {r["path"]: r for r in plan["manifest_files"]}
    require(len(manifests) == len(plan["manifest_files"]) == 2
            and set(manifests) == {plan["jobs_path"], plan["cases_path"]}, "Unfrozen manifest")
    for r in manifests.values():
        require(r["bytes"] > 0, "Empty required manifest")
        check_record(r)
    jobs = [json.loads(line) for line in Path(plan["jobs_path"]).read_bytes().splitlines() if line.strip()]
    cases = [json.loads(line) for line in Path(plan["cases_path"]).read_bytes().splitlines() if line.strip()]
    require(len(jobs) == len(cases) == 276 and dict(Counter(c["suite"] for c in cases)) == COUNTS,
            "Manifest coverage changed")
    cp = plan["checkpoint"]
    frozen = {r["path"]: r for r in plan["frozen_files"]}
    require(len(frozen) == len(plan["frozen_files"]) and frozen.get(cp["path"]) == cp, "Unfrozen checkpoint")
    root = Path(plan["jobs_path"]).parents[2]
    scripts = root / "scripts"
    required_code = {str(scripts / name) for name in (
        "gen_jobs.py", "prepare_e3_subset.py", "run_e3_subset_node.py", "run_gen.sh", "gpu_hold.py")}
    imports = {str(p) for p in (root / "DiffSynth-Studio/diffsynth").rglob("*.py")}
    frozen_imports = {p for p in frozen if Path(p).is_relative_to(root / "DiffSynth-Studio/diffsynth")}
    require(imports and imports == frozen_imports and required_code <= set(frozen), "Generator imports not frozen")
    runtime = {plan["runtime"]["python"]}
    for package in plan["runtime"]["packages"].values():
        runtime.add(package["entrypoint"])
    source_paths = {p for p in frozen if Path(p).is_relative_to(scripts)} | imports | runtime
    source_paths |= {p for p in frozen if p.endswith(".dist-info/METADATA")}
    require(source_paths <= set(frozen), "Runtime source not frozen")
    sources = [check_record(frozen[p]) for p in sorted(source_paths)]
    require(cp["bytes"] > 0, "Empty required checkpoint")
    check_record(cp)
    for j, c in zip(jobs, cases):
        require(j["model"] == TAG and j["ckpt"] == cp["path"] and j["seed"] == 0
                and j.get("steps", 50) == 50 and j.get("cfg", 5.0) == 5.0 and j.get("shift", 5.0) == 5.0
                and c["video"] == j["out"] and c["image"] == j["image"], "Generator contract changed")
        require(Path(j["out"]).is_relative_to(Path(plan_path).absolute().parent / "videos")
                and ".." not in Path(j["out"]).parts, "Output escapes private subset")
        require(j["image"] in frozen and frozen[j["image"]]["bytes"] > 0 and Path(j["image"]).is_file()
                and Path(j["image"]).stat().st_size == frozen[j["image"]]["bytes"], "Missing/incomplete node input")
    require(len({j["out"] for j in jobs}) == 276, "Duplicate outputs")
    cache = MODELS / "wm_checkpoints" / (sha(os.path.abspath(cp["path"]).encode()) + ".safetensors")
    require(cache.resolve().is_relative_to(MODELS.resolve()) and cache.resolve().is_relative_to(LOCAL_ROOT.resolve()),
            "Checkpoint cache is not local")
    cached = record(cache)
    require(cached["bytes"] > 0, "Empty required checkpoint cache")
    require(cached["sha256"] == cp["sha256"] and cached["bytes"] == cp["bytes"], "C checkpoint cache SHA256 mismatch")
    return plan, sources, cached


def preflight(plan_path, rank, approved_sha):
    output = node_path(plan_path, rank, "_preflight.json")
    require(not output.exists(), "Node preflight already exists; never overwrite")
    plan, sources, cached = verify_inputs(plan_path, approved_sha)
    expected_shards = {MODELS / p for p in ASSETS[:3]}
    require(set((MODELS / "Wan-AI/Wan2.2-TI2V-5B").glob("diffusion_pytorch_model*.safetensors"))
            == expected_shards, "Unexpected or missing DiT shards")
    models = []
    for asset in ASSETS:
        path = MODELS / asset
        require(path.resolve().is_relative_to(MODELS.resolve()) and path.resolve().is_relative_to(LOCAL_ROOT.resolve()),
                "Model asset must be node-local, not a shared symlink")
        model = record(path)
        require(model["bytes"] > 0, "Empty required model asset")
        models.append(dict(model, asset=asset))
    model_sha = sha(encoded([{k: r[k] for k in ("asset", "bytes", "sha256")} for r in models]))
    value = {"schema": "e3_subset_node_preflight_v1", "rank": rank, "plan_sha256": approved_sha,
             "created_at": time.time(), "hostname": os.uname().nodename, "models": models,
             "model_content_sha256": model_sha, "checkpoint_cache": cached,
             "source_imports_sha256": sha(encoded(sources)), "manifest_files": plan["manifest_files"],
             "GPU_used": False}
    exclusive(output, value)
    return output, sha(output.read_bytes())


def group_alive(pgid):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop_group(child, deadline):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # Keep a full second within the 3000-second cap for SIGKILL/reap.
    kill_at = min(time.monotonic() + 59, deadline - 1)
    while group_alive(child.pid) and time.monotonic() < kill_at:
        time.sleep(min(0.05, max(0, kill_at - time.monotonic())))
    if group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait(timeout=max(0.01, deadline - time.monotonic()))
    while group_alive(child.pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    require(not group_alive(child.pid), "Owned process tree did not release; holder remains paused")


def command_for(plan, rank):
    scripts = Path(plan["jobs_path"]).parents[2] / "scripts"
    command = [plan["runtime"]["python"], str(scripts / "gpu_hold.py"), "--state-dir", str(STATE),
               "--action", "exec", "--gpus", GPUS, "--", "bash", str(scripts / "run_gen.sh"),
               plan["jobs_path"], TAG, "--models", TAG]
    env = dict(os.environ, NODE_RANK=str(rank), NNODES="4")
    return command, env


def run_node(plan_path, rank, approved_sha, approved_preflight_sha):
    launch = node_path(plan_path, rank, ".launch.json")
    require(not launch.exists(), "Node rank already launched; never overwrite or resume")
    require(not node_path(plan_path, rank, ".finished.json").exists(), "Node receipt already exists; never overwrite")
    pf_path = node_path(plan_path, rank, "_preflight.json")
    data = pf_path.read_bytes()
    require(sha(data) == approved_preflight_sha, "Approved node preflight SHA256 mismatch")
    pf = json.loads(data)
    require(pf["schema"] == "e3_subset_node_preflight_v1" and pf["rank"] == rank
            and pf["plan_sha256"] == approved_sha and pf["hostname"] == os.uname().nodename, "Wrong node preflight")
    for peer in range(4):
        other = json.loads(node_path(plan_path, peer, "_preflight.json").read_bytes())
        require(other["rank"] == peer and other["plan_sha256"] == approved_sha
                and other["model_content_sha256"] == pf["model_content_sha256"]
                and other["checkpoint_cache"]["sha256"] == pf["checkpoint_cache"]["sha256"],
                "All four node model content hashes must match before launch")
    plan, sources, cached = verify_inputs(plan_path, approved_sha)
    require(sha(encoded(sources)) == pf["source_imports_sha256"] and cached == pf["checkpoint_cache"],
            "Source imports/checkpoint cache changed since preflight")
    require([r["asset"] for r in pf["models"]] == list(ASSETS), "Incomplete model preflight")
    require(set((MODELS / "Wan-AI/Wan2.2-TI2V-5B").glob("diffusion_pytorch_model*.safetensors"))
            == {MODELS / p for p in ASSETS[:3]}, "DiT shard set changed since preflight")
    for r in pf["models"]:
        p = MODELS / r["asset"]
        require(str(p) == r["path"] and p.resolve().is_relative_to(LOCAL_ROOT.resolve()), "Nonlocal model path")
        stat = p.stat()
        require((stat.st_size, stat.st_mtime_ns) == (r["bytes"], r["mtime_ns"]), "Local model drift since preflight")
    command, env = command_for(plan, rank)
    exclusive(launch, {"rank": rank, "plan_sha256": approved_sha, "preflight_sha256": approved_preflight_sha,
                       "command": command, "wall_cap_seconds": 3000, "complete_not_proven": True})
    child, owned_flags = None, []
    started, wall_start = time.monotonic(), time.time()
    interrupted, timed_out, error = None, False, None
    old = {}
    def interrupted_handler(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, interrupted_handler)
    try:
        STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Guard our pause flags against gpu_hold.exec clearing them before orphan cleanup.
        # Existing wrapped experiments are rejected, never stopped or unpaused.
        with contextlib.ExitStack() as stack:
            for gpu in range(8):
                lock = stack.enter_context((STATE / f"experiment.{gpu}.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for gpu in range(8):
                flag = STATE / f"pause.{gpu}"
                try:
                    fd = os.open(flag, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                owned_flags.append(flag)
        if time.monotonic() - started >= TERM_SECONDS:
            timed_out = True
        if interrupted is None and not timed_out:
            child = subprocess.Popen(command, env=env, shell=False, start_new_session=True)
            while child.poll() is None:
                if interrupted is not None:
                    break
                remaining = TERM_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                time.sleep(min(0.05, remaining))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        released = child is None
        try:
            if child is not None:
                if group_alive(child.pid):
                    stop_group(child, started + WALL_SECONDS)
                else:
                    child.wait()
                released = True
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        wall_end, elapsed = time.time(), time.monotonic() - started
        if released:
            for flag in owned_flags:
                flag.unlink(missing_ok=True)
        receipt = {"rank": rank, "start": wall_start, "end": wall_end, "elapsed_seconds": elapsed,
                   "gpu_hours": 8 * elapsed / 3600, "exit_code": child.returncode if child else None,
                   "timed_out": timed_out, "interrupted_signal": interrupted, "error": error,
                   "owned_tree_released": released, "cap_seconds": 3000, "cap_exceeded": elapsed > 3000,
                   "model_content_sha256": pf["model_content_sha256"], "plan_sha256": approved_sha,
                   "preflight_sha256": approved_preflight_sha, "complete_not_proven": True}
        try:
            exclusive(node_path(plan_path, rank, ".finished.json"), receipt)
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, choices=range(4), required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    parser.add_argument("--approved-preflight-sha256")
    args = parser.parse_args()
    if args.preflight:
        path, digest = preflight(args.plan, args.node_rank, args.approved_plan_sha256)
        print(json.dumps({"preflight": str(path), "sha256": digest, "GPU_used": False}))
    else:
        if not args.approved_preflight_sha256:
            parser.error("--run requires the lead-reviewed --approved-preflight-sha256")
        receipt = run_node(args.plan, args.node_rank, args.approved_plan_sha256, args.approved_preflight_sha256)
        print(json.dumps(receipt, allow_nan=False))
        if receipt["error"] or receipt["timed_out"] or receipt["interrupted_signal"] or receipt["exit_code"] != 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
