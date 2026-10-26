#!/usr/bin/env python3
"""One-shot C276 completion; CPU inspection by default, never scoring/publication."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import fcntl
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import run_e3_subset_node as n
import gpu_hold as hold
from prepare_e3_subset import encoded as input_encoded

PLAN_SHA = "afd1cd967ae03088930a1637ea93f66053f2adcaeae3b213f38e1c7160e57254"
RUN_ID = "global_completion_a1"
SCHEMA = "e3_subset_global_completion_v1"
RECOVERY_SHA = "4f17e00917e990b8341cc1d6907bc60453bd850acf51e481bd0ed7a9878ce5ac"
GPUS = (2, 3, 4)
NODE_RANK = 0
GEN_CAP, REPAIR_CAP, SHUTDOWN = 28, 2, 30
ORIGINAL_BUDGET = {"generation": 28, "score": 12, "total": 40}
AMENDED_BUDGET = {"generation": 30, "score": 12, "total": 42}
AMENDMENT_FILE = "budget_amendment_42.json"
TOTAL_FRAMES = 276 * 121
require = n.require


class BadVideo(ValueError):
    """Only failed media format/decode permits moving an approved output."""


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return n.record(path)["sha256"]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def budget_contract(experiment, plan_sha, amendment_sha=None):
    path = experiment / AMENDMENT_FILE
    if not path.exists():
        require(not path.is_symlink() and amendment_sha is None, "Missing approved budget amendment sidecar")
        return dict(ORIGINAL_BUDGET), None
    require(not path.is_symlink() and path.is_file() and isinstance(amendment_sha, str) and
            len(amendment_sha) == 64 and sha(path) == amendment_sha,
            "Budget amendment requires exact approved sidecar SHA")
    def unique(pairs):
        row = {}
        for key, value in pairs:
            require(key not in row, "Duplicate budget amendment key")
            row[key] = value
        return row
    row = json.loads(path.read_bytes(), object_pairs_hook=unique)
    require(row.get("schema") == "e3_subset_budget_amendment_v1" and row.get("user_authorized") is True and
            row.get("parent_plan_sha256") == plan_sha == PLAN_SHA and sha(experiment / "plan.json") == plan_sha,
            "Unapproved budget amendment schema/authorization/parent plan")
    require(row.get("original") == ORIGINAL_BUDGET and row.get("approved") == AMENDED_BUDGET and
            all(type(row[key][field]) is int for key in ("original", "approved") for field in ORIGINAL_BUDGET) and
            type(row.get("approved_additional_generation_gpu_hours")) is int and
            row["approved_additional_generation_gpu_hours"] == 2,
            "Only exact original28/12/40 to approved30/12/42 budget amendment is authorized")
    require(sha(path) == amendment_sha, "Budget amendment sidecar changed during verification")
    return dict(AMENDED_BUDGET), amendment_sha


def original_paths(experiment):
    return [experiment / f"node_{i}.finished.json" for i in range(4)] + [
        experiment / f"node_{i}_recovery_a1.finished.json" for i in (1, 2)]


def originals(experiment, plan_sha):
    paths = original_paths(experiment)
    require(set(experiment.glob("node_*.finished.json")) == set(paths), "Wait for exactly all six original receipts")
    rows, hashes = [], {}
    for index, path in enumerate(paths):
        digest, row = sha(path), read(path)
        rank = index if index < 4 else index - 3
        require(type(row.get("rank")) is int and row["rank"] == rank and row.get("plan_sha256") == plan_sha,
                "Original receipt identity mismatch")
        require(row.get("owned_tree_released") is True and row.get("cap_exceeded") is False,
                "Original receipt release/cap invalid")
        if index < 4:
            require(row.get("cap_seconds") == 3000, "Original cap missing")
        else:
            launch = read(path.with_name(path.name.replace(".finished.json", ".launch.json")))
            require(launch.get("rank") == rank and launch.get("plan_sha256") == plan_sha and
                    launch.get("run_id") == "recovery_a1" and launch.get("max_seconds") == 3000 and
                    launch.get("script_sha256") == RECOVERY_SHA and
                    sha(experiment.parents[1] / "scripts/run_e3_subset_recovery.py") == RECOVERY_SHA,
                    "Recovery cap/source evidence missing or changed")
        elapsed, cost = row.get("elapsed_seconds"), row.get("gpu_hours")
        require(finite(elapsed) and 0 < elapsed <= 3000 and finite(cost) and cost > 0 and
                math.isclose(cost, 8 * elapsed / 3600, abs_tol=1e-9), "Original actual billing invalid")
        require(finite(row.get("start")) and finite(row.get("end")) and row["end"] >= row["start"] and
                math.isclose(row["end"] - row["start"], elapsed, abs_tol=1), "Original wall clock evidence invalid")
        require("interrupted_signal" in row and "error" in row and row["interrupted_signal"] is None and row["error"] is None and
                type(row.get("exit_code")) is int and (index >= 4 or type(row.get("timed_out")) is bool),
                "User interrupt or unknown original error cannot be repaired")
        timed_out = row.get("timed_out") if index < 4 else row["exit_code"] != 0 and elapsed >= 2940
        if index in (1, 2):
            require(row["exit_code"] == 1 and row["timed_out"] is False and 90 <= elapsed <= 100,
                    "Unapproved original pause failure")
        elif timed_out:
            require(row["exit_code"] in (130, 143, -15, -9) and 2940 <= elapsed <= 3000,
                    "Unknown error is not an expected supervisor timeout")
        else:
            require(row["exit_code"] == 0, "Unknown generation failure")
        if index >= 4:
            require(row.get("run_id") == "recovery_a1", "Unapproved recovery history")
        require(sha(path) == digest, "Original receipt changed during verification")
        rows.append(row)
        hashes[str(path)] = digest
    used = sum(row["gpu_hours"] for row in rows)
    require(used <= GEN_CAP, "Original actual cost exceeds generation budget")
    return used, hashes


def allowance(used, approved, budget=None):
    budget = ORIGINAL_BUDGET if budget is None else budget
    require(budget in (ORIGINAL_BUDGET, AMENDED_BUDGET), "Unapproved generation budget")
    cap = REPAIR_CAP if budget == ORIGINAL_BUDGET else 3.4
    require(finite(used) and 0 <= used <= budget["generation"], "Negative/exceeded original budget")
    require(finite(approved) and 0 < approved <= cap, f"Repair authorization exceeds {cap} GPUh")
    remaining = min(budget["generation"] - used, approved)
    wall = math.floor(remaining * 3600 / len(GPUS))
    require(wall > SHUTDOWN, "BLOCKED: insufficient actual remaining generation budget")
    return remaining, wall, wall - SHUTDOWN


def protocol(plan_path, approved_sha):
    require(approved_sha == PLAN_SHA and sha(plan_path) == approved_sha, "Unapproved plan SHA")
    plan = read(plan_path)
    require(plan.get("schema") == "e3_subset_v1" and plan.get("private") is True and plan.get("tag") == n.TAG and
            plan["selection"]["counts"] == n.COUNTS, "Unapproved C276 selection")
    require(plan["budget"]["generation_gpu_hours"] == 28 and plan["budget"]["score_gpu_hours"] == 12 and
            plan["budget"]["total_gpu_hours"] == 40, "Frozen generation/score budgets changed")
    manifests = {r["path"]: r for r in plan["manifest_files"]}
    require(len(manifests) == len(plan["manifest_files"]) == 2 and
            set(manifests) == {plan["jobs_path"], plan["cases_path"]}, "Unfrozen jobs/cases")
    for rec in manifests.values():
        n.check_record(rec)
    jobs = [json.loads(line) for line in Path(plan["jobs_path"]).read_bytes().splitlines() if line.strip()]
    cases = [json.loads(line) for line in Path(plan["cases_path"]).read_bytes().splitlines() if line.strip()]
    require(len(jobs) == len(cases) == len({j["out"] for j in jobs}) == 276, "C276 coverage changed")
    require(dict(n.Counter(c["suite"] for c in cases)) == n.COUNTS, "Case suite coverage changed")
    frozen = {r["path"]: r for r in plan["frozen_files"]}
    tree = plan_path.parent / "videos"
    for job, case in zip(jobs, cases):
        out = Path(job["out"])
        require(out.is_absolute() and out.is_relative_to(tree) and ".." not in out.parts and out.suffix == ".mp4",
                "Output outside approved private manifest")
        require(case["video"] == job["out"] and case["image"] == job["image"] and
                job["model"] == n.TAG and job["ckpt"] == plan["checkpoint"]["path"] and
                job["seed"] == 0 and job.get("steps", 50) == 50 and job.get("cfg", 5.0) == 5.0 and
                job.get("shift", 5.0) == 5.0, "Same-protocol generator contract changed")
        require(case.get("input_contract_sha256") == n.sha(input_encoded({k: job[k] for k in
                ("image", "prompt", "seed", "cfg", "shift", "steps")})) and
                all(case.get(k) == job.get(k) for k in ("id", "prompt", "seed", "cfg", "shift", "steps", "neg")),
                "Case/source input contract changed")
        require(job["image"] in frozen, "Unfrozen static image")
        n.check_record(frozen[job["image"]])
    root = Path(plan["jobs_path"]).parents[2]
    for name in ("gen_jobs.py", "run_gen.sh", "gpu_hold.py", "run_e3_subset_node.py"):
        path = str(root / "scripts" / name)
        require(path in frozen, "Unfrozen original generator")
        n.check_record(frozen[path])
    for rec in plan["frozen_files"]:
        path = Path(rec["path"])
        if path.is_relative_to(root / "scripts") or path.is_relative_to(root / "DiffSynth-Studio/diffsynth"):
            n.check_record(rec)
    return plan, jobs


def no_writers():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        require(not any(Path(os.fsdecode(word)).name in ("gen_jobs.py", "run_gen.sh") for word in command),
                "Active generation writer; no cleanup/completion allowed")


def inventory(experiment, jobs):
    approved = {Path(j["out"]) for j in jobs}
    artifacts = {Path(str(p) + suffix) for p in approved for suffix in (".claim", ".tmp.mp4")}
    found, remnants = set(), []
    tree = experiment / "videos"
    require(not tree.is_symlink(), "Video tree symlink")
    for path in tree.rglob("*"):
        require(not path.is_symlink(), "Symlink in output tree")
        if path.is_dir():
            require(any(p.is_relative_to(path) for p in approved), "Unapproved output directory")
            continue
        require(path.is_file() and (path in approved or path in artifacts), "Unknown output/claim/tmp; preserve for review")
        if path in approved:
            found.add(path)
        else:
            remnants.append(path)
    return found, sorted(remnants)


def copy_local(source, destination):
    require(not Path(source).is_symlink() and Path(source).is_file(), "Nonregular generated output")
    before = n.record(source)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not destination.is_symlink() and not any(p.is_symlink() for p in destination.parents), "Staged video symlink")
    if destination.exists():
        require(destination.is_file() and sha(destination) == before["sha256"], "Existing local stage changed; preserve")
    else:
        with Path(source).open("rb") as src, destination.open("xb") as dst:
            os.chmod(destination, 0o600)
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    require(n.record(source) == before and sha(destination) == before["sha256"], "Source/local copy drift")
    return before


def decode(path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-show_streams",
                                "-of", "json", str(path)], check=True, capture_output=True, text=True, env=env, timeout=180)
        streams = json.loads(probe.stdout)["streams"]
        require(len(streams) == 1 and streams[0].get("codec_type") == "video", "Unexpected media streams")
        video = streams[0]
        count, fps = int(video["nb_read_frames"]), Fraction(video["avg_frame_rate"])
        require((count, fps, video["width"], video["height"]) == (121, 24, 1280, 704), "Wrong full-frame candidate format")
        completed = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "1",
                                   "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-threads", "1",
                                   "-filter_threads", "1", "-filter_complex_threads", "1", "-f", "null", "-"],
                                  check=True, capture_output=True, text=True, env=env, timeout=180)
        require(not probe.stderr.strip() and not completed.stderr.strip(), "Decoder reported errors")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, KeyError, ZeroDivisionError) as exc:
        raise BadVideo(str(exc)) from exc
    return {"sha256": sha(path), "all_frames_decoded": True, "actual_count_frames": count, "frames": count,
            "fps": float(fps), "width": video["width"], "height": video["height"], "decoder": "ffprobe-count_frames+ffmpeg-xerror"}


def stage_one(job, experiment, local, decoder):
    source = Path(job["out"])
    destination = local / source.relative_to(experiment / "videos")
    frozen = copy_local(source, destination)
    try:
        report = decoder(destination)
        if not (report.get("all_frames_decoded") is True and type(report.get("frames")) is int and
                report.get("actual_count_frames") == report["frames"] == 121 and
                (report.get("fps"), report.get("width"), report.get("height")) == (24, 1280, 704)):
            raise BadVideo("Full decode format invalid")
    except BadVideo as exc:
        require(n.record(source) == frozen and sha(destination) == frozen["sha256"], "Bad video changed during decode")
        return source, None, str(exc)
    require(report.get("sha256") == frozen["sha256"] == sha(destination) and n.record(source) == frozen,
            "Full decode content SHA drift")
    return source, dict(report, source=str(source), path=str(destination), source_stat=frozen), None


def stage_all(jobs, experiment, local, decoder=decode):
    require(local.is_absolute() and not local.resolve().is_relative_to(experiment.parents[1].resolve()) and
            str(local).startswith(("/dockerdata/", "/tmp/")), "Full decode stage must be launcher-local")
    require(not local.is_symlink(), "Local stage symlink")
    local.mkdir(mode=0o700, parents=True, exist_ok=True)
    outputs, bad = {}, {}
    available = [j for j in jobs if Path(j["out"]).is_file()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for source, row, error in pool.map(lambda j: stage_one(j, experiment, local, decoder), available):
            if error is None:
                outputs[str(source)] = row
            else:
                bad[source] = error
    return outputs, bad


def quarantine(experiment, paths):
    folder = experiment / (RUN_ID + ".quarantine")
    require(not folder.exists(), "Quarantine already exists; no retry")
    folder.mkdir(mode=0o700)
    records = []
    for path in sorted(paths):
        before = n.record(path)
        destination = folder / path.relative_to(experiment / "videos")
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        require(not destination.exists(), "Quarantine destination exists")
        record = dict(before, destination=str(destination), stat={k: getattr(path.stat(), k) for k in
                      ("st_mode", "st_ino", "st_dev", "st_uid", "st_gid", "st_ctime_ns")})
        n.exclusive(folder / f"move_{len(records):04d}.json", record)
        require(n.record(path) == before, "Quarantine source changed")
        os.rename(path, destination)
        require(sha(destination) == before["sha256"], "Quarantine SHA mismatch")
        records.append(record)
    n.exclusive(folder / "manifest.json", records)
    return records


def preflight(args):
    require(type(getattr(args, "node_rank", NODE_RANK)) is int and getattr(args, "node_rank", NODE_RANK) == NODE_RANK,
            "Completion requires node-rank 0 for launcher-local scoring stage")
    pfpath = args.experiment / "node_0_preflight.json"
    require(sha(pfpath) == args.preflight_sha256, "Approved node0 preflight SHA mismatch")
    pf = read(pfpath)
    require(pf["schema"] == "e3_subset_node_preflight_v1" and pf["rank"] == NODE_RANK and pf["plan_sha256"] == args.plan_sha256 and
            pf["hostname"] == os.uname().nodename == args.hostname, "Wrong node0 hostname/preflight")
    plan, sources, cached = n.verify_inputs(args.experiment / "plan.json", args.plan_sha256)
    require(n.sha(n.encoded(sources)) == pf["source_imports_sha256"] and cached == pf["checkpoint_cache"],
            "Original source/runtime/C cache changed")
    require([r["asset"] for r in pf["models"]] == list(n.ASSETS), "Incomplete node0 full model SHA preflight")
    require(set((n.MODELS / "Wan-AI/Wan2.2-TI2V-5B").glob("diffusion_pytorch_model*.safetensors")) ==
            {n.MODELS / p for p in n.ASSETS[:3]}, "Changed model shard set")
    for rec in pf["models"]:
        path = n.MODELS / rec["asset"]
        require(str(path) == rec["path"] and path.resolve().is_relative_to(n.LOCAL_ROOT), "Nonlocal model")
        n.check_record(rec)
    require(n.sha(n.encoded([{k: r[k] for k in ("asset", "bytes", "sha256")} for r in pf["models"]])) ==
            pf["model_content_sha256"], "Model content digest invalid")
    for rank in range(4):
        peer = read(args.experiment / f"node_{rank}_preflight.json")
        require(peer["rank"] == rank and peer["plan_sha256"] == args.plan_sha256 and
                peer["model_content_sha256"] == pf["model_content_sha256"] and
                peer["checkpoint_cache"]["sha256"] == cached["sha256"], "Peer original model/C hashes differ")
    return pf, plan


def repair_commands(plan, manifest, root):
    command = [plan["runtime"]["python"], str(root / "scripts/gen_jobs.py"), str(manifest)]
    base = dict(os.environ, PATH="/opt/conda/envs/torch-base/bin:" + os.environ.get("PATH", ""),
                PYTHONPATH=str(root / "DiffSynth-Studio"), DIFFSYNTH_MODEL_BASE_PATH="/dockerdata/models",
                DIFFSYNTH_SKIP_DOWNLOAD="true", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                TOKENIZERS_PARALLELISM="false", WM_CHECKPOINT_CACHE_DIR="/dockerdata/models/wm_checkpoints")
    require(plan["runtime"]["python"] == "/opt/conda/envs/torch-base/bin/python", "Original runtime changed")
    return [(command + [str(shard), "3", "--models", n.TAG], dict(base, CUDA_VISIBLE_DEVICES=str(gpu)))
            for shard, gpu in enumerate(GPUS)]


def free_gpus():
    mapping, processes = hold.query_gpus(), hold.query_processes()
    state = hold.read_state(n.STATE)
    for gpu in GPUS:
        require(gpu in mapping, "Required GPUs 2/3/4 unavailable")
        require(not (n.STATE / f"pause.{gpu}").exists(), "Existing GPU pause; never clear another scope")
        allowed = set()
        entry = state.get("gpus", {}).get(str(gpu), {})
        pid = entry.get("holder_pid")
        if pid is not None:
            require(state.get("running") is True and state.get("query_ok") is True and
                    finite(state.get("updated_at")) and 0 <= time.time() - state["updated_at"] < 10 and
                    hold.matching_process(state.get("pid", 0), "run") and hold.matching_process(pid, "worker"),
                    "Unknown GPU holder identity")
            allowed.add(pid)
        require(not (processes.get(mapping[gpu]["uuid"], set()) - allowed), "External/current-scope GPU process; no preemption")
    return mapping


def run_repair(args, plan, manifest, prior_hashes, approved):
    budget, amendment_sha = budget_contract(args.experiment, args.plan_sha256, getattr(args, "budget_amendment_sha256", None))
    remaining, wall, term = allowance(approved, args.max_repair_gpu_hours, budget)
    commands = repair_commands(plan, manifest, args.root)
    pf, _ = preflight(args)
    protocol(args.experiment / "plan.json", args.plan_sha256)
    manifest_sha = sha(manifest)
    frozen_stats = {r["path"]: (Path(r["path"]).stat().st_size, Path(r["path"]).stat().st_mtime_ns)
                    for r in plan["frozen_files"] + plan["manifest_files"] + pf.get("models", []) +
                    ([pf["checkpoint_cache"]] if "checkpoint_cache" in pf else [])}
    free_gpus()
    children, flags, codes, error = [], [], [], None
    interrupted, timed_out, released = None, False, True
    started, start = time.monotonic(), time.time()
    old = {}
    def handler(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, handler)
    try:
        with contextlib.ExitStack() as stack:
            for gpu in GPUS:
                lock = stack.enter_context((n.STATE / f"experiment.{gpu}.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                require(originals(args.experiment, args.plan_sha256) == (approved, prior_hashes), "Original receipt SHA drift before GPU")
                no_writers()
                free_gpus()
                for gpu in GPUS:
                    flag = n.STATE / f"pause.{gpu}"
                    fd = os.open(flag, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    os.close(fd)
                    flags.append(flag)
                pause_deadline = started + term - 22
                while True:
                    require(time.monotonic() < pause_deadline and interrupted is None, "Bounded holder pause deadline/interrupted")
                    state = hold.read_state(n.STATE)
                    if not state.get("running") or not hold.matching_process(state.get("pid", 0), "run"):
                        break
                    selected = [state.get("gpus", {}).get(str(g)) for g in GPUS]
                    if all(s and s["paused"] and s["holder_pid"] is None for s in selected):
                        break
                    time.sleep(.05)
                require(time.monotonic() + 21 < started + term, "Insufficient bounded idle-query interval")
                processes, mapping = hold.query_processes(), hold.query_gpus()
                require(all(not processes.get(mapping[g]["uuid"]) and mapping[g]["used_mib"] < 1024 for g in GPUS),
                        "GPU not truly idle after holder release")
                require(originals(args.experiment, args.plan_sha256) == (approved, prior_hashes), "Original receipts changed at GPU boundary")
                require(sha(args.experiment / "plan.json") == args.plan_sha256 and
                        sha(args.experiment / "node_0_preflight.json") == args.preflight_sha256 and sha(manifest) == manifest_sha,
                        "Plan/preflight/repair manifest drift at GPU boundary")
                require(budget_contract(args.experiment, args.plan_sha256, amendment_sha) == (budget, amendment_sha),
                        "Budget amendment drift at GPU boundary")
                for path, expected in frozen_stats.items():
                    current = Path(path).stat()
                    require((current.st_size, current.st_mtime_ns) == expected, "Frozen protocol/model/input drift at GPU boundary")
                require(time.monotonic() - started < term and interrupted is None, "Pause exhausted repair deadline/interrupted")
                for i, (command, env) in enumerate(commands):
                    log = stack.enter_context((args.experiment / f"{RUN_ID}.shard_{i}.log").open("xb"))
                    child = subprocess.Popen(command, env=env, cwd=args.root / "DiffSynth-Studio", shell=False,
                                             start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
                while any(c.poll() is None for c in children):
                    if interrupted is not None:
                        break
                    if time.monotonic() - started >= term:
                        timed_out = True
                        break
                    time.sleep(.05)
            finally:
                released = False
                failures = []
                for child in children:
                    if n.group_alive(child.pid):
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                while any(n.group_alive(c.pid) for c in children) and time.monotonic() < started + wall - 2:
                    time.sleep(.01)
                for child in children:
                    try:
                        if n.group_alive(child.pid):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=max(.01, started + wall - time.monotonic()))
                    except ProcessLookupError:
                        child.wait(timeout=max(.01, started + wall - time.monotonic()))
                    except BaseException as exc:
                        failures.append(str(exc))
                released = not failures and all(not n.group_alive(c.pid) for c in children)
                codes = [c.returncode for c in children]
                if released:
                    for flag in flags:
                        flag.unlink()
                require(released, "Owned repair tree unreleased: " + "; ".join(failures))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed, end = time.monotonic() - started, time.time()
        for sig, previous in old.items():
            signal.signal(sig, previous)
    cost = len(GPUS) * elapsed / 3600
    return {"start": start, "end": end, "elapsed_seconds": elapsed, "gpu_hours": cost, "gpus": list(GPUS),
            "wall_cap_seconds": wall, "term_seconds": term, "allowed_gpu_hours": remaining,
            "cap_exceeded": elapsed > wall or cost > remaining or approved + cost > budget["generation"],
            "owned_tree_released": released, "interrupted_signal": interrupted, "timed_out": timed_out,
            "error": error, "worker_exit_codes": codes, "commands": [r[0] for r in commands]}


def validate_global(experiment, plan_sha, budget_amendment_sha256=None):
    """Rehash full provenance and byte-copied stage, not just aggregate flags."""
    budget, amendment_sha = budget_contract(experiment, plan_sha, budget_amendment_sha256)
    repair_cap = REPAIR_CAP if amendment_sha is None else 3.4
    used, original_hashes = originals(experiment, plan_sha)
    path = experiment / (RUN_ID + ".finished.json")
    digest, row = sha(path), read(path)
    require(row.get("schema") == SCHEMA and row.get("run_id") == RUN_ID and row.get("plan_sha256") == plan_sha and
            row.get("original_six_sha256") == original_hashes and row.get("actual_prior_gpu_hours") == used,
            "Global receipt/original six SHA identity mismatch")
    require(row.get("source_sha256") == sha(Path(__file__)) and row.get("status") == "COMPLETE" and
            type(row.get("exit_code")) is int and row["exit_code"] == 0 and row.get("owned_tree_released") is True and
            row.get("cap_exceeded") is False and row.get("error") is None and row.get("interrupted_signal") is None and
            row.get("timed_out") is False, "Global completion failed/interrupted")
    require(type(row.get("node_rank")) is int and row["node_rank"] == NODE_RANK and
            row.get("preflight_path") == str(experiment / "node_0_preflight.json") and
            row.get("hostname") == os.uname().nodename, "Global stage must be verified on the same node0 hostname")
    require(row.get("budget_amendment_sha256") == amendment_sha and
            (amendment_sha is None or row.get("budget_gpu_hours") == budget), "Global budget amendment identity mismatch")
    stage, cost = row.get("repair_stage"), row.get("actual_repair_gpu_hours")
    require(finite(cost) and 0 <= cost <= repair_cap and finite(row.get("total_generation_gpu_hours")) and
            math.isclose(row["total_generation_gpu_hours"], used + cost, abs_tol=1e-9) and used + cost <= budget["generation"],
            "Global actual cost invalid/exceeded")
    if stage is None:
        require(cost == 0 and row.get("repair_job_count") == 0, "Unaccounted repair allocation")
    else:
        allowed, wall, term = allowance(used, row.get("approved_repair_gpu_hours"), budget)
        require(stage.get("gpus") == list(GPUS) and stage.get("wall_cap_seconds") == wall and stage.get("term_seconds") == term and
                stage.get("allowed_gpu_hours") == allowed and stage.get("owned_tree_released") is True and
                stage.get("cap_exceeded") is False and stage.get("interrupted_signal") is None and stage.get("timed_out") is False and
                stage.get("error") is None and stage.get("worker_exit_codes") == [0, 0, 0], "Invalid repair stage")
        elapsed = stage.get("elapsed_seconds")
        require(finite(elapsed) and 0 < elapsed <= wall and finite(stage.get("gpu_hours")) and
                math.isclose(stage["gpu_hours"], cost, abs_tol=1e-9) and math.isclose(cost, 3 * elapsed / 3600, abs_tol=1e-9) and
                finite(stage.get("start")) and finite(stage.get("end")) and
                math.isclose(stage["end"] - stage["start"], elapsed, abs_tol=1), "Repair elapsed/three GPU billing invalid")
    plan, jobs = protocol(experiment / "plan.json", plan_sha)
    no_writers()
    found, remnants = inventory(experiment, jobs)
    require(found == {Path(j["out"]) for j in jobs} and not remnants and all(p.stat().st_size > 0 for p in found),
            "Global zeroextra/empty/claims/temporary coverage failed")
    stage_path = Path(row["decode_manifest"])
    require(not stage_path.is_symlink() and sha(stage_path) == row.get("decode_manifest_sha256"), "Global decode manifest SHA drift")
    decoded = read(stage_path)
    require(decoded.get("schema") == "e3_subset_full_decode_v1" and decoded.get("plan_sha256") == plan_sha and
            decoded.get("node_rank") == NODE_RANK and decoded.get("hostname") == row["hostname"] and
            decoded.get("decode_ok") == row.get("decode_ok") == 276 and
            decoded.get("total_frames") == row.get("total_frames") == TOTAL_FRAMES and
            row.get("zeroextra") is True and row.get("zeroempty") is True and row.get("noactiveclaims_tmp") is True,
            "Global aggregate decode coverage invalid")
    outputs = decoded.get("outputs", {})
    require(set(outputs) == {j["out"] for j in jobs} and len(outputs) == 276, "Fake/partial whole completion")
    local = Path(row["local_stage"])
    require(local.is_absolute() and not local.resolve().is_relative_to(experiment.parents[1].resolve()) and
            str(local).startswith(("/dockerdata/", "/tmp/")) and not local.is_symlink() and
            stage_path == local.parent / "decode.json", "Global stage not launcher-local")
    require({p for p in local.rglob("*") if p.is_file()} == {Path(r["path"]) for r in outputs.values()}, "Local stage coverage changed")
    for job in jobs:
        report = outputs[job["out"]]
        destination = local / Path(job["out"]).relative_to(experiment / "videos")
        require(report.get("source") == job["out"] and report.get("path") == str(destination) and
                not any(p.is_symlink() for p in (destination, *destination.parents)) and report.get("all_frames_decoded") is True and
                report.get("decoder") == "ffprobe-count_frames+ffmpeg-xerror" and type(report.get("frames")) is int and
                report["frames"] == report.get("actual_count_frames") == 121 and
                (report.get("fps"), report.get("width"), report.get("height")) == (24, 1280, 704), "Global frame/format provenance failed")
        require(sha(job["out"]) == sha(destination) == report.get("sha256"), "Global source/local stage SHA drift")
    launch_path = experiment / (RUN_ID + ".launch.json")
    require(sha(launch_path) == row.get("launch_sha256"), "Global exclusive launch SHA drift")
    launch = read(launch_path)
    require(launch.get("plan_sha256") == plan_sha and launch.get("original_six_sha256") == original_hashes and
            launch.get("source_sha256") == row["source_sha256"] and launch.get("run_id") == RUN_ID,
            "Global launch identity changed")
    require(launch.get("budget_amendment_sha256") == amendment_sha and
            (amendment_sha is None or launch.get("budget_gpu_hours") == budget) and
            launch.get("approved_repair_gpu_hours") == row.get("approved_repair_gpu_hours"),
            "Global launch budget amendment changed")
    require(type(launch.get("node_rank")) is int and launch["node_rank"] == NODE_RANK and
            launch.get("preflight_path") == row["preflight_path"] and launch.get("hostname") == row["hostname"],
            "Global launch node0/hostname identity changed")
    require(sha(experiment / "node_0_preflight.json") == row.get("preflight_sha256") == launch.get("preflight_sha256"),
            "Global preflight SHA drift")
    pf = read(experiment / "node_0_preflight.json")
    require(pf.get("rank") == NODE_RANK and pf.get("plan_sha256") == plan_sha and pf.get("hostname") == row.get("hostname") and
            pf.get("model_content_sha256") == row.get("model_content_sha256") and
            all(read(p).get("model_content_sha256") == pf["model_content_sha256"] for p in original_paths(experiment)[:4]),
            "Global original/model/hostname provenance mismatch")
    if stage is not None:
        manifest = experiment / (RUN_ID + ".jobs.jsonl")
        require(sha(manifest) == row.get("repair_manifest_sha256") == launch.get("repair_manifest_sha256"), "Repair manifest SHA drift")
        repair = [json.loads(line) for line in manifest.read_bytes().splitlines()]
        approved_jobs = {j["out"]: j for j in jobs}
        require(len(repair) == row.get("repair_job_count") == launch.get("repair_job_count") and len(repair) > 0 and
                len({j["out"] for j in repair}) == len(repair) and all(j == approved_jobs.get(j["out"]) for j in repair),
                "Repair not identical approved missing-job protocol")
        commands = repair_commands(plan, manifest, Path(plan["jobs_path"]).parents[2])
        require(stage.get("commands") == [c[0] for c in commands], "Repair original generator argv changed")
    require(sha(path) == digest and originals(experiment, plan_sha) == (used, original_hashes), "Global receipt/history drift")
    return used + cost, {**original_hashes, str(path): digest}


def complete(args, decoder=decode):
    budget, amendment_sha = budget_contract(args.experiment, args.plan_sha256, getattr(args, "budget_amendment_sha256", None))
    used, prior = originals(args.experiment, args.plan_sha256)
    plan, jobs = protocol(args.experiment / "plan.json", args.plan_sha256)
    no_writers()
    found, remnants = inventory(args.experiment, jobs)
    if not args.run:
        outputs, bad = stage_all(jobs, args.experiment, args.work_dir / "inspect" / "videos", decoder)
        return {"status": "INSPECTED", "GPU_used": False, "actual_prior_gpu_hours": used,
                "remaining_generation_gpu_hours": budget["generation"] - used, "decode_ok": len(outputs),
                "missing": len(jobs) - len(found), "bad": len(bad), "remnants": len(remnants), "not_sota": True}
    require(args.gpu_authorized and args.hostname and args.preflight_sha256 and args.source_sha256 == sha(Path(__file__)),
            "Run requires explicit GPU authorization, hostname, full preflight and source SHA")
    pf, _ = preflight(args)
    launch_path, receipt_path = [args.experiment / (RUN_ID + suffix) for suffix in (".launch.json", ".finished.json")]
    require(not launch_path.exists() and not receipt_path.exists(), "Single global completion launch/receipt; no retry")
    outputs, bad = stage_all(jobs, args.experiment, args.work_dir / "initial" / "videos", decoder)
    repair = [j for j in jobs if Path(j["out"]) not in found or Path(j["out"]) in bad]
    if repair:
        allowance(used, args.max_repair_gpu_hours, budget)
    no_writers()
    require(originals(args.experiment, args.plan_sha256) == (used, prior), "Original receipts changed before exclusive launch")
    for path in original_paths(args.experiment):
        launch = read(path.with_name(path.name.replace(".finished.json", ".launch.json")))
        rank = read(path)["rank"]
        command, _ = n.command_for(plan, rank)
        require(launch.get("plan_sha256") == args.plan_sha256 and launch.get("rank") == rank and
                launch.get("command") == command and ("_recovery_" not in path.name or launch.get("run_id") == "recovery_a1"),
                "Original jobs/launch ownership not established")
    manifest = args.experiment / (RUN_ID + ".jobs.jsonl")
    if repair:
        with manifest.open("xb") as handle:
            for job in repair:
                handle.write(n.encoded(job))
            handle.flush()
            os.fsync(handle.fileno())
    launch = {"schema": SCHEMA, "run_id": RUN_ID, "plan_sha256": args.plan_sha256, "original_six_sha256": prior,
              "source_sha256": args.source_sha256, "preflight_sha256": args.preflight_sha256, "hostname": args.hostname,
              "node_rank": NODE_RANK, "preflight_path": str(args.experiment / "node_0_preflight.json"),
              "repair_job_count": len(repair), "repair_manifest_sha256": sha(manifest) if repair else None,
              "approved_repair_gpu_hours": args.max_repair_gpu_hours}
    if amendment_sha is not None:
        launch.update(budget_amendment_sha256=amendment_sha, budget_gpu_hours=budget)
    require(budget_contract(args.experiment, args.plan_sha256, amendment_sha) == (budget, amendment_sha),
            "Budget amendment drift before exclusive launch")
    n.exclusive(launch_path, launch)
    decode_manifest = args.work_dir / "final" / "decode.json"
    result = dict(launch, status="INCOMPLETE", exit_code=1, repair_stage=None, actual_prior_gpu_hours=used,
                  actual_repair_gpu_hours=0, total_generation_gpu_hours=used, owned_tree_released=True,
                  cap_exceeded=False, error=None, interrupted_signal=None, timed_out=False, decode_ok=0, total_frames=0,
                  launch_sha256=sha(launch_path), model_content_sha256=pf["model_content_sha256"],
                  decode_manifest=str(decode_manifest), local_stage=str(args.work_dir / "final" / "videos"),
                  zeroextra=False, zeroempty=False, noactiveclaims_tmp=False)
    try:
        if remnants or bad:
            no_writers()
            require(originals(args.experiment, args.plan_sha256) == (used, prior), "Original receipt drift before quarantine")
            result["quarantine"] = quarantine(args.experiment, set(remnants) | set(bad))
        if repair:
            stage = run_repair(args, plan, manifest, prior, used)
            result.update(repair_stage=stage, actual_repair_gpu_hours=stage["gpu_hours"],
                          total_generation_gpu_hours=used + stage["gpu_hours"], owned_tree_released=stage["owned_tree_released"],
                          cap_exceeded=stage["cap_exceeded"], interrupted_signal=stage["interrupted_signal"], timed_out=stage["timed_out"])
            require(not stage["error"] and stage["worker_exit_codes"] == [0, 0, 0] and not stage["timed_out"] and
                    stage["interrupted_signal"] is None and not stage["cap_exceeded"] and stage["owned_tree_released"],
                    "Repair stage failed: " + str(stage))
        protocol(args.experiment / "plan.json", args.plan_sha256)
        require(originals(args.experiment, args.plan_sha256) == (used, prior), "Original receipt drift at completion")
        no_writers()
        found, remnants = inventory(args.experiment, jobs)
        require(found == {Path(j["out"]) for j in jobs} and not remnants, "Global coverage/claim/tmp incomplete")
        outputs, bad = stage_all(jobs, args.experiment, args.work_dir / "final" / "videos", decoder)
        result["decode_ok"] = len(outputs)
        require(len(outputs) == 276 and not bad and all(Path(j["out"]).stat().st_size > 0 for j in jobs),
                "Global 276 full-frame decode incomplete")
        no_writers()
        final_found, final_remnants = inventory(args.experiment, jobs)
        require(final_found == found and not final_remnants and
                originals(args.experiment, args.plan_sha256) == (used, prior), "Final writer/coverage/receipt drift")
        protocol(args.experiment / "plan.json", args.plan_sha256)
        n.exclusive(decode_manifest, {"schema": "e3_subset_full_decode_v1", "plan_sha256": args.plan_sha256,
                                    "node_rank": NODE_RANK, "hostname": args.hostname,
                                    "decode_ok": 276, "total_frames": TOTAL_FRAMES, "outputs": outputs})
        require(budget_contract(args.experiment, args.plan_sha256, amendment_sha) == (budget, amendment_sha),
                "Budget amendment drift at completion")
        result.update(status="COMPLETE", exit_code=0, total_frames=TOTAL_FRAMES, zeroextra=True, zeroempty=True,
                      noactiveclaims_tmp=True, decode_manifest_sha256=sha(decode_manifest))
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        n.exclusive(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gpu-authorized", action="store_true")
    parser.add_argument("--root", type=Path, default=n.ROOT)
    parser.add_argument("--work-dir", type=Path, default=Path("/dockerdata/wm_eval/e3_subset_completion_a1"))
    parser.add_argument("--approved-plan-sha256", dest="plan_sha256")
    parser.add_argument("--approved-preflight-sha256", dest="preflight_sha256")
    parser.add_argument("--approved-source-sha256", dest="source_sha256")
    parser.add_argument("--hostname")
    parser.add_argument("--node-rank", type=int, choices=[NODE_RANK], default=NODE_RANK)
    parser.add_argument("--approved-budget-amendment-sha256", dest="budget_amendment_sha256")
    parser.add_argument("--max-repair-gpu-hours", type=float, default=2)
    args = parser.parse_args()
    if args.run and not args.plan_sha256:
        parser.error("--run needs explicit --approved-plan-sha256")
    args.plan_sha256 = args.plan_sha256 or PLAN_SHA
    args.root = args.root.resolve()
    args.experiment = args.root / "eval/e3_subset_20261025"
    args.work_dir = args.work_dir.absolute()
    os.umask(0o077)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with (args.experiment / (RUN_ID + ".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = complete(args)
        except Exception as exc:
            print(json.dumps({"status": "BLOCKED", "error": str(exc), "GPU_used": False, "not_sota": True}))
            raise SystemExit(1)
        print(json.dumps(result, allow_nan=False))
        if args.run and result["exit_code"] != 0:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
