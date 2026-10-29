#!/usr/bin/env python3
"""Private bounded Teacher scorer. CPU prepare/GT never reserves a GPU.

A GPU action is exactly one original VBench dimension under GNU timeout and
live gpu_hold exec. Local durable reservations count failures and lost parents;
unknown old outputs are never silently resumed, replaced or deleted.
"""
import argparse
import contextlib
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import score_teacher_devset as score
import score_e1 as e1

require = score.require
SCHEMA = "teacher_score_stage_ledger_v1"
ACTION_SCHEMA = "teacher_score_action_v1"
PARENT_SCHEMA = "teacher_score_parent_receipt_v1"
LOCAL_PREFIXES = ("/dockerdata/", "/tmp/", "/dev/shm/")
CLEANUP_SECONDS = 120


def local_path(path):
    path = Path(path).absolute()
    require(path.resolve() == path and str(path).startswith(LOCAL_PREFIXES), "Node-local canonical scoring path required")
    return path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def files_under(root):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), "Canonical dependency root required")
    paths = []
    for path in root.rglob("*"):
        require(not path.is_symlink(), "Dependency/model symlink coverage is not verified")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            paths.append(path)
    require(paths, "Empty dependency/model root")
    return sorted(paths)


PROBE = r'''
import contextlib,hashlib,importlib,importlib.metadata,importlib.util,io,json,os,pathlib,site,sys
sys.path.insert(0,sys.argv[1])
def offline(event,args):
 if event in ("socket.connect","socket.connect_ex","urllib.Request"): raise RuntimeError("CPU scoring runtime probe is offline")
sys.addaudithook(offline)
roots={};versions={"python":sys.version};loaded={}
for role,mod,dist in (("torch","torch","torch"),("dreamsim","dreamsim","dreamsim"),
 ("av","av","av"),("decord","decord","decord"),("PIL","PIL","Pillow"),("numpy","numpy","numpy")):
 with contextlib.redirect_stdout(io.StringIO()): module=importlib.import_module(mod)
 path=pathlib.Path(module.__file__).resolve()
 roots[role]=str(path.parent)
 try: versions[role]=importlib.metadata.version(dist)
 except importlib.metadata.PackageNotFoundError: versions[role]=str(module.__version__)
 loaded[role]=str(path)
spec=importlib.util.find_spec("vbench2_beta_i2v")
if spec is None: raise RuntimeError("Missing actual VBenchI2V source")
vp=pathlib.Path(spec.origin).resolve(); repo=pathlib.Path(sys.argv[1]).resolve()
if not vp.is_relative_to(repo): raise RuntimeError("VBenchI2V source is outside the explicit scoring repository")
roots["vbench"]=str(repo)
versions["vbench"]="source-sha256:"+hashlib.sha256(vp.read_bytes()).hexdigest();loaded["vbench"]=str(vp)
import torch
if torch.cuda.is_initialized(): raise RuntimeError("CPU runtime probe initialized CUDA")
libraries={"stdlib":str(pathlib.Path(os.__file__).resolve().parent)}
for i,path in enumerate(site.getsitepackages()): libraries["site_packages_"+str(i)]=str(pathlib.Path(path).resolve())
print(json.dumps(dict(versions=versions,dependency_roots=roots,loaded_modules=loaded,
 library_roots=libraries,python_invocation=sys.executable,python_prefix=sys.prefix,base_prefix=sys.base_prefix),sort_keys=True))
'''


def runtime_probe(python, vbench_repo, timeout_seconds=120):
    require(Path(python).is_absolute() and Path(python).is_file() and Path(vbench_repo).is_absolute(), "Explicit real scoring Python/repo required")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with tempfile.TemporaryDirectory(prefix="teacher-score-probe-") as prefix:
        env["PYTHONPYCACHEPREFIX"] = prefix
        result = subprocess.run([str(python), "-B", "-c", PROBE, str(vbench_repo)], env=env,
                                capture_output=True, text=True, check=True, timeout=timeout_seconds)
    return json.loads(result.stdout, object_pairs_hook=score.unique, parse_constant=score.reject_constant)


def collect_bundle(config_path, output):
    config = score.read_json(config_path)
    require(set(config) == {"python", "vbench_repo", "model_roots", "tools"}, "Exact explicit scoring runtime config required")
    require(set(config["tools"]) == {"ffmpeg", "ffprobe", "timeout", "nvidia_smi"} and
            set(config["model_roots"]) == {"vbench", "dreamsim"}, "Exact model/tool roles required")
    observation = runtime_probe(config["python"], config["vbench_repo"])
    roots = observation["dependency_roots"]
    dependencies = {str(p): score.file_record(p, allow_empty=True) for root in roots.values() for p in files_under(root)}
    libraries = {str(p): score.file_record(p, allow_empty=True) for root in observation["library_roots"].values() for p in files_under(root)}
    models = {str(p): score.file_record(p, allow_empty=p.suffix == ".py" or p.name == "trusted_list") for root in config["model_roots"].values() for p in files_under(root)}
    tools = {name: str(Path(path).resolve()) for name, path in config["tools"].items()}
    tools["python"] = str(Path(config["python"]).resolve())
    runtime_files = {p: score.file_record(p) for p in set(tools.values())}
    env_cfg = Path(config["python"]).parent.parent / "pyvenv.cfg"
    if env_cfg.is_file():
        row = score.file_record(env_cfg.resolve())
        runtime_files[row["path"]] = row
    bundle = {"schema": score.BUNDLE_SCHEMA, "private": True, "metric_protocol": "score_e1_all8_gt_v1",
              "dimensions": e1.DIMS, "gt_keys": list(e1.GT_KEYS),
              "sources": [score.file_record(path, allow_empty=True) for path in score.required_sources()],
              "dependency_roots": roots, "dependencies": list(dependencies.values()),
              "model_roots": config["model_roots"], "models": list(models.values()),
              "runtime": {"versions": observation["versions"], "tools": {k: tools[k] for k in ("python", "ffmpeg", "ffprobe")},
                          "files": list(runtime_files.values()), "library_roots": observation["library_roots"],
                          "library_files": list(libraries.values()), "observation": observation},
              "execution": {"python": config["python"], "vbench_repo": config["vbench_repo"], "tools": tools,
                            "offline": True, "gpu_dimensions": e1.DIMS, "gt_K": 12,
                            "vbench_local": True, "cache_dir": config["model_roots"]["vbench"]}}
    score.validate_bundle(bundle)
    return score.write_new(output, bundle)


def validate_execution_bundle(bundle, observe=True, verify_content=True):
    execution = bundle.get("execution", {})
    require(set(execution) == {"python", "vbench_repo", "tools", "offline", "gpu_dimensions", "gt_K", "vbench_local", "cache_dir"} and
            execution["offline"] is True and execution["gpu_dimensions"] == e1.DIMS and execution["gt_K"] == 12
            and execution["vbench_local"] is True and execution["cache_dir"] == bundle["model_roots"]["vbench"],
            "Actual scoring execution runtime must be frozen, not supplier booleans")
    require(set(execution["tools"]) == {"python", "ffmpeg", "ffprobe", "timeout", "nvidia_smi"}, "Scoring scope tools incomplete")
    runtime = bundle["runtime"]
    files = {r["path"]: r for r in runtime["files"]}
    require(set(execution["tools"].values()) <= set(files), "Unrecorded execution tool bytes")
    require(str(Path(execution["python"]).resolve()) == execution["tools"]["python"], "Python invocation resolved binary drift")
    score.coverage(runtime["library_roots"], runtime["library_files"], runtime["library_roots"], "Complete actual scoring environment")
    for row in runtime["library_files"]:
        score.check_record(row, content=verify_content, allow_empty=True)
    if observe:
        require(runtime_probe(execution["python"], execution["vbench_repo"]) == runtime["observation"], "Actual scoring environment/versions/prefix drift")
    return execution


@contextlib.contextmanager
def ledger_locked(path):
    path = local_path(path)
    require(path.is_file(), "Initialize approved local stage ledger first")
    import prepare_teacher_recovery as recovery
    with recovery.local_locked(path, local_root=path.parent):
        value = score.read_json(path)
        yield value
        recovery.durable_write(path, value)

def validate_ledger(ledger, plan, plan_sha):
    require(ledger.get("schema") == SCHEMA and ledger.get("plan_sha256") == plan_sha and
            ledger.get("common_execution_sha256") == plan["common_execution_sha256"] and
            ledger.get("bundle_sha256") == plan["bundle"]["sha256"] and
            ledger.get("caps") == score.SCORE_CAP_SECONDS and ledger.get("hostname") == os.uname().nodename,
            "Wrong node-local stage ledger identity/fixed budget")
    totals = {method: 0.0 for method in score.METHODS}
    tokens, actions = set(), set()
    schedule = {(a["method"], a["seed"], a["dimension"]): a["cap_seconds"] for a in plan["score_allocations"]}
    score.exact([(a["method"], a["seed"], a["dimension"]) for a in ledger["allocations"]], schedule, "Frozen ledger schedule")
    for item in ledger["allocations"]:
        method = item["method"]
        action = (method, item["seed"], item["dimension"])
        require(method in score.METHODS and item["seed"] in score.SEEDS and item["dimension"] in e1.DIMS and
                item["token"] not in tokens and action not in actions and
                item["status"] in ("reserved", "running", "finished") and type(item["allocated_gpus"]) is int and
                item["allocated_gpus"] == 1 and finite(item["cap_seconds"]) and item["cap_seconds"] > CLEANUP_SECONDS + 30,
                "Duplicate/unplanned scoring allocation or invalid cap")
        require(item["cap_seconds"] == schedule[action], "Per-action cap drift from approved scoring plan")
        tokens.add(item["token"]); actions.add(action)
        charged = item["cap_seconds"]
        if item.get("elapsed_observed_seconds") is not None:
            require(finite(item["elapsed_observed_seconds"]) and item["elapsed_observed_seconds"] >= 0, "Invalid failed allocation duration")
            charged = max(charged, item["elapsed_observed_seconds"])
        # Reservations stay fully charged until independently validated parent release evidence.
        if item["status"] == "finished":
            receipt = item.get("parent_receipt")
            require(isinstance(receipt, dict), "Finished allocation missing real release evidence; retain cap")
            score.check_record(receipt)
            proof = verify_parent_release(receipt["path"], receipt["sha256"], plan, plan_sha, require_success=False)
            require(proof["token"] == item["token"] and all(proof[k] == item[k] for k in
                    ("method", "seed", "dimension", "cap_seconds")), "Parent release token/task/fixed cap drift")
            require(proof["elapsed_seconds"] <= item["cap_seconds"], "Released action over cap; stop method scoring stage")
            charged = proof["elapsed_seconds"]
        totals[method] += charged
    require(all(totals[m] <= score.SCORE_CAP_SECONDS[m] for m in totals), "Independent method scoring stage cap exceeded")
    return totals


def validate_schedule(allocations):
    expected = {(m, s, d) for m in score.METHODS for s in score.SEEDS for d in e1.DIMS}
    score.exact([(a["method"], a["seed"], a["dimension"]) for a in allocations], expected, "All48 fixed score actions")
    require(all(set(a) == {"method", "seed", "dimension", "cap_seconds"} and finite(a["cap_seconds"]) and
                a["cap_seconds"] > CLEANUP_SECONDS + 30 for a in allocations), "Exact single-GPU scoring reservation schema required")
    for method in score.METHODS:
        require(sum(a["cap_seconds"] for a in allocations if a["method"] == method) <= score.SCORE_CAP_SECONDS[method],
                "Fixed two-seed stage reservations exceed method cap")
    return True


def owner_matches(plan, ledger_path):
    require(plan.get("score_owner") == {"hostname": os.uname().nodename, "ledger_path": str(local_path(ledger_path))},
            "Score host/ledger differs from immutable plan; cannot duplicate budget on another host/path")


def initialize_ledger(plan_path, plan_sha, path, allocations):
    require(score.sha256(plan_path) == plan_sha, "Approved scoring plan required")
    plan = score.read_json(plan_path)
    score.validate_plan(plan)
    owner_matches(plan, path)
    validate_schedule(allocations)
    require(allocations == plan["score_allocations"], "Ledger allowances differ from approved fixed schedule")
    rows = [{**a, "token": uuid.uuid4().hex, "status": "reserved", "allocated_gpus": 1} for a in allocations]
    ledger = {"schema": SCHEMA, "private": True, "plan_sha256": plan_sha,
              "common_execution_sha256": plan["common_execution_sha256"], "bundle_sha256": plan["bundle"]["sha256"],
              "caps": score.SCORE_CAP_SECONDS, "hostname": os.uname().nodename, "allocations": rows}
    validate_ledger(ledger, plan, plan_sha)
    return score.write_new(local_path(path), ledger)


def proc_identity(pid):
    path = Path("/proc") / str(pid)
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]), "sid": int(fields[3]),
            "start_ticks": int(fields[19]), "cmdline": [s.decode(errors="strict") for s in (path / "cmdline").read_bytes().split(b"\0") if s]}


def session_processes(sid):
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == sid and fields[0] not in ("Z", "X"):
                result.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(result)


def bounded_ancestry(cap, gpu, expected_parent):
    from run_teacher_inference_smoke import action_matches
    parents, pid = [], os.getppid()
    while pid > 1:
        identity = proc_identity(pid)
        parents.append(identity)
        pid = identity["ppid"]
    require(any(p["pid"] == expected_parent["pid"] and p["start_ticks"] == expected_parent["start_ticks"] for p in parents),
            "Worker is not owned by approved parent PID/starttime")
    holders = [p for p in parents if any(Path(a).name == "gpu_hold.py" for a in p["cmdline"]) and
               action_matches([a.encode() for a in p["cmdline"]], "exec")]
    require(any("--gpus" in p["cmdline"] and p["cmdline"][p["cmdline"].index("--gpus") + 1] == str(gpu) for p in holders),
            "Live one-GPU gpu_hold exec ancestry required")
    timeouts = [p for p in parents if p["cmdline"] and Path(p["cmdline"][0]).name == "timeout"]
    require(any("--signal=TERM" in p["cmdline"] and "--kill-after=30s" in p["cmdline"] and
                str(cap - CLEANUP_SECONDS) + "s" in p["cmdline"] for p in timeouts), "Caller GNU timeout TERM/kill30 required")
    return parents


def gpu_query(execution, state_dir, holder_pid, gpu, released):
    from run_teacher_inference_smoke import proc_args, action_matches
    args = proc_args(holder_pid)
    require(args and any(Path(a.decode()).name == "gpu_hold.py" for a in args if a) and action_matches(args, "run"), "Live GPU holder supervisor missing")
    status = score.read_json(Path(state_dir) / "status.json")
    require(status["pid"] == holder_pid and status["running"] is True and status.get("monitor_only") is True and status["query_ok"] is True and
            finite(status["updated_at"]) and 0 <= time.time() - status["updated_at"] < 10, "Stale GPU scope status")
    selected = status["gpus"][str(gpu)]
    require(not selected["external_pids"] and selected["holder_pid"] is None,
            "Monitor-only scope must have no idle holder or unrelated workload")
    query_command = [execution["tools"]["nvidia_smi"], "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    query_start = time.monotonic()
    result = subprocess.run(query_command,
                            capture_output=True, text=True, timeout=10, check=True)
    pids = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if row and row[0].strip() == selected["gpu_uuid"]:
            require(len(row) == 2, "Malformed raw GPU query")
            pids.append(int(row[1].strip()))
    allowed = []
    if released:
        require(selected["paused"] is True and selected["holder_pid"] is None, "Holder must be released before CUDA")
    require(set(pids) <= set(allowed), "GPU release/query shows an unrelated or still-owned process")
    return {"queried_epoch": time.time(), "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
            "gpu_uuid": selected["gpu_uuid"], "gpu_index": gpu, "observed_pids": sorted(pids),
            "allowed_holder_pids": allowed, "status": status, "holder_identity": proc_identity(holder_pid),
            "resumed_holder_identity": None,
            "command": query_command, "start_monotonic": query_start, "end_monotonic": time.monotonic()}


def prepare_stage(plan_path, plan_sha, method, seed, work, review_path, review_sha):
    require(method in score.METHODS and seed in score.SEEDS and score.sha256(plan_path) == plan_sha, "Exact approved score task required")
    plan = score.read_json(plan_path)
    cases = score.validate_plan(plan)
    bundle = score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    require("execution" in bundle, "Actual scoring execution runtime must be frozen")
    execution = bundle["execution"]
    from summarize_teacher_devset import validate_review_protocol
    require(score.sha256(review_path) == review_sha, "Approved review protocol byte SHA required before scores")
    review = validate_review_protocol(score.read_json(review_path), plan, plan_sha)
    entry = plan["groups"][method]["seeds"][seed]
    work = local_path(work) / method / seed
    binding = score.cache_binding(plan, method, seed)
    binding_path = work / "cache.binding.json"
    if binding_path.exists():
        require(score.read_json(binding_path) == binding, "Unknown/changed old cache; never overwrite")
    else:
        require(not (work / "scores").exists() and not (work / "gtref").exists(), "Old ID-only score/GT cache cannot be rebound")
        score.write_new(binding_path, binding)
    staged, local_cases, gt_reports = [], [], []
    for case, media in zip(cases, entry["media"]):
        require(case["id"] == media["case_id"], "Frozen case/media order drift")
        video = work / "videos" / method / (case["id"] + ".mp4")
        metric_video = work / "vbench" / "videos" / (case["id"] + "-0.mp4")
        gt = work / "inputs" / (case["id"] + ".mp4")
        image = work / "inputs" / (case["id"] + ".png")
        score.verified_copy(media["file"], video)
        staged.append({"case_id": case["id"], "file": score.verified_copy(media["file"], metric_video)})
        score.verified_copy(score.file_record(case["gt"]), gt)
        score.verified_copy(score.file_record(case["image"]), image)
        # Full CPU decode, no CUDA context and no GPU reservation.
        report = decode_cpu(video, execution, True)
        gt_report = decode_cpu(gt, execution, False)
        gt_reports.append({"case_id": case["id"], "generated": report, "gt": gt_report})
        local_cases.append({**case, "gt": str(gt), "image": str(image)})
    cases_path = work / "cases.jsonl"
    raw = "".join(json.dumps(c, sort_keys=True, allow_nan=False) + "\n" for c in local_cases)
    if cases_path.exists():
        require(cases_path.read_text() == raw, "Local staged GT metadata drift")
    else:
        work.mkdir(parents=True, exist_ok=True)
        with cases_path.open("x") as handle:
            handle.write(raw)
    info = [{"prompt_en": c["id"], "dimension": e1.DIMS, "custom_image_path": c["image"]} for c in local_cases]
    full_info = work / "vbench" / "full_info.json"
    if full_info.exists():
        require(score.read_json(full_info) == info, "Local full_info/source image drift")
    else:
        score.write_new(full_info, info)
    stage = {"schema": "teacher_score_stage_v1", "plan_sha256": plan_sha, "method": method, "seed": seed,
             "work": str(work), "source_media": entry["media"], "stage_media": staged,
             "cache_binding": score.file_record(binding_path), "cases": score.file_record(cases_path),
             "full_info": score.file_record(full_info), "decode": gt_reports,
             "review_protocol": score.file_record(review_path), "review_created_epoch": review["created_epoch"]}
    path = work / "stage.json"
    if path.exists():
        require(score.read_json(path) == stage, "Staged decode/hash evidence changed")
    else:
        score.write_new(path, stage)
    return stage


def decode_cpu(path, execution, generated):
    wrapper = r'''
import json,sys,av
from pathlib import Path
p=sys.argv[1]
with av.open(p) as c:
 s=c.streams.video[0]; fps=float(s.average_rate); w=s.width;h=s.height;n=0
 for f in c.decode(video=0):
  if (f.width,f.height)!=(w,h): raise RuntimeError("Frame size changed")
  n+=1
if sys.argv[2]=="generated":
 from decord import VideoReader
 if len(VideoReader(p,num_threads=1))!=n: raise RuntimeError("Official Decord/PyAV full decode counts differ")
print(json.dumps(dict(frames=n,fps=fps,width=w,height=h)))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    before = score.file_record(path)
    ffmpeg_command = [execution["tools"]["ffmpeg"], "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"]
    ffmpeg_result = subprocess.run(ffmpeg_command, env=env, capture_output=True, text=True, timeout=120, check=True)
    decoder_command = [execution["python"], "-B", "-c", wrapper, str(path), "generated" if generated else "gt"]
    result = subprocess.run(decoder_command, env=env,
                            capture_output=True, text=True, timeout=120, check=True)
    value = json.loads(result.stdout)
    require(type(value["frames"]) is int and value["frames"] > 1 and finite(value["fps"]) and value["fps"] > 0, "Invalid CPU full decode")
    if generated:
        require((value["frames"], value["fps"], value["width"], value["height"]) == (121, 24, 1280, 704), "Generated format drift")
    require(score.file_record(path) == before, "Decode media SHA drift")
    return {**value, "all_frames_decoded": True, "file": before,
            "decoder": {"ffmpeg_command": ffmpeg_command, "ffmpeg_exit_code": ffmpeg_result.returncode,
                        "ffmpeg_stdout": ffmpeg_result.stdout, "ffmpeg_stderr": ffmpeg_result.stderr,
                        "python_command": decoder_command, "python_exit_code": result.returncode,
                        "python_stdout": result.stdout, "python_stderr": result.stderr}}


def verify_stage(stage, plan, method, seed):
    require(score.is_sha(stage["plan_sha256"]), "Invalid stage plan binding")
    require(stage["method"] == method and stage["seed"] == seed and
            stage["source_media"] == plan["groups"][method]["seeds"][seed]["media"], "Stage source identity drift")
    score.verify_cache(plan, method, seed, stage["cache_binding"]["path"])
    for key in ("cache_binding", "cases", "full_info", "review_protocol"):
        score.check_record(stage[key])
    source = {r["case_id"]: r["file"] for r in stage["source_media"]}
    score.exact([r["case_id"] for r in stage["stage_media"]], source, "Stage exact media")
    for row in stage["stage_media"]:
        score.check_record(row["file"])
        require(row["file"]["sha256"] == source[row["case_id"]]["sha256"], "Local stage SHA drift")
    score.exact([r["case_id"] for r in stage["decode"]], source, "Full decode evidence")
    cases = {c["id"]: c for c in score.read_cases(plan["cases"]["path"])}
    for row in stage["decode"]:
        for kind in ("generated", "gt"):
            report = row[kind]
            score.check_record(report["file"])
            decoder = report["decoder"]
            decoded = json.loads(decoder["python_stdout"])
            require(decoder["ffmpeg_exit_code"] == decoder["python_exit_code"] == 0 and
                    decoder["ffmpeg_stderr"] == "" and all(decoded[k] == report[k] for k in ("frames", "fps", "width", "height")),
                    "Original full CPU decoder output/exit differs from stored report")
            require(report["all_frames_decoded"] is True and type(report["frames"]) is int and report["frames"] > 1,
                    "Missing full CPU decode")
        generated = row["generated"]
        require(generated["file"]["sha256"] == source[row["case_id"]]["sha256"] and
                (generated["frames"], generated["fps"], generated["width"], generated["height"]) == (121, 24, 1280, 704) and
                row["gt"]["file"]["sha256"] == cases[row["case_id"]]["video_sha256"], "Full decoded media/GT content/format drift")
    return True


def gtref_cpu(plan_path, plan_sha, method, seed, stage_path, max_seconds):
    require(finite(max_seconds) and max_seconds > 0 and score.sha256(plan_path) == plan_sha, "CPU GT bound/approved plan required")
    plan = score.read_json(plan_path); score.validate_plan(plan)
    stage = score.read_json(stage_path); verify_stage(stage, plan, method, seed)
    require(stage["plan_sha256"] == plan_sha, "Stage approved plan drift")
    bundle = score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    require("execution" in bundle, "Actual scoring execution runtime must be frozen")
    execution = bundle["execution"]
    destination = Path(stage["work"]) / "gtref" / (method + ".json")
    require(not destination.exists(), "Unknown/old GT cache preserved; use finished CPU GT receipt, not ID-only resume")
    wrapper = ("import sys,json;sys.path.insert(0,sys.argv[1]);import run_teacher_scoring as r;import score_e1 as e;"
               "b=r.score.read_json(sys.argv[5]);reads=r.install_asset_audit(b);e.VBREPO=sys.argv[2];"
               "e.cmd_gtref([sys.argv[3]],root=sys.argv[4]);"
               "r.score.write_new(sys.argv[6],list(reads.values()))")
    asset_path = Path(stage["work"]) / "gtref.assets.json"
    require(not asset_path.exists(), "Preserve old CPU GT asset evidence")
    command = [execution["tools"]["timeout"], "--signal=TERM", "--kill-after=30s", str(max_seconds) + "s",
               execution["python"], "-B", "-c", wrapper, str(score.ROOT / "scripts"), execution["vbench_repo"], method, stage["work"], plan["bundle"]["path"], str(asset_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", THREADS="8")
    gt_start_mono, gt_start_wall = time.monotonic(), time.time()
    with tempfile.TemporaryDirectory(prefix="teacher-score-gt-") as prefix:
        env["PYTHONPYCACHEPREFIX"] = prefix
        subprocess.run(command, env=env, check=True, timeout=max_seconds + 35)
    gt_end_mono, gt_end_wall = time.monotonic(), time.time()
    values = score.read_json(destination)
    cases = score.read_cases(plan["cases"]["path"])
    score.exact(values, [c["id"] for c in cases], "Original GT94 coverage")
    require(all(all(e1.finite(v.get(k)) for k in (*e1.GT_KEYS, "dur")) for v in values.values()), "GT nonfinite/missing duration")
    verify_stage(stage, plan, method, seed)
    score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    validate_gt_duration(values, stage, cases)
    result = {"schema": "teacher_cpu_gt_receipt_v1", "plan_sha256": plan_sha, "method": method, "seed": seed,
              "stage": score.file_record(stage_path), "bundle_sha256": plan["bundle"]["sha256"], "score_file": score.file_record(destination),
              "cache_binding": stage["cache_binding"], "CUDA_VISIBLE_DEVICES": "", "K": 12, "command": command,
              "asset_reads": score.read_json(asset_path), "asset_receipt": score.file_record(asset_path),
              "start_monotonic": gt_start_mono, "end_monotonic": gt_end_mono, "elapsed_seconds": gt_end_mono - gt_start_mono,
              "started_epoch": gt_start_wall, "ended_epoch": gt_end_wall, "max_seconds": max_seconds}
    return score.write_new(Path(stage["work"]) / "gtref.receipt.json", result)


def validate_gt_duration(values, stage, cases):
    expected = {r["case_id"]: min(r["generated"]["frames"] / r["generated"]["fps"],
                                  r["gt"]["frames"] / r["gt"]["fps"]) - 1e-3 for r in stage["decode"]}
    score.exact(values, expected, "GT original duration coverage")
    for case in cases:
        row = values[case["id"]]
        require(row.get("dataset") == case["dataset"] and all(e1.finite(row.get(k)) for k in (*e1.GT_KEYS, "dur")) and
                row["dur"] > 0 and math.isclose(row["dur"], expected[case["id"]], rel_tol=1e-9, abs_tol=1e-9),
                "GT metadata/nonfinite/original shared duration changed")
    return True


def command_for(action, execution):
    return [execution["tools"]["timeout"], "--signal=TERM", "--kill-after=30s", str(action["cap_seconds"] - CLEANUP_SECONDS) + "s",
            execution["python"], str(score.ROOT / "scripts/gpu_hold.py"), "--state-dir", action["state_dir"],
            "--action", "exec", "--gpus", str(action["gpu_index"]), "--",
            execution["python"], "-B", str(Path(__file__).resolve()), "worker", "--action", action["path"], "--approved-sha256", action["sha256"]]


_NATIVE_ORIGINALS = {}


def install_asset_audit(bundle):
    """Reject unfrozen weight files and any network access before model imports."""
    records = bundle.get("sources", []) + bundle["models"] + bundle["dependencies"] + bundle["runtime"]["files"] + bundle["runtime"]["library_files"]
    frozen = {r["path"]: r for r in records}
    reads, active = {}, False
    extensions = {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx", ".jit", ".npy", ".npz", ".pkl", ".pickle"}
    def audit(event, args):
        nonlocal active
        if event in ("socket.connect", "socket.connect_ex", "urllib.Request", "os.system"):
            raise RuntimeError("Metric scoring is offline; no downloads, shell commands or media uploads")
        if event == "subprocess.Popen" and args:
            executable = Path(os.fsdecode(args[0])).name
            require(executable not in {"wget", "curl", "git", "aria2c", "pip", "pip3", "sh", "bash"},
                    "Metric loading cannot spawn download or shell subprocesses")
        if active or event != "open" or not args or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        mode = args[1]
        reading = mode is None or isinstance(mode, str) and ("r" in mode or "+" in mode)
        if reading and path.suffix.lower() == ".pyc":
            raise FileNotFoundError("Use frozen source rather than unverified bytecode cache")
        source_file = path.suffix.lower() in {".py", ".so", ".dll", ".dylib"}
        if not reading or not source_file and path.suffix.lower() not in extensions:
            return
        path = path.resolve()
        require(str(path) in frozen, "Unfrozen metric weight/cache file: " + str(path))
        active = True
        try:
            score.check_record(frozen[str(path)], allow_empty=path.suffix == ".py")
            if not source_file:
                reads[str(path)] = frozen[str(path)]
        finally:
            active = False
    def validate_native_path(value):
        nonlocal active
        if not isinstance(value, (str, bytes, os.PathLike)):
            require(type(value) in (io.BufferedReader, io.FileIO), "Unbound native model stream; use a frozen file or regular file descriptor")
            descriptor = os.fstat(value.fileno())
            named = Path(value.name).absolute().resolve().stat()
            require((descriptor.st_dev, descriptor.st_ino) == (named.st_dev, named.st_ino), "Native model stream is not the frozen named file")
            value = value.name
        require(isinstance(value, (str, bytes, os.PathLike)), "Unbound native model stream; use a frozen file path")
        path = Path(os.fsdecode(value)).absolute().resolve()
        require(str(path) in frozen, "Unfrozen native model file: " + str(path))
        active = True
        try:
            score.check_record(frozen[str(path)], allow_empty=path.suffix == ".py")
            reads[str(path)] = frozen[str(path)]
        finally:
            active = False
        return path

    def wrap(module, name):
        if not hasattr(module, name):
            return
        key = (module.__name__, name)
        original = _NATIVE_ORIGINALS.setdefault(key, getattr(module, name))
        def checked(*args, **kwargs):
            value = args[0] if args else kwargs.get("filename", kwargs.get("f"))
            validate_native_path(value)
            return original(*args, **kwargs)
        setattr(module, name, checked)

    # Python audit hooks alone do not observe C++/Rust file opens.
    import torch
    import safetensors
    import safetensors.torch as safe_torch
    for module, name in ((torch, "load"), (torch.jit, "load"),
                         (safetensors, "safe_open"), (safe_torch, "safe_open"),
                         (safe_torch, "load_file")):
        wrap(module, name)
    for module, name in ((torch.hub, "load_state_dict_from_url"),):
        key = (module.__name__, name)
        _NATIVE_ORIGINALS.setdefault(key, getattr(module, name))
        def offline_url(*args, **kwargs):
            raise RuntimeError("Metric assets must be loaded from frozen local files, not URLs")
        setattr(module, name, offline_url)
    original_hub = _NATIVE_ORIGINALS.setdefault((torch.hub.__name__, "load"), torch.hub.load)
    def local_hub(repo_or_dir, model, *args, **kwargs):
        require(kwargs.get("source") == "local" and Path(repo_or_dir).is_dir(), "Only frozen local TorchHub source is allowed")
        root = Path(repo_or_dir).resolve()
        code = [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
        require(code, "Empty local TorchHub source")
        for path in code:
            validate_native_path(path)
        return original_hub(repo_or_dir, model, *args, **kwargs)
    torch.hub.load = local_hub
    sys.addaudithook(audit)
    return reads


def verify_asset_reads(reads, bundle):
    require(isinstance(reads, list) and reads, "Actual metric asset reads missing")
    frozen = {r["path"]: r for r in bundle["models"] + bundle["dependencies"] + bundle["runtime"]["files"] + bundle["runtime"]["library_files"]}
    score.exact([r["path"] for r in reads], {r["path"] for r in reads}, "Metric asset reads")
    for row in reads:
        require(frozen.get(row["path"]) == row, "Actual metric asset not in frozen complete bundle")
        score.check_record(row, allow_empty=Path(row["path"]).suffix == ".py")
    return True


def verify_parent_command(identity, action, plan):
    args = identity["cmdline"]
    indexes = [i for i, arg in enumerate(args) if Path(arg).name == "run_teacher_scoring.py"]
    require(len(indexes) == 1 and Path(args[indexes[0]]).absolute().resolve() == Path(__file__).resolve() and
            args[indexes[0] + 1:indexes[0] + 2] == ["run"],
            "Worker parent must be the frozen real scoring run entry, not a custom caller")
    expected = {"--approved-plan-sha256": action["plan"]["sha256"], "--method": action["method"],
                "--seed": action["seed"], "--dimension": action["dimension"], "--gpu": str(action["gpu_index"]),
                "--holder-pid": str(action["holder_pid"])}
    for key, value in expected.items():
        require(args.count(key) == 1 and args[args.index(key) + 1] == value, "Parent CLI task/SHA scope differs from approved action")
    paths = {"--plan": action["plan"]["path"], "--stage": action["stage"]["path"],
             "--ledger": plan["score_owner"]["ledger_path"], "--state-dir": action["state_dir"]}
    for key, value in paths.items():
        require(args.count(key) == 1 and str(Path(args[args.index(key) + 1]).absolute()) == value,
                "Parent CLI file/ledger scope differs from approved action")
    return True


def worker(action_path, action_sha):
    require(score.sha256(action_path) == action_sha, "Approved single score action SHA required")
    action = score.read_json(action_path)
    require(action.get("schema") == ACTION_SCHEMA and action.get("hostname") == os.uname().nodename, "Wrong worker action host/schema")
    worker_start_mono, worker_start_wall = time.monotonic(), time.time()
    launch_path = Path(action_path).parent / "launch.json"
    until = time.monotonic() + 5
    while not launch_path.exists():
        require(time.monotonic() < until, "Controlled parent launch evidence missing before worker admission")
        time.sleep(0.01)
    launch = score.read_json(launch_path)
    require(launch["action"]["path"] == str(Path(action_path).absolute()) and
            launch["action"]["sha256"] == action_sha, "Parent launch is not bound to this worker action")
    require(action["schema"] == ACTION_SCHEMA and action["hostname"] == os.uname().nodename, "Wrong worker action host/schema")
    require(finite(action["cap_seconds"]) and action["cap_seconds"] > CLEANUP_SECONDS + 30, "Invalid action cap")
    plan = score.read_json(action["plan"]["path"]); score.check_record(action["plan"])
    stage = score.read_json(action["stage"]["path"]); score.check_record(action["stage"])
    require(stage["plan_sha256"] == action["plan"]["sha256"], "Stage plan SHA drift")
    live_parent = proc_identity(action["parent_identity"]["pid"])
    require(live_parent == action["parent_identity"], "Original approved parent PID/starttime/command changed")
    verify_parent_command(live_parent, action, plan)
    require(plan["schema"] == score.PLAN_SCHEMA and set(plan["groups"]) == set(score.METHODS) and
            plan["cases"]["sha256"] == score.FROZEN_CASES_SHA256, "Worker needs the assembled frozen all3 generation contract")
    score.check_record(plan["cases"]); score.check_record(plan["comparison"])
    for group in plan["groups"].values():
        score.check_record(group["generation_completion"])
    score.check_record(plan["bundle"])
    bundle = score.read_json(plan["bundle"]["path"])
    execution = bundle["execution"]
    # Parent already verified the complete runtime before allocation. Worker rechecks source and all inputs.
    for record in bundle["sources"]:
        score.check_record(record, allow_empty=True)
    verify_stage(stage, plan, action["method"], action["seed"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == action["gpu_uuid"] and
            os.environ.get("PYTHONPYCACHEPREFIX") == action["pycache_prefix"] and
            not any(local_path(action["pycache_prefix"]).rglob("*.pyc")),
            "Exactly approved single GPU and clean source-only runtime required")
    ancestry = bounded_ancestry(action["cap_seconds"], action["gpu_index"], action["parent_identity"])
    expected_command = command_for({**action, "path": str(action_path), "sha256": action_sha}, execution)
    require(any(p["cmdline"] == expected_command for p in ancestry) and
            any(p["cmdline"] == expected_command[4:] for p in ancestry), "Exact approved timeout/gpu_hold exec command ancestry missing")
    before_query = gpu_query(execution, action["state_dir"], action["holder_pid"], action["gpu_index"], True)
    require(before_query["gpu_uuid"] == action["gpu_uuid"], "Worker GPU scope changed")
    destination = Path(stage["work"]) / "scores" / (action["method"] + "__" + action["dimension"] + "_eval_results.json")
    require(not destination.exists(), "Old/partial metric result must not be overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    os.environ["VBENCH_CACHE_DIR"] = execution["cache_dir"]
    os.chdir(execution["vbench_repo"])
    sys.path.insert(0, execution["vbench_repo"])
    asset_reads = install_asset_audit(bundle)
    import torch
    require(sys.prefix == bundle["runtime"]["observation"]["python_prefix"] and
            str(Path(torch.__file__).resolve()) == bundle["runtime"]["observation"]["loaded_modules"]["torch"],
            "Executing worker runtime prefix/torch origin differs from frozen actual observation")
    from vbench2_beta_i2v import VBenchI2V
    bench = VBenchI2V(torch.device("cuda"), stage["full_info"]["path"], str(destination.parent))
    bench.evaluate(videos_path=str(Path(stage["work"]) / "vbench/videos"), name=action["method"] + "__" + action["dimension"],
                   dimension_list=[action["dimension"]], resolution="16-9", mode="vbench_standard", local=True)
    validate_dimension(destination, action["dimension"], plan)
    verify_stage(stage, plan, action["method"], action["seed"])
    worker_end_mono, worker_end_wall = time.monotonic(), time.time()
    receipt = {"schema": "teacher_score_worker_receipt_v1", "action_sha256": action_sha, "token": action["token"],
               "method": action["method"], "seed": action["seed"], "dimension": action["dimension"],
               "worker_identity": proc_identity(os.getpid()), "ancestry": ancestry, "before_gpu_query": before_query,
               "score_file": score.file_record(destination), "elapsed_seconds": worker_end_mono - worker_start_mono,
               "start_monotonic": worker_start_mono, "end_monotonic": worker_end_mono,
               "start_epoch": worker_start_wall, "end_epoch": worker_end_wall, "model_begin_monotonic": begun,
               "success": True, "bundle_sha256": plan["bundle"]["sha256"], "cuda_visible": os.environ["CUDA_VISIBLE_DEVICES"],
               "asset_reads": list(asset_reads.values())}
    score.write_new(action["worker_receipt_path"], receipt)
    return receipt


def validate_dimension(path, dim, plan):
    result = score.read_json(path)
    require(set(result) == {dim} and isinstance(result[dim], list) and len(result[dim]) == 2 and e1.finite(result[dim][0]), "Dimension schema/aggregate drift")
    expected = [c["id"] + "-0.mp4" for c in score.read_cases(plan["cases"]["path"])]
    values = result[dim][1]
    score.exact([Path(r["video_path"]).name for r in values], expected, "Dimension94 coverage")
    require(all(e1.finite(r.get("video_results")) for r in values), "Dimension nonfinite scores")
    point = sum(e1.per_video_value(dim, r["video_results"]) for r in values) / len(values)
    require(math.isclose(point, float(result[dim][0]), rel_tol=1e-9, abs_tol=1e-9), "Original per-video mean/aggregate mismatch")
    return result


def stop_owned(child, deadline, owner=None, tracked=None):
    if owner is not None:
        from run_teacher_generation import cleanup_owned
        return cleanup_owned(child, owner, deadline, tracked)
    # Without a captured identity, never infer a process group from a reusable PID.
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=max(.01, min(29, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=max(.01, min(1, deadline - time.monotonic())))
    else:
        child.wait(timeout=.1)
    return []

def run_action(plan_path, plan_sha, method, seed, dim, stage_path, ledger_path, state_dir, gpu, holder_pid):
    require(method in score.METHODS and seed in score.SEEDS and dim in e1.DIMS and type(gpu) is int and 0 <= gpu < 8, "Exact fixed scoring task/GPU required")
    require(score.sha256(plan_path) == plan_sha, "Unapproved score plan")
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "Existing parent timer cannot be replaced")
    plan = score.read_json(plan_path); score.validate_plan(plan)
    bundle = score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    require("execution" in bundle, "Actual scoring execution runtime must be frozen")
    execution = bundle["execution"]
    stage = score.read_json(stage_path); verify_stage(stage, plan, method, seed)
    require(stage["plan_sha256"] == plan_sha, "Stage plan SHA drift")
    from summarize_teacher_devset import validate_review_protocol
    review = score.read_json(stage["review_protocol"]["path"])
    validate_review_protocol(review, plan, plan_sha)
    require(review["created_epoch"] <= time.time(), "Blind review protocol cannot be future dated")
    gt = score.read_json(Path(stage["work"]) / "gtref.receipt.json")
    require(gt["plan_sha256"] == plan_sha and gt["bundle_sha256"] == plan["bundle"]["sha256"] and
            gt["cache_binding"] == stage["cache_binding"], "Complete CPU GT stage required before reserving GPU")
    score.check_record(gt["score_file"])
    owner_matches(plan, ledger_path)
    query = gpu_query(execution, state_dir, holder_pid, gpu, False)
    require(query["status"]["gpus"][str(gpu)]["paused"] is False and
            not (Path(state_dir) / ("pause." + str(gpu))).exists(), "Another caller already paused/owns this GPU")
    destination = Path(stage["work"]) / "scores" / (method + "__" + dim + "_eval_results.json")
    require(not destination.exists(), "Existing/partial score preserved; cannot retry an allocated dimension")
    parent_identity = proc_identity(os.getpid())
    verify_parent_command(parent_identity, {"plan": score.file_record(plan_path), "stage": score.file_record(stage_path),
                         "method": method, "seed": seed, "dimension": dim, "gpu_index": gpu,
                         "holder_pid": holder_pid, "state_dir": str(Path(state_dir).absolute())}, plan)
    with ledger_locked(ledger_path) as ledger:
        validate_ledger(ledger, plan, plan_sha)
        require(not any(i["status"] == "running" for i in ledger["allocations"]), "Only one local score GPU action at a time")
        matches = [a for a in ledger["allocations"] if (a["method"], a["seed"], a["dimension"]) == (method, seed, dim)]
        require(len(matches) == 1 and matches[0]["status"] == "reserved", "Unreserved/already attempted score action; never retry/reset")
        item = matches[0]
        folder = local_path(stage["work"]) / "actions" / dim
        action_path = folder / "action.json"
        start_mono, start_wall = time.monotonic(), time.time()
        action = {"schema": ACTION_SCHEMA, "plan": score.file_record(plan_path), "stage": score.file_record(stage_path),
                  "token": item["token"], "method": method, "seed": seed, "dimension": dim, "hostname": os.uname().nodename,
                  "cap_seconds": item["cap_seconds"], "gpu_index": gpu, "gpu_uuid": query["gpu_uuid"], "holder_pid": holder_pid,
                  "state_dir": str(Path(state_dir).absolute()), "parent_identity": parent_identity,
                  "allocation_start_monotonic": start_mono, "allocation_start_epoch": start_wall,
                  "worker_receipt_path": str(folder / "worker.json"), "pycache_prefix": str(folder / "unused_pycache")}
        record = score.write_new(action_path, action)
        command = command_for({**action, "path": record["path"], "sha256": record["sha256"]}, execution)
        item.update(status="running", action=record, command=command, parent_identity=parent_identity,
                    allocation_start_monotonic=start_mono, allocation_start_epoch=start_wall)
    child, child_identity, parent, error, interrupted, timed_out = None, None, {}, None, None, False
    child_owner, tracked, signalled = None, [], []
    old = {signal.SIGALRM: signal.getsignal(signal.SIGALRM)}
    def expired(_signum, _frame):
        raise TimeoutError("Allocated scoring parent hard cap exhausted; reservation remains charged")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, action["cap_seconds"] - (time.monotonic() - start_mono)))
    def interrupt(signum, _frame):
        nonlocal interrupted
        interrupted = signum
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, interrupt)
    try:
        pycache = local_path(action["pycache_prefix"])
        pycache.mkdir(mode=0o700, parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=action["gpu_uuid"], PYTHONDONTWRITEBYTECODE="1", PYTHONPYCACHEPREFIX=str(pycache),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1", OMP_NUM_THREADS="1")
        with (folder / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child_identity = proc_identity(child.pid)
            from run_teacher_generation import identity, owned_members
            child_owner = identity(child.pid)
            score.write_new(folder / "launch.json", {"action": record, "parent_identity": parent_identity,
                             "child_identity": child_identity, "command": command, "before_gpu_query": query,
                             "observed_monotonic": time.monotonic(), "observed_epoch": time.time()})
            while child.poll() is None and interrupted is None:
                owned_members(child_owner, tracked)
                if time.monotonic() - start_mono >= action["cap_seconds"] - CLEANUP_SECONDS:
                    timed_out = True
                    break
                time.sleep(0.05)
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            if child is not None:
                signalled = stop_owned(child, start_mono + action["cap_seconds"], child_owner, tracked)
            release_queries = [gpu_query(execution, state_dir, holder_pid, gpu, False) for _ in range(2)]
            after_query = release_queries[-1]
            require(after_query["gpu_uuid"] == action["gpu_uuid"], "GPU UUID changed during cleanup")
            require(time.monotonic() - start_mono < action["cap_seconds"],
                    "Allocated scoring parent hard cap exhausted after cleanup; retain failure cost")
            # Rehash score source/model/environment bytes after CUDA and cleanup, inside the charged clock.
            score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
            observed_after = bundle["runtime"]["observation"]
            elapsed = time.monotonic() - start_mono
            end_wall = time.time()
            worker_record = score.file_record(action["worker_receipt_path"]) if Path(action["worker_receipt_path"]).is_file() else None
            worker_value = score.read_json(worker_record["path"]) if worker_record else None
            succeeded = (error is None and interrupted is None and not timed_out and child is not None and child.returncode == 0 and
                         worker_value is not None and worker_value.get("success") is True and elapsed <= action["cap_seconds"])
            parent = {"schema": PARENT_SCHEMA, "action": record, "launch": score.file_record(folder / "launch.json"),
                      "worker": worker_record, "token": action["token"], "method": method, "seed": seed, "dimension": dim,
                      "parent_identity": parent_identity, "child_identity": child_identity,
                      "allocation_start_monotonic": start_mono, "allocation_end_monotonic": start_mono + elapsed,
                      "start_epoch": start_wall, "end_epoch": end_wall, "elapsed_seconds": elapsed,
                      "allocated_gpus": 1, "cap_seconds": action["cap_seconds"], "exit_code": child.returncode if child else None,
                      "timed_out": timed_out or child is not None and child.returncode in (124, 137),
                      "interrupted_signal": interrupted, "error": error, "after_gpu_query": after_query,
                      "session_remaining_pids": session_processes(child.pid) if child else [], "success": succeeded,
                      "runtime_observation": observed_after,
                      "bundle_content_sha256": score.digest(bundle),
                      "tracked_owned_processes": tracked, "cleanup_signals": signalled, "child_owner": child_owner,
                      "release_queries": release_queries}
            parent_record = score.write_new(folder / "supervised.json", parent)
            verify_parent_release(parent_record["path"], parent_record["sha256"], plan, plan_sha, require_success=False)
            with ledger_locked(ledger_path) as ledger:
                chosen = next(a for a in ledger["allocations"] if a["token"] == action["token"])
                require(chosen["status"] == "running" and chosen["action"] == record, "Allocation owner/action changed")
                chosen.update(status="finished", parent_receipt=parent_record)
                validate_ledger(ledger, plan, plan_sha)
        except BaseException as exc:
            error = "Release unproven; cap retained: " + type(exc).__name__ + ": " + str(exc)
            signal.setitimer(signal.ITIMER_REAL, 0)
            if child is not None and child.poll() is None:
                try:
                    stop_owned(child, time.monotonic() + 1, child_owner, tracked)
                except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                    error += "; owned child release remains unproven"
            observed_elapsed = max(0.0, time.monotonic() - start_mono)
            failure = {"schema": "teacher_score_parent_failure_v1", "success": False,
                       "action": record, "plan_sha256": plan_sha, "token": action["token"],
                       "method": method, "seed": seed, "dimension": dim, "parent_identity": parent_identity,
                       "allocation_start_monotonic": start_mono, "allocation_end_monotonic": start_mono + observed_elapsed,
                       "start_epoch": start_wall, "end_epoch": time.time(), "elapsed_observed_seconds": observed_elapsed,
                       "actual_allocated_seconds": None, "charged_upper_seconds": max(action["cap_seconds"], observed_elapsed),
                       "cap_seconds": action["cap_seconds"], "release_verified": False,
                       "budget_exceeded": observed_elapsed > action["cap_seconds"], "error": error,
                       "interrupted_signal": interrupted, "exit_code": child.returncode if child else None,
                       "quality_pass": False, "sota_claim": False}
            failure_record = score.write_new(folder / "supervised.failure.json", failure)
            try:
                with ledger_locked(ledger_path) as ledger:
                    chosen = next(a for a in ledger["allocations"] if a["token"] == action["token"])
                    require(chosen.get("action") == record, "Failed action ownership changed")
                    chosen.update(failure_receipt=failure_record, elapsed_observed_seconds=observed_elapsed,
                                  charged_upper_seconds=max(action["cap_seconds"], observed_elapsed), release_verified=False)
            except (OSError, RuntimeError, ValueError, KeyError):
                pass  # The independent failure receipt survives even if the original ledger is unreadable.
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old.items():
            signal.signal(sig, handler)
    require(error is None and parent["success"] is True, error or "Score action failed; actual failure cost preserved")
    return parent_record


def verify_query(record, gpu_uuid):
    require(record["exit_code"] == 0 and record.get("stderr", "") == "" and record["gpu_uuid"] == gpu_uuid and finite(record["queried_epoch"]), "GPU query identity/exit missing")
    pids = [int(r[1].strip()) for r in csv.reader(io.StringIO(record["stdout"])) if r and r[0].strip() == gpu_uuid]
    require(sorted(pids) == record["observed_pids"] and set(pids) <= set(record["allowed_holder_pids"]), "Raw GPU query release mismatch")
    status = record["status"]; selected = status["gpus"][str(record["gpu_index"])]
    require(status["pid"] == record["holder_identity"]["pid"] and status["running"] is True and status.get("monitor_only") is True and status["query_ok"] is True and
            0 <= record["queried_epoch"] - status["updated_at"] < 10 and selected["gpu_uuid"] == gpu_uuid and
            not selected["external_pids"] and selected["holder_pid"] is None and record["allowed_holder_pids"] == [],
            "GPU release monitor-only/no-holder status unbound/stale")
    require(set(record["allowed_holder_pids"]) == ({selected["holder_pid"]} if selected["holder_pid"] else set()), "Unknown allowed GPU occupant")
    require(any(Path(a).name == "gpu_hold.py" for a in record["holder_identity"]["cmdline"]) and
            ("run" in record["holder_identity"]["cmdline"] or "--action=run" in record["holder_identity"]["cmdline"]),
            "Recorded GPU holder is not original live run scope")
    require(record["resumed_holder_identity"] is None, "No idle reservation holder is allowed in scoring scope")
    return True


def verify_parent_release(path, approved_sha, plan, plan_sha, require_success=True):
    require(score.sha256(path) == approved_sha, "Parent receipt SHA mismatch")
    proof = score.read_json(path)
    require(proof.get("schema") == PARENT_SCHEMA and proof["allocated_gpus"] == 1 and
            proof["method"] in score.METHODS and proof["seed"] in score.SEEDS and proof["dimension"] in e1.DIMS,
            "Wrong scoring parent release schema/task")
    for key in ("action", "launch"):
        score.check_record(proof[key])
    action = score.read_json(proof["action"]["path"])
    verify_parent_command(proof["parent_identity"], action, plan)
    task_caps = {(a["method"], a["seed"], a["dimension"]): a["cap_seconds"] for a in plan["score_allocations"]}
    require(action["hostname"] == plan["score_owner"]["hostname"] and
            task_caps.get((action["method"], action["seed"], action["dimension"])) == action["cap_seconds"],
            "Parent score host/task cap differs from immutable plan")
    launch = score.read_json(proof["launch"]["path"])
    require(action["plan"]["sha256"] == plan_sha and action["method"] == proof["method"] and action["seed"] == proof["seed"] and
            action["dimension"] == proof["dimension"] and action["token"] == proof["token"] and
            launch["action"] == proof["action"] and launch["parent_identity"] == proof["parent_identity"] == action["parent_identity"] and
            launch["child_identity"] == proof["child_identity"], "Parent launch/action/plan/owner drift")
    score.check_record(action["plan"]); score.check_record(action["stage"])
    bundle = score.read_json(plan["bundle"]["path"])
    require(proof.get("runtime_observation") == bundle["runtime"]["observation"] and
            proof.get("bundle_content_sha256") == score.digest(bundle), "Actual parent scoring runtime/bundle observation drift")
    command = command_for({**action, "path": proof["action"]["path"], "sha256": proof["action"]["sha256"]}, bundle["execution"])
    require(launch["command"] == command and proof["child_identity"]["cmdline"] == command and finite(proof["elapsed_seconds"]) and proof["elapsed_seconds"] > 0 and
            finite(proof["start_epoch"]) and proof["start_epoch"] > 0 and finite(proof["end_epoch"]) and proof["end_epoch"] > proof["start_epoch"] and
            proof["cap_seconds"] == action["cap_seconds"] and proof["cap_seconds"] <= score.SCORE_CAP_SECONDS[proof["method"]] and
            finite(proof["allocation_start_monotonic"]) and proof["allocation_start_monotonic"] > 0 and
            finite(proof["allocation_end_monotonic"]) and proof["allocation_end_monotonic"] > proof["allocation_start_monotonic"] and
            abs(proof["allocation_end_monotonic"] - proof["allocation_start_monotonic"] - proof["elapsed_seconds"]) <= 1e-6 and
            abs(proof["end_epoch"] - proof["start_epoch"] - proof["elapsed_seconds"]) < 5 and
            proof["session_remaining_pids"] == [] and proof["child_identity"]["sid"] == proof["child_identity"]["pid"],
            "Parent bounded command/clock/session release invalid")
    require(finite(launch.get("observed_monotonic")) and finite(launch.get("observed_epoch")) and
            proof["allocation_start_monotonic"] <= launch["observed_monotonic"] <= proof["allocation_end_monotonic"] and
            proof["start_epoch"] <= launch["observed_epoch"] <= proof["end_epoch"],
            "Observed controlled launch clock outside allocated lifecycle")
    verify_query(launch["before_gpu_query"], action["gpu_uuid"])
    verify_query(proof["after_gpu_query"], action["gpu_uuid"])
    require(isinstance(proof.get("release_queries"), list) and len(proof["release_queries"]) >= 2 and
            proof["release_queries"][-1] == proof["after_gpu_query"], "Repeated original GPU release queries required")
    for query in proof["release_queries"][-2:]:
        verify_query(query, action["gpu_uuid"])
        require(finite(query["start_monotonic"]) and finite(query["end_monotonic"])
                and query["end_monotonic"] > query["start_monotonic"], "Release query must have a positive observed duration")
        require(query["command"] == [bundle["execution"]["tools"]["nvidia_smi"], "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"] and
                finite(query["start_monotonic"]) and finite(query["end_monotonic"]) and
                proof["allocation_start_monotonic"] <= query["start_monotonic"] <= query["end_monotonic"] <= proof["allocation_end_monotonic"],
                "Original release query command/monotonic clock drift")
    previous_query, last_query = proof["release_queries"][-2:]
    require(previous_query["end_monotonic"] < last_query["start_monotonic"], "Release queries must be distinct and chronologically ordered")
    owner = proof.get("child_owner", {})
    require(owner.get("pid") == proof["child_identity"]["pid"] and owner.get("session") == owner["pid"] and
            owner.get("start_ticks") == proof["child_identity"]["start_ticks"] and
            isinstance(proof.get("tracked_owned_processes"), list) and isinstance(proof.get("cleanup_signals"), list),
            "Original owned-session identity/cleanup signal evidence missing")
    require(proof["after_gpu_query"]["queried_epoch"] >= proof["start_epoch"] and
            proof["after_gpu_query"]["queried_epoch"] <= proof["end_epoch"] + 1,
            "GPU release query occurred outside parent allocation clock")
    require(proof["start_epoch"] == action["allocation_start_epoch"] and
            proof["allocation_start_monotonic"] == action["allocation_start_monotonic"] and
            proof["child_identity"]["ppid"] == proof["parent_identity"]["pid"] and
            proof["after_gpu_query"]["holder_identity"] == launch["before_gpu_query"]["holder_identity"],
            "Parent allocation clock/PID lineage/holder identity drift")
    if require_success:
        require(proof["success"] is True and proof["exit_code"] == 0 and proof["timed_out"] is False and
                proof["interrupted_signal"] is None and proof["error"] is None and proof["elapsed_seconds"] <= proof["cap_seconds"] and
                proof["worker"] is not None, "Parent action failed/missing worker; retain conservative cost")
        score.check_record(proof["worker"])
        worker_value = score.read_json(proof["worker"]["path"])
        require(worker_value["schema"] == "teacher_score_worker_receipt_v1" and worker_value["success"] is True and
                worker_value["action_sha256"] == proof["action"]["sha256"] and worker_value["token"] == action["token"] and
                worker_value["bundle_sha256"] == plan["bundle"]["sha256"] and worker_value["cuda_visible"] == action["gpu_uuid"] and
                worker_value["worker_identity"]["sid"] == proof["child_identity"]["sid"] and
                worker_value["worker_identity"]["cmdline"] == command[command.index("--") + 1:] and
                any(p["cmdline"] == command and p["pid"] == proof["child_identity"]["pid"] and
                    p["start_ticks"] == proof["child_identity"]["start_ticks"] for p in worker_value["ancestry"]) and
                any(p["cmdline"] == command[4:] for p in worker_value["ancestry"]) and
                any(p["pid"] == proof["parent_identity"]["pid"] and p["start_ticks"] == proof["parent_identity"]["start_ticks"]
                    for p in worker_value["ancestry"]), "Worker parent ownership/bundle/GPU scope mismatch")
        require(all(finite(worker_value.get(k)) for k in ("start_monotonic", "end_monotonic", "elapsed_seconds", "start_epoch", "end_epoch")) and
                proof["allocation_start_monotonic"] <= worker_value["start_monotonic"] <= worker_value["model_begin_monotonic"] <=
                worker_value["end_monotonic"] <= proof["allocation_end_monotonic"] and
                abs(worker_value["end_monotonic"] - worker_value["start_monotonic"] - worker_value["elapsed_seconds"]) < 0.001 and
                abs(worker_value["end_epoch"] - worker_value["start_epoch"] - worker_value["elapsed_seconds"]) < 5 and
                all(q["start_monotonic"] >= worker_value["end_monotonic"] for q in proof["release_queries"][-2:]),
                "Actual worker lifecycle/release query clock mismatch")
        stage = score.read_json(action["stage"]["path"])
        require(worker_value["score_file"]["path"] == str(Path(stage["work"]) / "scores" /
                (action["method"] + "__" + action["dimension"] + "_eval_results.json")) and
                all(worker_value[k] == action[k] for k in ("method", "seed", "dimension")),
                "Actual metric output belongs to another method/seed/dimension")
        verify_query(worker_value["before_gpu_query"], action["gpu_uuid"])
        score.check_record(worker_value["score_file"])
        verify_asset_reads(worker_value["asset_reads"], bundle)
        validate_dimension(worker_value["score_file"]["path"], action["dimension"], plan)
    if action["hostname"] == os.uname().nodename:
        from run_teacher_generation import same_process
        require(not session_processes(proof["child_identity"]["sid"]) and
                not any(same_process(row) for row in proof.get("tracked_owned_processes", [])),
                "Owned session or observed escaped process still live; retain full reservation")
    require(score.sha256(path) == approved_sha, "Parent release receipt changed during verification")
    return proof


def finalize_seed(plan_path, plan_sha, method, seed, stage_path, ledger_path):
    require(score.sha256(plan_path) == plan_sha, "Approved scoring plan required")
    plan = score.read_json(plan_path); score.validate_plan(plan)
    bundle = score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    require("execution" in bundle, "Actual scoring execution runtime must be frozen")
    stage = score.read_json(stage_path); verify_stage(stage, plan, method, seed)
    with ledger_locked(ledger_path) as ledger:
        validate_ledger(ledger, plan, plan_sha)
        chosen = [a for a in ledger["allocations"] if a["method"] == method and a["seed"] == seed]
        score.exact([a["dimension"] for a in chosen], e1.DIMS, "All8 finalized dimensions")
        require(all(a["status"] == "finished" for a in chosen), "All8 scoring parent receipts required")
        parents = [a["parent_receipt"] for a in chosen]
    proofs = [verify_parent_release(p["path"], p["sha256"], plan, plan_sha) for p in parents]
    work = Path(stage["work"])
    gt = score.read_json(work / "gtref.receipt.json"); score.check_record(gt["score_file"])
    source_root = Path(plan["groups"][method]["seeds"][seed]["root"])
    score_files = {}
    for proof in proofs:
        worker_value = score.read_json(proof["worker"]["path"])
        record = worker_value["score_file"]
        dest = source_root / "scores" / Path(record["path"]).name
        # Byte-exact completed output promotion, not rerunning a metric or overwriting old caches.
        require(dest.resolve() == dest and not dest.exists(), "Preserve old/symlink shared scoring output")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as out, Path(record["path"]).open("rb") as inp:
            for chunk in iter(lambda: inp.read(1024 * 1024), b""):
                out.write(chunk)
        score_files[proof["dimension"]] = score.file_record(dest)
        require(score_files[proof["dimension"]]["sha256"] == record["sha256"], "Score promotion SHA drift")
    dest = source_root / "gtref" / (method + ".json")
    require(dest.resolve() == dest and not dest.exists(), "Preserve old/symlink shared GT scoring output")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as out, Path(gt["score_file"]["path"]).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            out.write(chunk)
    score_files["gtref"] = score.file_record(dest)
    require(score_files["gtref"]["sha256"] == gt["score_file"]["sha256"], "GT promotion SHA drift")
    result = {"schema": score.RECEIPT_SCHEMA, "success": True, "plan_sha256": plan_sha, "method": method, "seed": seed,
              "bundle_sha256": plan["bundle"]["sha256"], "common_execution_sha256": plan["common_execution_sha256"],
              "parent_release_verified": True, "budget_verified": True, "scope_verified": True, "owned_tree_released": True,
              "stage_cap_seconds": score.SCORE_CAP_SECONDS[method], "allocated_gpu_seconds": sum(p["elapsed_seconds"] for p in proofs),
              "source_media": stage["source_media"], "stage_media": stage["stage_media"], "cache_binding": stage["cache_binding"],
              "score_files": score_files, "started_epoch": min(p["start_epoch"] for p in proofs),
              "execution_evidence": {"parents": parents, "stage": score.file_record(stage_path), "gt": score.file_record(work / "gtref.receipt.json"),
                                     "ledger": score.file_record(ledger_path)}}
    score.verify_scoring_execution(result, plan)
    e1.load_config_rows(method, score.read_cases(plan["cases"]["path"]), str(source_root))
    return score.write_new(source_root / "teacher_score.receipt.json", result)


def verify_execution(receipt, plan):
    evidence = receipt.get("execution_evidence")
    require(isinstance(evidence, dict) and set(evidence) == {"parents", "stage", "gt", "ledger"}, "Independent scoring evidence missing")
    score.validate_plan(plan)
    bundle = score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    require("execution" in bundle, "Actual scoring execution runtime must be frozen")
    for key in ("stage", "gt"):
        score.check_record(evidence[key])
    stage = score.read_json(evidence["stage"]["path"]); verify_stage(stage, plan, receipt["method"], receipt["seed"])
    require(stage["plan_sha256"] == receipt["plan_sha256"], "Final stage plan drift")
    proofs = []
    for row in evidence["parents"]:
        score.check_record(row)
        proofs.append(verify_parent_release(row["path"], row["sha256"], plan, receipt["plan_sha256"]))
    score.exact([p["dimension"] for p in proofs], e1.DIMS, "Eight real parent receipts")
    require(all(p["method"] == receipt["method"] and p["seed"] == receipt["seed"] for p in proofs) and
            math.isclose(sum(p["elapsed_seconds"] for p in proofs), receipt["allocated_gpu_seconds"], abs_tol=1e-6),
            "Parent scoring task/cost mismatch")
    # Ledger is mutable while other fixed actions finish; use live locked contents, not stale byte equality.
    ledger_path = local_path(evidence["ledger"]["path"])
    owner_matches(plan, ledger_path)
    with ledger_locked(ledger_path) as ledger:
        validate_ledger(ledger, plan, receipt["plan_sha256"])
        recorded = {r["token"]: r for r in ledger["allocations"]}
        require(all(p["token"] in recorded and recorded[p["token"]]["status"] == "finished" and
                    recorded[p["token"]]["parent_receipt"] == row for p, row in zip(proofs, evidence["parents"])), "Unaccounted finalized parent cost")
    gt = score.read_json(evidence["gt"]["path"])
    require(gt["plan_sha256"] == receipt["plan_sha256"] and gt["method"] == receipt["method"] and gt["seed"] == receipt["seed"] and
            gt["bundle_sha256"] == receipt["bundle_sha256"] and gt["cache_binding"] == receipt["cache_binding"] and
            gt["K"] == 12 and gt["CUDA_VISIBLE_DEVICES"] == "", "Independent original CPU GT evidence drift")
    score.check_record(gt["score_file"])
    require(all(finite(gt.get(k)) for k in ("start_monotonic", "end_monotonic", "elapsed_seconds", "started_epoch", "ended_epoch", "max_seconds")) and
            gt["started_epoch"] >= stage["review_created_epoch"] and gt["elapsed_seconds"] > 0 and
            gt["elapsed_seconds"] <= gt["max_seconds"] + 35 and
            abs(gt["end_monotonic"] - gt["start_monotonic"] - gt["elapsed_seconds"]) < .001 and
            abs(gt["ended_epoch"] - gt["started_epoch"] - gt["elapsed_seconds"]) < 5,
            "CPU GT lifecycle/rubric-before-scoring clock mismatch")
    score.check_record(gt["asset_receipt"])
    require(score.read_json(gt["asset_receipt"]["path"]) == gt["asset_reads"], "CPU GT real asset audit record drift")
    bundle = score.verify_bundle(plan["bundle"]["path"], plan["bundle"]["sha256"])
    verify_asset_reads(gt["asset_reads"], bundle)
    validate_gt_duration(score.read_json(gt["score_file"]["path"]), stage, score.read_cases(plan["cases"]["path"]))
    return True


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect-bundle"); p.add_argument("--runtime-config", required=True, type=Path); p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("initialize-ledger"); p.add_argument("--plan", required=True, type=Path); p.add_argument("--approved-plan-sha256", required=True)
    p.add_argument("--ledger", required=True, type=Path); p.add_argument("--allocations", required=True, type=Path)
    for name in ("prepare-stage", "gtref", "run", "finalize"):
        p = sub.add_parser(name); p.add_argument("--plan", required=True, type=Path); p.add_argument("--approved-plan-sha256", required=True)
        p.add_argument("--method", choices=score.METHODS, required=True); p.add_argument("--seed", choices=score.SEEDS, required=True)
        if name == "prepare-stage":
            p.add_argument("--work", required=True, type=Path); p.add_argument("--review-protocol", required=True, type=Path); p.add_argument("--approved-review-sha256", required=True)
        else:
            p.add_argument("--stage", required=True, type=Path)
        if name == "gtref": p.add_argument("--max-seconds", type=float, required=True)
        if name in ("run", "finalize"): p.add_argument("--ledger", required=True, type=Path)
        if name == "run":
            p.add_argument("--dimension", choices=e1.DIMS, required=True); p.add_argument("--gpu", type=int, required=True)
            p.add_argument("--holder-pid", type=int, required=True); p.add_argument("--state-dir", type=Path, required=True)
    p = sub.add_parser("worker"); p.add_argument("--action", type=Path, required=True); p.add_argument("--approved-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "collect-bundle": return collect_bundle(args.runtime_config, args.out)
    if args.command == "initialize-ledger": return initialize_ledger(args.plan, args.approved_plan_sha256, args.ledger, score.read_json(args.allocations))
    if args.command == "worker": return worker(args.action, args.approved_sha256)
    if args.command == "prepare-stage": return prepare_stage(args.plan, args.approved_plan_sha256, args.method, args.seed, args.work, args.review_protocol, args.approved_review_sha256)
    if args.command == "gtref": return gtref_cpu(args.plan, args.approved_plan_sha256, args.method, args.seed, args.stage, args.max_seconds)
    if args.command == "run": return run_action(args.plan, args.approved_plan_sha256, args.method, args.seed, args.dimension, args.stage, args.ledger, args.state_dir, args.gpu, args.holder_pid)
    return finalize_seed(args.plan, args.approved_plan_sha256, args.method, args.seed, args.stage, args.ledger)


if __name__ == "__main__":
    main()
