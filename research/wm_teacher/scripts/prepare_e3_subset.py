#!/usr/bin/env python3
"""Prepare/verify a private, score-blind 276-video E3 diagnostic subset. CPU only."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TAG = "e3C_subset_20261025"
SELECTION_SEED = 20261025
SOURCE_SHA256 = {
    "jobs_vbench.jsonl": "ac015711fff307b89edeedb9275da8cd1c6485feda07f15afe8ae170ffd215d5",
    "jobs_vbench_po.jsonl": "811818d017d5ae506294f7f54a325f6d9ba96d2346c7aac4b30c03db0e284fbe",
    "jobs_wm.jsonl": "0498fd13d86ff14745f4db9252ae179f360fbe6835d1856a795279247d12e2e6",
    "jobs_extra_base.jsonl": "0946154b23349eeb91211c4bddf51d3ed3f5ce872622673915eb1adeef24e4f8",
}
VB_INFO = Path("/dockerdata/wm_eval/VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json")
PHYS_INFO = Path("/dockerdata/wm_eval/bench/physics-IQ-benchmark/descriptions/descriptions_original.csv")
WMB_INFO = Path("/dockerdata/wm_eval/bench/WorldModelBench/worldmodelbench.json")
EXPECTED_COUNTS = {"vbench": 71, "vbench_po": 71, "wm": 22, "physiq": 42, "wmb": 70}
SAFETY_RULES = {
    "sexualized_or_erotic": re.compile(
        r"\b(?:intimate|romantic\s+(?:kiss|scene|embrace)|kiss(?:ing)?\s+passionately|"
        r"passionate(?:ly)?\s+(?:kiss|embrace)|seduct(?:ive|ively|ion)|sensual(?:ly)?|"
        r"erotic|sexual(?:ly)?|pornograph\w*|making\s+love|make\s+love|"
        r"naked|nude|nudity|lingerie|striptease|topless|lustful|steamy\s+(?:kiss|romance))\b",
        re.IGNORECASE),
    "graphic_or_harmful": re.compile(r"\b(?:gore|gory|behead\w*|tortur\w*|rape|raping)\b", re.IGNORECASE),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Non-finite JSON value")
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


def loads(data):
    def reject(value):
        raise ValueError("Non-finite JSON constant")
    value = json.loads(data, parse_constant=reject)
    finite(value)
    return value


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def jsonl(data):
    require(bool(data.strip()), "Empty manifest")
    return [loads(line) for line in data.splitlines() if line.strip()]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    path = Path(path).absolute()
    require(path.is_file(), "Missing frozen file/input")
    h = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Frozen file changed while hashing")
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": after.st_size}


def verify_records(records):
    require(len({r["path"] for r in records}) == len(records), "Duplicate frozen path")
    for record in records:
        require(file_record(record["path"]) == record, "Frozen SHA256/size mismatch")


def validate_jobs(jobs, candidate=False):
    require(bool(jobs), "Empty job list")
    identities, outs = set(), set()
    required = {"id", "model", "ckpt", "image", "prompt", "out", "seed"}
    optional = {"cfg", "shift", "steps", "neg", "cfg_zero_star", "cfg_zero_steps"}
    for j in jobs:
        finite(j)
        require(isinstance(j, dict) and required <= set(j) <= required | optional, "Malformed job fields")
        for key in ("id", "model", "image", "prompt", "out", "ckpt"):
            require(isinstance(j[key], str) and bool(j[key]), "Empty/non-string job field")
        for key in ("image", "out"):
            p = Path(j[key])
            require(p.is_absolute() and ".." not in p.parts, "Job paths must be absolute and canonical")
        require(j["ckpt"] == "NONE" or Path(j["ckpt"]).is_absolute(), "Invalid checkpoint path")
        require(type(j["seed"]) is int and j["seed"] == 0, "Historical seed must remain zero")
        require(type(j.get("steps", 50)) is int and j.get("steps", 50) == 50, "Historical steps changed")
        for key in ("cfg", "shift"):
            require(type(j.get(key, 5.0)) in (int, float) and j.get(key, 5.0) == 5.0,
                    "Historical s5c5 changed")
        require(j.get("cfg_zero_star", False) is False and j.get("cfg_zero_steps", 0) == 0,
                "Historical sampler override changed")
        ident = (str(Path(j["out"]).parent), j["id"]) if candidate else (j["model"], j["id"])
        require(ident not in identities and j["out"] not in outs, "Duplicate job id/output")
        identities.add(ident)
        outs.add(j["out"])
        require(Path(j["image"]).is_file(), "Missing input image/video")
        if candidate:
            require(not unsafe_reasons(j["prompt"]), "Unsafe candidate prompt")
            require(j["model"] == TAG and j["ckpt"] != "NONE", "Unexpected candidate arm")


def unsafe_reasons(prompt):
    return [name for name, pattern in SAFETY_RULES.items() if pattern.search(prompt)]


def strata_sample(rows, n, key, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    require(len(groups) <= n <= len(rows), "Insufficient strata/sample coverage")
    labels = sorted(groups)
    targets = {k: n * len(groups[k]) / len(rows) for k in labels}
    quotas = {k: max(1, int(targets[k])) for k in labels}
    while sum(quotas.values()) < n:
        candidates = [k for k in labels if quotas[k] < len(groups[k])]
        k = max(candidates, key=lambda x: (targets[x] - quotas[x], -labels.index(x)))
        quotas[k] += 1
    while sum(quotas.values()) > n:
        candidates = [k for k in labels if quotas[k] > 1]
        require(bool(candidates), "Cannot preserve all strata")
        k = max(candidates, key=lambda x: (quotas[x] - targets[x], -labels.index(x)))
        quotas[k] -= 1
    picked = []
    for k in labels:
        pool = sorted(groups[k], key=lambda x: x["id"])
        rng.shuffle(pool)
        picked.extend(pool[:quotas[k]])
    return picked, {str(k): {"eligible": len(groups[k]), "selected": quotas[k]} for k in labels}


def generator_settings(path):
    tree = ast.parse(Path(path).read_text())
    settings = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple) and [x.id for x in target.elts] == ["W", "H", "NUM_FRAMES"]:
                    settings.update(zip(("width", "height", "num_frames"), ast.literal_eval(node.value)))
                elif isinstance(target, ast.Name) and target.id == "NEG":
                    settings["negative_prompt"] = ast.literal_eval(node.value)
    require((settings.get("width"), settings.get("height"), settings.get("num_frames")) == (1280, 704, 121),
            "Historical resolution/frame constants changed")
    lora = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "load_lora"]
    require(len(lora) == 1 and any(k.arg == "alpha" and ast.literal_eval(k.value) == 1
                                  for k in lora[0].keywords), "LoRA alpha is not frozen to 1")
    settings.update(cfg=5.0, shift=5.0, steps=50, seed=0, lora_alpha=1, fps=24)
    return settings


def select_cases(sources, vb_info, wm_info, phys_info, wmb_info):
    rng = random.Random(SELECTION_SEED)
    exclusions, strata = {}, {}

    def screen(rows, suite):
        kept, reasons = [], Counter()
        for row in rows:
            flagged = sorted(set(unsafe_reasons(row["prompt"]) + unsafe_reasons(row.get("po_prompt", ""))))
            if flagged:
                reasons.update(flagged)
            else:
                kept.append(row)
        exclusions[suite] = {"input_cases": len(rows), "excluded_cases": len(rows) - len(kept),
                             "eligible_cases": len(kept), "reason_counts": dict(reasons)}
        return kept

    raw, po = sources["vbench"], sources["vbench_po"]
    require(set(raw) == set(po) and len(raw) == 355, "Historical VBench paired coverage changed")
    metadata = {}
    for row in vb_info:
        dims = sorted(d for d in row["dimension"] if d != "camera_motion")
        if not dims:
            continue
        cid = row["prompt_en"].replace("/", "_")
        require(cid not in metadata, "Duplicate official VBench identity")
        metadata[cid] = dict(row, dims=dims)
    require(set(metadata) == set(raw), "Official VBench metadata coverage mismatch")
    vrows = []
    for cid in sorted(raw):
        a, b, m = raw[cid], po[cid], metadata[cid]
        require(a["image"] == b["image"] and a["seed"] == b["seed"] and a["prompt"] == m["prompt_en"]
                and Path(a["image"]).name == m["image_name"], "VBench raw/po historical input drift")
        vrows.append(dict(a, dims=m["dims"], image_type=m["image_type"], po_prompt=b["prompt"], suite="vbench"))
    vpicked, strata["vbench"] = strata_sample(screen(vrows, "vbench_paired"), 71,
                                             lambda r: (tuple(r["dims"]), r["image_type"]), rng)
    selected = []
    for r in vpicked:
        selected.append({k: v for k, v in r.items() if k != "po_prompt"})
        selected.append(dict(po[r["id"]], suite="vbench_po", dims=r["dims"], image_type=r["image_type"]))

    wm = sources["wm"]
    wmeta = {r["id"]: r for r in wm_info}
    require(len(wmeta) == len(wm_info) and set(wmeta) == set(wm) and len(wm) == 108,
            "WM metadata coverage changed")
    wrows = []
    for cid, j in wm.items():
        m = wmeta[cid]
        require(j["image"] == m["image"] and j["prompt"] == m["prompt"] and m["gt"] == m["image"],
                "WM historical input drift")
        wrows.append(dict(j, suite="wm", dataset=m["dataset"], gt=m["gt"]))
    wpicked, strata["wm"] = strata_sample(screen(wrows, "wm"), 22, lambda r: r["dataset"], rng)
    selected.extend(wpicked)

    physics = sources["physiq"]
    pmeta = {r["scenario"][:4]: r for r in phys_info if "_take-1_" in r["scenario"]}
    require(len(pmeta) == 198 and set(pmeta) == set(physics), "Physics take-1 coverage changed")
    scenes = defaultdict(list)
    for cid, j in physics.items():
        m = pmeta[cid]
        match = re.fullmatch(r"\d{4}_perspective-(left|center|right)_take-1_(.+)\.mp4", m["scenario"])
        require(match is not None, "Invalid Physics scene/view")
        require(Path(j["out"]).name == m["generated_video_name"] and j["prompt"] == m["description"],
                "Physics historical prompt/name drift")
        scenes[match[2]].append(dict(j, suite="physiq", scene=match[2], view=match[1], category=m["category"]))
    require(len(scenes) == 66, "Physics scene count changed")
    scene_rows = []
    for scene, views in scenes.items():
        require(len(views) == 3 and {r["view"] for r in views} == {"left", "center", "right"}
                and len({r["category"] for r in views}) == 1, "Incomplete/inconsistent Physics scene")
        scene_rows.append({"id": scene, "prompt": " ".join(r["prompt"] for r in views),
                           "category": views[0]["category"], "views": views})
    ppicked, strata["physiq_scenes"] = strata_sample(screen(scene_rows, "physiq_scenes"), 14,
                                                    lambda r: r["category"], rng)
    for scene in ppicked:
        selected.extend(sorted(scene["views"], key=lambda r: r["view"]))

    world = sources["wmb"]
    world_meta = {Path(r["first_frame"]).stem: r for r in wmb_info}
    require(len(world_meta) == len(wmb_info) == 350 and set(world_meta) == set(world),
            "WMB metadata coverage changed")
    world_rows = []
    for cid, j in world.items():
        m = world_meta[cid]
        require(Path(j["image"]).name == Path(m["first_frame"]).name and j["prompt"] == m["text_instruction"],
                "WMB historical input drift")
        world_rows.append(dict(j, suite="wmb", domain=m["domain"]))
    eligible = screen(world_rows, "wmb")
    domains = sorted({r["domain"] for r in world_rows})
    require(len(domains) == 7, "WMB domain coverage changed")
    strata["wmb"] = {}
    for domain in domains:
        pool = sorted([r for r in eligible if r["domain"] == domain], key=lambda r: r["id"])
        require(len(pool) >= 10, "Insufficient safe WMB domain coverage")
        rng.shuffle(pool)
        selected.extend(pool[:10])
        strata["wmb"][domain] = {"eligible": len(pool), "selected": 10}
    require(dict(Counter(r["suite"] for r in selected)) == EXPECTED_COUNTS, "Subset count mismatch")
    return selected, strata, exclusions


def derive_jobs(selected, out_dir, checkpoint, settings):
    jobs, cases = [], []
    for r in selected:
        model = TAG + ("_po" if r["suite"] == "vbench_po" else "")
        suite_dir = "vbench" if r["suite"] == "vbench_po" else r["suite"]
        model_dir = model + ("-op-run_01" if r["suite"] == "physiq" else "")
        out = str(Path(out_dir) / "videos" / suite_dir / model_dir / Path(r["out"]).name)
        j = {k: r[k] for k in ("id", "image", "prompt", "seed")}
        j.update(model=TAG, ckpt=str(checkpoint), out=out, cfg=5.0, shift=5.0, steps=50)
        if "neg" in r:
            require(r["neg"] == settings["negative_prompt"], "Historical negative prompt override differs")
            j["neg"] = r["neg"]
        jobs.append(j)
        c = dict(j, model=model, suite=r["suite"], video=out, base_video=r["out"])
        for key in ("dims", "image_type", "dataset", "gt", "scene", "view", "category", "domain"):
            if key in r:
                c[key] = r[key]
        c["input_contract_sha256"] = sha_bytes(encoded({k: r.get(k, settings.get(k))
                                                       for k in ("image", "prompt", "seed", "cfg", "shift", "steps")}))
        cases.append(c)
    validate_jobs(jobs, candidate=True)
    require(len({(c["suite"], c["id"]) for c in cases}) == len(cases), "Duplicate case identity")
    return jobs, cases


def exclusive(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_checkpoint_cache(checkpoint, checkpoint_sha, cache_dir):
    cache = Path(cache_dir) / (hashlib.sha256(str(Path(checkpoint).absolute()).encode()).hexdigest() + ".safetensors")
    record = file_record(cache) if cache.exists() else None
    if record:
        require(record["sha256"] == checkpoint_sha, "Checkpoint cache content SHA256 mismatch")
    return {"cache_dir": str(Path(cache_dir).absolute()), "cache_path": str(cache),
            "cache_present": record is not None, "cache_record": record,
            "effective_checkpoint": str(cache) if record else str(checkpoint),
            "effective_content_sha256": checkpoint_sha}


def runtime_snapshot():
    require(Path(sys.executable).absolute() == Path("/opt/conda/envs/torch-base/bin/python"),
            "Prepare using the frozen torch-base generator Python")
    packages = {"torch": "torch", "numpy": "numpy", "imageio": "imageio", "Pillow": "PIL",
                "safetensors": "safetensors", "imageio-ffmpeg": "imageio_ffmpeg", "av": "av"}
    info = {"python": str(Path(sys.executable).absolute()), "python_version": sys.version, "packages": {}}
    records = [file_record(sys.executable)]
    for package, module in packages.items():
        dist = importlib.metadata.distribution(package)
        metadata = [f for f in dist.files if str(f).endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "Missing/ambiguous installed dependency metadata")
        entry = Path(dist.locate_file(module + "/__init__.py"))
        records.extend([file_record(entry), file_record(dist.locate_file(metadata[0]))])
        info["packages"][package] = {"version": dist.version, "entrypoint": str(entry)}
    return info, records


def prepare(root=ROOT, out_dir=None, vb_info=VB_INFO, phys_info=PHYS_INFO, wmb_info=WMB_INFO,
            cache_dir=None):
    root = Path(root).resolve()
    out = Path(out_dir or root / "eval/e3_subset_20261025").absolute()
    require(out == root / "eval/e3_subset_20261025" and not out.exists() and not out.is_symlink(),
            "Output must be the new private E3 subset directory")
    checkpoint = root / "models/train/e3_arms_20261023/armC_fixed_anchor.safetensors"
    cache_dir = cache_dir or os.environ.get("WM_CHECKPOINT_CACHE_DIR", "/dockerdata/models/wm_checkpoints")
    frozen = {}
    sources = {}
    for name, expected in SOURCE_SHA256.items():
        p = root / "eval" / name
        data = p.read_bytes()
        require(sha_bytes(data) == expected, "Historical source manifest SHA256 mismatch")
        frozen[str(p)] = file_record(p)
        model = "base_po" if name == "jobs_vbench_po.jsonl" else "base"
        rows = [j for j in jsonl(data) if j["model"] == model]
        if name == "jobs_extra_base.jsonl":
            for suite in ("physiq", "wmb"):
                group = [j for j in rows if (root / "eval" / suite / "videos") in Path(j["out"]).parents]
                validate_jobs(group)
                sources[suite] = {j["id"]: j for j in group}
            require(sum(len(sources[s]) for s in ("physiq", "wmb")) == len(rows), "Unknown extra suite")
        else:
            validate_jobs(rows)
            suite = {"jobs_vbench.jsonl": "vbench", "jobs_vbench_po.jsonl": "vbench_po", "jobs_wm.jsonl": "wm"}[name]
            sources[suite] = {j["id"]: j for j in rows}
        require(all(j["ckpt"] == "NONE" for j in rows), "Historical reference is not base")
    metadata_paths = [Path(vb_info), root / "eval/wm_bench/cases.jsonl", Path(phys_info), Path(wmb_info)]
    metadata = []
    for p in metadata_paths:
        data = p.read_bytes()
        frozen[str(p.absolute())] = file_record(p)
        metadata.append(list(csv.DictReader(io.StringIO(data.decode()))) if p.suffix == ".csv"
                        else jsonl(data) if p.suffix == ".jsonl" else loads(data))
    settings = generator_settings(root / "scripts/gen_jobs.py")
    selected, strata, exclusions = select_cases(sources, *metadata)
    jobs, cases = derive_jobs(selected, out, checkpoint, settings)
    require(len(jobs) == 276, "Expected exactly 276 generation jobs")
    require(all(not Path(j["out"]).exists() and not Path(j["out"] + ".claim").exists() for j in jobs),
            "Candidate output/claim already exists")
    for p in sorted({r["image"] for r in selected} | {r["out"] for r in selected} | {str(checkpoint)}):
        frozen[p] = file_record(p)
        require(frozen[p]["bytes"] > 0, "Empty input/reference/checkpoint")
    cp = frozen[str(checkpoint)]
    cache = verify_checkpoint_cache(checkpoint, cp["sha256"], cache_dir)
    code = [root / "scripts" / name for name in ("gen_jobs.py", "build_eval_jobs.py", "build_extra_bench_jobs.py",
                                                "prepare_e3_subset.py")]
    imports = sorted((root / "DiffSynth-Studio/diffsynth").rglob("*.py"))
    require(bool(imports), "Missing DiffSynth import source")
    for p in code + imports:
        frozen[str(p)] = file_record(p)
    runtime, runtime_records = runtime_snapshot()
    for record in runtime_records:
        frozen[record["path"]] = record
    blobs = {"jobs.jsonl": b"".join(encoded(j) for j in jobs), "cases.jsonl": b"".join(encoded(c) for c in cases)}
    files = {str(out / name): {"path": str(out / name), "sha256": sha_bytes(data), "bytes": len(data)}
             for name, data in blobs.items()}
    plan = {
        "schema": "e3_subset_v1", "status": "prepared_pending_lead_review_NOT_launched", "private": True,
        "tag": TAG, "checkpoint": cp, "checkpoint_cache": cache, "generation": settings, "runtime": runtime,
        "selection": {"seed": SELECTION_SEED, "score_blind": True, "score_files_read": 0,
                      "method": "proportional largest-remainder with >=1 per observed stratum; sorted pools then seeded shuffle",
                      "vbench_stratum": "official non-camera dimension signature x image_type; raw/po common IDs",
                      "wm_stratum": "historical dataset", "physics_unit": "scene, keep all three views; category strata",
                      "wmb_stratum": "domain, exactly 10 each", "counts": EXPECTED_COUNTS, "strata": strata,
                      "vbench_dimension_counts": dict(Counter(d for c in cases if c["suite"] == "vbench" for d in c["dims"])),
                      "safety_exclusions": exclusions,
                      "exclusion_policy": "conservative exclusions before sampling: intimate whole word, romantic kiss/scene, erotic/sexualized expressions, nude/naked; apply union of VBench raw/po and exclude whole Physics scene if any view unsafe. May exclude non-sexual intimate close-ups or nude sculpture; ordinary non-sexual sculpture/art remains eligible. Diagnostic subset, not official distribution.",
                      "distribution_claim": "diagnostic safety-screened subset, NOT official complete benchmark distribution"},
        "budget": {"total_gpu_hours": 40, "generation_gpu_hours": 28, "score_gpu_hours": 12,
                   "generation_nodes": 4, "gpus_per_node": 8, "total_generation_gpus": 32,
                   "per_node_wall_seconds": 3000, "scheduled_generation_gpu_hours": 32 * 3000 / 3600,
                   "generation_reserve_gpu_hours": 28 - 32 * 3000 / 3600,
                   "full_1366_authorized": False, "training_authorized": False},
        "launch": {"authorized_to_launch": False, "GPU_used": False,
                   "required_preflight": "verify_plan before generator manifest loading and before scoring; content-verify checkpoint cache on each node",
                   "base_directories_read_only": True},
        "frozen_files": list(frozen.values()), "manifest_files": list(files.values()),
        "jobs_path": str(out / "jobs.jsonl"), "cases_path": str(out / "cases.jsonl"),
        "code_import_scope": "builder/gen_jobs, all DiffSynth Python sources, runtime binary, imported dependency entrypoints and version METADATA; native shared libraries not exhaustively hashed; no torch/CUDA import",
    }
    verify_records(plan["frozen_files"])
    out.mkdir(mode=0o700)
    for name, data in blobs.items():
        exclusive(out / name, data)
    data = encoded(plan)
    exclusive(out / "plan.json", data)
    exclusive(out / "plan.sha256", (sha_bytes(data) + "\n").encode())
    verify_plan(out / "plan.json")
    return plan


def verify_plan(plan_path, cache_dir=None, expected_plan_sha256=None):
    """Fail closed before any GPU import/manifest use; never read scores or write files."""
    path = Path(plan_path).absolute()
    data = path.read_bytes()
    expected = expected_plan_sha256 or path.with_name("plan.sha256").read_text().strip()
    require(sha_bytes(data) == expected, "Frozen plan SHA256 mismatch")
    plan = loads(data)
    require(plan["schema"] == "e3_subset_v1" and plan["tag"] == TAG and plan["private"] is True,
            "Unexpected subset plan")
    require(plan["selection"]["counts"] == EXPECTED_COUNTS and plan["selection"]["seed"] == SELECTION_SEED
            and plan["selection"]["score_blind"] is True, "Subset selection contract changed")
    verify_records(plan["frozen_files"])
    verify_records(plan["manifest_files"])
    require({r["path"] for r in plan["manifest_files"]} == {plan["jobs_path"], plan["cases_path"]},
            "Unfrozen manifest path")
    cp = plan["checkpoint"]
    require(cp in plan["frozen_files"], "Unfrozen checkpoint")
    verify_checkpoint_cache(cp["path"], cp["sha256"], cache_dir or os.environ.get("WM_CHECKPOINT_CACHE_DIR",
                                                                                 plan["checkpoint_cache"]["cache_dir"]))
    jobs, cases = jsonl(Path(plan["jobs_path"]).read_bytes()), jsonl(Path(plan["cases_path"]).read_bytes())
    validate_jobs(jobs, candidate=True)
    require(len(jobs) == len(cases) == 276 and dict(Counter(c["suite"] for c in cases)) == EXPECTED_COUNTS,
            "Manifest coverage differs")
    require(len({(c["suite"], c["id"]) for c in cases}) == len(cases), "Duplicate score case identity")
    frozen_paths = {r["path"] for r in plan["frozen_files"]}
    for j, c in zip(jobs, cases):
        score_model = TAG + ("_po" if c["suite"] == "vbench_po" else "")
        require(all(c.get(k) == v for k, v in j.items() if k != "model") and c["model"] == score_model
                and j["model"] == TAG and c["video"] == j["out"] and j["ckpt"] == cp["path"],
                "Job/case pairing changed")
        require(c["image"] in frozen_paths and c["base_video"] in frozen_paths, "Unfrozen score input/reference")
        require(Path(c["out"]).is_relative_to(path.parent / "videos")
                and c["out"] != c["base_video"], "Output escapes private subset or overwrites base")
    return plan, jobs, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    plan = verify_plan(args.verify, args.cache_dir)[0] if args.verify else prepare(args.root, cache_dir=args.cache_dir)
    print(json.dumps({"status": plan["status"], "counts": plan["selection"]["counts"],
                      "safety_exclusions": plan["selection"]["safety_exclusions"],
                      "GPU_used": False, "budget_gpu_hours": plan["budget"]["total_gpu_hours"]}, allow_nan=False))


if __name__ == "__main__":
    main()
