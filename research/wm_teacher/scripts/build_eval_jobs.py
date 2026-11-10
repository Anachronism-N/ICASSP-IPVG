#!/usr/bin/env python3
"""
Build the two evaluation benchmarks + generation job lists.

1. WM-Bench (in-domain): up to PER_DS held-out round-2 val clips per dataset, each from a distinct
   source sequence (val split is sequence-disjoint from training). Input = GT first frame + caption.
   -> eval/wm_bench/cases.jsonl
2. VBench-I2V core: the 355 image-prompt pairs that carry the non-camera dimensions
   (vbench2_i2v_full_info.json), 16:9 crops, 1 sample per pair (seed 0).
   -> eval/vbench_i2v/cases.jsonl

Jobs for gen_jobs.py -> eval/jobs_<suite>.jsonl for the given models.
Usage: python build_eval_jobs.py [--suites wm,vbench] [--models base,round1,round2]
"""
import argparse
import csv
import glob
import hashlib
from pathlib import Path
import json
import os
import random
from collections import defaultdict

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval"
VB = "/dockerdata/wm_eval/VBench/vbench2_beta_i2v"
VB_IMG = f"{EV}/vbench_i2v/images_16-9"  # copied from VB/data/crop/16-9 so both nodes can read it
PER_DS = 10
OFFICIAL_DIMENSIONS = frozenset(('i2v_subject', 'i2v_background', 'subject_consistency',
    'background_consistency', 'motion_smoothness', 'dynamic_degree',
    'aesthetic_quality', 'imaging_quality', 'camera_motion'))
MODELS = {
    "base": "NONE",
    "round1": f"{ROOT}/models/train/round1_lora_r32/step-14921.safetensors",
    "round2": f"{ROOT}/models/train/round2_5s_lora_r32/run2/step-1937.safetensors",
}
_r3 = sorted(glob.glob(f"{ROOT}/models/train/round3_anneal_lora_r32/run*/step-*.safetensors"), key=os.path.getmtime)
if _r3:
    MODELS["round3"] = _r3[-1]  # newest = final checkpoint of the annealing run
_r4 = sorted(glob.glob(f"{ROOT}/models/train/round4_aes_lora_r32/run*/step-*.safetensors"), key=os.path.getmtime)
if _r4:
    MODELS["round4"] = _r4[-1]  # aesthetic-filtered quality tuning
_r5 = sorted(glob.glob(f"{ROOT}/models/train/round5_dyn_lora_r32/run*/step-*.safetensors"), key=os.path.getmtime)
if _r5:
    MODELS["round5"] = _r5[-1]  # motion+aesthetic selection, short/long caption mix
_f1 = sorted(glob.glob(f"{ROOT}/models/train/full1_5b/run*/step-*.safetensors"), key=os.path.getmtime)
if _f1:
    MODELS["full1"] = _f1[-1]  # full fine-tune of the 5B DiT
_f2 = sorted(glob.glob(f"{ROOT}/models/train/ft2_5b/run*/step-*.safetensors"), key=os.path.getmtime)
if _f2:
    MODELS["ft2"] = _f2[-1]  # full FT from base: dense captions, camera-filtered, + general/driving/robot data
for _m in sorted(glob.glob(f"{ROOT}/models/train/merge/*.safetensors")):
    MODELS["m_" + os.path.basename(_m)[:-12]] = _m  # weight interpolation base<->ckpt (merge_ckpt.py)


def safe(s):
    return s.replace("/", "_")


def wm_cases():
    rng = random.Random(0)
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"{ROOT}/training_metadata/round2/round2_val.csv")):
        ds = r["video"].split("/")[2].rsplit("_5s", 1)[0]
        seq = os.path.basename(r["video"]).rsplit("_w", 1)[0]
        by_ds[ds][seq].append(r)
    cases = []
    for ds in sorted(by_ds):
        seqs = sorted(by_ds[ds])
        rng.shuffle(seqs)
        for s in seqs:
            rng.shuffle(by_ds[ds][s])
        # round robin over sequences: distinct sequences first; datasets with few val sequences
        # (noxi, seamless, h2o) get further windows of the same session, which are different moments
        picked, k = [], 0
        while len(picked) < PER_DS and any(k < len(by_ds[ds][s]) for s in seqs):
            for s in seqs:
                if k < len(by_ds[ds][s]) and len(picked) < PER_DS:
                    picked.append(by_ds[ds][s][k])
            k += 1
        for r in picked:
            cid = f"{ds}__{os.path.basename(r['video'])[:-4]}"
            cases.append({"id": cid, "dataset": ds, "gt": f"{ROOT}/{r['video']}",
                          "image": f"{ROOT}/{r['video']}", "prompt": r["prompt"]})
    return cases


def vbench_cases():
    info = json.load(open(f"{VB}/vbench2_i2v_full_info.json"))
    cases = []
    for x in info:
        dims = [d for d in x["dimension"] if d != "camera_motion"]
        if not dims:
            continue
        cases.append({"id": safe(x["prompt_en"]), "dataset": x["image_type"], "dims": dims,
                      "image": f"{VB_IMG}/{x['image_name']}", "image_name": x["image_name"],
                      "prompt": x["prompt_en"]})
    return cases


def file_hash(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def official_generation_plan(preflight_path, frozen_models):
    raw = Path(preflight_path).read_bytes()
    preflight = json.loads(raw)
    if preflight.get('schema') != 'official_vbench_i2v_complete_coverage_CPU_preflight_v1':
        raise ValueError('Official complete-coverage preflight required')
    if preflight.get('execution_allowed') is not False or preflight.get('samples_per_pair') != 5:
        raise ValueError('CPU-only five-sample preflight required')
    rows = preflight['coverage_records']
    if len(rows) != 5590 or len({row['filename'] for row in rows}) != 5590:
        raise ValueError('Complete 5590-video coverage required')
    groups = defaultdict(set)
    pair_inputs = {}
    input_pairs = {}
    for row in rows:
        if type(row['sample_index']) is not int or not 0 <= row['sample_index'] < 5:
            raise ValueError('Strict integer sample index required')
        if type(row['pair_key']) is not str or not row['pair_key']:
            raise ValueError('Nonempty pair key required')
        identity = (row['raw_prompt'], row['image_key'])
        if row['pair_key'] in pair_inputs and pair_inputs[row['pair_key']] != identity:
            raise ValueError('Pair key maps to different image/prompt inputs')
        if identity in input_pairs and input_pairs[identity] != row['pair_key']:
            raise ValueError('Image/prompt pair split across different pair keys')
        pair_inputs[row['pair_key']] = identity
        input_pairs[identity] = row['pair_key']
        groups[row['pair_key']].add(row['sample_index'])
        if row['execution_allowed'] is not False or Path(row['filename']).name != row['filename']:
            raise ValueError('Non-dispatchable safe output names required')
    if len(groups) != 1118 or any(indices != set(range(5)) for indices in groups.values()):
        raise ValueError('Every official pair requires all five sample indices')
    metadata = preflight['metadata']
    if file_hash(metadata['path']) != metadata['sha256']:
        raise ValueError('Official metadata changed')
    info = json.loads(Path(metadata['path']).read_bytes())
    expected = {(case['prompt_en'].replace('/', '_') + '-' + str(i) + '.mp4', i, case['prompt_en'], case['image_name']) for case in info for i in range(5)}
    observed = {(row['filename'], row['sample_index'], row['raw_prompt'], row['image_key']) for row in rows}
    if len(info) != 1118 or len(expected) != 5590 or observed != expected:
        raise ValueError('Plan rows differ from official metadata')
    metadata_cases = {(case['prompt_en'], case['image_name']): case for case in info}
    if len(metadata_cases) != 1118 or set(input_pairs) != set(metadata_cases):
        raise ValueError('Pair identities differ from official metadata')
    for row in rows:
        dimensions = row['dimensions']
        expected_dimensions = set(metadata_cases[(row['raw_prompt'], row['image_key'])]['dimension']) & OFFICIAL_DIMENSIONS
        if type(dimensions) is not list or any(type(d) is not str for d in dimensions) or len(dimensions) != len(set(dimensions)) or set(dimensions) != expected_dimensions:
            raise ValueError('Per-pair scoring dimensions differ from official metadata')
    if any(type(row['proposed_seed']) is not int or not 0 <= row['proposed_seed'] < (1 << 63) for row in rows):
        raise ValueError('Explicit finite integer seeds required')
    for image in preflight['image_files'].values():
        if Path(image['path']).stat().st_size != image['bytes'] or file_hash(image['path']) != image['sha256']:
            raise ValueError('Official input image bytes changed')
    if type(frozen_models) is not dict or not frozen_models:
        raise ValueError('Explicit model names and checkpoints required')
    checkpoints, jobs = {}, []
    for name, checkpoint in frozen_models.items():
        if type(name) is not str or not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Safe explicit model name required')
        if checkpoint == 'NONE':
            checkpoints[name] = {'kind': 'base_checkpoint', 'base_asset_admission_required': True}
        else:
            if type(checkpoint) is not str or not Path(checkpoint).is_absolute():
                raise ValueError('Explicit absolute checkpoint path required')
            path = Path(checkpoint)
            before = path.stat()
            if not path.is_file() or before.st_size == 0:
                raise ValueError('Nonempty frozen checkpoint required')
            digest = file_hash(path)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError('Checkpoint changed while freezing')
            checkpoints[name] = {'path': str(path), 'bytes': after.st_size, 'sha256': digest}
        for row in rows:
            jobs.append({
                'id': row['pair_key'] + '-' + str(row['sample_index']),
                'model': name, 'ckpt': checkpoint,
                'image': preflight['image_files'][row['image_key']]['path'],
                'prompt': row['raw_prompt'], 'seed': row['proposed_seed'],
                'sample_index': row['sample_index'], 'dimensions': row['dimensions'],
                'out': str(Path(EV) / 'vbench_i2v/videos_official5' / name / row['filename']),
                'execution_allowed': False,
            })
    return {
        'schema': 'official_vbench_generation_CPU_plan_v1',
        'preflight': {'path': str(Path(preflight_path).resolve()), 'sha256': hashlib.sha256(raw).hexdigest()},
        'builder_source_sha256': file_hash(__file__), 'frozen_checkpoints': checkpoints, 'jobs': jobs,
        'videos_per_model': 5590, 'samples_per_pair': 5,
        'seed_policy': preflight['seed_policy'], 'prompt_protocol': 'raw',
        'execution_allowed': False, 'GPU_used': False, 'generation_performed': False,
        'sampler_budget_and_backend_admission_required': True,
        'crop_provenance_admission_required': True,
        'semantic_and_supervised_cache_gates_required': False,
        'supervised_training_data_used': False, 'formal_training_allowed': False,
        'checkpoint_training_provenance_admission_required': any(cp != 'NONE' for cp in frozen_models.values()),
        'remote_source_byte_gate_required': True, 'sota_proven': False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="wm,vbench")
    ap.add_argument("--models", default="base,round1,round2")
    ap.add_argument("--tag", default="", help="suffix for the jobs file, e.g. _round3")
    ap.add_argument("--prompt-map", default="", help="json {case_id: prompt} overriding the case prompt")
    ap.add_argument("--name-suffix", default="", help="appended to the model's output dir, e.g. _po")
    ap.add_argument("--official-five-plan", help="CPU-only complete-coverage preflight path; never dispatches gen_jobs.py")
    ap.add_argument("--frozen-models", help="JSON mapping explicit model names to absolute checkpoints or NONE")
    ap.add_argument("--output", help="New JSON plan path, never a legacy jobs/cases file")
    a = ap.parse_args()
    if not a.official_five_plan and (a.frozen_models or a.output):
        ap.error("Frozen models and output require --official-five-plan")
    if a.official_five_plan:
        if not a.frozen_models or not a.output or a.prompt_map or a.name_suffix or a.tag:
            ap.error("Official mode requires frozen models and a new output; prompt overrides and legacy tags are forbidden")
        output = Path(a.output)
        if not output.is_absolute() or output.exists():
            ap.error("Official CPU plan output must be a new absolute path")
        frozen_models = json.loads(Path(a.frozen_models).read_bytes())
        plan = official_generation_plan(a.official_five_plan, frozen_models)
        with output.open("x", encoding="utf-8") as target:
            json.dump(plan, target, ensure_ascii=False, sort_keys=True)
            target.write(chr(10))
        output.chmod(0o600)
        print(f"CPU official plan: {len(plan['jobs'])} jobs; execution remains blocked")
        return
    for suite in a.suites.split(","):
        cases = wm_cases() if suite == "wm" else vbench_cases()
        d = f"{EV}/{'wm_bench' if suite == 'wm' else 'vbench_i2v'}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/cases.jsonl", "w") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        missing = [c["image"] for c in cases if not os.path.exists(c["image"])]
        jobs = []
        pmap = json.load(open(a.prompt_map)) if a.prompt_map else {}
        for m in a.models.split(","):
            for c in cases:
                # VBench expects <prompt>-<index>.mp4
                name = f"{c['id']}-0.mp4" if suite == "vbench" else f"{c['id']}.mp4"
                jobs.append({"id": c["id"], "model": m + a.name_suffix, "ckpt": MODELS[m], "image": c["image"],
                             "prompt": pmap.get(c["id"], c["prompt"]),
                             "out": f"{d}/videos/{m}{a.name_suffix}/{name}", "seed": 0})
        with open(f"{EV}/jobs_{suite}{a.tag}.jsonl", "w") as f:
            for j in jobs:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        per = defaultdict(int)
        for c in cases:
            per[c["dataset"]] += 1
        print(f"{suite}: {len(cases)} cases, {len(jobs)} jobs, missing images={len(missing)} {dict(per)}")


if __name__ == "__main__":
    main()
