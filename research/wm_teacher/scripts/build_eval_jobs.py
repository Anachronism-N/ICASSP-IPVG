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
import json
import os
import random
from collections import defaultdict

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval"
VB = "/dockerdata/wm_eval/VBench/vbench2_beta_i2v"
VB_IMG = f"{EV}/vbench_i2v/images_16-9"  # copied from VB/data/crop/16-9 so both nodes can read it
PER_DS = 10
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="wm,vbench")
    ap.add_argument("--models", default="base,round1,round2")
    ap.add_argument("--tag", default="", help="suffix for the jobs file, e.g. _round3")
    ap.add_argument("--prompt-map", default="", help="json {case_id: prompt} overriding the case prompt")
    ap.add_argument("--name-suffix", default="", help="appended to the model's output dir, e.g. _po")
    a = ap.parse_args()
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
