#!/usr/bin/env python3
"""
E1 generation jobs (docs/43 section 3): sigma_shift x cfg grid on the E1 dev set, Wan2.2-TI2V-5B base, raw caption,
default negative prompt (gen_jobs.NEG, no "neg" field), seed 0, 50 steps, 121 frames 1280x704 (gen_jobs constants).
All jobs share model="base"/ckpt="NONE", so each gen_jobs shard loads the pipeline once. Case-major order keeps a
partial run balanced across configs. Each config writes to its own directory; existing jobs files are never
overwritten and the run aborts if any output already exists.

Usage: /opt/conda/envs/torch-base/bin/python scripts/build_e1_jobs.py
       /opt/conda/envs/torch-base/bin/python scripts/build_e1_jobs.py --smoke OUT_DIR   (2 jobs into a private dir)
"""
import argparse
import json
import os
import sys

ROOT = "/srv/wm_teacher"
E1 = f"{ROOT}/eval/e1_devset_20261021"
SHIFTS = (3, 5, 8, 12)
CFGS = (4, 5, 6)
DEFAULT = (5, 5)
SEED = 0
STEPS = 50
SMOKE_CONFIGS = ((5, 5), (12, 6))


def config_name(shift, cfg):
    return f"s{shift}_c{cfg}"


def build_jobs(cases, video_root, configs=None):
    configs = configs or [(s, c) for s in SHIFTS for c in CFGS]
    jobs = []
    for case in cases:
        for shift, cfg in configs:
            name = config_name(shift, cfg)
            jobs.append({"id": f"{case['id']}__{name}", "model": "base", "ckpt": "NONE", "image": case["image"],
                         "prompt": case["prompt"], "out": f"{video_root}/{name}/{case['id']}.mp4", "seed": SEED,
                         "cfg": float(cfg), "shift": float(shift), "steps": STEPS})
    outs = [j["out"] for j in jobs]
    if len(set(outs)) != len(outs) or len({j["id"] for j in jobs}) != len(jobs):
        raise RuntimeError("Duplicate job id/output")
    return jobs


def write_jobs(jobs, path):
    existing = [j["out"] for j in jobs if os.path.exists(j["out"]) or os.path.exists(j["out"] + ".claim")]
    if existing:
        raise RuntimeError(f"{len(existing)} outputs/claims already exist, e.g. {existing[0]}")
    with open(path, "x") as f:  # never overwrite an existing jobs file
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=f"{E1}/cases.jsonl")
    ap.add_argument("--smoke", metavar="OUT_DIR", help="write 2 smoke jobs (first case, s5_c5 and s12_c6) here")
    a = ap.parse_args()
    cases = [json.loads(line) for line in open(a.cases)]
    if a.smoke:
        jobs = build_jobs(cases[:1], f"{a.smoke}/videos", SMOKE_CONFIGS)
        path = f"{a.smoke}/jobs_smoke.jsonl"
    else:
        jobs = build_jobs(cases, f"{E1}/videos")
        path = f"{E1}/jobs_e1.jsonl"
        if len(jobs) != len(cases) * len(SHIFTS) * len(CFGS):
            sys.exit("Unexpected job count")
    write_jobs(jobs, path)
    print(f"{len(jobs)} jobs ({len(cases)} cases) -> {path}")


if __name__ == "__main__":
    main()
