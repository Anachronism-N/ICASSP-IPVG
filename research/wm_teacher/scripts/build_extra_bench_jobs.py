"""Generation jobs (gen_jobs.py format) for two extra public I2V benchmarks.

physiq : Physics-IQ ORIGINAL (public gs://physics-iq-benchmark; the Verified set is HF-gated), i2v,
         198 take-1 scenarios, original prompts (descriptions_original.csv, 'op'). Outputs
         eval/physiq/videos/<model>-op-run_<NN>/<generated_video_name>; seeds run_01..: 0,1,2,3.
         (trim to exactly 5 s before scoring.) Original leaderboard: Wan2.2 38.3 %, Sora2 42.3 %.
wmb    : WorldModelBench (350 instances, 7 domains), i2v protocol: image=first_frame,
         prompt=text_instruction. Outputs eval/wmb/videos/<model>/<first_frame basename>.mp4

Usage: python build_extra_bench_jobs.py --suites physiq,wmb --models base,round3 [--runs 1]
"""
import argparse
import csv
import json
import os

from build_eval_jobs import MODELS

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval"
BENCH = "/dockerdata/wm_eval/bench"  # local copy; images are staged into EV/*/inputs so the worker can read them


def stage(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    dst = f"{dst_dir}/{os.path.basename(src)}"
    if not os.path.exists(dst):
        import shutil
        shutil.copy(src, dst)
    return dst


def pmap(suite, po):
    """prompt-optimized map {case_id: prompt} written by rewrite_prompts_vbench.py --suite <suite>"""
    return json.load(open(f"{EV}/{suite}/prompts_po.json")) if po else {}


def physiq(models, runs, po=False):
    rows = list(csv.DictReader(open(f"{BENCH}/physics-IQ-benchmark/descriptions/descriptions_original.csv")))
    rows = [r for r in rows if "_take-1_" in r["scenario"]]
    assert len(rows) == 198, len(rows)
    frames = {f[:4]: f for f in os.listdir(f"{BENCH}/physics-IQ-benchmark-data/physics-IQ-benchmark/switch-frames")}
    jobs, P, sfx = [], pmap("physiq", po), ("_po" if po else "")
    for m in models:
        for run in range(1, runs + 1):
            for r in rows:
                img = stage(f"{BENCH}/physics-IQ-benchmark-data/physics-IQ-benchmark/switch-frames/{frames[r['scenario'][:4]]}",
                            f"{EV}/physiq/inputs")
                if po and r["scenario"][:4] not in P:
                    continue
                jobs.append({"id": r["scenario"][:4], "model": m + sfx, "ckpt": MODELS[m], "image": img,
                             "prompt": P.get(r["scenario"][:4], r["description"]), "seed": run - 1,
                             "out": f"{EV}/physiq/videos/{m}{sfx}-op-run_{run:02d}/{r['generated_video_name']}"})
    return jobs


def wmb(models, po=False):
    d = json.load(open(f"{BENCH}/WorldModelBench/worldmodelbench.json"))
    jobs, P, sfx = [], pmap("wmb", po), ("_po" if po else "")
    for m in models:
        for x in d:
            img = stage(f"{BENCH}/WorldModelBench/{x['first_frame']}", f"{EV}/wmb/inputs")
            name = os.path.splitext(os.path.basename(x["first_frame"]))[0] + ".mp4"
            if po and name[:-4] not in P:
                continue
            jobs.append({"id": name[:-4], "model": m + sfx, "ckpt": MODELS[m], "image": img,
                         "prompt": P.get(name[:-4], x["text_instruction"]), "seed": 0, "out": f"{EV}/wmb/videos/{m}{sfx}/{name}"})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="physiq,wmb")
    ap.add_argument("--models", default="base,round3")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--po", action="store_true", help="use the prompt-optimized maps (eval/<suite>/prompts_po.json)")
    a = ap.parse_args()
    models = a.models.split(",")
    jobs = []
    if "physiq" in a.suites:
        jobs += physiq(models, a.runs, a.po)
    if "wmb" in a.suites:
        jobs += wmb(models, a.po)
    out = f"{EV}/jobs_extra{a.tag}.jsonl"
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"{len(jobs)} jobs -> {out}")


if __name__ == "__main__":
    main()
