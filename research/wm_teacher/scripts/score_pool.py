#!/usr/bin/env python3
"""
Parallel VBench scoring: one subprocess per (suite, model, dim), up to CONC at a time,
one per available scoring GPU. Pass only GPUs that are not being used for generation.
Already-finished (suite, model, dim) results are skipped by vbench_eval.py itself.

Usage: /dockerdata/venv_vbench/bin/python score_pool.py SUITE:MODEL[,MODEL..] [SUITE:MODELS ...] [--conc 16]
"""
import argparse
import os
import subprocess
import sys
import time
import queue
from concurrent.futures import ThreadPoolExecutor

ROOT = "/srv/wm_teacher"
L = f"{ROOT}/dataset/download_logs"
DIMS = ["i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]
PY = "/dockerdata/venv_vbench/bin/python"
EVAL = "/dockerdata/wm_tmp/vbench_eval.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    a = ap.parse_args()
    reserved = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
            if not any(os.path.basename(x) == b"evaluation.py" for x in cmd) or b"--model_name" not in cmd:
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                process_env = f.read().split(b"\0")
            for entry in process_env:
                if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                    reserved.update(entry.split(b"=", 1)[1].decode().split(","))
        except OSError:
            continue
    gpus = [g for g in dict.fromkeys(a.gpus.split(",")) if g not in reserved]
    if not gpus:
        raise RuntimeError("All requested GPUs are reserved by WorldModelBench scoring")
    available_gpus = queue.Queue()
    for g in gpus:
        available_gpus.put(g)
    print(f"VBench GPUs: {','.join(gpus)}; WMB reserved: {','.join(sorted(reserved))}", flush=True)
    env = dict(os.environ, VBENCH_CACHE_DIR="/dockerdata/vbench_cache")

    # stage videos once per (suite, model) before fanning out (avoids concurrent copies)
    tasks = []
    for spec in a.specs:
        suite, models = spec.split(":")
        for m in models.split(","):
            r = subprocess.run([PY, EVAL, "--suite", suite, "--model", m, "--dims", "none"],
                               cwd="/dockerdata/wm_eval/VBench", env=env, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"Staging {suite}/{m} failed: {r.stderr[-1000:]}")
            print(f"staged {suite}/{m}: {r.stdout.strip().splitlines()[:1]}", flush=True)
            tasks += [(suite, m, d) for d in DIMS]

    def run(task):
        suite, m, d = task
        g = available_gpus.get()
        try:
            log = f"{L}/score_{suite}_{m}_{d}.log"
            t = time.time()
            with open(log, "w") as f:
                rc = subprocess.run([PY, EVAL, "--suite", suite, "--model", m, "--dims", d],
                                    cwd="/dockerdata/wm_eval/VBench", env=dict(env, CUDA_VISIBLE_DEVICES=g),
                                    stdout=f, stderr=subprocess.STDOUT).returncode
            print(f"{time.strftime('%H:%M:%S')} {suite}/{m}/{d} rc={rc} {(time.time() - t) / 60:.0f}min", flush=True)
            return rc
        finally:
            available_gpus.put(g)

    with ThreadPoolExecutor(min(a.conc, len(gpus))) as ex:
        results = list(ex.map(run, tasks))
    if any(results):
        raise SystemExit("VBench scoring failed; see per-dimension logs")
    print("[pool done]", flush=True)


if __name__ == "__main__":
    main()
