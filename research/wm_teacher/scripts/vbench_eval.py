#!/usr/bin/env python3
"""
Score generated videos with the official VBench-I2V code (vbench2_beta_i2v), all non-camera dims.

  suite=vbench : eval/vbench_i2v/videos/<model>/<prompt>-0.mp4, official full_info (16:9 crops)
  suite=wm     : eval/wm_bench/videos/<model>/<id>.mp4 ; a custom full_info is written with the GT
                 first frame (1280x704 png) as the conditioning image, so the same i2v_subject /
                 i2v_background / quality dims apply to the in-domain benchmark.
                 model "gt" scores the real clips themselves (reference ceiling for real footage).

Run with /dockerdata/venv_vbench/bin/python from /dockerdata/wm_eval/VBench (relative crop paths).
  python vbench_eval.py --suite vbench --model round2
"""
import argparse
import fcntl
import json
import math
import os
import shutil
import sys

import torch

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval"
VBREPO = "/dockerdata/wm_eval/VBench"
WORK = "/dockerdata/wm_eval/work"
DIMS = ["i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]


def prepare_wm(model):
    """Custom full_info + local copies (VBench decodes every frame; avoid the fuse mount)."""
    cases = [json.loads(l) for l in open(f"{EV}/wm_bench/cases.jsonl")]
    vdir = f"{WORK}/wm/{model}/videos"
    idir = f"{WORK}/wm/images"
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(idir, exist_ok=True)
    import imageio.v2 as imageio
    from PIL import Image
    info = []
    for c in cases:
        img = f"{idir}/{c['id']}.png"
        if not os.path.exists(img):
            r = imageio.get_reader(c["gt"])
            Image.fromarray(r.get_data(0)).save(img)
            r.close()
        src = c["gt"] if model == "gt" else f"{EV}/wm_bench/videos/{model}/{c['id']}.mp4"
        dst = f"{vdir}/{c['id']}-0.mp4"
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        info.append({"prompt_en": c["id"], "dimension": DIMS, "custom_image_path": img})
    fi = f"{WORK}/wm/{model}/full_info.json"
    json.dump(info, open(fi, "w"))
    return vdir, fi, len(info)


def prepare_vbench(model):
    vdir = f"{WORK}/vbench/{model}/videos"
    os.makedirs(vdir, exist_ok=True)
    src = f"{EV}/vbench_i2v/videos/{model}"
    for f in os.listdir(src):
        if f.endswith(".mp4") and not os.path.exists(f"{vdir}/{f}"):
            shutil.copyfile(f"{src}/{f}", f"{vdir}/{f}")
    return vdir, f"{VBREPO}/vbench2_beta_i2v/vbench2_i2v_full_info.json", len(os.listdir(vdir))


def valid_result(result, expected):
    try:
        names = [os.path.basename(v["video_path"]) for v in result[1]]
        return (len(names) == len(expected) and set(names) == expected
                and math.isfinite(float(result[0]))
                and all(math.isfinite(float(v["video_results"])) for v in result[1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["vbench", "wm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dims", default=",".join(DIMS))
    a = ap.parse_args()
    os.chdir(VBREPO)
    sys.path.insert(0, VBREPO)
    from vbench2_beta_i2v import VBenchI2V

    suite_dir = f"{EV}/{'wm_bench' if a.suite == 'wm' else 'vbench_i2v'}"
    cases = [json.loads(line) for line in open(f"{suite_dir}/cases.jsonl")]
    missing = []
    for case in cases:
        if a.model == "gt" and a.suite == "wm":
            path = case["gt"]
        else:
            name = case["id"] + ("-0.mp4" if a.suite == "vbench" else ".mp4")
            path = f"{suite_dir}/videos/{a.model}/{name}"
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            missing.append(case["id"])
    if missing:
        raise RuntimeError(f"Incomplete {a.suite}/{a.model}: {len(missing)}/{len(cases)} missing or empty videos")
    expected = {dim: {c["id"] + "-0.mp4" for c in cases if dim in c.get("dims", DIMS)} for dim in DIMS}
    vdir, full_info, n = (prepare_wm if a.suite == "wm" else prepare_vbench)(a.model)
    out = f"{EV}/{'wm_bench' if a.suite == 'wm' else 'vbench_i2v'}/scores"
    os.makedirs(out, exist_ok=True)
    print(f"[{a.suite}/{a.model}] {n} videos, full_info={full_info}", flush=True)
    if a.dims == "none":  # staging only (score_pool.py)
        return
    vb = VBenchI2V(torch.device("cuda"), full_info, out)
    for dim in a.dims.split(","):  # one json per dim -> partial results survive a crash
        result_path = f"{out}/{a.model}__{dim}_eval_results.json"
        # All scorers run on this launcher; serialize a dimension across early and main pools.
        with open(f"{os.path.dirname(vdir)}/{dim}.score.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.exists(result_path):
                result = json.load(open(result_path))[dim]
                if valid_result(result, expected[dim]):
                    continue
                print(f"Re-scoring incomplete/invalid {dim}; expected {len(expected[dim])} cases", flush=True)
            vb.evaluate(videos_path=vdir, name=f"{a.model}__{dim}", dimension_list=[dim],
                        resolution="16-9", mode="vbench_standard")
            result = json.load(open(result_path))[dim]
            if not valid_result(result, expected[dim]):
                raise RuntimeError(f"Incomplete/invalid {dim} results; expected {len(expected[dim])} cases")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
