#!/usr/bin/env python3
"""
GT-referenced metrics on WM-Bench (what VBench cannot measure: how close a generation stays to
the real continuation of the same first frame + caption).

Models differ in fps / length (ours 121f@24, Wan2.1 81f@16, CogVideoX 49f@8, Hunyuan 129f@24), so
both videos are sampled at K time points spread over the shared duration
(min(len_gen, len_gt), in seconds), resized to 256x144, frame 0 (the given image) excluded.

Per (model, clip):
  dreamsim      mean DreamSim distance gen vs GT at matched times   (lower = closer)  [perceptual]
  dreamsim_last DreamSim at the last shared time point               (drift at the end)
  motion_ratio  |log(mean |dF| gen / mean |dF| GT)|  (64x36 gray frame differences at the matched
                times); 0 = same amount of motion as the real clip    (lower = better)
CPU only (runs next to GPU jobs). Output: eval/wm_bench/gtref_<model>.json
Usage: /dockerdata/venv_vbench/bin/python gt_ref_metrics.py MODEL [MODEL ...]
"""
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = "/srv/wm_teacher"
EV = f"{ROOT}/eval/wm_bench"
K = 12
torch.set_num_threads(int(os.environ.get("THREADS", "32")))


def read(path):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 24)
        frames = [f.to_image() for f in c.decode(video=0)]
    return frames, fps


def sample(frames, fps, dur):
    ts = np.linspace(0, dur, K + 1)[1:]  # skip t=0 (conditioning frame)
    idx = [min(len(frames) - 1, int(round(t * fps))) for t in ts]
    return [frames[i] for i in idx]


def motion(frames):
    g = [np.asarray(f.convert("L").resize((64, 36)), dtype=np.float32) for f in frames]
    return float(np.mean([np.abs(a - b).mean() for a, b in zip(g[1:], g[:-1])])) + 1e-3


def main():
    from dreamsim import dreamsim
    os.chdir("/dockerdata/wm_eval/VBench")  # dreamsim weights live in ./models
    model, preprocess = dreamsim(pretrained=True, device="cpu", cache_dir="./models")
    cases = [json.loads(l) for l in open(f"{EV}/cases.jsonl")]
    gt_cache = {}
    for name in sys.argv[1:]:
        out = f"{EV}/gtref_{name}.json"
        res = json.load(open(out)) if os.path.exists(out) else {}
        for c in cases:
            vp = f"{EV}/videos/{name}/{c['id']}.mp4"
            if c["id"] in res or not os.path.exists(vp):
                continue
            if c["id"] not in gt_cache:
                gt_cache[c["id"]] = read(c["gt"])
            gf, gfps = gt_cache[c["id"]]
            try:
                vf, vfps = read(vp)
            except Exception as e:
                print(f"[{name}] unreadable {c['id']}: {e}", flush=True)
                continue
            dur = min(len(gf) / gfps, len(vf) / vfps) - 1e-3
            a, b = sample(gf, gfps, dur), sample(vf, vfps, dur)
            a = [x.resize((256, 144)) for x in a]
            b = [x.resize((256, 144)) for x in b]
            with torch.no_grad():
                ta = torch.cat([preprocess(x) for x in a])
                tb = torch.cat([preprocess(x) for x in b])
                d = model(ta, tb).flatten().tolist()
            res[c["id"]] = {"dataset": c["dataset"], "dreamsim": float(np.mean(d)), "dreamsim_last": d[-1],
                            "motion_ratio": abs(math.log(motion(b) / motion(a))), "dur": dur}
            json.dump(res, open(out, "w"))
        print(f"[{name}] {len(res)} clips: dreamsim={np.mean([r['dreamsim'] for r in res.values()]):.4f} "
              f"motion_err={np.mean([r['motion_ratio'] for r in res.values()]):.3f}", flush=True)


if __name__ == "__main__":
    main()
