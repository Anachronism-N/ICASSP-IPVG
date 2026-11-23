#!/usr/bin/env python3
"""Check that reslice_5s._process_block cuts frame-accurate windows from mid-source blocks."""
import os
import random
import sys
import time

import av
import numpy as np

sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reslice_5s as R


def frame_at(segs, idx):
    s, k = idx // R.SEG, idx % R.SEG
    with av.open(os.path.join(R.ROOT, segs[s])) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == k:
                return f.to_ndarray(format="gray").astype(np.float32)


def first(p):
    with av.open(p) as c:
        for f in c.decode(video=0):
            return f.to_ndarray(format="gray").astype(np.float32)


random.seed(3)
for ds in ["celebvhq", "mira", "noxi"]:
    runs = R.load_runs({ds}, "landscape")[ds]
    r = random.choice([x for x in runs if x["windows"] >= 6])
    t = time.time()
    rows, fails = R._process_block(r, [3, 4, 5], "dataset/wan22_5s_test")
    for row in rows:
        w = row["window_idx"]
        out = first(os.path.join(R.ROOT, row["video_path"]))
        ref = frame_at(r["segs"], w * R.W)
        prev = frame_at(r["segs"], w * R.W - 1)
        print(f"{ds} w{w}: |out-exact|={np.abs(out - ref).mean():.2f} |out-prevframe|={np.abs(out - prev).mean():.2f}",
              flush=True)
    print(f"  {ds}: rows={len(rows)} fails={fails} {time.time() - t:.1f}s", flush=True)
