#!/usr/bin/env python3
"""
CPU scan for hard cuts / abrupt changes / flicker in videos (training clips or generated samples).
Per video: decode to 64x36 gray, d_t = mean|f_t - f_{t-1}|.
  cut      : d_t > max(25, 6 * median(d))   (shot boundary or teleport-like jump)
  flicker  : std of per-frame mean brightness after removing a 9-frame moving average
Usage:
  scan_cuts.py csv TRAIN.csv --per-source 60 --out out.jsonl     (source = dataset/wan22_5s/<src>_5s/...)
  scan_cuts.py dirs NAME=DIR [NAME=DIR ...] --out out.jsonl
"""
import argparse
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "/srv/wm_teacher"
W, H = 64, 36


def stats(path):
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H},format=gray",
                              "-f", "rawvideo", "-"], capture_output=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return None
    f = np.frombuffer(raw, np.uint8)
    n = f.size // (W * H)
    if n < 10:
        return None
    f = f[: n * W * H].reshape(n, H, W).astype(np.float32)
    d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    med = float(np.median(d))
    cuts = int((d > max(25.0, 6 * med)).sum())
    lum = f.mean(axis=(1, 2))
    k = 9
    smooth = np.convolve(np.pad(lum, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return {"frames": n, "cuts": cuts, "max_jump": float(d.max()), "med_diff": med,
            "flicker": float(np.std(lum - smooth))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["csv", "dirs"])
    ap.add_argument("items", nargs="+")
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--threads", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    todo = []  # (group, path)
    if a.mode == "csv":
        import csv
        by = defaultdict(list)
        for r in csv.DictReader(open(a.items[0])):
            m = re.search(r"wan22_5s(?:_extra)?/([a-z0-9]+?)(?:_5s)?/", r["video"])
            by[m.group(1) if m else "other"].append(os.path.join(ROOT, r["video"]))
        rng = random.Random(0)
        for g, v in by.items():
            todo += [(g, p) for p in rng.sample(v, min(a.per_source, len(v)))]
    else:
        for it in a.items:
            name, d = it.split("=", 1)
            todo += [(name, os.path.join(d, x)) for x in sorted(os.listdir(d)) if x.endswith(".mp4") and ".tmp" not in x]

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda t: (t[0], t[1], stats(t[1])), todo))
    agg = defaultdict(list)
    with open(a.out, "w") as fo:
        for g, p, s in res:
            if s:
                fo.write(json.dumps({"group": g, "path": p, **s}) + "\n")
                agg[g].append(s)
    print(f"{'group':14s} {'n':>5s} {'cut-rate':>9s} {'cuts/clip':>9s} {'flicker':>8s} {'med_diff':>8s}")
    for g in sorted(agg):
        v = agg[g]
        print(f"{g:14s} {len(v):5d} {np.mean([s['cuts'] > 0 for s in v]):9.3f} {np.mean([s['cuts'] for s in v]):9.2f} "
              f"{np.mean([s['flicker'] for s in v]):8.3f} {np.mean([s['med_diff'] for s in v]):8.2f}")


if __name__ == "__main__":
    main()
