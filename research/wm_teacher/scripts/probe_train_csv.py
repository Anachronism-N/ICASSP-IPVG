#!/usr/bin/env python3
"""Probe real video params (resolution / fps / frames / duration) of a training CSV, per dataset."""
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import av

ROOT = "/srv/wm_teacher"
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/training_metadata/round1_train.csv"
PER_DS = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def probe(rel):
    try:
        with av.open(os.path.join(ROOT, rel)) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate) if s.average_rate else 0.0
            n = s.frames
            dur = float(c.duration) / 1e6 if c.duration else (n / fps if fps else 0)
            if not n:
                n = sum(1 for _ in c.decode(video=0))
            return s.width, s.height, round(fps, 2), n, round(dur, 2)
    except Exception:
        return None


by_ds = defaultdict(list)
with open(CSV) as f:
    for row in csv.DictReader(f):
        by_ds[row["video"].split("/")[2].rsplit("_", 1)[0]].append(row["video"])

rng = random.Random(0)
print(f"{'dataset':10s} {'n_train':>7s} {'probed':>6s} {'resolution':22s} {'fps':14s} "
      f"{'frames min/med/max':>18s} {'<49f%':>6s} {'dur med':>7s}")
for ds, vids in sorted(by_ds.items(), key=lambda x: -len(x[1])):
    sample = rng.sample(vids, min(PER_DS, len(vids)))
    with ThreadPoolExecutor(32) as ex:
        res = [r for r in ex.map(probe, sample) if r]
    if not res:
        print(f"{ds:10s} {len(vids):7d} probe failed"); continue
    reso = Counter(f"{w}x{h}" for w, h, *_ in res).most_common(2)
    fpsc = Counter(r[2] for r in res).most_common(2)
    fr = sorted(r[3] for r in res)
    lt49 = 100 * sum(1 for x in fr if x < 49) / len(fr)
    durs = sorted(r[4] for r in res)
    reso_s = ",".join(f"{k}({100*v//len(res)}%)" for k, v in reso)
    fps_s = ",".join(f"{k}({100*v//len(res)}%)" for k, v in fpsc)
    print(f"{ds:10s} {len(vids):7d} {len(res):6d} {reso_s:22s} {fps_s:14s} "
          f"{fr[0]:>5d}/{fr[len(fr)//2]:>4d}/{fr[-1]:>5d}   {lt49:6.1f} {durs[len(durs)//2]:7.2f}", flush=True)
