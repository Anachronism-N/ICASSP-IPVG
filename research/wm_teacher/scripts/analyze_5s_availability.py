#!/usr/bin/env python3
"""
Estimate how much 5s (121-frame @24fps) training data each dataset can yield.

1. Group processed clips by source video: strip trailing segment index `_NNN.mp4`
   (added by transcode_and_clip.py when source > 5s and was cut into 4s segments).
2. Probe real frame counts on a random sample per dataset (PyAV).
3. For split sources, consecutive 4s segments are contiguous (ffmpeg -f segment),
   so N consecutive segments can be concatenated and re-cut into 5s windows:
   windows ~= floor(total_seconds / 5) per contiguous run.
"""
import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import av

ROOT = "/srv/wm_teacher"
SRC = f"{ROOT}/training_metadata/recaptioned"
DATASETS = ["hoigen1m", "egoexo4d", "charades", "noxi", "direct", "mira", "celebvhq",
            "matrix", "vfhq", "dexycb", "easycom", "h2o", "seamless", "openvidhd"]
SEG_RE = re.compile(r"^(.*)_(\d{3})\.mp4$")
N_PROBE = 120
FPS = 24
FRAMES_5S = 121


def probe(path):
    try:
        with av.open(os.path.join(ROOT, path)) as c:
            s = c.streams.video[0]
            n = s.frames
            if not n:
                dur = float(s.duration * s.time_base) if s.duration else float(c.duration) / 1e6
                n = int(round(dur * FPS))
            return n
    except Exception:
        return None


def analyze(name):
    paths = []
    with open(f"{SRC}/{name}_recaptioned.jsonl") as f:
        for line in f:
            try:
                paths.append(json.loads(line)["video_path"])
            except Exception:
                pass
    # group by source
    split_src = defaultdict(list)  # source -> list of segment idx
    unsplit = []
    for p in paths:
        m = SEG_RE.match(os.path.basename(p))
        if m:
            split_src[m.group(1)].append(int(m.group(2)))
        else:
            unsplit.append(p)

    # probe sample
    rng = random.Random(0)
    sample = rng.sample(paths, min(N_PROBE, len(paths)))
    with ThreadPoolExecutor(16) as ex:
        frames = [n for n in ex.map(probe, sample) if n]

    def frac(cond):
        return sum(1 for n in frames if cond(n)) / max(len(frames), 1)

    # 5s windows from split sources (contiguous runs of segment idx, 4s each)
    win_split = 0
    src_ge5 = 0
    for idxs in split_src.values():
        idxs = sorted(set(idxs))
        run = 1
        runs = []
        for a, b in zip(idxs, idxs[1:]):
            if b == a + 1:
                run += 1
            else:
                runs.append(run); run = 1
        runs.append(run)
        w = sum((4 * r) // 5 for r in runs)  # conservative: every segment counted as 4s
        win_split += w
        if w > 0:
            src_ge5 += 1

    # unsplit clips: only usable if already >= 121 frames
    unsplit_ge5 = int(round(len(unsplit) * frac(lambda n: n >= FRAMES_5S))) if unsplit else 0
    med = sorted(frames)[len(frames) // 2] if frames else 0
    return {
        "dataset": name, "clips": len(paths),
        "split_sources": len(split_src), "unsplit_clips": len(unsplit),
        "probe_n": len(frames), "median_frames": med,
        "pct_ge121": 100 * frac(lambda n: n >= FRAMES_5S),
        "pct_ge97": 100 * frac(lambda n: n >= 97),
        "pct_lt96": 100 * frac(lambda n: n < 96),
        "win5s_from_concat": win_split, "split_src_ge5s": src_ge5,
        "unsplit_ge5s_est": unsplit_ge5,
    }


if __name__ == "__main__":
    names = sys.argv[1:] or DATASETS
    print(f"{'dataset':10s} {'clips':>9s} {'split_src':>9s} {'unsplit':>9s} "
          f"{'med_fr':>6s} {'%>=121':>6s} {'%<96':>6s} {'5s_win(concat)':>14s} {'unsplit>=5s':>11s}", flush=True)
    for n in names:
        r = analyze(n)
        print(f"{r['dataset']:10s} {r['clips']:9d} {r['split_sources']:9d} {r['unsplit_clips']:9d} "
              f"{r['median_frames']:6d} {r['pct_ge121']:6.1f} {r['pct_lt96']:6.1f} "
              f"{r['win5s_from_concat']:14d} {r['unsplit_ge5s_est']:11d}", flush=True)
