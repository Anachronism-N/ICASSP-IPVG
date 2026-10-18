#!/usr/bin/env python3
"""
Exact inventory of 5s (121-frame @24fps) windows recoverable from processed clips.

Processed clips come from transcode_and_clip.py:
  - source > 5s  -> contiguous 4s segments  <stem>_NNN.mp4 (all 96 frames except the last)
  - source <= 5s -> single clip <stem>.mp4 (no segment suffix)
For every contiguous run of segments we probe only the LAST segment (real frames + resolution),
total_frames = 96*(len-1) + last_frames. Unsplit clips are probed individually.
Windows = floor(total_frames / 121), non-overlapping.

MIRA is handled from its chunk scheme (<match>_cNNNNN.pK_720p.mp4, all chunks 96 frames @24fps,
contiguity verified via physics.jsonl time_remaining).

Output: training_metadata/inventory_5s.jsonl (one line per contiguous run) + summary table.
"""
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = "/srv/wm_teacher"
Q = f"{ROOT}/training_metadata/caption_queues"
OUT = f"{ROOT}/training_metadata/inventory_5s.jsonl"
W = 121
SEG = 96
QUEUES = {
    "hoigen1m": "hoigen1m.jsonl", "celebvhq": "celebvhq.jsonl", "charades": "charades.jsonl",
    "noxi": "noxi.jsonl", "vfhq": "vfhq.jsonl", "direct": "direct.jsonl",
    "egoexo4d": "egoexo4d_full.jsonl", "matrix": "matrix_full.jsonl", "easycom": "easycom.jsonl",
    "dexycb": "dexycb.jsonl", "h2o": "h2o.jsonl", "seamless": "seamless.jsonl",
    "openvidhd": "openvidhd.jsonl",
}
SEG_RE = re.compile(r"^(.*)_(\d{3})\.mp4$")
MIRA_RE = re.compile(r"(.+)_c(\d{5})\.p(\d+)_720p\.mp4$")


def probe(rel):
    import av
    try:
        with av.open(os.path.join(ROOT, rel)) as c:
            s = c.streams.video[0]
            n = s.frames
            if not n:
                n = sum(1 for _ in c.decode(video=0))
            return rel, n, s.width, s.height
    except Exception:
        return rel, 0, 0, 0


def runs_of(idxs):
    idxs = sorted(set(idxs))
    out, cur = [], [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            out.append(cur); cur = [b]
    out.append(cur)
    return out


def main():
    runs = []  # dicts: dataset, source, dir, segs(list of rel paths), probe_rel
    for ds, fn in QUEUES.items():
        split = defaultdict(list)
        with open(f"{Q}/{fn}") as f:
            for line in f:
                try:
                    vp = json.loads(line)["video_path"]
                except Exception:
                    continue
                d, b = os.path.split(vp)
                m = SEG_RE.match(b)
                if m:
                    split[(d, m.group(1))].append(int(m.group(2)))
                else:
                    runs.append({"dataset": ds, "source": b[:-4], "segs": [vp], "split": False})
        for (d, stem), idxs in split.items():
            for r in runs_of(idxs):
                runs.append({"dataset": ds, "source": stem, "split": True,
                             "segs": [f"{d}/{stem}_{i:03d}.mp4" for i in r]})
        print(f"[list] {ds}: {sum(1 for r in runs if r['dataset']==ds)} runs", flush=True)

    to_probe = [r["segs"][-1] for r in runs]
    print(f"[probe] {len(to_probe)} files ...", flush=True)
    res = {}
    with ProcessPoolExecutor(128) as ex:
        for i, (rel, n, w, h) in enumerate(ex.map(probe, to_probe, chunksize=64), 1):
            res[rel] = (n, w, h)
            if i % 20000 == 0:
                print(f"  probed {i}/{len(to_probe)}", flush=True)

    agg = defaultdict(lambda: defaultdict(float))
    with open(OUT, "w") as fo:
        for r in runs:
            n_last, w, h = res.get(r["segs"][-1], (0, 0, 0))
            if n_last == 0:
                agg[r["dataset"]]["probe_fail"] += 1
                continue
            total = SEG * (len(r["segs"]) - 1) + n_last
            win = total // W
            orient = "portrait" if h > w else "landscape"
            r.update(total_frames=total, width=w, height=h, orient=orient, windows=win)
            fo.write(json.dumps(r) + "\n")
            a = agg[r["dataset"]]
            a["runs"] += 1
            a["hours"] += total / 24 / 3600
            a[f"win_{orient}"] += win
            a[f"runs_ge5_{orient}"] += 1 if win > 0 else 0

    # MIRA (no probe needed: all chunks 96 frames @24fps, 1280x720)
    streams = defaultdict(list)
    with open(f"{ROOT}/training_metadata/manifest_mira_full.jsonl") as f:
        for line in f:
            try:
                vp = json.loads(line)["video_path"]
            except Exception:
                continue
            m = MIRA_RE.search(vp)
            if m:
                streams[(m.group(1), m.group(3))].append(int(m.group(2)))
    a = agg["mira"]
    for (match, p), idxs in streams.items():
        for r in runs_of(idxs):
            total = SEG * len(r)
            a["runs"] += 1
            a["hours"] += total / 24 / 3600
            a["win_landscape"] += total // W
            a["runs_ge5_landscape"] += 1 if total >= W else 0

    print(f"\n{'dataset':10s} {'runs':>8s} {'hours':>8s} {'src>=5s(L)':>10s} {'5s_win_L':>10s} "
          f"{'src>=5s(P)':>10s} {'5s_win_P':>9s} {'fail':>5s}")
    tot_l = tot_p = 0
    for ds in list(QUEUES) + ["mira"]:
        a = agg[ds]
        tot_l += a["win_landscape"]; tot_p += a["win_portrait"]
        print(f"{ds:10s} {int(a['runs']):8d} {a['hours']:8.1f} {int(a['runs_ge5_landscape']):10d} "
              f"{int(a['win_landscape']):10d} {int(a['runs_ge5_portrait']):10d} "
              f"{int(a['win_portrait']):9d} {int(a['probe_fail']):5d}")
    print(f"{'TOTAL':10s} {'':8s} {'':8s} {'':10s} {int(tot_l):10d} {'':10s} {int(tot_p):9d}")
    print(f"\ninventory -> {OUT}")


if __name__ == "__main__":
    main()
