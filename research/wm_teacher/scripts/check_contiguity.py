#!/usr/bin/env python3
"""
Verify that consecutive processed segments (_NNN, _NNN+1) of a source are really contiguous.

For a sampled boundary: d_cross = mean|last frame of seg k - first frame of seg k+1|,
d_intra = median of consecutive-frame diffs inside seg k (last 8 frames), plus the
same-orientation check. A boundary is "broken" if d_cross > max(4*d_intra, 12) or the
two segments have different resolution.
"""
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = "/srv/wm_teacher"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def tail_frames(p, k=9):
    import av, numpy as np
    with av.open(os.path.join(ROOT, p)) as c:
        s = c.streams.video[0]
        fr = [f.to_ndarray(format="gray").astype("float32") for f in c.decode(video=0)]
    return fr[-k:], (s.width, s.height)


def head_frame(p):
    import av
    with av.open(os.path.join(ROOT, p)) as c:
        s = c.streams.video[0]
        for f in c.decode(video=0):
            return f.to_ndarray(format="gray").astype("float32"), (s.width, s.height)


def check(pair):
    import numpy as np
    a, b = pair
    try:
        ta, ra = tail_frames(a)
        hb, rb = head_frame(b)
    except Exception as e:
        return "error", 0, 0
    if ra != rb:
        return "broken_res", 0, 0
    intra = float(np.median([np.abs(x - y).mean() for x, y in zip(ta, ta[1:])]))
    cross = float(np.abs(ta[-1] - hb).mean())
    ok = cross <= max(4 * intra, 12.0)
    return ("ok" if ok else "broken"), intra, cross


def main():
    rng = random.Random(1)
    by = defaultdict(list)
    with open(f"{ROOT}/training_metadata/inventory_5s.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if len(r["segs"]) >= 2:
                by[r["dataset"]].append(r)
    pairs = []
    for ds, runs in by.items():
        for r in rng.sample(runs, min(N, len(runs))):
            i = rng.randrange(len(r["segs"]) - 1)
            pairs.append((ds, (r["segs"][i], r["segs"][i + 1])))
    with ProcessPoolExecutor(64) as ex:
        res = list(ex.map(check, [p for _, p in pairs]))
    agg = defaultdict(lambda: defaultdict(int))
    ex_bad = {}
    for (ds, p), (st, intra, cross) in zip(pairs, res):
        agg[ds][st] += 1
        if st.startswith("broken") and ds not in ex_bad:
            ex_bad[ds] = (p[0].split("/")[-1], p[1].split("/")[-1], round(intra, 1), round(cross, 1))
    print(f"{'dataset':10s} {'checked':>7s} {'ok':>4s} {'broken':>6s} {'bad_res':>7s} {'err':>4s}  example")
    for ds in sorted(agg):
        a = agg[ds]
        n = sum(a.values())
        print(f"{ds:10s} {n:7d} {a['ok']:4d} {a['broken']:6d} {a['broken_res']:7d} {a['error']:4d}  {ex_bad.get(ds, '')}")


if __name__ == "__main__":
    main()
