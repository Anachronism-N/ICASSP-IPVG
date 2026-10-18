"""Round 5 mixture: target the VBench gap to DreamX-World-1.0 (dynamic_degree is the largest term).

From the aesthetic-scored pool (training_metadata/round4/aes_scores_*.jsonl; val sequences excluded):
per dataset, rank windows by  z(aesthetic) + MOTION_W * z(log diff_median)  computed within the dataset,
drop the dataset's least-moving --min-motion-q fraction, keep <= --per-seq windows per source sequence,
and fill the Round-2 share scaled to --n-total. Windows already used by Round4 are skipped first and
only reused if a dataset runs short.
Output: training_metadata/round5/round5_pool.csv (video,prompt,dataset,aes,motion) — the short-caption
step adds a 'short' column, then make_round5_csv() writes round5_train.csv with a short/long mix.
"""
import argparse
import csv
import glob
import json
import math
import random
import statistics as st
from collections import defaultdict

ROOT = "/srv/wm_teacher"
R4 = f"{ROOT}/training_metadata/round4"
OUT = f"{ROOT}/training_metadata/round5"
SHARE = {
    "hoigen1m": 20000, "celebvhq": 12000, "mira": 12000, "egoexo4d": 10000, "charades": 10000,
    "noxi": 8000, "direct": 8000, "vfhq": 6000, "matrix": 6000, "seamless": 3000,
    "easycom": 2800, "openvidhd": 300, "h2o": 196,
}


def z(xs):
    m, s = st.mean(xs), (st.pstdev(xs) or 1.0)
    return [(x - m) / s for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=40000)
    ap.add_argument("--motion-w", type=float, default=1.0)
    ap.add_argument("--min-motion-q", type=float, default=0.3)
    ap.add_argument("--per-seq", type=int, default=6)
    a = ap.parse_args()
    import os
    os.makedirs(OUT, exist_ok=True)
    aes = {}
    for f in glob.glob(f"{R4}/aes_scores_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            aes[r["video"]] = r["aes"]
    used4 = {r["video"] for r in csv.DictReader(open(f"{R4}/round4_train.csv"))}
    by = defaultdict(list)
    for l in open(f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl"):
        m = json.loads(l)
        if m["video_path"] in aes:
            by[m["source_dataset"]].append(m)
    scale = a.n_total / sum(SHARE.values())
    rows = []
    print(f"{'dataset':10s} {'pool':>6s} {'target':>6s} {'taken':>6s} {'reuse4':>6s} {'mot_all':>7s} {'mot_kept':>8s} {'aes_kept':>8s}")
    for ds, share in SHARE.items():
        cand = by[ds]
        if not cand:
            continue
        mot = [math.log((m.get("diff_median") or 0) + 0.05) for m in cand]
        floor = sorted(mot)[int(a.min_motion_q * (len(mot) - 1))]
        za, zm = z([aes[m["video_path"]] for m in cand]), z(mot)
        score = {m["video_path"]: za[i] + a.motion_w * zm[i] for i, m in enumerate(cand) if mot[i] >= floor}
        ranked = sorted((m for m in cand if m["video_path"] in score), key=lambda m: -score[m["video_path"]])
        target = round(share * scale)
        per_seq, got = defaultdict(int), []
        for fresh_pass in (True, False):  # first windows not used by Round4, then reuse if short
            for m in ranked:
                if len(got) >= target:
                    break
                if (m["video_path"] in used4) == fresh_pass:
                    continue
                if per_seq[m["source_sequence_id"]] >= a.per_seq:
                    continue
                per_seq[m["source_sequence_id"]] += 1
                got.append(m)
        reuse = sum(m["video_path"] in used4 for m in got)
        mk = st.mean(m.get("diff_median") or 0 for m in got)
        ma = st.mean(m.get("diff_median") or 0 for m in cand)
        print(f"{ds:10s} {len(cand):6d} {target:6d} {len(got):6d} {reuse:6d} {ma:7.2f} {mk:8.2f} "
              f"{st.mean(aes[m['video_path']] for m in got):8.3f}")
        rows += [dict(video=m["video_path"], prompt=m["prompt"], dataset=ds, aes=aes[m["video_path"]],
                      motion=m.get("diff_median") or 0) for m in got]
    random.Random(5).shuffle(rows)
    with open(f"{OUT}/round5_pool.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "prompt", "dataset", "aes", "motion"])
        w.writeheader()
        w.writerows(rows)
    json.dump(vars(a), open(f"{OUT}/round5_pool_args.json", "w"))
    print("total", len(rows), "->", f"{OUT}/round5_pool.csv")


def make_round5_csv(short_frac=0.3, seed=5):
    """round5_pool_short.csv (with 'short') -> round5_train.csv where short_frac rows use the short caption."""
    rows = list(csv.DictReader(open(f"{OUT}/round5_pool_short.csv")))
    rng = random.Random(seed)
    n_short = 0
    with open(f"{OUT}/round5_train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        for r in rows:
            use_short = r.get("short") and rng.random() < short_frac
            n_short += bool(use_short)
            w.writerow([r["video"], r["short"] if use_short else r["prompt"]])
    print(f"round5_train.csv: {len(rows)} rows, {n_short} short captions")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        make_round5_csv(float(sys.argv[2]) if len(sys.argv) > 2 else 0.3)
    else:
        main()
