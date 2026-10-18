"""FT2 training mixture (full fine-tune from the Wan2.2-TI2V-5B base), targeting the out-of-domain
regressions diagnosed on WorldModelBench / Physics-IQ / VBench (more camera motion + shake, forgetting):
  * in-domain windows from the round-4 aesthetic pool (val sequences already excluded), same per-dataset
    shares as round 5 but NOT motion-selected; per dataset drop the top-20% camera-shift windows
    (cam_scores.jsonl, phase-correlation px/4 frames) and cap first-person 'direct' + 'egoexo4d'
  * supplementary sources (EXTRA_SHARE: OpenVidHD general 1080p, UK dashcam, RoboMIND, EgoDex) as
    --general-frac of the total, all with dense Qwen3.5 captions (dataset/wan22_5s_extra/captions_5s.jsonl)
  * captions: full dense training-style captions only (--short-frac 0); short benchmark prompts are
    expanded at eval time instead (rewrite_prompts_vbench.py)
Output: training_metadata/ft2/ft2_train.csv (video,prompt), shuffled.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict

ROOT = "/srv/wm_teacher"
# supplementary-source shares of the non-in-domain part (general OpenVidHD 1080p, UK dashcam driving,
# RoboMIND static-camera robot manipulation, EgoDex egocentric dexterous manipulation)
EXTRA_SHARE = {"openvidgen": 0.5, "ukdash": 0.17, "robomind": 0.17, "egodex": 0.16}
SHARE = {"hoigen1m": 8139, "mira": 4883, "celebvhq": 4883, "charades": 4069, "direct": 3255, "vfhq": 2442,
         "matrix": 2442, "easycom": 1139, "noxi": 894, "seamless": 398, "egoexo4d": 294, "openvidhd": 122, "h2o": 80}


def cam_sentence(shift):
    if shift < 0.5:
        return "The camera is static."
    if shift < 2.0:
        return "The camera moves slightly."
    return "The camera moves noticeably."


def first_sentence(t, max_words=20):
    s = re.split(r"(?<=[.!?])\s", t.strip())[0]
    return " ".join(s.split()[:max_words]).rstrip(",;") + ("" if s.endswith(".") else ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=33000)
    ap.add_argument("--general-frac", type=float, default=0.40, help="share of supplementary (non-in-domain) rows")
    ap.add_argument("--short-frac", type=float, default=0.0, help="keep 0: train on full dense captions only (user policy)")
    ap.add_argument("--cam-drop-q", type=float, default=0.20, help="per dataset, drop windows above this camera-shift quantile")
    ap.add_argument("--cam-abs-max", type=float, default=8.0, help="absolute cap on camera shift (px / 4 frames @320w)")
    ap.add_argument("--ego-cap", type=float, default=0.5, help="multiply direct/egoexo4d shares by this")
    ap.add_argument("--share-mult", default="",
                    help="per-dataset share multipliers, e.g. 'matrix=0.3,mira=0.5' (FT3: high-motion sources, "
                         "eval/diag/cuts_ft2_train_full.jsonl: frame diff 8.5 / 5.3 vs <=2.4 elsewhere)")
    ap.add_argument("--out", default=f"{ROOT}/training_metadata/ft2/ft2_train.csv")
    a = ap.parse_args()
    rng = random.Random(0)
    mult = {k: float(v) for k, v in (x.split("=") for x in a.share_mult.split(",") if x)}

    cam = {}
    for l in open(f"{ROOT}/training_metadata/ft2/cam_scores.jsonl"):
        r = json.loads(l)
        if "error" not in r:
            cam[r["video"]] = r["cam_shift"]
    meta = {}
    for o in ("landscape", "portrait"):
        for l in open(f"{ROOT}/dataset/wan22_5s/final_5s_{o}.jsonl"):
            m = json.loads(l)
            if m["video_path"] in cam:
                meta[m["video_path"]] = m
    short = {json.loads(l)["video"]: json.loads(l)["short"] for l in open(f"{ROOT}/training_metadata/round5/short_cache.jsonl")}
    bad = set(open(f"{ROOT}/training_metadata/round5/round5_pool.csv.bad.txt").read().split())

    by = defaultdict(list)
    for v, m in meta.items():
        if v not in bad and not m.get("scene_cut"):
            by[m["source_dataset"]].append(m)
    share = dict(SHARE)
    for d in ("direct", "egoexo4d"):
        share[d] = int(share[d] * a.ego_cap)
    for d, f in mult.items():
        share[d] = int(share[d] * f)
    n_in = round(a.n_total * (1 - a.general_frac))
    scale = n_in / sum(share.values())
    rows, stat = [], []
    for d, s in share.items():
        cand = by[d]
        if not cand:
            continue
        sh = sorted(cam[m["video_path"]] for m in cand)
        thr = sh[min(len(sh) - 1, int((1 - a.cam_drop_q) * len(sh)))]
        thr = min(thr, a.cam_abs_max)  # egoexo4d etc. have violent head-mounted shake (median shift >100 px)
        cand = [m for m in cand if cam[m["video_path"]] <= thr]
        rng.shuffle(cand)
        per_seq, got = Counter(), []
        for m in cand:  # spread over source sequences, <= 4 windows each
            if per_seq[m["source_sequence_id"]] < 4:
                per_seq[m["source_sequence_id"]] += 1
                got.append(m)
            if len(got) >= round(s * scale):
                break
        stat.append((d, len(by[d]), thr, round(s * scale), len(got),
                     sum(cam[m["video_path"]] >= 0.5 for m in got) / max(1, len(got))))
        for m in got:
            v = m["video_path"]
            long_p = m["prompt"].strip()
            short_p = short.get(v) or first_sentence(m["prompt"])
            rows.append({"video": v, "long": long_p, "short": short_p})
    # supplementary sources, all densely recaptioned by Qwen3.5 in the training style (caption_5s_extra.py)
    extra = defaultdict(list)
    for l in open(f"{ROOT}/dataset/wan22_5s_extra/captions_5s.jsonl"):
        r = json.loads(l)
        extra[r["source_dataset"]].append(r)
    n_extra = a.n_total - len(rows)
    want = {k: round(n_extra * w) for k, w in EXTRA_SHARE.items()}
    n_gen = 0
    for k in EXTRA_SHARE:
        lst = extra.get(k, [])
        rng.shuffle(lst)
        per_seq, got = Counter(), []
        for r in lst:
            if per_seq[r["source_sequence_id"]] < 6:
                per_seq[r["source_sequence_id"]] += 1
                got.append(r)
            if len(got) >= want[k]:
                break
        stat.append((k, len(lst), 0.0, want[k], len(got), 0.0))
        n_gen += len(got)
        for r in got:
            rows.append({"video": r["video_path"], "long": r["prompt"].strip(), "short": first_sentence(r["prompt"])})
    rng.shuffle(rows)
    k = int(a.short_frac * len(rows))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "prompt"])
        for i, r in enumerate(rows):
            w.writerow([r["video"], r["short"] if i < k else r["long"]])
    print(f"{'dataset':10s} {'pool':>7s} {'camthr':>7s} {'target':>7s} {'taken':>7s} {'moving':>7s}")
    for d, n, thr, t, g, mv in stat:
        print(f"{d:10s} {n:7d} {thr:7.2f} {t:7d} {g:7d} {mv:7.2f}")
    print(f"supplementary: {n_gen}; total {len(rows)} ({n_gen / len(rows):.0%} supplementary), short captions {k}")
    print("example short:", rows[0]["short"][:200])


if __name__ == "__main__":
    main()
