#!/usr/bin/env python3
"""
E3 motion scoring with VBench's own RAFT dynamic_degree implementation (vbench/dynamic_degree.py: DynamicDegree
model loading, get_frames (cv2, interval=max(1, round(fps/8))), set_params, get_score, iters=20) and the VBench RAFT
weights. Per clip:
  dynamic        VBench boolean: >= count_num = round(4*n_sampled/16) pairs with top-5% flow mean > thres,
                 thres = 6*min(H,W)/256 of the frames fed to RAFT (so --scale 0.5 halves the threshold, same as
                 VBench would at that resolution)
  n_over         number of pairs above thres (VBench stops early; we count all)
  top5_mean      mean over pairs of VBench's per-pair score (top 5% flow magnitude mean, padded frame as in VBench)
  cam_median     median over pairs of |median(u), median(v)| -- global (camera) component
  resid_top5     mean over pairs of top-5% magnitude of (flow - per-pair median flow) -- subject motion
All flow values are in px of the RAFT input resolution; field 'scale' tells the factor (divide by it for full-res px).

Run with /dockerdata/venv_vbench/bin/python. Sharding (--shard i --nshards n over the sorted unique list), resume
(skips videos with a successful record in --out), jsonl output (one line per clip, flushed).
  score:      score_e3_motion.py --list prefilter.csv --out m.jsonl --scale 0.5 --shard 0 --nshards 32
  calibrate:  score_e3_motion.py --list calib.txt --out c.jsonl --calibrate      (decodes once, scores 1.0 and 0.5)
  report:     score_e3_motion.py --report 'c*.jsonl'                              (CPU, agreement of dynamic)
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = "/srv/wm_teacher"
VBREPO = "/dockerdata/wm_eval/VBench"
RAFT_WEIGHTS = "/dockerdata/vbench_cache/raft_model/models/raft-things.pth"
TOP_FRAC = 0.05


# ---------------- pure functions (CPU-testable) ----------------
def frame_interval(fps):
    return max(1, round(fps / 8))


def vbench_params(h, w, n_sampled):
    """Same as DynamicDegree.set_params: threshold from min side of the RAFT input, count from sampled frames."""
    return {"thres": 6.0 * (min(h, w) / 256.0), "count_num": round(4 * (n_sampled / 16.0))}


def n_sampled_frames(n_frames, fps):
    return len(range(0, n_frames, frame_interval(fps)))


def check_move(scores, thres, count_num):
    """VBench check_move (early-exit semantics are equivalent to n_over >= count_num)."""
    return count_over(scores, thres) >= count_num


def count_over(scores, thres):
    return int(sum(s > thres for s in scores))


def top_mean(rad, frac=TOP_FRAC):
    """VBench get_score: mean of the int(h*w*frac) largest magnitudes."""
    flat = np.asarray(rad, dtype=np.float64).ravel()
    k = int(flat.size * frac)
    if k <= 0:
        return 0.0
    return float(np.mean(np.partition(flat, flat.size - k)[flat.size - k:]))


def flow_stats(u, v, frac=TOP_FRAC):
    """(top5 of |flow|, |median flow| = camera component, top5 of |flow - median flow| = residual)."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    mu, mv = float(np.median(u)), float(np.median(v))
    return (top_mean(np.hypot(u, v), frac), float(np.hypot(mu, mv)), top_mean(np.hypot(u - mu, v - mv), frac))


def aggregate(pair_top5, pair_cam, pair_resid, h, w, scale):
    p = vbench_params(h, w, len(pair_top5) + 1)
    r3 = lambda xs: [round(float(x), 3) for x in xs]
    return {"scale": scale, "raft_hw": [h, w], "n_sampled": len(pair_top5) + 1, "n_pairs": len(pair_top5),
            "thres": round(p["thres"], 4), "count_num": p["count_num"],
            "dynamic": check_move(pair_top5, p["thres"], p["count_num"]),
            "n_over": count_over(pair_top5, p["thres"]),
            "top5_mean": round(float(np.mean(pair_top5)), 4) if pair_top5 else None,
            "top5_max": round(float(np.max(pair_top5)), 4) if pair_top5 else None,
            "cam_median": round(float(np.median(pair_cam)), 4) if pair_cam else None,
            "cam_mean": round(float(np.mean(pair_cam)), 4) if pair_cam else None,
            "resid_top5": round(float(np.mean(pair_resid)), 4) if pair_resid else None,
            "pair_top5": r3(pair_top5), "pair_cam": r3(pair_cam), "pair_resid": r3(pair_resid)}


def read_list(path):
    """txt (one path per line), csv with a 'video' column, or jsonl with 'video'. Paths relative to ROOT allowed."""
    if path.endswith(".csv"):
        vids = [r["video"] for r in csv.DictReader(open(path, newline=""))]
    elif path.endswith(".jsonl"):
        vids = [json.loads(l)["video"] for l in open(path) if l.strip()]
    else:
        vids = [l.strip() for l in open(path) if l.strip()]
    return sorted(set(vids))


def shard(items, i, n):
    if not 0 <= i < n:
        raise ValueError("need 0 <= shard < nshards")
    return items[i::n]


def load_records(paths):
    """Last successful record per video wins; corrupt (partial) lines ignored."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in d:
                out[d["video"]] = d
    return out


def calibration_report(records, a="s1.0", b="s0.5"):
    rs = [r for r in records if a in r and b in r]
    if not rs:
        return {"n": 0}
    da, db = np.array([r[a]["dynamic"] for r in rs]), np.array([r[b]["dynamic"] for r in rs])
    ta = np.array([r[a]["top5_mean"] / r[a]["scale"] for r in rs])
    tb = np.array([r[b]["top5_mean"] / r[b]["scale"] for r in rs])
    rank = lambda x: np.argsort(np.argsort(x))
    spear = float(np.corrcoef(rank(ta), rank(tb))[0, 1]) if len(rs) > 2 else None
    return {"n": len(rs), "agreement": round(float((da == db).mean()), 4),
            "dynamic_rate": {a: round(float(da.mean()), 4), b: round(float(db.mean()), 4)},
            "confusion": {"both_true": int((da & db).sum()), "only_" + a: int((da & ~db).sum()),
                          "only_" + b: int((~da & db).sum()), "both_false": int((~da & ~db).sum())},
            "top5_fullres_px_ratio_median": round(float(np.median(tb / np.maximum(ta, 1e-6))), 4),
            "top5_spearman": round(spear, 4) if spear is not None else None}


# ---------------- GPU part ----------------
def load_dynamic(device):
    sys.path.insert(0, VBREPO)
    from easydict import EasyDict as edict
    from vbench.dynamic_degree import DynamicDegree
    return DynamicDegree(edict({"model": RAFT_WEIGHTS, "small": False, "mixed_precision": False,
                                "alternate_corr": False}), device)


def score_frames(dyn, frames, scale):
    import torch
    import torch.nn.functional as F
    from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
    if scale != 1.0:
        frames = [F.interpolate(f, scale_factor=scale, mode="bilinear", align_corners=False, antialias=True)
                  for f in frames]
    h, w = frames[0].shape[-2:]
    top5, cam, resid = [], [], []
    with torch.no_grad():
        for image1, image2 in zip(frames[:-1], frames[1:]):
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = dyn.model(image1, image2, iters=20, test_mode=True)
            top5.append(dyn.get_score(image1, flow_up))  # VBench per-pair score, unchanged
            fl = padder.unpad(flow_up)[0].float().cpu().numpy()
            _, c, r = flow_stats(fl[0], fl[1])
            cam.append(c)
            resid.append(r)
    return aggregate(top5, cam, resid, int(h), int(w), scale)


def video_fps(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list")
    ap.add_argument("--out")
    ap.add_argument("--scale", type=float, default=0.5, choices=[0.5, 1.0])
    ap.add_argument("--calibrate", action="store_true", help="score each clip at 1.0 and 0.5 from one decode")
    ap.add_argument("--report", help="glob of calibration jsonl files -> agreement summary (CPU)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.report:
        print(json.dumps(calibration_report(list(load_records(sorted(glob.glob(a.report))).values())), indent=1))
        return
    vids = shard(read_list(a.list), a.shard, a.nshards)
    done = load_records([a.out])
    todo = [v for v in vids if v not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[motion] shard {a.shard}/{a.nshards}: {len(vids)} videos, {len(todo)} to score", flush=True)
    import torch
    dyn = load_dynamic(torch.device(a.device))
    scales = [1.0, 0.5] if a.calibrate else [a.scale]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "a") as f:
        for i, v in enumerate(todo, 1):
            path = v if v.startswith("/") else f"{ROOT}/{v}"
            rec = {"video": v}
            try:
                t = time.time()
                fps, n = video_fps(path)
                frames = dyn.get_frames(path)  # VBench decode + interval subsampling
                rec.update({"fps": fps, "n_frames": n, "interval": frame_interval(fps)})
                for s in scales:
                    rec[f"s{s}"] = score_frames(dyn, frames, s)
                if not a.calibrate:
                    rec.update(rec.pop(f"s{a.scale}"))
                rec["sec"] = round(time.time() - t, 2)
                del frames
            except Exception as e:  # recorded; retried on resume
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[motion] {i}/{len(todo)} {(time.time() - t0) / i:.2f}s/clip", flush=True)
    print(f"[motion] done {len(todo)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
