"""Camera-motion score for 5s windows (CPU): mean global translation between frames i and i+4 via phase
correlation on 320x176 grayscale (px per 4 frames). Also the share of 4-frame steps with shift>1px.
Usage: python score_camera.py LIST.txt OUT.jsonl [--procs N]   (resume-safe: skips videos already in OUT)"""
import argparse, json, os, sys
import numpy as np, av, cv2
from multiprocessing import Pool
ROOT = "/srv/wm_teacher"
cv2.setNumThreads(1)

def score(v):
    try:
        c = av.open(f"{ROOT}/{v}")
        fr = [cv2.resize(x.to_ndarray(format="gray"), (320, 176)).astype(np.float32) for x in c.decode(video=0)]
        c.close()
        s = np.array([np.hypot(*cv2.phaseCorrelate(fr[i], fr[i + 4])[0]) for i in range(0, len(fr) - 4, 4)])
        return {"video": v, "cam_shift": round(float(np.median(s)), 3), "cam_mean": round(float(s.mean()), 3),
                "cam_frac_moving": round(float((s > 1).mean()), 3)}
    except Exception as e:
        return {"video": v, "error": str(e)[:100]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("list"); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    vids = [l.strip() for l in open(a.list) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["video"] for l in open(a.out)}
    todo = [v for v in vids if v not in done]
    print(f"{len(vids)} videos, {len(todo)} to score", flush=True)
    with open(a.out, "a") as f, Pool(a.procs) as p:
        for i, r in enumerate(p.imap_unordered(score, todo, chunksize=8), 1):
            f.write(json.dumps(r) + "\n")
            if i % 2000 == 0:
                f.flush(); print(f"{i}/{len(todo)}", flush=True)
    print("done", flush=True)
