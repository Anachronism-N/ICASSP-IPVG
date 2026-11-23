"""CPU pre-flight: fully decode every video of a training CSV and drop the ones that are unreadable or
shorter than --min-frames (a bad file crashes all DDP ranks mid-run).

Usage: python check_decodable.py IN.csv [--min-frames 121] [--procs 48]
Writes IN.bad.txt (path<TAB>reason) and, if anything is bad, rewrites IN.csv without those rows (backup IN.csv.orig).
"""
import argparse
import csv
import os
import shutil
from multiprocessing import Pool

ROOT = "/srv/wm_teacher"


def check(args):
    v, min_frames = args
    import av
    for attempt in range(3):  # fuse hiccups: retry before declaring bad
        try:
            with av.open(f"{ROOT}/{v}") as c:
                n = sum(1 for _ in c.decode(video=0))
            return v, None if n >= min_frames else f"only {n} frames"
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:80]}"
            import time
            time.sleep(5 * (attempt + 1))
    return v, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--min-frames", type=int, default=121)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    bad = {}
    with Pool(a.procs) as p:
        for i, (v, err) in enumerate(p.imap_unordered(check, [(r["video"], a.min_frames) for r in rows], chunksize=8), 1):
            if err:
                bad[v] = err
            if i % 2000 == 0:
                print(f"{i}/{len(rows)} checked, bad {len(bad)}", flush=True)
    with open(a.csv + ".bad.txt", "w") as f:
        for v, e in bad.items():
            f.write(f"{v}\t{e}\n")
    if bad:
        shutil.copy(a.csv, a.csv + ".orig")
        with open(a.csv + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(r for r in rows if r["video"] not in bad)
        os.replace(a.csv + ".tmp", a.csv)
    print(f"done: {len(rows)} rows, {len(bad)} bad removed", flush=True)


if __name__ == "__main__":
    main()
