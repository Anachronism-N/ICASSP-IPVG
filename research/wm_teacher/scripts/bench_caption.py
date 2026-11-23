#!/usr/bin/env python3
"""Throughput / quality benchmark of caption_5s.caption() on already-sliced files (no manifest needed)."""
import glob
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/srv/wm_teacher/scripts")
import caption_5s as C

ports = [int(p) for p in sys.argv[1].split(",")]
n = int(sys.argv[2])
conc = int(sys.argv[3])
files = []
for d in glob.glob("/srv/wm_teacher/dataset/wan22_5s/*_5s"):
    files += glob.glob(f"{d}/*.mp4")[:max(n, 50)]
random.Random(0).shuffle(files)
files = files[:n]
pool = C.Ports(ports)
t = time.time()
with ThreadPoolExecutor(conc) as ex:
    res = list(ex.map(lambda p: C.caption({"video_path": p}, pool, 450), files))
el = time.time() - t
ok = [r for r in res if r[0]]
print(f"n={n} conc={conc} ports={ports} ok={len(ok)} fail={n-len(ok)} {el:.1f}s -> {len(ok)/el:.2f} cap/s, "
      f"avg_words={sum(r[2] for r in ok)/max(len(ok),1):.0f}, truncated={sum(r[1]=='length' for r in ok)}")
for f, r in list(zip(files, res))[:2]:
    print("---", f.split("/")[-2], f.split("/")[-1]); print(r[0] or r[3])
