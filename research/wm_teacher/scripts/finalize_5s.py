#!/usr/bin/env python3
"""
Finalize the 5s dataset after reslice_5s.py + caption_5s.py are done.

- joins manifests (landscape/portrait) with captions by video_path
- re-applies caption_5s.sanitize() to every caption (idempotent; earlier captions predate the
  clause-level cleaner), drops captions that fall under 60 words after cleaning
- writes DiffSynth-ready CSVs (video,prompt) per orientation + a full JSONL with metadata
- writes the list of windows still lacking a caption (to be re-queued)
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caption_5s import sanitize

ROOT = "/srv/wm_teacher"
D = f"{ROOT}/dataset/wan22_5s"

caps = {}
with open(f"{D}/captions_5s.jsonl") as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        caps[r["video_path"]] = r  # last wins

stats = defaultdict(Counter)
missing = []
for orient in ("landscape", "portrait"):
    man = f"{D}/manifest_5s_{orient}.jsonl"
    if not os.path.exists(man):
        continue
    out_jsonl = open(f"{D}/final_5s_{orient}.jsonl", "w")
    out_csv = open(f"{D}/final_5s_{orient}.csv", "w", newline="")
    w = csv.writer(out_csv)
    w.writerow(["video", "prompt"])
    seen = set()
    with open(man) as f:
        for line in f:
            try:
                m = json.loads(line)
            except Exception:
                continue
            vp = m["video_path"]
            if vp in seen:
                continue
            seen.add(vp)
            ds = m["source_dataset"]
            stats[ds][f"{orient}_windows"] += 1
            c = caps.get(vp)
            if not c:
                missing.append(vp)
                stats[ds]["no_caption"] += 1
                continue
            p = sanitize(c["prompt"])
            if len(p.split()) < 60:
                missing.append(vp)
                stats[ds]["short_after_clean"] += 1
                continue
            m.update(prompt=p, caption_model=c.get("caption_model"), caption_words=len(p.split()))
            m.pop("src_segments", None)
            out_jsonl.write(json.dumps(m, ensure_ascii=False) + "\n")
            w.writerow([vp, p])
            stats[ds][f"{orient}_final"] += 1
            stats[ds]["scene_cut"] += bool(m.get("scene_cut"))
    out_jsonl.close()
    out_csv.close()

with open(f"{D}/missing_captions.txt", "w") as f:
    f.write("\n".join(missing))

cols = ["landscape_windows", "landscape_final", "portrait_windows", "portrait_final",
        "no_caption", "short_after_clean", "scene_cut"]
print(f"{'dataset':10s} " + " ".join(f"{c:>17s}" for c in cols))
tot = Counter()
for ds in sorted(stats, key=lambda d: -sum(stats[d].values())):
    tot.update(stats[ds])
    print(f"{ds:10s} " + " ".join(f"{stats[ds][c]:17d}" for c in cols))
print(f"{'TOTAL':10s} " + " ".join(f"{tot[c]:17d}" for c in cols))
print(f"hours (final): {(tot['landscape_final'] + tot['portrait_final']) * 121 / 24 / 3600:.1f}")
print(f"missing/short captions: {len(missing)} -> {D}/missing_captions.txt")
