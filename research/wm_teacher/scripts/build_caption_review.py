#!/usr/bin/env python3
"""
Build a self-contained HTML page for human review of 5s captions.
Samples N clips per dataset, embeds a downscaled preview video (base64) next to its caption.
Usage: python3 scripts/build_caption_review.py [per_dataset] [out_html]
"""
import base64
import collections
import json
import os
import random
import subprocess
import sys

ROOT = "/srv/wm_teacher"
CAPS = f"{ROOT}/dataset/wan22_5s/captions_5s.jsonl"
FFMPEG = ("/srv/PRIVATE_STORAGE"
          "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")

per_ds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
out_html = sys.argv[2] if len(sys.argv) > 2 else f"{ROOT}/dataset/wan22_5s/caption_review.html"

by_ds = collections.defaultdict(list)
with open(CAPS) as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue
        by_ds[r["source_dataset"]].append(r)

rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
picks = []
for ds in sorted(by_ds, key=lambda d: -len(by_ds[d])):
    rows = rng.sample(by_ds[ds], min(per_ds, len(by_ds[ds])))
    picks += [(ds, r) for r in rows]
rng.shuffle(picks)


def preview_b64(path):
    """Downscaled mp4 preview as base64 (~300KB)."""
    cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", path,
           "-vf", "scale=512:-2", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "30", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    return base64.b64encode(r.stdout).decode() if r.stdout else None


cards = []
for i, (ds, r) in enumerate(picks, 1):
    p = os.path.join(ROOT, r["video_path"])
    b64 = preview_b64(p) if os.path.exists(p) else None
    vid = (f'<video controls loop muted style="width:100%;max-width:520px;background:#000" '
           f'src="data:video/mp4;base64,{b64}"></video>') if b64 else "<i>video missing</i>"
    meta = (f"{r['source_dataset']} · {r['caption_words']} words · scene_cut={r.get('scene_cut')} · "
            f"{r.get('width')}x{r.get('height')} · {os.path.basename(r['video_path'])}")
    cards.append(f"""
    <div class="card"><div class="vid">{vid}</div>
      <div class="cap"><div class="meta">{meta}</div><p>{r['prompt']}</p></div></div>""")

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>5s caption review</title>
<style>
body{{font-family:sans-serif;margin:24px;background:#f5f5f5}}
h1{{font-size:20px}} .sub{{color:#666;margin-bottom:20px}}
.card{{display:flex;gap:16px;background:#fff;border-radius:8px;padding:14px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.12)}}
.vid{{flex:0 0 44%}} .cap{{flex:1}} .meta{{color:#888;font-size:12px;margin-bottom:8px;word-break:break-all}}
p{{font-size:14px;line-height:1.55;margin:0}}
</style></head><body>
<h1>5s caption review</h1>
<div class="sub">{len(picks)} samples, {per_ds} per dataset, from dataset/wan22_5s/captions_5s.jsonl</div>
{''.join(cards)}
</body></html>"""

with open(out_html, "w") as f:
    f.write(html)
print(f"{len(picks)} samples -> {out_html} ({os.path.getsize(out_html)/1e6:.1f} MB)")
