"""Download supplementary sources for FT2 (out-of-domain coverage: driving, robotics, egocentric hands).
  uk_dashcam : aap9002/UK-Road-DashCam (MIT), 274 x ~3 min 1080p30 front dashcam            ~170 GB
  robomind   : Traly/RoboMIND-lerobot (CC-BY-NC-4.0), 720p30, only static/front/top cameras  ~115 GB
  egodex     : griffinlabs/EgoDex-LeRobot-v3.0 (CC-BY-NC-ND-4.0, Apple EgoDex), 1080p30 egocentric
               dexterous manipulation; first 3 video files of each of 134 train tasks         ~80 GB
-> dataset/new_sources/<name>/ (resume-safe)."""
import os, re, sys
from collections import defaultdict
from huggingface_hub import HfApi, snapshot_download
TOK = os.environ.get("HF_TOKEN")
OUT = "/srv/wm_teacher/dataset/new_sources"
api = HfApi(token=TOK)

def egodex_patterns():
    files = [s.rfilename for s in api.dataset_info("griffinlabs/EgoDex-LeRobot-v3.0").siblings]
    per = defaultdict(list)
    for f in files:
        m = re.match(r"train/([^/]+)/videos/", f)
        if m and f.endswith(".mp4"):
            per[m.group(1)].append(f)
    keep = [f for t in per for f in sorted(per[t])[:3]]
    return keep + [f for f in files if f.startswith("train/") and "/meta/" in f]

JOBS = {
    "uk_dashcam": ("aap9002/UK-Road-DashCam", None),
    "robomind": ("Traly/RoboMIND-lerobot", ["*/meta/*", "agilex_3rgb/videos/observation.images.camera_front/*",
                 "franka_1rgb/videos/observation.images.camera_top/*", "franka_fr3_dual/videos/observation.images.camera_front/*",
                 "franka_fr3_dual/videos/observation.images.camera_top/*", "ur_1rgb/videos/observation.images.camera_top/*"]),
    "egodex": ("griffinlabs/EgoDex-LeRobot-v3.0", "EGODEX"),
}
for name in (sys.argv[1:] or JOBS):
    repo, pat = JOBS[name]
    if pat == "EGODEX":
        pat = egodex_patterns()
        print(f"[{name}] {len(pat)} files", flush=True)
    for attempt in range(5):
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=f"{OUT}/{name}", allow_patterns=pat,
                              token=TOK, max_workers=16)
            print(f"[{name}] DONE", flush=True)
            break
        except Exception as e:
            print(f"[{name}] attempt {attempt + 1} failed: {str(e)[:200]}", flush=True)
