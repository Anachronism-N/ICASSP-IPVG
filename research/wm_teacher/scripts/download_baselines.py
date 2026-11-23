#!/usr/bin/env python3
"""Download related-work I2V baselines (diffusers format) to /dockerdata/models_bl/<name>."""
import sys
import time

from huggingface_hub import snapshot_download

REPOS = {
    "CogVideoX-5b-I2V": "zai-org/CogVideoX-5b-I2V",
    "HunyuanVideo-I2V": "hunyuanvideo-community/HunyuanVideo-I2V",
    "Wan2.1-I2V-14B-720P": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
}
names = sys.argv[1:] or list(REPOS)
for n in names:
    for attempt in range(5):
        try:
            t = time.time()
            p = snapshot_download(REPOS[n], local_dir=f"/dockerdata/models_bl/{n}", max_workers=16)
            print(f"[done] {n} -> {p} ({(time.time() - t) / 60:.0f} min)", flush=True)
            break
        except Exception as e:
            print(f"[retry {attempt + 1}] {n}: {str(e)[:200]}", flush=True)
            time.sleep(60)
print("[all done]", flush=True)
