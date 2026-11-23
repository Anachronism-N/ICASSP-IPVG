#!/usr/bin/env python3
"""Upload the finalized 5s metadata (captions + manifests + round-2 split) to nzc-415/wan22-5s/metadata/."""
import os
import time

from huggingface_hub import HfApi

ROOT = "/srv/wm_teacher"
REPO = "nzc-415/wan22-5s"
FILES = {
    f"{ROOT}/dataset/wan22_5s/final_5s_landscape.jsonl": "metadata/final_5s_landscape.jsonl",
    f"{ROOT}/dataset/wan22_5s/final_5s_portrait.jsonl": "metadata/final_5s_portrait.jsonl",
    f"{ROOT}/dataset/wan22_5s/missing_captions.txt": "metadata/missing_captions.txt",
    f"{ROOT}/training_metadata/round2/round2_train.csv": "metadata/round2_train.csv",
    f"{ROOT}/training_metadata/round2/round2_val.csv": "metadata/round2_val.csv",
}

api = HfApi(token=os.environ["HF_TOKEN"])
for local, remote in FILES.items():
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset")
            print(f"ok {remote} ({os.path.getsize(local) / 1e9:.2f}GB)", flush=True)
            break
        except Exception as e:
            print(f"retry {attempt + 1} {remote}: {str(e)[:200]}", flush=True)
            time.sleep(30)
print("[done]", flush=True)
