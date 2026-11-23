#!/usr/bin/env python3
"""
Upload all caption files and related metadata to HF.

Uploads:
1. training_metadata/recaptioned/*.jsonl → captions/ (VLM captions, 14 files)
2. dataset/manifests/mira_captions_final.jsonl → captions/ (MIRA full manifest)

Usage:
    HF_TOKEN=... nohup python3 scripts/upload_captions.py > /tmp/upload_captions.log 2>&1 &
"""
import os, sys, time
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: HF_TOKEN env var not set")
    sys.exit(1)

REPO_ID = "NZC415/wan22-processed-clips"
PROJECT_ROOT = Path("/srv/wm_teacher")

def upload_with_retry(api, local_path, repo_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=TOKEN,
            )
            return True
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(15)
    return False

def main():
    api = HfApi(token=TOKEN)
    print(f"[{time.strftime('%H:%M:%S')}] Uploading captions to {REPO_ID}", flush=True)

    # 1. All recaptioned caption files
    recap_dir = PROJECT_ROOT / "training_metadata" / "recaptioned"
    recap_files = sorted(recap_dir.glob("*_recaptioned.jsonl"))
    print(f"\n[1/2] VLM caption files ({len(recap_files)}):", flush=True)
    for f in recap_files:
        name = f.name  # e.g. celebvhq_recaptioned.jsonl
        repo_path = f"captions/{name}"
        size_mb = f.stat().st_size / 1e6
        print(f"  uploading {name} ({size_mb:.0f}MB)...", flush=True)
        if upload_with_retry(api, f, repo_path):
            print(f"  ✅ {name}", flush=True)
        else:
            print(f"  ❌ {name} FAILED", flush=True)

    # 2. MIRA full captions manifest (with metadata)
    mira_manifest = PROJECT_ROOT / "dataset" / "manifests" / "mira_captions_final.jsonl"
    if mira_manifest.exists():
        print(f"\n[2/2] MIRA full manifest:", flush=True)
        size_mb = mira_manifest.stat().st_size / 1e6
        print(f"  uploading mira_captions_final.jsonl ({size_mb:.0f}MB)...", flush=True)
        if upload_with_retry(api, mira_manifest, "captions/mira_captions_final.jsonl"):
            print(f"  ✅ mira_captions_final.jsonl", flush=True)
        else:
            print(f"  ❌ mira_captions_final.jsonl FAILED", flush=True)

    print(f"\n=== CAPTIONS UPLOAD DONE at {time.strftime('%H:%M:%S')} ===", flush=True)

if __name__ == "__main__":
    main()
