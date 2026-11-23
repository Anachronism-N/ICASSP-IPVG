#!/usr/bin/env python3
"""
Upload MIRA raw tars to HF.

4,858 tars (~8.3TB) in general_action/mira_rocket/{train,test}/.
Each tar is already in tar format, uploaded directly as LFS files (no repack).
Supports resume via repo file listing.

Usage:
    HF_TOKEN=... nohup python3 scripts/upload_mira.py > /tmp/upload_mira.log 2>&1 &
"""
import os, sys, time, glob
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: HF_TOKEN env var not set")
    sys.exit(1)

REPO_ID = "nzc-415/wan22-processed-clips"
MIRA_ROOT = Path("/srv/wm_teacher/dataset/general_action/mira_rocket")

def get_uploaded(api):
    for attempt in range(5):
        try:
            files = api.list_repo_files(REPO_ID, repo_type="dataset", token=TOKEN)
            return set(f for f in files if f.startswith("shards/mira_raw/"))
        except Exception as e:
            print(f"  list attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(10)
    return set()

def main():
    api = HfApi(token=TOKEN)

    # Collect all tars: train/000-004/*.tar + test/*.tar
    tars = sorted(glob.glob(str(MIRA_ROOT / "train" / "*" / "*.tar")))
    tars += sorted(glob.glob(str(MIRA_ROOT / "test" / "*.tar")))
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(tars)} MIRA tars", flush=True)

    uploaded = get_uploaded(api)
    print(f"Already uploaded: {len(uploaded)}", flush=True)

    # Build upload list: repo path mirrors relative structure
    todo = []
    for tar_path in tars:
        p = Path(tar_path)
        rel = p.relative_to(MIRA_ROOT)  # train/000/dataset_00000.tar
        repo_path = f"shards/mira_raw/{rel}"
        if repo_path not in uploaded:
            todo.append((tar_path, repo_path))

    print(f"To upload: {len(todo)} tars", flush=True)
    if not todo:
        print("=== MIRA UPLOAD COMPLETE (nothing to do) ===", flush=True)
        return

    total_bytes = sum(os.path.getsize(t) for t, _ in todo)
    print(f"Total size: {total_bytes/1e9:.0f}GB", flush=True)

    done_bytes = 0
    start_time = time.time()
    fails = 0

    for i, (tar_path, repo_path) in enumerate(todo, 1):
        size = os.path.getsize(tar_path)
        success = False
        for attempt in range(3):
            try:
                api.upload_file(
                    path_or_fileobj=tar_path,
                    path_in_repo=repo_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=TOKEN,
                )
                success = True
                break
            except Exception as e:
                print(f"  [{i}/{len(todo)}] attempt {attempt+1} failed: {str(e)[:100]}", flush=True)
                if attempt < 2:
                    time.sleep(30)

        done_bytes += size
        elapsed = time.time() - start_time
        speed = done_bytes / elapsed / 1e6 if elapsed > 0 else 0
        eta_h = (total_bytes - done_bytes) / (speed * 1e6) / 3600 if speed > 0 else 0

        if success:
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ✅ {Path(repo_path).name} | {done_bytes/1e9:.0f}/{total_bytes/1e9:.0f}GB | {speed:.0f}MB/s | ETA {eta_h:.1f}h", flush=True)
        else:
            fails += 1
            print(f"  [{i}/{len(todo)}] ❌ FAILED {Path(repo_path).name}", flush=True)

    print(f"\n=== MIRA UPLOAD DONE at {time.strftime('%H:%M:%S')} ===", flush=True)
    print(f"Uploaded: {len(todo)-fails}/{len(todo)}, failed: {fails}", flush=True)

if __name__ == "__main__":
    main()
