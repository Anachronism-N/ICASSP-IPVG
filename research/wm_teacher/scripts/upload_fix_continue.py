#!/usr/bin/env python3
"""
Fix missing shards and continue HF upload.

Tasks:
1. hoigen1m: fix 2 missing shards (#4, #57) - resume skips the other 59
2. egoexo4d: full upload (43GB, ~6 shards)
3. celebvhq: full upload (~2.9TB, ~343 shards) - videos ready, captions separate

Usage:
    HF_TOKEN=... nohup python3 scripts/upload_fix_continue.py > /tmp/upload_hf.log 2>&1 &
"""
import os, sys, tarfile, math, time
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: HF_TOKEN env var not set")
    sys.exit(1)

REPO_ID = "nzc-415/wan22-processed-clips"
PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"
SHARD_SIZE = 8 * 1024 * 1024 * 1024  # 8GB

# Upload order: fix first, then new datasets
UPLOAD_ORDER = [
    ("hoigen1m",   WT_DIR / "hoigen1m_wan22"),    # fix 2 missing shards
    ("egoexo4d",   WT_DIR / "egoexo4d_wan22"),    # new: 43GB
    ("celebvhq",   WT_DIR / "celebvhq_wan22"),    # new: ~2.9TB
]

def get_uploaded_shards(api):
    """Get set of already uploaded shard names (with retry)."""
    for attempt in range(5):
        try:
            files = api.list_repo_files(REPO_ID, repo_type="dataset", token=TOKEN)
            return set(f for f in files if f.startswith("shards/"))
        except Exception as e:
            print(f"  list_repo_files attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(10)
    return set()

def upload_with_retry(api, tmp_path, repo_path, max_retries=3):
    """Upload a file with retries."""
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                token=TOKEN,
            )
            return True
        except Exception as e:
            print(f"    upload attempt {attempt+1} failed: {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(30)
    return False

def upload_dataset(api, name, dir_path, uploaded_set):
    """Pack and upload one dataset, skip already uploaded shards."""
    if not dir_path.exists():
        print(f"  SKIP {name}: not found", flush=True)
        return

    print(f"\n[{time.strftime('%H:%M:%S')}] Scanning {name} files...", flush=True)
    # Flat directory: list and sort mp4s
    mp4s = sorted(f for f in os.listdir(str(dir_path)) if f.endswith('.mp4'))
    mp4s = [os.path.join(str(dir_path), f) for f in mp4s]

    if not mp4s:
        print(f"  SKIP {name}: no mp4", flush=True)
        return

    # Get sizes (this takes a while for large dirs)
    print(f"[{time.strftime('%H:%M:%S')}] Stat-ing {len(mp4s)} files...", flush=True)
    sizes = []
    total_size = 0
    for p in mp4s:
        try:
            s = os.path.getsize(p)
            sizes.append(s)
            total_size += s
        except OSError:
            sizes.append(0)

    n_shards = max(1, math.ceil(total_size / SHARD_SIZE))
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}: {len(mp4s)} clips, {total_size/1e9:.1f}GB, {n_shards} shards", flush=True)
    print(f"{'='*60}", flush=True)

    shard_idx = 0
    current_shard = []
    current_size = 0
    uploaded_count = 0

    for i, mp4_path in enumerate(mp4s):
        file_size = sizes[i]
        current_shard.append((os.path.basename(mp4_path), mp4_path))
        current_size += file_size

        if current_size >= SHARD_SIZE or i == len(mp4s) - 1:
            shard_idx += 1
            shard_name = f"{name}_shard_{shard_idx:04d}-of-{n_shards:04d}.tar"
            repo_path = f"shards/{shard_name}"

            if repo_path in uploaded_set:
                uploaded_count += 1
                if shard_idx % 20 == 0 or shard_idx == n_shards:
                    print(f"  [{shard_idx}/{n_shards}] skip (uploaded)", flush=True)
                current_shard = []
                current_size = 0
                continue

            # Pack
            tmp_path = f"/tmp/{shard_name}"
            print(f"  [{shard_idx}/{n_shards}] packing {shard_name} ({current_size/1e9:.1f}GB, {len(current_shard)} files)...", flush=True)
            try:
                with tarfile.open(tmp_path, "w") as tf:
                    for fname, fpath in current_shard:
                        tf.add(fpath, arcname=fname)
            except Exception as e:
                print(f"  [{shard_idx}/{n_shards}] pack FAIL: {e}", flush=True)
                current_shard = []
                current_size = 0
                continue

            # Upload with retry
            print(f"  [{shard_idx}/{n_shards}] uploading {shard_name}...", flush=True)
            if upload_with_retry(api, tmp_path, repo_path):
                print(f"  [{shard_idx}/{n_shards}] ✅ {shard_name}", flush=True)
                uploaded_set.add(repo_path)
                uploaded_count += 1
            else:
                print(f"  [{shard_idx}/{n_shards}] ❌ FAILED after retries", flush=True)

            try:
                os.remove(tmp_path)
            except OSError:
                pass
            current_shard = []
            current_size = 0

    # Upload manifest
    manifest = dir_path / "manifest_wan22.jsonl"
    if manifest.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(manifest),
                path_in_repo=f"manifests/{name}_manifest_wan22.jsonl",
                repo_id=REPO_ID, repo_type="dataset", token=TOKEN,
            )
            print(f"  ✅ manifest", flush=True)
        except Exception as e:
            print(f"  manifest upload failed: {e}", flush=True)

    print(f"  {name}: DONE ({uploaded_count}/{n_shards} shards in repo)", flush=True)


def main():
    api = HfApi(token=TOKEN)
    create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, token=TOKEN, exist_ok=True)
    print(f"Repo: {REPO_ID}", flush=True)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    uploaded_set = get_uploaded_shards(api)
    print(f"Already uploaded: {len(uploaded_set)} shards", flush=True)

    for name, dir_path in UPLOAD_ORDER:
        try:
            upload_dataset(api, name, dir_path, uploaded_set)
        except Exception as e:
            print(f"  ❌ {name}: {e}", flush=True)
            uploaded_set = get_uploaded_shards(api)

    print(f"\n=== ALL UPLOADS DONE at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)


if __name__ == "__main__":
    main()
