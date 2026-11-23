#!/usr/bin/env python3
"""
Sequential upload: smallest dataset first, one at a time.
Supports resume (skips already uploaded shards).
"""
import os, sys, tarfile, math, json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN", "<HF_TOKEN>")
REPO_ID = "NZC415/wan22-processed-clips"
PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"
SHARD_SIZE = 8 * 1024 * 1024 * 1024  # 8GB

# Ordered by size: smallest first (already uploaded ones excluded)
UPLOAD_ORDER = [
    ("dexycb",     WT_DIR / "dexycb_processed"),
    ("matrix",     WT_DIR / "matrix_wan22"),
    ("egoexo4d",   WT_DIR / "egoexo4d_wan22"),
    ("direct",     WT_DIR / "direct_wan22"),
    ("seamless",   WT_DIR / "seamless_wan22"),
    ("vfhq",       WT_DIR / "vfhq_wan22"),
    ("noxi",       WT_DIR / "noxi_wan22"),
    ("charades",   WT_DIR / "charades_wan22"),
    ("hoigen1m",   WT_DIR / "hoigen1m_wan22"),
    ("mira",       WT_DIR / "mira_processed"),
]

def get_uploaded_shards(api):
    """Get set of already uploaded shard names."""
    try:
        files = api.list_repo_files(REPO_ID, repo_type="dataset", token=TOKEN)
        return set(f for f in files if f.startswith("shards/"))
    except:
        return set()

def upload_dataset(api, name, dir_path, uploaded_set):
    """Pack and upload one dataset, skip already uploaded shards."""
    if not dir_path.exists():
        print(f"  SKIP {name}: not found", flush=True)
        return

    # Get all mp4 files
    mp4s = []
    for root, dirs, files in os.walk(str(dir_path)):
        for f in sorted(files):
            if f.endswith('.mp4'):
                mp4s.append(os.path.join(root, f))

    if not mp4s:
        print(f"  SKIP {name}: no mp4", flush=True)
        return

    total_size = sum(os.path.getsize(f) for f in mp4s)
    n_shards = max(1, math.ceil(total_size / SHARD_SIZE))
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}: {len(mp4s)} clips, {total_size/1e9:.1f}GB, {n_shards} shards", flush=True)
    print(f"{'='*60}", flush=True)

    shard_idx = 0
    current_shard = []
    current_size = 0

    for i, mp4_path in enumerate(mp4s):
        file_size = os.path.getsize(mp4_path)
        current_shard.append((os.path.basename(mp4_path), mp4_path))
        current_size += file_size

        if current_size >= SHARD_SIZE or i == len(mp4s) - 1:
            shard_idx += 1
            shard_name = f"{name}_shard_{shard_idx:04d}-of-{n_shards:04d}.tar"
            repo_path = f"shards/{shard_name}"

            if repo_path in uploaded_set:
                print(f"  [{shard_idx}/{n_shards}] skip {shard_name} (already uploaded)", flush=True)
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

            # Upload
            print(f"  [{shard_idx}/{n_shards}] uploading {shard_name}...", flush=True)
            try:
                api.upload_file(
                    path_or_fileobj=tmp_path,
                    path_in_repo=repo_path,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=TOKEN,
                )
                print(f"  [{shard_idx}/{n_shards}] ✅ {shard_name}", flush=True)
                uploaded_set.add(repo_path)
            except Exception as e:
                print(f"  [{shard_idx}/{n_shards}] upload FAIL: {e}", flush=True)

            os.remove(tmp_path)
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
        except: pass

    print(f"  {name}: DONE ({shard_idx}/{n_shards} shards)", flush=True)


def main():
    api = HfApi(token=TOKEN)
    create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, token=TOKEN, exist_ok=True)
    print(f"Repo: {REPO_ID}", flush=True)

    # Get already uploaded shards
    uploaded_set = get_uploaded_shards(api)
    print(f"Already uploaded: {len(uploaded_set)} shards", flush=True)

    for name, dir_path in UPLOAD_ORDER:
        try:
            upload_dataset(api, name, dir_path, uploaded_set)
        except Exception as e:
            print(f"  ❌ {name}: {e}", flush=True)
            # Re-fetch uploaded set in case of error
            uploaded_set = get_uploaded_shards(api)

    print("\n=== ALL UPLOADS DONE ===", flush=True)


if __name__ == "__main__":
    main()
