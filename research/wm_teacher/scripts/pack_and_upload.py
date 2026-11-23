#!/usr/bin/env python3
"""
Pack processed wan22 clips into tar shards and upload to HuggingFace.

Packs each dataset's *_wan22/ directory into 8GB tar shards,
uploads to NZC415/wan22-processed-clips as LFS files.
This avoids HF file count limits (231K individual files → ~30 tar shards).

Usage:
    python3 scripts/pack_and_upload.py --datasets h2o easycom openvidhd matrix
    python3 scripts/pack_and_upload.py --datasets charades noxi seamless
    python3 scripts/pack_and_upload.py --all
"""

import argparse
import os
import tarfile
import tempfile
import math
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN", "<HF_TOKEN>")
REPO_ID = "NZC415/wan22-processed-clips"
PROJECT_ROOT = Path("/srv/wm_teacher")
WT_DIR = PROJECT_ROOT / "dataset" / "wan22_training"
SHARD_SIZE = 8 * 1024 * 1024 * 1024  # 8GB per shard

DATASETS = {
    "h2o": WT_DIR / "h2o_wan22",
    "easycom": WT_DIR / "easycom_wan22",
    "openvidhd": WT_DIR / "openvidhd_wan22",
    "matrix": WT_DIR / "matrix_wan22",
    "seamless": WT_DIR / "seamless_wan22",
    "noxi": WT_DIR / "noxi_wan22",
    "charades": WT_DIR / "charades_wan22",
    "hoigen1m": WT_DIR / "hoigen1m_wan22",
    "egoexo4d": WT_DIR / "egoexo4d_wan22",
    "dexycb": WT_DIR / "dexycb_processed",
    "mira": WT_DIR / "mira_processed",
    "vfhq": WT_DIR / "vfhq_wan22",
    "direct": WT_DIR / "direct_wan22",
}


def pack_and_upload_dataset(api, name, dir_path):
    """Pack clips into tar shards and upload each shard."""
    if not dir_path.exists():
        print(f"  SKIP {name}: not found")
        return

    # Get all mp4 files
    mp4s = sorted([f for f in os.listdir(dir_path) if f.endswith('.mp4')])
    if not mp4s:
        print(f"  SKIP {name}: no mp4 files")
        return

    # Calculate shards
    total_size = sum(os.path.getsize(dir_path / f) for f in mp4s)
    n_shards = max(1, math.ceil(total_size / SHARD_SIZE))
    print(f"  {name}: {len(mp4s)} clips, {total_size/1e9:.1f}GB, {n_shards} shards")

    # Pack and upload shard by shard
    shard_idx = 0
    current_shard = []
    current_size = 0

    for i, mp4_name in enumerate(mp4s):
        mp4_path = dir_path / mp4_name
        file_size = os.path.getsize(mp4_path)

        current_shard.append((mp4_name, mp4_path))
        current_size += file_size

        # Shard full or last file
        if current_size >= SHARD_SIZE or i == len(mp4s) - 1:
            shard_idx += 1
            shard_name = f"{name}_shard_{shard_idx:04d}-of-{n_shards:04d}.tar"

            # Check if already uploaded
            try:
                files_in_repo = api.list_repo_files(REPO_ID, repo_type="dataset", token=TOKEN)
                if shard_name in files_in_repo:
                    print(f"    skip {shard_name} (already uploaded)")
                    current_shard = []
                    current_size = 0
                    continue
            except:
                pass

            # Pack shard to temp file
            tmp_path = f"/tmp/{shard_name}"
            print(f"    packing {shard_name} ({current_size/1e9:.1f}GB, {len(current_shard)} files)...")
            with tarfile.open(tmp_path, "w") as tf:
                for fname, fpath in current_shard:
                    tf.add(str(fpath), arcname=fname)

            # Upload shard
            print(f"    uploading {shard_name}...")
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=f"shards/{shard_name}",
                repo_id=REPO_ID,
                repo_type="dataset",
                token=TOKEN,
            )
            print(f"    ✅ {shard_name}")
            os.remove(tmp_path)

            current_shard = []
            current_size = 0

    # Upload manifest
    manifest = dir_path / "manifest_wan22.jsonl"
    if manifest.exists():
        api.upload_file(
            path_or_fileobj=str(manifest),
            path_in_repo=f"manifests/{name}_manifest_wan22.jsonl",
            repo_id=REPO_ID, repo_type="dataset", token=TOKEN,
        )
        print(f"    ✅ manifest")

    print(f"  {name}: DONE ({shard_idx} shards)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, choices=list(DATASETS.keys()))
    args = ap.parse_args()

    api = HfApi(token=TOKEN)
    create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, token=TOKEN, exist_ok=True)
    print(f"Repo: {REPO_ID}")

    for name in args.datasets:
        print(f"\n=== {name} ===")
        try:
            pack_and_upload_dataset(api, name, DATASETS[name])
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
