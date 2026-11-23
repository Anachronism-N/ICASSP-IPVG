#!/usr/bin/env python3
"""
Download missing OpenVidHD parts from HuggingFace dataset nkp37/OpenVid-1M.
Sets proxy and downloads with resume support.
"""
import os
import sys
import time

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['http_proxy'] = 'http://star-proxy.oa.com:3128'
os.environ['https_proxy'] = 'http://star-proxy.oa.com:3128'

from huggingface_hub import hf_hub_download, HfApi

REPO_ID = 'nkp37/OpenVid-1M'
LOCAL_DIR = '/srv/wm_teacher/dataset/general_video/openvidhd/'

# Build list of files to download
files_to_download = [
    # Part 33: only _ab exists locally, need _aa
    'OpenVidHD/OpenVidHD_part_33_part_aa',
    # Part 65: completely missing
    'OpenVidHD/OpenVidHD_part_65_part_aa',
    'OpenVidHD/OpenVidHD_part_65_part_ab',
]

# Parts 67-81 (inclusive): completely missing
for part in range(67, 82):
    files_to_download.append(f'OpenVidHD/OpenVidHD_part_{part}_part_aa')
    files_to_download.append(f'OpenVidHD/OpenVidHD_part_{part}_part_ab')

print(f'=== OpenVidHD Repair Download ===')
print(f'Source: {REPO_ID}')
print(f'Destination: {LOCAL_DIR}')
print(f'Total files to download: {len(files_to_download)}')
print()

# Verify which files still need downloading
need_download = []
skip_count = 0
for fname in files_to_download:
    local_path = os.path.join(LOCAL_DIR, fname)
    if os.path.exists(local_path):
        fsize = os.path.getsize(local_path)
        print(f'  SKIP (exists, {fsize/1e9:.1f}GB): {fname}')
        skip_count += 1
    else:
        print(f'  NEED: {fname}')
        need_download.append(fname)

print(f'\nAlready have: {skip_count}, Need to download: {len(need_download)}')
print()

if not need_download:
    print('All files already present. Nothing to download.')
    sys.exit(0)

# Download
success_count = 0
fail_count = 0
start_time = time.time()

for i, fname in enumerate(need_download, 1):
    local_path = os.path.join(LOCAL_DIR, fname)
    print(f'[{i}/{len(need_download)}] Downloading: {fname}')
    try:
        result = hf_hub_download(
            repo_id=REPO_ID,
            repo_type='dataset',
            filename=fname,
            local_dir=LOCAL_DIR,
            resume_download=True,
        )
        if os.path.exists(local_path):
            fsize = os.path.getsize(local_path)
            print(f'  OK: {fname} ({fsize/1e9:.1f}GB)')
        else:
            print(f'  OK: {fname} (downloaded to {result})')
        success_count += 1
    except Exception as e:
        print(f'  FAILED: {fname} - {e}')
        fail_count += 1

elapsed = time.time() - start_time
print(f'\n=== Download Complete ===')
print(f'Success: {success_count}, Failed: {fail_count}')
print(f'Time: {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)')
sys.exit(0 if fail_count == 0 else 1)
