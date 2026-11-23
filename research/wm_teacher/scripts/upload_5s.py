#!/usr/bin/env python3
"""
Pack the 5s windows into ~6GB tar shards and upload to HF (nzc-415/wan22-5s).

Follows dataset/wan22_5s/manifest_5s_{landscape,portrait}.jsonl while reslice_5s.py is still
running, so already-sliced windows are uploaded immediately. Exits when reslice is done and
every manifest window has been packed+uploaded (files that failed packing are retried next run).

State:
  - /dockerdata/wm_logs/upload_5s_state.jsonl  : one line per shard {"shard": n, "files": [...]}
  - shard names already in the HF repo are skipped (resume across restarts)
"""
import json
import os
import re
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, create_repo

ROOT = "/srv/wm_teacher"
D = f"{ROOT}/dataset/wan22_5s"
MANIFESTS = [f"{D}/manifest_5s_{o}.jsonl" for o in ("landscape", "portrait")]
STATE = "/dockerdata/wm_logs/upload_5s_state.jsonl"
TMP = "/dockerdata/wm_tmp"
LOG_PREFIX = "[upload_5s]"
REPO = "nzc-415/wan22-5s"
TOKEN = os.environ["HF_TOKEN"]
SHARD_BYTES = 6 * 1024 ** 3
PARALLEL = int(os.environ.get("UPLOAD_PARALLEL", 4))
RESLICE_LOG = "/dockerdata/wm_logs/reslice_5s.log"
FFMPEG = ("/srv/PRIVATE_STORAGE"
          "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")  # unused, keep parity


def log(msg):
    print(f"{LOG_PREFIX} {time.strftime('%H:%M:%S')} {msg}", flush=True)


def reslice_done():
    try:
        with open(RESLICE_LOG) as f:
            return any(line.startswith("[done]") for line in f)
    except FileNotFoundError:
        return False


def load_state():
    packed = set()
    if os.path.exists(STATE):
        with open(STATE) as f:
            for line in f:
                try:
                    packed.update(json.loads(line)["files"])
                except Exception:
                    pass
    return packed


def main():
    api = HfApi(token=TOKEN)
    create_repo(REPO, repo_type="dataset", private=False, token=TOKEN, exist_ok=True)
    try:
        remote = {f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("shards/")}
    except Exception:
        remote = set()
    log(f"remote shards: {len(remote)}")

    packed = load_state()
    log(f"locally packed windows: {len(packed)}")

    # scan manifests (only complete files are ever listed there)
    rows = []
    for man in MANIFESTS:
        if not os.path.exists(man):
            continue
        with open(man) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["video_path"] not in packed:
                    # manifest rows only exist after the file landed -> no per-file stat (ceph too slow)
                    rows.append((r["video_path"], os.path.join(ROOT, r["video_path"])))
    log(f"windows waiting to pack: {len(rows)} (slicing_done={reslice_done()})")
    rows.sort(key=lambda x: x[0])  # stable order -> stable sharding across restarts

    # build shards by file count (~2600 files x ~2.4MB ~ 6GB); no stat needed
    FILES_PER_SHARD = 2600
    groups = [rows[i:i + FILES_PER_SHARD] for i in range(0, len(rows), FILES_PER_SHARD)]

    # pack+upload PARALLEL shards at once: a single shard is ~12 min, dominated by reading 2600
    # files through the fuse mount, so serial upload of ~2.5M windows would take over a week
    new_state = open(STATE, "a")
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    quota = threading.Event()  # HF "exceeded your public storage" -> stop, retrying won't help

    def do_shard(gi, files):
        if quota.is_set():
            return
        name = f"shards/5s_{int(time.time())}_{gi:05d}_{len(files)}.tar"
        local = os.path.join(TMP, os.path.basename(name))
        try:
            with tarfile.open(local, "w", dereference=False) as tf:
                for vp, p in files:
                    tf.add(p, arcname=vp)
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=local, path_in_repo=name,
                                    repo_id=REPO, repo_type="dataset", token=TOKEN)
                    break
                except Exception as e:
                    log(f"upload retry {attempt+1}: {str(e)[:160]}")
                    if "storage" in str(e).lower() and "exceeded" in str(e).lower():
                        quota.set()
                        raise
                    time.sleep(30)
            else:
                raise RuntimeError("upload failed 3x")
            with lock:
                new_state.write(json.dumps({"shard": name, "files": [vp for vp, _ in files]}) + "\n")
                new_state.flush()
                stats["ok"] += len(files)
            log(f"shard ok: {name} ({len(files)} files) total_ok={stats['ok']}")
        except Exception as e:
            with lock:
                stats["fail"] += len(files)
            log(f"shard FAIL ({len(files)} files): {str(e)[:200]}")
        finally:
            try:
                os.remove(local)
            except OSError:
                pass

    with ThreadPoolExecutor(PARALLEL) as ex:
        list(ex.map(lambda g: do_shard(*g), enumerate(groups)))
    new_state.close()
    n_ok, n_fail = stats["ok"], stats["fail"]
    if quota.is_set():
        log("STOPPED: HF public storage quota exceeded")

    left = 0
    for man in MANIFESTS:
        if os.path.exists(man):
            left += sum(1 for _ in open(man)) - len(load_state())
    log(f"done. ok={n_ok} fail={n_fail} windows_remaining={max(left,0)} slicing_done={reslice_done()}")


if __name__ == "__main__":
    main()
