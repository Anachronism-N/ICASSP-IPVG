"""Upload the train/val 5s windows that did not fit into nzc-415/wan22-5s (quota hit)
to an overflow dataset repo, so Round2/Round3 training stays reproducible from HF.

Input : /dockerdata/wm_logs/hf_missing_trainval.txt (relative video paths)
Output: <REPO>/shards/*.tar (tar member names = relative video paths) + index.jsonl
Resume: state in /dockerdata/wm_logs/upload_overflow_state.jsonl
Env   : HF_TOKEN (write token for REPO's owner), OVERFLOW_REPO
"""
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

ROOT = "/srv/wm_teacher"
LIST = "/dockerdata/wm_logs/hf_missing_trainval.txt"
STATE = "/dockerdata/wm_logs/upload_overflow_state.jsonl"
TMP = "/dockerdata/wm_tmp"
REPO = os.environ.get("OVERFLOW_REPO", "NZC415/wan22-5s-trainval-overflow")
TOKEN = os.environ["HF_TOKEN"]
SHARD_FILES = 2600


def log(m):
    print(f"[overflow] {time.strftime('%H:%M:%S')} {m}", flush=True)


def main():
    api = HfApi(token=TOKEN)
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    done = set()
    if os.path.exists(STATE):
        for l in open(STATE):
            done.update(json.loads(l)["files"])
    todo = [v for v in open(LIST).read().split() if v not in done and os.path.exists(f"{ROOT}/{v}")]
    groups = [todo[i:i + SHARD_FILES] for i in range(0, len(todo), SHARD_FILES)]
    log(f"repo={REPO} done={len(done)} todo={len(todo)} shards={len(groups)}")

    def one(idx_files):
        idx, files = idx_files
        name = f"shards/overflow_{int(time.time())}_{idx:03d}.tar"
        local = f"{TMP}/{os.path.basename(name)}"
        try:
            with tarfile.open(local, "w") as tf:
                for v in files:
                    tf.add(f"{ROOT}/{v}", arcname=v)
            for attempt in range(4):
                try:
                    api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=REPO, repo_type="dataset")
                    return name, files, None
                except Exception as e:
                    if "exceeded" in str(e) and "storage" in str(e):
                        return name, files, "QUOTA"
                    log(f"retry {attempt + 1} {name}: {str(e)[:120]}")
                    time.sleep(60)
            return name, files, "upload failed"
        finally:
            if os.path.exists(local):
                os.remove(local)

    with open(STATE, "a") as st, ThreadPoolExecutor(4) as ex:
        for name, files, err in ex.map(one, enumerate(groups)):
            if err:
                log(f"FAIL {name}: {err}")
                if err == "QUOTA":
                    log("STOPPED: quota")
                    break
                continue
            st.write(json.dumps({"shard": name, "files": files}) + "\n")
            st.flush()
            log(f"ok {name} ({len(files)})")
    # index so users can map each window -> (repo, shard)
    rows = [json.loads(l) for l in open(STATE)]
    with open(f"{TMP}/overflow_index.jsonl", "w") as f:
        for r in rows:
            for v in r["files"]:
                f.write(json.dumps({"video_path": v, "repo": REPO, "shard": r["shard"]}) + "\n")
    api.upload_file(path_or_fileobj=f"{TMP}/overflow_index.jsonl", path_in_repo="index.jsonl",
                    repo_id=REPO, repo_type="dataset")
    log(f"done: {sum(len(r['files']) for r in rows)} windows in {len(rows)} shards")


if __name__ == "__main__":
    main()
