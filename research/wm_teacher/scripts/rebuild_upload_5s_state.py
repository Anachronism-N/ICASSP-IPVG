#!/usr/bin/env python3
"""
Rebuild /dockerdata/wm_logs/upload_5s_state.jsonl (lost with the old node) from the shards
already in the HF repo: stream each remote tar once (nothing written to disk) and record its
member names. Output format matches upload_5s.py: {"shard": name, "files": [...]}.
(Walking headers with per-member HTTP range requests got rate-limited; streaming is ~8MB/s per
connection, so shards are processed in parallel.)
"""
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from huggingface_hub import HfApi

REPO = "nzc-415/wan22-5s"
TOKEN = os.environ["HF_TOKEN"]
STATE = sys.argv[1] if len(sys.argv) > 1 else "/dockerdata/wm_logs/upload_5s_state.jsonl"
PARALLEL = 16


class _Stream:
    """File-like reader over HTTP for tarfile 'r|' mode; reconnects with a Range header when the
    connection drops mid-stream (long downloads through the proxy get cut regularly)."""
    def __init__(self, url):
        self.url, self.pos, self.buf, self.it, self.size, self.drops = url, 0, b"", None, None, 0

    def _open(self):
        r = requests.get(self.url, headers={"Authorization": f"Bearer {TOKEN}",
                                            "Range": f"bytes={self.pos}-"}, stream=True, timeout=120)
        r.raise_for_status()
        if self.size is None:
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.it = r.iter_content(chunk_size=1 << 20)

    def read(self, n=-1):
        while n < 0 or len(self.buf) < n:
            if self.size is not None and self.pos >= self.size:
                break
            try:
                if self.it is None:
                    self._open()
                chunk = next(self.it)
                self.buf += chunk
                self.pos += len(chunk)
            except StopIteration:
                self.it = None  # server closed early -> reopen at self.pos (loop exits at EOF)
            except Exception:
                self.drops += 1
                if self.drops > 50:
                    raise
                self.it = None
                time.sleep(5)
        if n < 0:
            out, self.buf = self.buf, b""
        else:
            out, self.buf = self.buf[:n], self.buf[n:]
        return out


def members(shard):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{shard}"
    for attempt in range(3):
        try:
            with tarfile.open(fileobj=_Stream(url), mode="r|") as tf:
                return [m.name for m in tf if m.isfile()]
        except Exception as e:
            print(f"{shard}: attempt {attempt + 1} failed: {str(e)[:120]}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"{shard}: gave up")


def main():
    api = HfApi(token=TOKEN)
    shards = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("shards/"))
    done = set()
    if os.path.exists(STATE):
        for line in open(STATE):
            done.add(json.loads(line)["shard"])
    todo = [s for s in shards if s not in done]
    print(f"remote shards={len(shards)} already in state={len(done)} todo={len(todo)}", flush=True)
    n_fail = 0
    with open(STATE, "a") as out, ThreadPoolExecutor(PARALLEL) as ex:
        futs = {ex.submit(members, s): s for s in todo}
        for fu in as_completed(futs):
            shard = futs[fu]
            try:
                names = fu.result()
            except Exception as e:
                n_fail += 1
                print(f"{shard}: FAILED {e}", flush=True)
                continue
            out.write(json.dumps({"shard": shard, "files": names}) + "\n")
            out.flush()
            print(f"{shard}: {len(names)} files", flush=True)
    print(f"[done] failed={n_fail}", flush=True)


if __name__ == "__main__":
    main()
