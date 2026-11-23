#!/usr/bin/env python3
"""Copy a finalized training cache to node-local disk (parallel), driven by its cache_index.json.

Only the files named in the index (plus the index and binding) are copied, so no directory listing or per-file stat
is needed on the shared filesystem (both stall under load). Each copied file must match the size recorded in the
index; train_e3_lora.load_sample re-verifies each file's SHA256 at load. Never overwrites: destination must not exist.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import time


def index_files(index):
    for s in index['samples']:
        yield s['x0']['path'], s['x0']['bytes']
        for c in s['contexts'].values():
            yield c['path'], c['bytes']


def main(src, dst, workers=16):
    src, dst = Path(src), Path(dst)
    raw = (src / 'cache_index.json').read_bytes()
    index = json.loads(raw)
    assert index.get('complete') is True, 'source cache not finalized'
    assert not dst.exists(), 'destination exists'
    tmp = dst.with_name(dst.name + '.partial')
    files = list(index_files(index))
    assert all(not os.path.isabs(p) and '..' not in Path(p).parts for p, _ in files), 'index paths must be relative'
    t0 = time.time()

    def one(item):
        rel, size = item
        out = tmp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and out.stat().st_size == size):
            shutil.copyfile(src / rel, out)
            os.chmod(out, 0o600)
        assert out.stat().st_size == size, 'size differs: ' + rel
        return size
    with ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(one, files))
    for name in ('binding.json',):
        if (src / name).exists():
            shutil.copyfile(src / name, tmp / name)
    (tmp / 'cache_index.json').write_bytes(raw)
    os.rename(tmp, dst)
    print(json.dumps(dict(src=str(src), dst=str(dst), files=len(files), bytes=total, seconds=time.time() - t0)))


if __name__ == '__main__':
    main(*sys.argv[1:3], *(int(x) for x in sys.argv[3:4]))
