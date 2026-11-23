#!/usr/bin/env python3
"""Run one shard process per visible GPU (CUDA_VISIBLE_DEVICES split); '{shard}' and '{nshards}' in argv are substituted.

Exit status is nonzero if any shard fails. Used under teacher_guarded_launch.py; grants no approval.
"""
import os
from pathlib import Path
import subprocess
import sys


def main():
    log_dir = Path(sys.argv[1]); argv = sys.argv[2:]
    assert argv and log_dir.is_dir(), 'usage: teacher_per_gpu_spawn.py LOG_DIR CMD...'
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    offset = int(os.environ.get('SPAWN_SHARD_OFFSET', '0'))       # multi-node: global shard = offset + local index
    total = int(os.environ.get('SPAWN_SHARD_TOTAL', str(len(uuids))))
    assert 0 <= offset and offset + len(uuids) <= total
    procs = []
    for k, u in enumerate(uuids):
        cmd = [a.replace('{shard}', str(offset + k)).replace('{nshards}', str(total)) for a in argv]
        log = open(log_dir / f'shard{k}.log', 'xb')
        procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=u), stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print({'returncodes': codes}, flush=True)
    sys.exit(1 if any(codes) else 0)


if __name__ == '__main__':
    main()
