import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/teacher_per_gpu_spawn.py'


def test_one_process_per_visible_device_with_shard_substitution(tmp_path):
    env = {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b,GPU-c', 'PATH': '/usr/bin:/bin'}
    code = "import os,sys; print(sys.argv[1], sys.argv[2], os.environ['CUDA_VISIBLE_DEVICES'])"
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), sys.executable, '-c', code, '{shard}', '{nshards}'], env=env)
    assert r.returncode == 0
    assert [(tmp_path / f'shard{k}.log').read_text().split() for k in range(3)] == [[str(k), '3', u] for k, u in enumerate(['GPU-a', 'GPU-b', 'GPU-c'])]
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / 'x'), 'true'], env=env)
    assert r.returncode != 0


def test_global_shard_offset_for_multinode(tmp_path):
    env = {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b', 'PATH': '/usr/bin:/bin', 'SPAWN_SHARD_OFFSET': '8', 'SPAWN_SHARD_TOTAL': '16'}
    code = "import sys; print(sys.argv[1], sys.argv[2])"
    r = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), sys.executable, '-c', code, '{shard}', '{nshards}'], env=env)
    assert r.returncode == 0
    assert [(tmp_path / f'shard{k}.log').read_text().split() for k in range(2)] == [['8', '16'], ['9', '16']]
