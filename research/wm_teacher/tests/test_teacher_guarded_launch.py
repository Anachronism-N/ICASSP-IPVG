import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import teacher_guarded_launch as gl  # noqa: E402


def test_descendants_include_new_session_grandchildren_but_not_parent():
    import subprocess, time
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True); print(p.pid,flush=True); time.sleep(30)"
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    try:
        grandchild = int(child.stdout.readline())
        pids = gl.descendant_pids(child.pid)
        assert {child.pid, grandchild} <= pids and os.getpid() not in pids and 1 not in pids
    finally:
        os.kill(grandchild, 9); child.kill(); child.wait()


def test_ledger_prior_reads_phase_keys_and_defaults_training_to_zero():
    b = dict(current_label_reserved_upper_GPUh=48.5, current_technical_reserved_upper_GPUh=6.0)
    assert gl.ledger_prior(b, 'label_and_cache') == 48.5
    assert gl.ledger_prior(b, 'technical_admission') == 6.0
    assert gl.ledger_prior(b, 'gate_conditioned_teacher_training_and_evaluation') == 0.0
    with pytest.raises(KeyError):
        gl.ledger_prior(b, 'borrowed_phase')


def test_save_new_never_overwrites(tmp_path):
    p = tmp_path / 'x.json'
    row = gl.save_new(p, dict(a=1))
    assert row['bytes'] == p.stat().st_size and oct(p.stat().st_mode)[-3:] == '600'
    with pytest.raises(FileExistsError):
        gl.save_new(p, dict(a=2))


def test_reserve_marker_and_pending_holder_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(gl, 'RESERVE', str(tmp_path / 'od_reserve_%d'))
    gl.touch_reserve([3, 5])
    assert (tmp_path / 'od_reserve_3').exists() and (tmp_path / 'od_reserve_5').exists()

    class G:
        @staticmethod
        def holder_identity(pid, gpu):
            return dict(pid=pid) if pid == os.getpid() and gpu == 2 else None
    assert gl.pending_holders(G, 2) == [os.getpid()] and gl.pending_holders(G, 1) == []


def test_locked_ledger_update_is_atomic(tmp_path, monkeypatch):
    import json
    p = tmp_path / 'budget.live.json'
    p.write_text(json.dumps(dict(allocations=[], x=1.0)))
    monkeypatch.setattr(gl, 'BUDGET', p)
    assert gl.update_budget(lambda b: b.update(x=b['x'] + 2.5) or b['x']) == 3.5
    assert json.loads(p.read_text())['x'] == 3.5 and gl.ledger_prior(dict(current_official_eval_reserved_upper_GPUh=2.0), 'official_protocol_evaluation') == 2.0


def test_foreign_ms12k_locks_are_held_and_never_overridden(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(gl, 'MS12K_LOCK', str(tmp_path / 'ms12k_gpu_%d.lock'))
    (tmp_path / 'ms12k_gpu_0.lock').write_text(''); (tmp_path / 'ms12k_gpu_1.lock').write_text('')
    fds = gl.hold_foreign_locks([0, 1, 2])
    assert len(fds) == 2
    other = os.open(str(tmp_path / 'ms12k_gpu_0.lock'), os.O_RDWR)
    with pytest.raises(OSError):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for fd in fds: os.close(fd)
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(RuntimeError):
        gl.hold_foreign_locks([0])
    os.close(other)


def test_terminate_tree_kills_new_session_descendants():
    import signal
    import subprocess
    import time
    p = subprocess.Popen([sys.executable, '-c', 'import subprocess,sys,time\n'
                          'c=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"], start_new_session=True)\n'
                          'print(c.pid, flush=True); time.sleep(60)'], stdout=subprocess.PIPE, text=True,
                         start_new_session=True)
    grand = int(p.stdout.readline())
    tree = gl.descendant_pids(p.pid)
    assert grand in tree
    os.killpg(p.pid, signal.SIGKILL); p.wait()
    time.sleep(0.2)
    assert Path('/proc/%d' % grand).exists()  # orphaned: outside the child's process group
    assert gl.terminate_tree(tree - {p.pid}, grace=5) == []
    assert not Path('/proc/%d' % grand).exists() or \
        Path('/proc/%d/stat' % grand).read_text().rsplit(')', 1)[1].split()[0] == 'Z'
