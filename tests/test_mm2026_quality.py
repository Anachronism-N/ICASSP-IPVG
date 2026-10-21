import importlib.util
from pathlib import Path
import socket
import subprocess
import urllib.request
import pytest
spec=importlib.util.spec_from_file_location('mmquality',Path(__file__).resolve().parents[1]/'scripts/mm2026_quality_eval.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mos_normalized_once():
    assert module.normalized_imaging(65.)==.65
    for v in (True,None,'65',float('nan'),float('inf')):
        with pytest.raises(ValueError):module.normalized_imaging(v)


def test_network_and_subprocess_blocked_and_restored():
    original=socket.create_connection,urllib.request.urlopen,subprocess.Popen
    with module.offline_only():
        for fn,args in [(socket.create_connection,(('127.0.0.1',9),)),(urllib.request.urlopen,('https://example.invalid',)),(subprocess.Popen,(['wget'],))]:
            with pytest.raises(RuntimeError,match='forbidden'):fn(*args)
    assert (socket.create_connection,urllib.request.urlopen,subprocess.Popen)==original


def test_dirty_upstream_rejected_before_model_reads(monkeypatch,tmp_path):
    calls=[]
    def run(command, **kwargs):
        calls.append(command)
        if 'diff' in command:
            raise subprocess.CalledProcessError(1,command)
        return type('Result',(),{'stdout':module.VBENCH_REVISION+'\n'})()
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(subprocess.CalledProcessError):
        module.validate_local(tmp_path,tmp_path/'missing.pth')
    assert any('diff' in command for command in calls)


def test_offline_restored_on_error():
    original=socket.socket.connect
    with pytest.raises(ValueError):
        with module.offline_only():raise ValueError('expected')
    assert socket.socket.connect==original
