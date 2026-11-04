"""CPU contracts for synthetic encoder technical admission, never supervised cache approval."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest

S=Path(__file__).resolve().parents[1]/'scripts/run_teacher_encoder_cache_probe.py'
spec=importlib.util.spec_from_file_location('encoder_probe',S);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_import_and_help_do_not_initialize_torch():
    code='import runpy,sys;runpy.run_path('+repr(str(S))+',run_name="not_main");assert "torch" not in sys.modules'
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0
    assert subprocess.run([sys.executable,str(S),'--help'],capture_output=True).returncode==0


def test_unsigned_scope_rejected_without_GPU_claim(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'E',tmp_path)
    p.save(tmp_path/'authorization.json',dict(synthetic=True))
    with pytest.raises(AssertionError):p.authorize()
    assert not (tmp_path/'attempt.claim').exists()


def test_unauthorized_run_never_initializes_resource(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'E',tmp_path)
    with patch.object(p.subprocess,'Popen') as spawn,pytest.raises(AssertionError):p.run(SimpleNamespace(gpu_authorized=False))
    spawn.assert_not_called();assert not list(tmp_path.iterdir())


def test_existing_evidence_never_overwritten(tmp_path):
    row=p.save(tmp_path/'terminal.json',dict(success=False))
    with pytest.raises((FileExistsError,ValueError)):p.save(tmp_path/'terminal.json',dict(success=True))
    assert p.read(row)['success'] is False


def test_scope_preserves_full_synthetic_shape_and_forbids_training():
    import ast
    tree=ast.parse(S.read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    assert p.CAP==720
    text=ast.get_source_segment(S.read_text(),worker)
    assert "range(121)" in text and "(1280,704)" in text
    assert 'cache.encode_video' in text and text.count('cache.encode_prompt')==2
    assert 'model_updates=0' in text and 'supervised_cache_admitted=False' in text
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    text=ast.get_source_segment(S.read_text(),run)
    assert "time.monotonic()-started" in text and 'max(.2,elapsed/3600)' in text
    assert '4.8+max(.2,elapsed/3600)' in text
