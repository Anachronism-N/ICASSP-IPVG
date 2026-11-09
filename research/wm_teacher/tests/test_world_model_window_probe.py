import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_world_model_window_probe as probe


def test_import_CPU_and_no_hidden_model_load():
    code = 'import runpy,sys;runpy.run_path('+repr(probe.__file__)+');assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-c',code], check=True, capture_output=True)
    subprocess.run([sys.executable,probe.__file__,'--help'], check=True, capture_output=True)


def test_run_requires_explicit_execution_and_publication_before_reading_plan():
    with pytest.raises(ValueError, match='GPU'):
        probe.run(SimpleNamespace(gpu_authorized=False, publication=None))


def test_new_receipt_exclusive_and_private(tmp_path):
    path=tmp_path/'receipt.json';bound=probe.write_new(path, dict(success=False))
    assert bound == probe.record(path) and path.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError): probe.write_new(path, dict(success=True))
    assert json.loads(path.read_bytes())['success'] is False


def test_source_preclaim_gates_full_spec_and_no_training():
    text=Path(probe.__file__).read_text();tree=ast.parse(text)
    worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
    source=ast.get_source_segment(text,worker)
    assert 'skip_download=True' in source and 'redirect_common_files=False' in source
    assert 'save_video' in source and 'validate_video' in source and 'max_calls=MAX_CALLS' in source
    assert 'references[1] == references[2] == last' in source
    assert 'train_step(' not in source and '.backward(' not in source
    assert probe.PHASE=='technical_admission' and probe.CAP==3600 and probe.MAX_CALLS==3


def test_forged_plan_limits_rejected(tmp_path):
    p=tmp_path/'plan.json'
    p.write_text(json.dumps(dict(schema='world_model_window_probe_plan_v1',phase='technical_admission',
        hostname='wrong', cap_seconds=probe.CAP, allocated_cards=1, upper_GPUh=1, max_calls=3)))
    with pytest.raises(ValueError,match='scope'):
        probe.verify(str(p),probe.record(p)['sha256'],assets=False)


def test_bad_plan_sha_refused_before_scope(tmp_path):
    p=tmp_path/'plan.json';p.write_text('{}')
    with pytest.raises(ValueError,match='SHA'): probe.verify(str(p),'a'*64,assets=False)
