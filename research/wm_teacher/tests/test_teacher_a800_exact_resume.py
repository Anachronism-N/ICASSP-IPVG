import importlib.util,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import pytest
S=Path(__file__).resolve().parents[1]/'scripts/run_teacher_a800_exact_resume.py'
spec=importlib.util.spec_from_file_location('admission',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_help_and_import_do_not_initialize_model():
    r=subprocess.run([sys.executable,str(S),'--help'],capture_output=True);assert r.returncode==0
    code='import runpy,sys;runpy.run_path('+repr(str(S))+',run_name="not_main");assert "torch" not in sys.modules'
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0

def test_exact_cfg_fixed_physical_and_logical_scope():
    cfg=a.cfg();assert cfg['max_steps']==cfg['horizon']==4 and cfg['warmup']==1
    assert cfg['offload'] is True and cfg['anchor_lambda']==.5 and cfg['lora_rank']==cfg['lora_alpha']==32
    assert cfg['per_rank_batch']==cfg['grad_accum']==1
    assert a.CAP==5040

def test_unsigned_allocation_rejected_before_claim(tmp_path):
    with patch.object(a,'E',tmp_path):
        a.save(tmp_path/'authorization.json',{'schema':'unsigned'})
        with pytest.raises(AssertionError):a.authorize()
        assert not (tmp_path/'attempt.claim').exists()

def test_checkpoint_evidence_writer_never_overwrites(tmp_path):
    row=a.save(tmp_path/'evidence.json',{'exact':True})
    with pytest.raises(FileExistsError):a.save(tmp_path/'evidence.json',{'exact':False})
    assert a.record(row['path'])==row
