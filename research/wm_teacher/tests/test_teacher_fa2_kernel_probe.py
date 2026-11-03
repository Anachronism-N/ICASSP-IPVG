import importlib.util,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import pytest
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_teacher_fa2_kernel_probe.py'
spec=importlib.util.spec_from_file_location('probe',SCRIPT);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

def test_import_and_help_do_not_initialize_torch():
    result=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True)
    assert result.returncode==0
    code='import runpy,sys;runpy.run_path('+repr(str(SCRIPT))+',run_name="not_main");assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True)
    assert result.returncode==0

def test_new_auth_and_existing_claim_not_reused(tmp_path):
    with patch.object(probe,'E',tmp_path):
        row=probe.write(tmp_path/'authorization.json',{'schema':'unapproved'})
        with pytest.raises(AssertionError):probe.authorize()
        assert not (tmp_path/'attempt.claim').exists()
        assert probe.bound(tmp_path/'authorization.json')==row

def test_record_writer_is_immutable(tmp_path):
    row=probe.write(tmp_path/'receipt.json',{'technical_only':True})
    with pytest.raises(FileExistsError):probe.write(tmp_path/'receipt.json',{'technical_only':False})
    assert probe.bound(tmp_path/'receipt.json')==row
