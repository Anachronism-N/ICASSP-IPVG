import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
spec=importlib.util.spec_from_file_location('local_story',Path(__file__).resolve().parents[1]/'scripts/local_story_eval.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_frame_endpoints_and_reproducible_sampling():
    assert module.frame_indices(49)==[0,7,14,21,27,34,41,48]
    for count in (True,7,49.0):
        with pytest.raises(ValueError):module.frame_indices(count)


def test_story_similarity_is_raw_unit_dot_not_official_score():
    assert module.normalized_similarity([1,0],[0,1])==0
    assert module.normalized_similarity([1,0],[-1,0])==-1
    for a,b in (([2,0],[1,0]),([1,float('nan')],[1,0]),([1,0],[1])):
        with pytest.raises(ValueError):module.normalized_similarity(a,b)
    assert module.PROTOCOL['official_story_score'] is None


def test_model_lock_rejects_path_escape_and_changed_content(tmp_path):
    from ipvg.common import sha256_file
    file=tmp_path/'model';file.write_bytes(b'fixture')
    module.checked_files(tmp_path,{'model':sha256_file(file)})
    with pytest.raises(ValueError):module.checked_files(tmp_path,{'../model':'x'})
    with pytest.raises(ValueError):module.checked_files(tmp_path,{'model':'0'*64})
