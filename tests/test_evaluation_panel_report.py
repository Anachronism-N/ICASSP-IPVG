import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('panelreport',Path(__file__).resolve().parents[1]/'scripts/evaluation_panel_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_missing_scores_have_separate_denominator_not_zero():
    assert module.finite_values([.5,None,.7])=={'count':2,'mean':.6}
    assert module.finite_values([None])=={'count':0,'mean':None}
    for values in ([True],[float('nan')],['.5'],[float('inf')]):
        with pytest.raises(ValueError):module.finite_values(values)


def test_cur_missing_reference_and_all_missing_distinct_values():
    r={'records':[{'sample_id':'a','cur_score':None,'arc_score':None,'face_detection_coverage':0,
                   'cur_scoring_coverage':0,'arc_scoring_coverage':0}]}
    rows=module.report_records(r)
    assert rows[0][1]['cur_score'] is None
    r['records'][0]['cur_score']=0
    assert module.report_records(r)[0][1]['cur_score']==0


def test_score_hash_and_failed_run_rejected(tmp_path):
    from ipvg.common import write_json,sha256_file
    write_json(tmp_path/'scores.json',{'records':[]})
    write_json(tmp_path/'result.json',{'status':'completed','score_sha256':sha256_file(tmp_path/'scores.json'),
               'backend_elapsed_seconds':36,'gpus':1,'allocated_gpu_hours':.01})
    assert module.job(tmp_path)[0]['status']=='completed'
    write_json(tmp_path/'scores.json',{'records':['drift']})
    with pytest.raises(ValueError):module.job(tmp_path)
