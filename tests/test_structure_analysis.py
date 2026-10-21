import importlib.util
import os
from pathlib import Path
import pytest
pytest.importorskip('numpy')
path=Path(os.environ.get('IPVG_STRUCTURE_ANALYZER', str(Path(__file__).resolve().parents[1]/'scripts/analyze_reference_structure.py')))
spec=importlib.util.spec_from_file_location('structure_audit',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def observation(gate):
    return {'mode':'reference_difference_temporal_v1','gate':gate,'reference_frames':1,
            'outer_calls':40,'native_calls':80,'hint_layers':15,'grid':[14,22,30],
            'reference_prefix_exact':True,'body_max_abs_change':.25 if gate==0 else 0,
            'reference_temporal_centered_max_abs':.25,'body_temporal_mean_max_abs_error':.001 if gate==0 else 0}


def test_actual_structure_change_and_identity_control():
    module.routing_check(observation(0),0)
    module.routing_check(observation(1),1)


@pytest.mark.parametrize('problem',['calls','native','prefix','layers','grid','unchanged','nonfinite','negative','wronggate','wrongmode'])
def test_routing_evidence_error_rejected(problem):
    row=observation(0)
    if problem=='calls':row['outer_calls']=39
    elif problem=='native':row['native_calls']=40
    elif problem=='prefix':row['reference_prefix_exact']=False
    elif problem=='layers':row['hint_layers']=14
    elif problem=='grid':row['grid']=[14,30,22]
    elif problem=='unchanged':row['body_max_abs_change']=0
    elif problem=='nonfinite':row['body_temporal_mean_max_abs_error']=float('nan')
    elif problem=='negative':row['reference_temporal_centered_max_abs']=-1
    elif problem=='wronggate':row['gate']=1
    elif problem=='wrongmode':row['mode']='alpha'
    with pytest.raises(ValueError):module.routing_check(row,0)


def test_control_cannot_mutate_prefix_or_body():
    row=observation(1);row['body_max_abs_change']=.001
    with pytest.raises(ValueError):module.routing_check(row,1)
