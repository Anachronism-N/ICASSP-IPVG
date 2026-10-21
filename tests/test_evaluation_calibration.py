import importlib.util
from pathlib import Path
import pytest
np=pytest.importorskip('numpy')
spec=importlib.util.spec_from_file_location('evalcal',Path(__file__).resolve().parents[1]/'scripts/calibrate_evaluation_panel.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_actor_split_is_disjoint_stable_source_only():
    actors={str(i) for i in range(32)}
    left,right=module.actor_split(actors)
    assert len(left)==len(right)==16 and set(left).isdisjoint(right)
    assert (left,right)==module.actor_split(actors)
    with pytest.raises(ValueError):module.actor_split({'one'})


def test_source_positive_and_cross_actor_negative_pairs():
    refs={'a':np.array([1.,0.]),'b':np.array([0.,1.])}
    records=[{'actor':'a','vectors':[refs['a']]*16},{'actor':'b','vectors':[refs['b']]*16}]
    p,n,_,_=module.pairs(['a','b'],refs,records)
    assert p==[1,1] and n==[0,0]
    records[0]['vectors']=refs['a'][:1].tolist()
    records=records[1:]
    p,n,_,_=module.pairs(['a','b'],refs,records)
    assert p==[1] and n==[0]
