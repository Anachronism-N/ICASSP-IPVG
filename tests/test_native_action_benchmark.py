import importlib.util
from pathlib import Path
from collections import Counter
import pytest
spec=importlib.util.spec_from_file_location('action_prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_native_action_benchmark.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def records():
    rows=[]
    for i in range(32):
        for suffix in ('a','b'):
            rows.append({'sample_id':f's{i}{suffix}','identity_id':f'actor{i}','source_video_id':f'source{i}',
                         'reference_images':['ref.png'],'reference_sha256':'hash','prompt':'acted talking',
                         'video_path':'target.mp4','media_sha256':'targethash','track':1,'split':'dev'})
    rows.append({'sample_id':'train','identity_id':'trainactor','split':'train'})
    rows.append({'sample_id':'holdout','identity_id':'holdoutactor','split':'holdout'})
    prior=[r for r in rows if r['identity_id'] in {f'actor{i}' for i in range(8)}]
    return rows,prior


def test_additional_actors_balanced_action_categories_and_no_target_fields():
    rows,prior=records();selected=module.selected_rows(rows,prior)
    assert len(selected)==16 and len({r['identity_id'] for r in selected})==8
    assert not {r['identity_id'] for r in selected}&{r['identity_id'] for r in prior}
    assert Counter(r['action_category'] for r in selected)==Counter({k:4 for k in module.PROMPTS})
    assert all('video_path' not in r and 'media_sha256' not in r and r['split']=='dev' for r in selected)
    assert all(r['prompt']==module.PROMPTS[r['action_category']] for r in selected)
    assert len({r['sample_id'] for r in selected})==16
    assert selected==module.selected_rows(list(reversed(rows)),list(reversed(prior)))


def test_insufficient_extra_development_actors_fails():
    rows,prior=records()
    with pytest.raises(ValueError):module.selected_rows(rows,rows)


def test_same_reference_selected_in_a_stable_order():
    rows,prior=records();first=module.selected_rows(rows,prior)
    rows[10]['video_path']='another_target.mp4'
    assert first==module.selected_rows(rows,prior)
