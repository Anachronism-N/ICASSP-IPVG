"""CPU-only prospective job plans, not generation or SOTA evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

S=Path(__file__).resolve().parents[1]/'scripts/build_eval_jobs.py'
spec=importlib.util.spec_from_file_location('eval_plan',S)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

@pytest.fixture
def preflight(tmp_path):
    image=tmp_path/'image.jpg';image.write_bytes(b'fixture image, not decoded')
    core=[dict(prompt_en='scene '+str(i),image_name='image',dimension=['i2v_subject']) for i in range(355)]
    camera=[dict(prompt_en='scene '+str(i)+', camera motion '+str(k),image_name='image',dimension=['camera_motion']) for i in range(109) for k in range(7)]
    info=core+camera;metadata=tmp_path/'metadata.json';metadata.write_text(json.dumps(info))
    rows=[dict(filename=x['prompt_en']+'-'+str(i)+'.mp4',pair_key=x['prompt_en'],sample_index=i,raw_prompt=x['prompt_en'],image_key='image',proposed_seed=i,dimensions=x['dimension'],execution_allowed=False) for x in info for i in range(5)]
    value=dict(schema='official_vbench_i2v_complete_coverage_CPU_preflight_v1',execution_allowed=False,samples_per_pair=5,
        coverage_records=rows,metadata=dict(path=str(metadata),sha256=b.file_hash(metadata)),
        image_files={'image':dict(path=str(image),bytes=image.stat().st_size,sha256=b.file_hash(image))},
        seed_policy={'status':'CPU_proposal_only'})
    path=tmp_path/'preflight.json';path.write_text(json.dumps(value));return path,value


def test_full_five_seed_plan_freezes_explicit_models_and_never_dispatches(preflight,tmp_path,monkeypatch):
    path,_=preflight;ckpt=tmp_path/'candidate.safetensors';ckpt.write_bytes(b'synthetic checkpoint bytes')
    monkeypatch.setattr(b,'EV',str(tmp_path/'eval'))
    result=b.official_generation_plan(path,{'base':'NONE','candidate':str(ckpt)})
    assert len(result['jobs'])==11180 and result['videos_per_model']==5590
    assert result['frozen_checkpoints']['candidate']['sha256']==b.file_hash(ckpt)
    assert all(not row['execution_allowed'] for row in result['jobs']) and not result['execution_allowed']
    left=result['jobs'][:5590];right=result['jobs'][5590:]
    assert [(r['id'],r['prompt'],r['seed'],r['image']) for r in left]==[(r['id'],r['prompt'],r['seed'],r['image']) for r in right]
    assert len({r['out'] for r in result['jobs']})==11180 and not (tmp_path/'eval').exists()


@pytest.mark.parametrize('change',['partial','duplicate','prompt','image','sample','seed','dispatch'])
def test_modified_coverage_rejected_before_model_plan(preflight,change):
    path,value=preflight;v=copy.deepcopy(value)
    if change=='partial':v['coverage_records'].pop()
    elif change=='duplicate':v['coverage_records'][-1]=v['coverage_records'][0]
    elif change=='prompt':v['coverage_records'][0]['raw_prompt']='changed'
    elif change=='image':v['coverage_records'][0]['image_key']='changed'
    elif change=='sample':v['coverage_records'][0]['sample_index']=5
    elif change=='seed':v['coverage_records'][0]['proposed_seed']=True
    else:v['coverage_records'][0]['execution_allowed']=True
    path.write_text(json.dumps(v))
    with pytest.raises(ValueError):b.official_generation_plan(path,{'base':'NONE'})


def test_modified_image_bytes_rejected(preflight):
    path,v=preflight;Path(v['image_files']['image']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='image bytes'):b.official_generation_plan(path,{'base':'NONE'})


@pytest.mark.parametrize('models',[{}, {'../bad':'NONE'},{'candidate':'relative.pt'},{'candidate':None}])
def test_unfrozen_or_unsafe_model_rejected(preflight,models):
    path,_=preflight
    with pytest.raises(ValueError):b.official_generation_plan(path,models)


def test_new_plan_never_overwrites_legacy_case_or_jobs(preflight,tmp_path,monkeypatch):
    path,_=preflight;models=tmp_path/'models.json';models.write_text('{"base":"NONE"}')
    legacy=tmp_path/'cases.jsonl';legacy.write_bytes(b'original legacy data')
    monkeypatch.setattr(sys,'argv',[str(S),'--official-five-plan',str(path),'--frozen-models',str(models),'--output',str(legacy)])
    with patch.object(b,'official_generation_plan') as build,pytest.raises(SystemExit):b.main()
    build.assert_not_called();assert legacy.read_bytes()==b'original legacy data'


def test_official_options_never_fall_through_to_legacy_writes(monkeypatch):
    monkeypatch.setattr(sys,'argv',[str(S),'--frozen-models','models.json','--output','new.json'])
    with patch.object(b,'wm_cases') as cases,pytest.raises(SystemExit):b.main()
    cases.assert_not_called()
