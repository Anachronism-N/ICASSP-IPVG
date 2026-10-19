import copy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('anyid_context',Path(__file__).resolve().parents[1]/'scripts/prepare_anyid_context.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    original=[]
    for actor in range(1000,1032):
        for prompt in ('neutral','happy'):
            sid='diagnostic_neutral' if actor==1009 and prompt=='neutral' else f'actor{actor}_{prompt}'
            original.append({'sample_id':sid,'identity_id':f'actor{actor}','split':'dev','prompt':prompt,
                             'reference_images':[f'/original/{actor}.png'],'reference_sha256':str(actor),
                             'source_video_id':sid,'reference_source_video_id':f'ref{actor}'})
    legacy=[copy.deepcopy(r) for r in original if r['identity_id'] in {'actor1000','actor1001','actor1002','actor1009'}]
    def condition(margin):
        return [dict(copy.deepcopy(r),original_reference_images=r['reference_images'][:],
                     reference_images=[f'/crop/{margin}/{r["identity_id"]}.png'],reference_crop={'margin':margin}) for r in original]
    return original,condition(.3),condition(.6),legacy


def test_deterministic_cohort_and_paired_seed_schedule():
    original,narrow,wide,legacy=inputs()
    manifests,protocol=module.paired_manifests(original,narrow,wide,legacy,'diagnostic_neutral')
    assert protocol['output_count']==96
    assert len(protocol['additional_dev_actors'])==8 and len(protocol['legacy_actors'])==4
    assert module.select_actors(original,legacy)==module.select_actors(list(reversed(original)),list(reversed(legacy)))
    left=sorted((r['sample_id'],r['generation_seed']) for key,rows in manifests.items() if key.startswith('narrow') for r in rows)
    right=sorted((r['sample_id'],r['generation_seed']) for key,rows in manifests.items() if key.startswith('wide') for r in rows)
    assert left==right and len(left)==len(set(left))==48
    for seed in (42,43):
        for shard in (0,1):
            a=manifests[f'narrow_s{seed}_shard{shard}']
            b=manifests[f'wide_s{seed}_shard{shard}']
            assert [(r['sample_id'],r['prompt'],r['generation_seed']) for r in a]==[(r['sample_id'],r['prompt'],r['generation_seed']) for r in b]
    assert protocol['holdout_used'] is False


@pytest.mark.parametrize('mutation',['prompt','anchor','margin','source','duplicate','split'])
def test_conditioning_drift_rejected(mutation):
    original,narrow,wide,legacy=inputs()
    row=next(r for r in wide if r['identity_id']=='actor1009')
    if mutation=='prompt': row['prompt']='changed'
    elif mutation=='anchor': row['original_reference_images']=['/wrong.png']
    elif mutation=='margin': row['reference_crop']['margin']=.7
    elif mutation=='source': row['source_video_id']='wrong'
    elif mutation=='duplicate': wide.append(copy.deepcopy(row))
    else: row['split']='holdout'
    with pytest.raises(ValueError): module.paired_manifests(original,narrow,wide,legacy,'diagnostic_neutral')


def test_legacy_cohort_must_match_dev():
    original,narrow,wide,legacy=inputs()
    legacy[0]['prompt']='changed'
    with pytest.raises(ValueError): module.paired_manifests(original,narrow,wide,legacy,'diagnostic_neutral')


def test_explicit_diagnostic_pilot_must_exist():
    original,narrow,wide,legacy=inputs()
    with pytest.raises(ValueError,match='pilot is absent'):
        module.paired_manifests(original,narrow,wide,legacy,'absent_sample')
