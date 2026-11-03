"""CPU contracts only; these tests never certify real-video semantic correctness."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from unittest.mock import patch

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_teacher_label_semantic_review as review


def good(): return {k:'yes' for k in review.FIELDS}


def test_strict_review_no_missing_fields_or_free_text():
    assert review.verdict(good())=='adequate'
    for key in review.FIELDS:
        value=good();value[key]='no';assert review.verdict(value)=='needs_edit'
        value[key]='uncertain';assert review.verdict(value)=='uncertain'
    for bad in ({},dict(good(),explanation='untrusted prose'),dict(good(),entity_match=True),
                dict(good(),entity_match='probably')):
        with pytest.raises(ValueError):review.validate_review(bad)


def test_synthetic_controls_are_distinct_known_motion_evidence():
    from PIL import ImageChops
    controls=review.controls()
    assert len(controls)==4 and {c['expected'] for c in controls}=={'adequate','needs_edit'}
    still,moving=review.control_images(False),review.control_images(True)
    assert len(still)==len(moving)==16 and all(i.size==(640,352) for i in still+moving)
    assert ImageChops.difference(still[0],still[-1]).getbbox() is None
    assert ImageChops.difference(moving[0],moving[-1]).getbbox() is not None
    assert still[0].getpixel((80,176))==(255,0,0) and moving[-1].getpixel((560,176))==(255,0,0)


@pytest.mark.parametrize('complete',[True,False])
def test_complete_eos_required_without_retry(complete):
    import torch
    class Inputs(dict):
        def to(self,device): assert device=='cuda:0';return self
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            assert len(messages[0]['content'])==17
            assert review.PROMPT in messages[0]['content'][-1]['text']
            return Inputs(input_ids=torch.tensor([[10,11]]))
        def batch_decode(self,output,**kwargs):return [json.dumps(good())]
    class Model:
        generation_config=SimpleNamespace(eos_token_id=[151645,151643])
        calls=0
        def generate(self,**kwargs):
            self.calls+=1;assert kwargs['do_sample'] is False and kwargs['max_new_tokens']==384
            return torch.tensor([[10,11,5,151645 if complete else 5]])
    model=Model()
    if complete:assert review.validate_review(json.loads(review.judge(model,Processor(),review.control_images(False),'Candidate')))==good()
    else:
        with pytest.raises(ValueError):review.judge(model,Processor(),review.control_images(False),'Candidate')
    assert model.calls==1 and not torch.cuda.is_initialized()


def test_pre_dispatch_verification_failure_has_full_charge_and_no_gpu(tmp_path,monkeypatch):
    r,process=review.helpers()
    output=tmp_path/'review_run01'
    p=dict(output_root=str(output),gpu_uuid='GPU-0')
    row=review.write(tmp_path/'plan.json',p)
    args=SimpleNamespace(gpu_authorized=True,plan=row['path'],sha=row['sha256'],commit='a'*40,receipt='missing')
    monkeypatch.setattr(sys,'argv',[str(Path(review.__file__).resolve())])
    monkeypatch.setattr(review,'verify',lambda *a: (_ for _ in ()).throw(ValueError('mock verification drift')))
    with patch.object(review.subprocess,'Popen') as spawn,pytest.raises(ValueError): review.run(args)
    spawn.assert_not_called()
    cost=json.loads((output/'terminal.json').read_bytes())
    assert cost['terminal'] and not cost['success'] and cost['GPUs_released']
    assert cost['charged_upper_GPUh']==1 and cost['cumulative_label_upper_GPUh']==21.165898210853467
    assert cost['elapsed_seconds']>=0 and cost['review_semantic_pass'] is False
    assert (output/'attempt.claim').is_file()
    with pytest.raises(FileExistsError):review.run(args)


def test_unauthorized_allocation_rejected_without_claim(tmp_path):
    args=SimpleNamespace(gpu_authorized=False)
    with pytest.raises(ValueError):review.run(args)
    assert not list(tmp_path.iterdir())



def test_followup_scope_fits_remaining_budget_without_semantic_gate_change(monkeypatch):
    for k in ('CAP','TAIL','ALLOCATION_GPUH','PRIOR_GPUH','AUTH','AUTH_SHA','FOLLOWUP'):
        monkeypatch.setattr(review,k,getattr(review,k))
    prompt=review.PROMPT;fields=review.FIELDS
    review.configure_followup()
    assert review.CAP==540 and review.TAIL==60 and review.ALLOCATION_GPUH==0.15
    assert review.PRIOR_GPUH+review.ALLOCATION_GPUH<=24
    assert review.PROMPT==prompt and review.FIELDS==fields and review.FOLLOWUP
    assert review.verdict(good())=='adequate'
    assert review.AUTH.parent.name=='teacher_label_review_followup_20261103'


def test_campaign003_does_not_reset_old_budget_or_semantic_gate(monkeypatch):
    for k in ('CAP','TAIL','ALLOCATION_GPUH','PRIOR_GPUH','AUTH','AUTH_SHA','CAMPAIGN003'):
        monkeypatch.setattr(review,k,getattr(review,k))
    prompt=review.PROMPT;fields=review.FIELDS
    review.configure_campaign003()
    assert review.CAP==3600 and review.ALLOCATION_GPUH==1
    assert review.PRIOR_GPUH==41.832564877520134
    assert review.AUTH.parent.name=='semantic_review003'
    assert review.PROMPT==prompt and review.FIELDS==fields
    assert review.verdict(good())=='adequate'


@pytest.mark.parametrize('change',[{}, {'all_cases_denominator':30}, {'jobs':[{}]*29},
                                   {'stage_cap_GPUh':24}, {'new_label_stage_prior_GPUh':0}])
def test_campaign003_verify_retains_all_32_cases_and_phase_budget(monkeypatch,change):
    import hashlib,os
    for k in ('CAP','TAIL','ALLOCATION_GPUH','PRIOR_GPUH','AUTH','AUTH_SHA','CAMPAIGN003'):
        monkeypatch.setattr(review,k,getattr(review,k))
    review.configure_campaign003()
    p=dict(schema='teacher_label_independent_review_plan_v1',private=True,hostname=os.uname().nodename,
           node_ip=os.environ.get('NODE_IP'),cap_seconds=3600,allocated_cards=1,upper_GPUh=1,
           prior_charged_label_GPUh=review.PRIOR_GPUH,cumulative_label_upper_GPUh=review.PRIOR_GPUH+1,
           prompt_sha256=hashlib.sha256(review.PROMPT.encode()).hexdigest(),controls=review.controls(),
           automatic_retry_allowed=False,semantic_review_pass=False,formal_training_allowed=False,
           AR_allowed=False,student_distillation_allowed=False,authorization={'sha256':'auth'},
           all_cases_denominator=32,jobs=[{}]*30,new_label_stage_prior_GPUh=18,stage_cap_GPUh=96,
           inputs=[],previous_costs=[],source={},model={'files':[]},runtime={'fixture':True})
    p.update(change)
    fake=SimpleNamespace(pinned=lambda path,digest,end=None:({'sha256':'auth'} if path==review.AUTH else {'sha256':'plan'}),
                         check=lambda *args:None,runtime=lambda end=None:{'fixture':True})
    monkeypatch.setattr(review,'helpers',lambda:(fake,None))
    monkeypatch.setattr(review,'read',lambda row,end=None:({'parent_campaign_sha256':'campaign'} if row['sha256']=='auth' else p))
    monkeypatch.setattr(review,'record',lambda *args:{'sha256':'campaign'})
    if change:
        with pytest.raises(ValueError):review.verify('synthetic_plan','digest')
    else:assert review.verify('synthetic_plan','digest')[0] is p
