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
