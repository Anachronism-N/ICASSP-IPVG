import importlib.util
from pathlib import Path
import sys
import pytest
from ipvg.common import load_json,write_json,sha256_file

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('native_audit',SCRIPTS/'analyze_native_baselines.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def identity_record(score,valid):
    values=[score]*valid+[None]*(16-valid)
    return {'group':'standin','cur_frame_cosines':values,'arc_frame_cosines':list(values),
            'cur_score':score if valid else 0.,'arc_score':score if valid else 0.,
            'cur_scoring_coverage':valid/16,'arc_scoring_coverage':valid/16,
            'face_detection_coverage':valid/16}


def test_missing_faces_separate_from_author_zero_mean(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    rows=[identity_record(.7,16),identity_record(0.,0)]
    result=module.identity_coverage_summary(rows)['standin']
    assert result['videos']==2 and result['detected_identity_videos']==1 and result['all_missing_identity_videos']==1
    assert result['mean_face_frame_coverage']==.5
    assert result['valid_video_cur_mean_diagnostic_only']==.7
    assert 'method-specific denominators' in result['notice']


@pytest.mark.parametrize('error',['coverage','count','mean','nonfinite'])
def test_identity_coverage_or_mean_drift_rejected(error,monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    row=identity_record(.5,16)
    if error=='coverage':row['cur_scoring_coverage']=.5
    elif error=='count':row['arc_frame_cosines'].pop()
    elif error=='mean':row['cur_score']=.7
    else:row['cur_frame_cosines'][0]=float('nan')
    with pytest.raises(ValueError):module.identity_coverage_summary([row])


def evidence(tmp_path,monkeypatch):
    (tmp_path/'outputs').mkdir()
    ref=tmp_path/'ref.png';ref.write_bytes(b'reference fixture')
    video=tmp_path/'outputs/one.mp4';video.write_bytes(b'encoded fixture')
    row={'sample_id':'one','identity_id':'actor','source_video_id':'separate_source',
         'reference_images':[str(ref)],'prompt':'A person moves.','track':1,'split':'dev','timeline':[]}
    (tmp_path/'manifest.jsonl').write_text(__import__('json').dumps(row)+'\n')
    write_json(tmp_path/'config.json',{'parameters':{}})
    result={'run_id':'fixture','status':'completed','source_sha256':'source','resources':{'gpus':1},
            'backend_elapsed_seconds':36.,'estimated_backend_gpu_hours':.01,
            'output_sha256':{'one.mp4':sha256_file(video)}}
    profile={'matched_ip_tensors':240,
             'standin_checkpoint_sha256':'52f47630b4fefbdac841d3fd1ce146cca2a6a94ec0a6ab9407d16f2f635177e9',
             'samples':[{'sample_id':'one','native_denoiser_calls':40,'ip_latent_calls':[{'ip_latent_shape':[1,16,1,64,64]}],
                         'ip_q_lora_calls':[[1,1024,5120]],'reference_original_sha256':sha256_file(ref)}]}
    write_json(tmp_path/'result.json',result);write_json(tmp_path/'backend_metrics.json',profile)
    entry={'run_id':'fixture','arm':'standin','source_sha256':'source','count':1,'reserved_gpu_hours':.5,
           'config_sha256':sha256_file(tmp_path/'config.json'),'manifest_sha256':sha256_file(tmp_path/'manifest.jsonl')}
    import ipvg.evaluate
    monkeypatch.setattr(ipvg.evaluate,'evaluate_outputs',lambda *a,**kw:{'ok':True,'samples':[{'media':{'decoded_frames':49}}]})
    return result,profile,entry


def test_real_ip_loading_and_token_call_contract_audited(tmp_path,monkeypatch):
    _,_,entry=evidence(tmp_path,monkeypatch)
    assert module.generation_evidence(tmp_path,entry)[-1]==.01


@pytest.mark.parametrize('error',['source','keys','calls','ip_tokens','checkpoint','video_hash','budget','missing_observations'])
def test_invalid_native_execution_evidence_rejected(error,tmp_path,monkeypatch):
    result,profile,entry=evidence(tmp_path,monkeypatch)
    if error=='source':result['source_sha256']='drift'
    elif error=='keys':profile['matched_ip_tensors']=239
    elif error=='calls':profile['samples'][0]['native_denoiser_calls']=39
    elif error=='ip_tokens':profile['samples'][0]['ip_q_lora_calls']=[[1,1,5120]]
    elif error=='checkpoint':profile['standin_checkpoint_sha256']='drift'
    elif error=='video_hash':(tmp_path/'outputs/one.mp4').write_bytes(b'changed')
    elif error=='budget':result['estimated_backend_gpu_hours']=.5
    elif error=='missing_observations':profile['samples']=[]
    write_json(tmp_path/'result.json',result);write_json(tmp_path/'backend_metrics.json',profile)
    with pytest.raises(ValueError):module.generation_evidence(tmp_path,entry)
