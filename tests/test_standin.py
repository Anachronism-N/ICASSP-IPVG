import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('standin_adapter',Path(__file__).resolve().parents[1]/'scripts/adapters/standin.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def request():
    return {'task':'inference','resources':{'gpus':1},'parameters':{'width':480,'height':352,'num_frames':49,
      'fps':16,'num_inference_steps':20,'cfg_scale':5,'seed':42},'samples':[{'track':1,'reference_images':['original.png']}]}


def test_native_benchmark_spec_is_fixed():
    module.validate_request(request())
    for key,value in [('width',512),('num_frames',48),('seed',True),('cfg_scale',True)]:
        r=request();r['parameters'][key]=value
        with pytest.raises(ValueError):module.validate_request(r)
    r=request();r['samples'][0]['control_video_path']='target.mp4'
    with pytest.raises(ValueError):module.validate_request(r)
    r=request();r['samples'][0]['reference_images'].append('other.png')
    with pytest.raises(ValueError):module.validate_request(r)


def test_native_state_requires_complete_all240pairs():
    class Tensor:
        shape=(128,5120)
    class FakeTorch:
        @staticmethod
        def is_tensor(t):return isinstance(t,Tensor)
        @staticmethod
        def isfinite(t):return type('Finite',(),{'all':lambda self:True})()
    with pytest.raises(ValueError):module.validate_state({},FakeTorch)
    values={}
    for i in range(40):
        for name in ('q_loras','k_loras','v_loras'):
            for part in ('down','up'):
                t=Tensor();t.shape=(128,5120) if part=='down' else (5120,128)
                values[f'blocks.{i}.self_attn.{name}.{part}.weight']=t
    assert len(module.validate_state(values,FakeTorch))==240
    values.pop(next(iter(values)))
    with pytest.raises(ValueError):module.validate_state(values,FakeTorch)


def test_model_lock_checks_cached_weights_not_only_file_exists(tmp_path):
    from ipvg.common import sha256_file
    base=tmp_path/'base';base.mkdir();weights=tmp_path/'weights';weights.mkdir()
    parser=tmp_path/'parser';parser.write_bytes(b'parser')
    detector=tmp_path/'det';detector.write_bytes(b'det')
    (base/'config').write_bytes(b'config')
    (weights/module.CHECKPOINT).write_bytes(b'changed')
    lock={'wan_files':{'config':sha256_file(base/'config')},'standin_files':{module.CHECKPOINT:sha256_file(weights/module.CHECKPOINT)},
          'reuse_files':{'config':sha256_file(base/'config')},'tokenizer_files':{},
          'parser_sha256':sha256_file(parser),'detector_sha256':sha256_file(detector)}
    with pytest.raises(ValueError,match='checkpoint differs'):module.verify_lock(tmp_path,base,weights,parser,detector,lock)
