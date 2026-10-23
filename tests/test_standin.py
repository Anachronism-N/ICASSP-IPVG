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


class MetadataTensor:
    def __init__(self,length,batch=1,dim=4,device='cpu',dtype='float32'):
        self.shape=(batch,length,dim);self.device=device;self.dtype=dtype

    def __getitem__(self,key):
        raise AssertionError('Native mode must not slice or operate on tensors')


class FakeAttention:
    def __init__(self):
        self.calls=[];self.result=object()

    def forward(self,q,k,v):
        self.calls.append((q,k,v))
        return self.result

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)


def fake_blocks():
    from types import SimpleNamespace
    return [SimpleNamespace(self_attn=SimpleNamespace(cond_size=1024,kv_cache=None,attn=FakeAttention())) for _ in range(40)]


def test_attention_mode_defaults_and_rejects_unknown():
    r=request();module.validate_request(r)
    assert r['parameters'].get('attention_mode','native')=='native'
    for mode in ('native','partition_balanced'):
        r['parameters']['attention_mode']=mode;module.validate_request(r)
    r['parameters']['attention_mode']='joint_scaled'
    with pytest.raises(ValueError,match='attention_mode'):module.validate_request(r)
    with pytest.raises(ValueError,match='attention_mode'):
        with module.attention_partitioning(fake_blocks(),'joint_scaled'):pass


@pytest.mark.parametrize('fa3,fa2,sage,expected',[(False,True,False,'FA2'),(False,False,False,'SDPA'),
                                                             (False,False,True,'SAGE'),(True,True,True,'FA3')])
def test_actual_kernel_backend_is_checked_without_torch(fa3,fa2,sage,expected):
    from types import SimpleNamespace
    upstream=SimpleNamespace(FLASH_ATTN_3_AVAILABLE=fa3,FLASH_ATTN_2_AVAILABLE=fa2,SAGE_ATTN_AVAILABLE=sage)
    assert module.validate_attention_backend(upstream)==expected
    if expected=='FA2':assert module.validate_attention_backend(upstream,'FA2')=='FA2'
    else:
        with pytest.raises(ValueError,match='attention backend'):module.validate_attention_backend(upstream,'FA2')
    r=request();r['parameters']['expected_attention_backend']='FA2';module.validate_request(r)
    r['parameters']['expected_attention_backend']='unknown'
    with pytest.raises(ValueError,match='expected_attention_backend'):module.validate_request(r)


def test_native_40blocks_cfg_cache_counts_restore_and_sample_reset():
    blocks=fake_blocks();q=MetadataTensor(3);full=MetadataTensor(1027);ip=MetadataTensor(1024)
    for _ in range(2):
        with module.attention_partitioning(blocks) as stats:
            for call in range(40):
                for block in blocks:
                    owner=block.self_attn
                    if call==0:
                        owner.kv_cache=None
                        assert owner.attn(ip,ip,ip) is owner.attn.result
                        owner.kv_cache={'k_ip':ip,'v_ip':ip}
                    # Positive and negative CFG calls both read the original cached IP.
                    assert owner.kv_cache['k_ip'] is ip
                    assert owner.attn(q=q,k=full,v=full) is owner.attn.result
        module.validate_attention_stats(stats,'native')
        assert {name:stats[name] for name in ('joint_video_reads','ip_self_reads','plain_reads','kernel_calls','partition_reads')}==dict(joint_video_reads=1600,ip_self_reads=40,plain_reads=0,kernel_calls=1640,partition_reads=0)
        assert all('forward' not in vars(block.self_attn.attn) for block in blocks)
        assert all(block.self_attn.attn.calls[-1]==(q,full,full) for block in blocks)


@pytest.mark.parametrize('mode',['native','partition_balanced'])
def test_no_ip_is_original_call_without_tensor_ops(mode):
    blocks=fake_blocks();blocks[0].self_attn.cond_size=None
    q=MetadataTensor(3);k=MetadataTensor(7)
    with module.attention_partitioning(blocks,mode) as stats:
        assert blocks[0].self_attn.attn(q,k,k) is blocks[0].self_attn.attn.result
    assert {name:stats[name] for name in ('joint_video_reads','ip_self_reads','plain_reads','kernel_calls','partition_reads')}==dict(joint_video_reads=0,ip_self_reads=0,plain_reads=1,kernel_calls=1,partition_reads=0)


@pytest.mark.parametrize('mode',['native','partition_balanced'])
@pytest.mark.parametrize('bad',['cond_size','rank','batch','dim','device','dtype','kv_length','unexpected_read','empty_main'])
def test_conditioned_invalid_shape_rejects_before_kernel_and_restores(mode,bad):
    blocks=fake_blocks();owner=blocks[0].self_attn
    q=MetadataTensor(3);k=MetadataTensor(1027);v=MetadataTensor(1027)
    if bad=='cond_size':owner.cond_size=32
    elif bad=='rank':q.shape=(3,4)
    elif bad=='batch':v.shape=(2,1027,4)
    elif bad=='dim':k.shape=(1,1027,8)
    elif bad=='device':k.device='different'
    elif bad=='dtype':v.dtype='float64'
    elif bad=='kv_length':v.shape=(1,1026,4)
    elif bad=='unexpected_read':k.shape=v.shape=(1,3,4)
    else:q.shape=(1,0,4);k.shape=v.shape=(1,1024,4)
    with pytest.raises(ValueError,match='Stand-In attention'):
        with module.attention_partitioning(blocks,mode) as stats:owner.attn(q,k,v)
    assert owner.attn.calls==[]
    assert stats['kernel_calls']==0
    assert all('forward' not in vars(block.self_attn.attn) for block in blocks)


def test_original_instance_forward_and_owner_closures_restore_on_exception():
    blocks=fake_blocks();saved=lambda q,k,v:'instance forward'
    blocks[0].self_attn.attn.forward=saved
    blocks[0].self_attn.cond_size=None
    q=MetadataTensor(3);full=MetadataTensor(1027)
    with pytest.raises(RuntimeError,match='pipeline failed'):
        with module.attention_partitioning(blocks) as stats:
            assert blocks[0].self_attn.attn(q,q,q)=='instance forward'
            assert blocks[1].self_attn.attn(q,full,full) is blocks[1].self_attn.attn.result
            raise RuntimeError('pipeline failed')
    assert blocks[0].self_attn.attn.forward is saved
    assert all('forward' not in vars(block.self_attn.attn) for block in blocks[1:])
    assert stats['blocks'][0]['plain_reads']==1 and stats['blocks'][0]['joint_video_reads']==0
    assert stats['blocks'][1]['joint_video_reads']==1 and stats['blocks'][1]['plain_reads']==0
    assert stats['kernel_calls']==2


def test_partial_install_failure_restores_preceding_blocks():
    blocks=fake_blocks();del blocks[10].self_attn.attn
    with pytest.raises(AttributeError):
        with module.attention_partitioning(blocks):pass
    assert all('forward' not in vars(block.self_attn.attn) for block in blocks[:10])
    with pytest.raises(ValueError,match='exactly40'):
        with module.attention_partitioning(blocks[:39]):pass


def sdpa_blocks(torch):
    from types import SimpleNamespace
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=[]
        def forward(self,q,k,v):
            self.calls.append((q,k,v))
            return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1)).squeeze(1)
    return [SimpleNamespace(self_attn=SimpleNamespace(cond_size=1024,attn=Attention())) for _ in range(40)]


def test_torch_native_exact_module_binding_ip_self_and_no_new_parameters():
    torch=pytest.importorskip('torch');blocks=sdpa_blocks(torch);attn=blocks[0].self_attn.attn
    generator=torch.Generator().manual_seed(17)
    q=torch.randn(1,3,4,generator=generator);k=torch.randn(1,1027,4,generator=generator);v=torch.randn(1,1027,4,generator=generator)
    ip=torch.randn(1,1024,4,generator=generator)
    expected=attn(q,k,v);expected_ip=attn(ip,ip,ip)
    parameters=list(attn.named_parameters());state=dict(attn.state_dict())
    with module.attention_partitioning(blocks) as stats:
        assert torch.equal(attn(q,k,v),expected)
        assert torch.equal(attn(q=ip,k=ip,v=ip),expected_ip)
        assert list(attn.named_parameters())==parameters and dict(attn.state_dict())==state
    assert 'forward' not in vars(attn)
    assert stats['joint_video_reads']==1 and stats['ip_self_reads']==1


def test_torch_balanced_matches_analytic_partition_not_joint_and_keeps_strides():
    torch=pytest.importorskip('torch');blocks=sdpa_blocks(torch);attn=blocks[0].self_attn.attn
    # Non-contiguous inputs and unequal partition sizes expose joint-softmax dilution.
    q=torch.zeros(2,3,8)[:,:,::2];k=torch.zeros(2,1027,8)[:,:,::2]
    v=torch.zeros(2,1027,8)[:,:,::2];v[:,3:]=2
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    expected=0.5*(attn(q,k[:,:3],v[:,:3])+attn(q,k[:,3:],v[:,3:]))
    joint=attn(q,k,v);attn.calls.clear()
    with module.attention_partitioning(blocks,'partition_balanced') as stats:actual=attn(q,k,v)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(actual,torch.ones_like(actual),rtol=0,atol=0)
    assert torch.isfinite(actual).all() and not torch.equal(actual,joint)
    assert len(attn.calls)==2
    for query,key,value in attn.calls:
        assert query is q and key.stride()==k.stride() and value.stride()==v.stride()
        assert key.untyped_storage().data_ptr()==k.untyped_storage().data_ptr()
        assert value.untyped_storage().data_ptr()==v.untyped_storage().data_ptr()
    assert {name:stats[name] for name in ('joint_video_reads','ip_self_reads','plain_reads','kernel_calls','partition_reads')}==dict(joint_video_reads=1,ip_self_reads=0,plain_reads=0,kernel_calls=2,partition_reads=1)


def test_torch_balanced_matches_explicit_random_softmax_decomposition():
    torch=pytest.importorskip('torch');blocks=sdpa_blocks(torch);attn=blocks[0].self_attn.attn
    generator=torch.Generator().manual_seed(29)
    q=torch.randn(1,3,4,generator=generator)
    k=torch.randn(1,1027,4,generator=generator);v=torch.randn(1,1027,4,generator=generator)
    video_weights=torch.softmax(q@k[:,:3].transpose(-1,-2)/2,dim=-1)
    ip_weights=torch.softmax(q@k[:,3:].transpose(-1,-2)/2,dim=-1)
    expected=0.5*(video_weights@v[:,:3]+ip_weights@v[:,3:])
    joint=attn(q,k,v)
    with module.attention_partitioning(blocks,'partition_balanced'):actual=attn(q,k,v)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    assert torch.isfinite(actual).all() and not torch.allclose(actual,joint)


def test_torch_candidate_ip_self_unchanged_and_kernel_exception_restores():
    torch=pytest.importorskip('torch');blocks=sdpa_blocks(torch);attn=blocks[0].self_attn.attn
    ip=torch.zeros(1,1024,2);expected=attn(ip,ip,ip)
    with module.attention_partitioning(blocks,'partition_balanced') as stats:
        assert torch.equal(attn(ip,ip,ip),expected)
    assert stats['ip_self_reads']==1 and stats['partition_reads']==0
    old_forward=attn.forward
    def fail_second(q,k,v):
        if k.shape[1]==1024:raise RuntimeError('IP kernel failed')
        return old_forward(q,k,v)
    attn.forward=fail_second;q=torch.zeros(1,3,2);full=torch.zeros(1,1027,2)
    with pytest.raises(RuntimeError,match='IP kernel failed'):
        with module.attention_partitioning(blocks,'partition_balanced') as stats:attn(q,full,full)
    assert attn.forward is fail_second
    assert stats['kernel_calls']==2 and stats['partition_reads']==1
    assert all('forward' not in vars(block.self_attn.attn) for block in blocks[1:])


def test_candidate_40blocks_cfg_counts_with_original_kernel_fakes():
    blocks=fake_blocks()
    class SliceTensor(MetadataTensor):
        def __getitem__(self,key):
            start,stop,step=key[1].indices(self.shape[1])
            return SliceTensor(len(range(start,stop,step)))
    for block in blocks:
        block.self_attn.attn.forward=lambda q,k,v:float(k.shape[1])
    saved=[block.self_attn.attn.forward for block in blocks]
    q=SliceTensor(3);full=SliceTensor(1027);ip=SliceTensor(1024)
    with module.attention_partitioning(blocks,'partition_balanced') as stats:
        for step in range(20):
            for branch in ('positive','negative'):
                for block in blocks:
                    owner=block.self_attn
                    if step==0 and branch=='positive':
                        assert owner.attn(ip,ip,ip)==1024
                        owner.kv_cache={'k_ip':ip,'v_ip':ip}
                    assert owner.kv_cache['k_ip'] is ip and owner.kv_cache['v_ip'] is ip
                    assert owner.attn(q,full,full)==0.5*(3+1024)
    module.validate_attention_stats(stats,'partition_balanced')
    assert {name:stats[name] for name in ('joint_video_reads','ip_self_reads','plain_reads','kernel_calls','partition_reads')}==dict(joint_video_reads=1600,ip_self_reads=40,plain_reads=0,kernel_calls=3240,partition_reads=1600)
    assert all(block.self_attn.attn.forward is old for block,old in zip(blocks,saved))
    stats['blocks'][0]['joint_video_reads']-=1
    with pytest.raises(RuntimeError,match='structure evidence'):module.validate_attention_stats(stats,'partition_balanced')
