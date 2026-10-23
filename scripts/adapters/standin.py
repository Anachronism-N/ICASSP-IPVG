"""Pinned native Stand-In/Wan2.1-14B inference, with local-only preprocessing."""
import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from ipvg.common import load_json, write_json, resolve_path, sha256_file

REVISION='e351224366be169076e94af1454115d91d458313'
CHECKPOINT='Stand-In_wan2.1_T2V_14B.ckpt'
CHECKPOINT_SHA='52f47630b4fefbdac841d3fd1ce146cca2a6a94ec0a6ab9407d16f2f635177e9'
NEGATIVE='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
ATTENTION_MECHANISMS={
    'native':'original joint video/IP softmax; original IP self-read',
    'partition_balanced':'0.5*(original video-only softmax + original IP-only softmax) for joint video reads; original IP self-read',
}


def validate_request(request):
    p=request['parameters']
    if request.get('task')!='inference' or request['resources']['gpus']!=1:
        raise ValueError('Stand-In adapter only supports single-GPU inference')
    for k,v in [('width',480),('height',352),('num_frames',49),('fps',16),('num_inference_steps',20),('cfg_scale',5)]:
        if isinstance(p.get(k),bool) or p.get(k)!=v:raise ValueError('Unverified Stand-In specification: '+k)
    if any(r['track']!=1 or len(r['reference_images'])!=1 or r.get('control_video_path') for r in request['samples']):
        raise ValueError('One original Track1 reference only; no target/control video')
    seed=p.get('seed',42)
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**31:raise ValueError('Invalid seed')
    if p.get('attention_mode','native') not in ATTENTION_MECHANISMS:raise ValueError('Invalid attention_mode')
    if 'expected_attention_backend' in p and p['expected_attention_backend'] not in ('FA2','SDPA'):
        raise ValueError('Invalid expected_attention_backend')


def validate_state(state, torch):
    if not isinstance(state,dict):raise ValueError('Checkpoint state must be a dictionary')
    expected={f'blocks.{i}.self_attn.{name}.{part}.weight':(128,5120) if part=='down' else (5120,128)
              for i in range(40) for name in ('q_loras','k_loras','v_loras') for part in ('down','up')}
    if set(state)!=set(expected):raise ValueError('Stand-In checkpoint must contain exactly240 IP-branch tensors')
    for k,shape in expected.items():
        if not torch.is_tensor(state[k]) or tuple(state[k].shape)!=shape or not torch.isfinite(state[k]).all():
            raise ValueError('Invalid Stand-In tensor: '+k)
    return expected


def verify_lock(root, base, weights, parser, detector, lock):
    for field,folder in [('wan_files',base),('standin_files',weights),('reuse_files',base)]:
        if not lock.get(field):raise ValueError('Incomplete native model lock')
        for name,digest in lock[field].items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(folder/name)!=digest:
                raise ValueError('Model file hash mismatch: '+name)
    for name,digest in lock['tokenizer_files'].items():
        if sha256_file(base/'google/umt5-xxl'/name)!=digest:raise ValueError('Tokenizer hash mismatch')
    if sha256_file(parser)!=lock['parser_sha256'] or sha256_file(detector)!=lock['detector_sha256']:
        raise ValueError('Preprocessor hash mismatch')
    if sha256_file(weights/CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Stand-In checkpoint differs')


def validate_attention_backend(module, expected=None):
    backend=('FA3' if module.FLASH_ATTN_3_AVAILABLE else 'FA2' if module.FLASH_ATTN_2_AVAILABLE
             else 'SAGE' if module.SAGE_ATTN_AVAILABLE else 'SDPA')
    if expected is not None and backend!=expected:raise ValueError('Unexpected Stand-In attention backend: '+backend)
    return backend


@contextmanager
def attention_partitioning(blocks, attention_mode='native'):
    if attention_mode not in ATTENTION_MECHANISMS:raise ValueError('Invalid attention_mode')
    if len(blocks)!=40:raise ValueError('Require exactly40 Stand-In blocks')
    names=('joint_video_reads','ip_self_reads','plain_reads','kernel_calls','partition_reads')
    stats={'mode':attention_mode,'block_count':40,
           'blocks':[dict(block_index=i,**{name:0 for name in names}) for i in range(40)]}
    previous=[]

    def wrap(owner, original, counts):
        def forward(q,k,v):
            cond_size=owner.cond_size
            if cond_size is None:
                counts['plain_reads']+=1;counts['kernel_calls']+=1
                return original(q,k,v)
            shapes=q.shape,k.shape,v.shape
            if (cond_size!=1024 or any(len(shape)!=3 for shape in shapes)
                    or any(shape[0]!=shapes[0][0] or shape[2]!=shapes[0][2] for shape in shapes[1:])
                    or k.device!=q.device or v.device!=q.device or k.dtype!=q.dtype or v.dtype!=q.dtype
                    or k.shape[1]!=v.shape[1]):
                raise ValueError('Invalid conditioned Stand-In attention shape/device/dtype')
            main=q.shape[1]
            if main==1024 and k.shape[1]==1024:
                counts['ip_self_reads']+=1
            elif main>0 and k.shape[1]==main+1024:
                counts['joint_video_reads']+=1
                if attention_mode=='partition_balanced':
                    counts['kernel_calls']+=1;counts['partition_reads']+=1
                    video=original(q,k[:,:main],v[:,:main])
                    counts['kernel_calls']+=1
                    ip=original(q,k[:,main:],v[:,main:])
                    return 0.5*(video+ip)
            else:
                raise ValueError('Unexpected conditioned Stand-In attention read')
            counts['kernel_calls']+=1
            return original(q,k,v)
        return forward

    try:
        for block,counts in zip(blocks,stats['blocks']):
            owner=block.self_attn;attn=owner.attn
            had_forward='forward' in vars(attn)
            old_forward=vars(attn).get('forward')
            original=attn.forward
            previous.append((attn,had_forward,old_forward))
            attn.forward=wrap(owner,original,counts)
        yield stats
    finally:
        for attn,had_forward,old_forward in reversed(previous):
            if had_forward:attn.forward=old_forward
            else:delattr(attn,'forward')
        stats.update({name:sum(block[name] for block in stats['blocks']) for name in names})


def validate_attention_stats(stats, attention_mode):
    expected={'joint_video_reads':40,'ip_self_reads':1,'plain_reads':0,
              'kernel_calls':81 if attention_mode=='partition_balanced' else 41,
              'partition_reads':40 if attention_mode=='partition_balanced' else 0}
    if (stats['mode']!=attention_mode or stats['block_count']!=40 or len(stats['blocks'])!=40
            or any(block['block_index']!=i or any(block[name]!=value for name,value in expected.items()) for i,block in enumerate(stats['blocks']))):
        raise RuntimeError('Stand-In attention structure evidence failed')
    if any(stats[name]!=40*value for name,value in expected.items()):
        raise RuntimeError('Stand-In attention totals evidence failed')


@contextmanager
def native_preprocessing(module, detector_file, parsing_file):
    import torch
    import onnxruntime as ort
    from insightface.model_zoo.scrfd import SCRFD
    from insightface.app.common import Face
    from facexlib.parsing.bisenet import BiSeNet
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(detector_file),sess_options=options,providers=['CPUExecutionProvider'])
    detector=SCRFD(model_file=str(detector_file),session=session)
    class App:
        def prepare(self,ctx_id=0,det_size=(640,640)):
            detector.prepare(ctx_id=ctx_id,input_size=det_size,det_thresh=.5)
        def get(self,image):
            boxes,kps=detector.detect(image,max_num=0,metric='default')
            return [Face(bbox=box[:4],kps=point,det_score=box[4]) for box,point in zip(boxes,kps)]
    def face_factory(**kwargs):return App()
    def parser_factory(model_name,device):
        if model_name!='bisenet':raise ValueError('Only nativeBiSeNet parser supported')
        model=BiSeNet(num_class=19)
        model.load_state_dict(torch.load(str(parsing_file),map_location='cpu',weights_only=True),strict=True)
        return model.to(device).eval()
    previous=module.FaceAnalysis,module.init_parsing_model
    module.FaceAnalysis=face_factory;module.init_parsing_model=parser_factory
    try:yield
    finally:module.FaceAnalysis,module.init_parsing_model=previous


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);a=p.parse_args()
    request=load_json(a.request);validate_request(request)
    params=request['parameters'];root=request['repo_root']
    attention_mode=params.get('attention_mode','native')
    upstream=resolve_path(params['upstream_dir'],root)
    revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if revision!=REVISION:raise ValueError('Stand-In native code revision differs')
    subprocess.run(['git','-C',str(upstream),'diff','--quiet','HEAD'],check=True)
    base=resolve_path(params['base_dir'],root);weights=resolve_path(params['standin_dir'],root)
    parser=resolve_path(params['parser_path'],root);detector=resolve_path(params['detector_path'],root)
    lock=load_json(resolve_path(params['model_lock'],root));verify_lock(root,base,weights,parser,detector,lock)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',TOKENIZERS_PARALLELISM='false')
    import torch
    import numpy as np
    import numpy.testing
    from PIL import Image
    torch.set_num_threads(4)
    if torch.cuda.device_count()!=1:raise ValueError('Require one visible A100GPU')
    # This optional native FA3 build targets Hopper, not A100; leave original FA2/SDPA fallback unchanged.
    saved=sys.modules.get('flash_attn_interface');sys.modules['flash_attn_interface']=None
    sys.path.insert(0,str(upstream))
    try:
        from wan_loader import load_wan_pipe
        from models.set_condition_branch import set_stand_in
        from models import wan_video_dit
        from preprocessor import image_input_preprocessor as processor_module
    finally:
        if saved is None:sys.modules.pop('flash_attn_interface',None)
        else:sys.modules['flash_attn_interface']=saved
    backend=validate_attention_backend(wan_video_dit,params.get('expected_attention_backend'))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from mm2026_quality_eval import offline_only
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    with offline_only():
        state=torch.load(str(weights/CHECKPOINT),map_location='cpu',weights_only=True)
        state=state.get('state_dict',state);expected=validate_state(state,torch)
        with native_preprocessing(processor_module,detector,parser):
            face_processor=processor_module.FaceProcessor(device=torch.device('cpu'))
        pipe=load_wan_pipe(str(base),wan_version='2.1',torch_dtype=torch.bfloat16)
        set_stand_in(pipe,model_path=str(weights),wan_version='2.1')
        matched=0
        for k in expected:
            block=k.split('.')[1];name=k.split('.')[3];part=k.split('.')[4]
            parameter=getattr(getattr(pipe.dit.blocks[int(block)].self_attn,name),part).weight
            if not torch.equal(parameter.detach().cpu(),state[k].to(dtype=parameter.dtype)):
                raise RuntimeError('Native IP-branch weights not loaded: '+k)
            matched+=1
        del state
    report={'backend':'Stand-In native Wan2.1-T2V14B','code_revision':revision,'model_lock_sha256':sha256_file(resolve_path(params['model_lock'],root)),
            'standin_checkpoint_sha256':CHECKPOINT_SHA,'matched_ip_tensors':matched,'native_vram_management':True,
            'attention_backend':backend,
            'attention_mode':attention_mode,'mechanism':ATTENTION_MECHANISMS[attention_mode],
            'preprocessor':'native1.5bbox/whitepad512/BiSeNet; identicalSCRFD detector-only CPUORT4threads',
            'loading_seconds':time.monotonic()-started,'samples':[]}
    path=Path(request['run_dir'])/'backend_metrics.json';write_json(path,report)
    calls=[];q_calls=[]
    original=pipe.model_fn
    def observe_fn(**kw):
        ip=kw.get('ip_image');calls.append({'ip_latent_shape':list(ip.shape) if ip is not None else None})
        return original(**kw)
    pipe.model_fn=observe_fn
    handle=pipe.dit.blocks[0].self_attn.q_loras.register_forward_pre_hook(lambda m,args:q_calls.append(list(args[0].shape)))
    try:
        for row in request['samples']:
            ref=Path(row['reference_images'][0]);before=sha256_file(ref)
            with offline_only():
                with Image.open(ref) as image:ip=face_processor.process(image.convert('RGB'))
            if ip.size!=(512,512):raise ValueError('Nativereferenceprocessingnot512square')
            seed=(params.get('seed',42)+int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16))%2**31
            sample={'sample_id':row['sample_id'],'seed':seed,'reference_original_sha256':before,
                    'reference_pixels_sha256':hashlib.sha256(ip.tobytes()).hexdigest()}
            report['samples'].append(sample)
            calls.clear();q_calls.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
            stats=None
            try:
                with attention_partitioning(pipe.dit.blocks,attention_mode) as stats:
                    with offline_only():
                        frames=pipe(prompt=row['prompt'],negative_prompt=params.get('negative_prompt',NEGATIVE),ip_image=ip,
                                    seed=seed,rand_device='cpu',width=480,height=352,num_frames=49,num_inference_steps=20,
                                    cfg_scale=5,tiled=True)
                torch.cuda.synchronize();seconds=time.monotonic()-start
            finally:
                sample.update(native_denoiser_calls=len(calls),
                              ip_latent_calls=[c for c in calls if c['ip_latent_shape'] is not None],
                              ip_q_lora_calls=list(q_calls),attention_structure=stats)
                write_json(path,report)
            validate_attention_stats(stats,attention_mode)
            if len(frames)!=49 or any(np.asarray(f).shape!=(352,480,3) for f in frames):raise ValueError('Wrong outputframes/dimensions')
            if (len(calls)!=40 or sum(c['ip_latent_shape'] is not None for c in calls)!=1
                    or not q_calls or q_calls[0]!=[1,1024,5120]):
                raise RuntimeError('NativeIPconditionalbranch/CFGstep evidence failed')
            if sha256_file(ref)!=before:raise ValueError('Originalreferencechanged')
            from data.video import save_video
            output=Path(request['output_dir'])/(row['sample_id']+'.mp4')
            save_video(frames,str(output),fps=16,quality=5)
            sample.update(generated_frames=49,generation_seconds=seconds,
                          peak_allocated_vram_gb=torch.cuda.max_memory_allocated()/1e9,
                          peak_reserved_vram_gb=torch.cuda.max_memory_reserved()/1e9)
            write_json(path,report)
    finally:
        handle.remove();pipe.model_fn=original


if __name__=='__main__':main()
