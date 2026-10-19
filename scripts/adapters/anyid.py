"""Thin single-GPU adapter for the pinned AnyID preview; no upstream code vendoring."""
import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from ipvg.common import load_json,resolve_path,write_json

UPSTREAM_REVISION='6c714f71af6e08d2af29aecd74a3a9fd9e8f94a3'


def reference_latent_lengths(output_frames):
    if isinstance(output_frames,bool) or not isinstance(output_frames,int) or output_frames<1 or output_frames%4!=1:
        raise ValueError('Output frames must be 4k+1')
    return [1,(output_frames-1)//4+1]


def text_encoder_on_cpu(parameters):
    value=parameters.get('t5_cpu',True)
    if not isinstance(value,bool):
        raise ValueError('t5_cpu must be a JSON boolean')
    return value


class TimedTextEncoder:
    """Delegate the original encoder unchanged while recording synchronized wall time."""
    def __init__(self, encoder, synchronize):
        self.encoder=encoder
        self.synchronize=synchronize
        self.calls=[]

    def __getattr__(self, name):
        return getattr(self.encoder,name)

    def __call__(self, texts, device):
        self.synchronize()
        started=time.monotonic()
        result=self.encoder(texts,device)
        self.synchronize()
        self.calls.append({'device':str(device),'elapsed_seconds':time.monotonic()-started,
                           'context_shapes':[list(context.shape) for context in result]})
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    args=parser.parse_args()
    request=load_json(args.request)
    if request.get('task')!='inference' or request['resources']['gpus']!=1:
        raise ValueError('AnyID adapter supports single-GPU inference only')
    parameters=request['parameters']
    t5_cpu=text_encoder_on_cpu(parameters)
    upstream=resolve_path(parameters['upstream_dir'],request['repo_root'])
    revision=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!=UPSTREAM_REVISION: raise ValueError('AnyID upstream revision mismatch')
    base=resolve_path(parameters['base_dir'],request['repo_root'])
    merged=resolve_path(parameters['merged_dit'],request['repo_root'])
    if base==merged or not (merged/'config.json').is_file():
        raise ValueError('Require a separate premerged AnyID DiT directory')
    for row in request['samples']:
        if len(row['reference_images'])!=1 or row.get('control_video_path'):
            raise ValueError('Compare one image only, without target/control videos')
    sys.path[:0]=[str(upstream/'trainer/models/wan22'),str(upstream)]
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    import imageio.v2 as imageio
    from PIL import Image
    import wan
    from wan.configs import WAN_CONFIGS
    from generate import process_image
    width,height=parameters.get('width',480),parameters.get('height',480)
    if width%32 or height%32 or width<32 or height<32:
        raise ValueError('AnyID dimensions must be positive multiples of 32')
    output_frames=parameters.get('num_frames',49)
    inner_t=reference_latent_lengths(output_frames)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started=time.monotonic()
    pipeline=wan.WanVPT2V(config=WAN_CONFIGS['ti2v-5B'],checkpoint_dir=str(base),device_id=0,rank=0,
                         t5_cpu=t5_cpu,convert_model_dtype=True,dit_dir=str(merged))
    torch.cuda.synchronize()
    report={'backend':'AnyID Wan2.2 TI2V-5B preview','upstream_revision':revision,
            'loading_seconds':time.monotonic()-started,'samples':[],
            't5_cpu':t5_cpu,'offload_model':True,'kind':'resource_profile_not_quality_score'}
    pipeline.text_encoder=TimedTextEncoder(pipeline.text_encoder,torch.cuda.synchronize)
    metrics_path=Path(request['run_dir'])/'backend_metrics.json'
    write_json(metrics_path,report)
    output=Path(request['output_dir'])
    output.mkdir(exist_ok=True,parents=True)
    for row in request['samples']:
        seed=(int(parameters.get('seed',42))+int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16))%2**31
        with Image.open(row['reference_images'][0]) as image:
            reference=process_image(image.convert('RGB'),height,width)
        pipeline.text_encoder.calls.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        generated_at=time.monotonic()
        video=pipeline.generate(row['prompt'],img=[reference],size=(width,height),max_area=width*height,
                                frame_num=output_frames+4,inner_t=inner_t,shift=parameters.get('sample_shift',3.0),
                                sampling_steps=parameters.get('num_inference_steps',20),
                                guide_scale=parameters.get('cfg_scale',5.0),seed=seed,offload_model=True,
                                n_prompt=parameters.get('negative_prompt',''))
        torch.cuda.synchronize()
        elapsed=time.monotonic()-generated_at
        if tuple(video.shape)!=(3,output_frames,height,width) or not torch.isfinite(video).all():
            raise RuntimeError('AnyID returned invalid video dimensions or values')
        frames=((video.detach().float().clamp(-1,1)+1)*127.5).to(torch.uint8).permute(1,2,3,0).cpu().numpy()
        path=output/(row['sample_id']+'.mp4')
        with imageio.get_writer(str(path),fps=parameters.get('fps',16),codec='libx264',quality=5) as writer:
            for frame in frames: writer.append_data(frame)
        text_calls=list(pipeline.text_encoder.calls)
        if len(text_calls)!=2:
            raise RuntimeError('Expected exactly two native positive/negative text encodings')
        report['samples'].append({'sample_id':row['sample_id'],'seed':seed,'generated_frames':len(frames),
                                  'generation_seconds':elapsed,'text_encoding_calls':text_calls,
                                  'text_encoding_seconds':sum(call['elapsed_seconds'] for call in text_calls),
                                  'peak_allocated_vram_gb':torch.cuda.max_memory_allocated()/1e9,
                                  'peak_reserved_vram_gb':torch.cuda.max_memory_reserved()/1e9})
        write_json(Path(request['run_dir'])/'backend_metrics.json',report)


if __name__=='__main__':
    main()
