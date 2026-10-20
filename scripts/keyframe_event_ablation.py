"""Fixed-budget sparse AMT reconstruction; independent KeyID-inspired selection component only."""
import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ipvg.common import load_json,sha256_file,write_json


def interval_error(frames,left,right):
    if right<=left:raise ValueError('Interval endpoints must increase')
    ids=np.arange(left,right+1)
    phase=(ids-left)/(right-left)
    predicted=(1-phase[:,None,None,None])*frames[left]+phase[:,None,None,None]*frames[right]
    return float(np.abs(frames[left:right+1]-predicted).mean())*(right-left)


def select_keyframes(frames,boundary=16,mode='uniform'):
    if len(frames)!=33 or boundary!=16:raise ValueError('Require fixed33frame two-event probe with boundary16')
    if mode=='uniform':return [0,8,16,24,32]
    if mode!='event_error':raise ValueError('Unknown keyframe policy')
    small=np.stack([cv2.resize(f,(32,32),interpolation=cv2.INTER_AREA) for f in frames]).astype(np.float32)
    result=[0,boundary,32]
    for left,right in [(0,boundary),(boundary,32)]:
        chosen=min(range(left+1,right),key=lambda k:(interval_error(small,left,k)+interval_error(small,k,right),k))
        result.append(chosen)
    return sorted(result)


def reconstruction_schedule(keys,count=33):
    if keys!=sorted(set(keys)) or len(keys)!=5 or keys[0]!=0 or keys[-1]!=count-1 or keys[2]!=16:
        raise ValueError('Require five ordered keys preserving endpoints and event boundary16')
    schedule=[]
    for left,right in zip(keys[:-1],keys[1:]):
        for index in range(left+1,right):schedule.append((index,left,right,(index-left)/(right-left)))
    if len(schedule)!=28:raise ValueError('Every nonkey frame must be reconstructed exactly once')
    return schedule


def decode(path):
    cap=cv2.VideoCapture(str(path))
    frames=[]
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        while True:
            ok,frame=cap.read()
            if not ok:break
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    if len(frames)!=33 or abs(fps-8)>1e-6 or any(f.shape!=(192,320,3) for f in frames):
        raise ValueError('Require33frames320x1928fps source')
    return frames


def main():
    from mm2026_quality_eval import validate_local,offline_only,load_amt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request=load_json(args.request)
    if request['resources']['gpus']!=1:raise ValueError('Single allocated GPU required')
    p=request['parameters']
    vbench,amt=Path(p['vbench_dir']),Path(p['amt_path'])
    validate_local(vbench,amt)
    sys.path.insert(0,str(vbench))
    os.environ.update(VBENCH_CACHE_DIR=p['vbench_cache'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    import imageio.v2 as imageio
    torch.set_num_threads(4)
    if torch.cuda.device_count()!=1:raise ValueError('Require one visible GPU')
    # Initialize bundled output encoder before the network/subprocess guard.
    import imageio_ffmpeg
    imageio_ffmpeg.get_ffmpeg_exe()
    report={'status':'running','keyframe_count':5,'interpolator':'AMT-S, not KeyID LTX2.3','samples':[]}
    started=time.monotonic()
    with offline_only():motion=load_amt(vbench,amt)
    metrics_path=Path(request['run_dir'])/'backend_metrics.json'
    write_json(metrics_path,report)
    output=Path(request['output_dir'])
    for row in request['samples']:
        source=Path(row['input_video_path'])
        if sha256_file(source)!=row['input_video_sha256']:raise ValueError('Frozen draft changed')
        frames=decode(source)
        keys=select_keyframes(frames,mode=row['keyframe_mode'])
        schedule=reconstruction_schedule(keys)
        result=[f.copy() if i in keys else None for i,f in enumerate(frames)]
        torch.cuda.reset_peak_memory_stats()
        sample_started=time.monotonic()
        calls=0
        with offline_only(),torch.inference_mode():
            for index,left,right,phase in schedule:
                a=torch.from_numpy(frames[left].copy()).permute(2,0,1).unsqueeze(0).float().to('cuda')/255
                b=torch.from_numpy(frames[right].copy()).permute(2,0,1).unsqueeze(0).float().to('cuda')/255
                t=torch.tensor(phase,device='cuda').reshape(1,1,1,1)
                predicted=motion.model(a,b,t,scale_factor=1.,eval=True)['imgt_pred']
                result[index]=(predicted[0].permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
                calls+=1
        torch.cuda.synchronize()
        if calls!=28 or any(f is None for f in result) or not all(np.array_equal(result[k],frames[k]) for k in keys):
            raise RuntimeError('Sparse reconstruction or anchor preservation failed')
        # Lossless RGB H.264 avoids encoder noise confounding reconstruction error and anchors.
        path=output/(row['sample_id']+'.mp4')
        with imageio.get_writer(str(path),fps=8,codec='libx264rgb',pixelformat='rgb24',macro_block_size=1,
                                ffmpeg_params=['-crf','0','-preset','fast']) as writer:
            for frame in result:writer.append_data(frame)
        final=decode(path)
        if not all(np.array_equal(final[k],frames[k]) for k in keys):raise RuntimeError('Encoded keyframes changed')
        report['samples'].append({'sample_id':row['sample_id'],'base_sample_id':row['base_sample_id'],'group':row['group'],
                                  'mode':row['keyframe_mode'],'keyframe_indices':keys,'model_calls':calls,
                                  'input_sha256':row['input_video_sha256'],'output_sha256':sha256_file(path),
                                  'keyframes_exact_after_encode':True,'elapsed_seconds':time.monotonic()-sample_started,
                                  'peak_allocated_vram_gb':torch.cuda.max_memory_allocated()/1e9})
        write_json(metrics_path,report)
    report.update(status='completed',elapsed_seconds=time.monotonic()-started)
    write_json(metrics_path,report)


if __name__=='__main__':main()
