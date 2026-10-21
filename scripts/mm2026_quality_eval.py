"""Pinned offline VBench quality dimensions used by MM2026 papers, not official final score."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from ipvg.common import load_json,sha256_file,write_json

VBENCH_REVISION='fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
AMT_SHA256='07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77'
MUSIQ_SHA256='358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50'
PROTOCOL={'version':'vbench_mm26_quality_v1','vbench_revision':VBENCH_REVISION,
          'motion_smoothness':'original AMT-S odd-frame reconstruction, 1-MAE/255, all decoded frames',
          'imaging_quality':'original MUSIQ-SPAQ all-frame mean MOS/100, longer-side preprocessing<=512',
          'precision':'float32, single CUDA device','aggregation':'video equal weight; normalized per-video dimensions',
          'official_final_score':None,'human_score':None,
          'missing_components':['CurricularFace author checkpoint','GME','StoryScore','official human evaluation'],
          'warning':'Named paper-related dimensions, not exact KeyID face/text protocol or official challenge score.'}


def normalized_imaging(mos):
    import math
    if isinstance(mos,bool) or not isinstance(mos,(float,int)) or not math.isfinite(mos):
        raise ValueError('Require finite raw MUSIQ MOS')
    return mos/100


@contextmanager
def offline_only():
    def denied(*args,**kwargs):
        raise RuntimeError('Network and subprocess downloads forbidden in offline evaluation')
    targets=[(socket.socket,'connect'),(socket.socket,'connect_ex'),(socket,'create_connection'),
             (urllib.request,'urlopen'),(subprocess,'Popen')]
    original=[(owner,name,getattr(owner,name)) for owner,name in targets]
    try:
        for owner,name,_ in original:setattr(owner,name,denied)
        yield
    finally:
        for owner,name,value in original:setattr(owner,name,value)


def validate_local(vbench,amt,musiq=None):
    revision=subprocess.run(['git','-C',str(vbench),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!=VBENCH_REVISION:raise ValueError('VBench revision differs')
    subprocess.run(['git','-C',str(vbench),'diff','--quiet','HEAD','--','vbench'],check=True)
    if not amt.is_file() or sha256_file(amt)!=AMT_SHA256:raise ValueError('Local AMT weight mismatch')
    if musiq is not None and (not musiq.is_file() or sha256_file(musiq)!=MUSIQ_SHA256):raise ValueError('Local MUSIQ weight mismatch')
    return vbench/'vbench/third_party/amt/cfgs/AMT-S.yaml'


def load_amt(vbench,amt):
    from vbench.motion_smoothness import MotionSmoothness
    return MotionSmoothness(str(vbench/'vbench/third_party/amt/cfgs/AMT-S.yaml'),str(amt),'cuda')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--vbench',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--amt',type=Path,required=True)
    parser.add_argument('--musiq',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose fresh quality score output')
    config=validate_local(args.vbench,args.amt,args.musiq)
    request=load_json(args.manifest)
    rows=request['samples']
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Nonempty unique quality IDs required')
    from ipvg.evaluate import probe_video
    for row in rows:
        path=Path(row['video_path'])
        if not path.is_file() or sha256_file(path)!=row['video_sha256']:raise ValueError('Frozen quality input hash changed')
        if not probe_video(path)['ok']:raise ValueError('Invalid quality media')
    sys.path.insert(0,str(args.vbench))
    os.environ.update(VBENCH_CACHE_DIR=str(args.cache),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import torch
    import numpy.testing  # Initialize NumPy's local CPU-capability probe before subprocess lockdown.
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Require exactly one visible CUDA device')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started=time.monotonic()
    records=[]
    with offline_only():
        from pyiqa.archs.musiq_arch import MUSIQ
        from vbench.imaging_quality import technical_quality
        motion=load_amt(args.vbench,args.amt)
        imaging=MUSIQ(pretrained_model_path=str(args.musiq)).to('cuda').eval()
        for row in rows:
            path=row['video_path']
            with torch.inference_mode():
                smooth=float(motion.motion_score(path))
                aggregate,detail=technical_quality(imaging,[path],'cuda',imaging_quality_preprocessing_mode='longer')
            mos=float(detail[0]['video_results'])
            score=normalized_imaging(mos)
            if abs(score-float(aggregate))>1e-7:raise RuntimeError('MUSIQ normalization mismatch')
            if sha256_file(path)!=row['video_sha256']:raise ValueError('Media changed during quality evaluation')
            records.append({'sample_id':row['sample_id'],'group':row.get('group'),
                            'scores':{'motion_smoothness':smooth,'imaging_quality':score},
                            'musiq_raw_mos':mos,'video_sha256':row['video_sha256'],'score_kind':'local_proxy'})
    torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    write_json(args.output,{'protocol':PROTOCOL,'manifest_sha256':sha256_file(args.manifest),
                            'weight_hashes':{'amt':AMT_SHA256,'musiq':MUSIQ_SHA256,'config':sha256_file(config)},
                            'records':records,'resource':{'gpus':1,'elapsed_seconds':elapsed,'allocated_gpu_hours':elapsed/3600,
                            'peak_allocated_vram_gb':torch.cuda.max_memory_allocated()/1e9},
                            'status':'completed','official_score':None})
    print('Offline paper-related quality dimensions scored',len(records),'videos')


if __name__=='__main__':main()
