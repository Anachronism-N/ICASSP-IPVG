"""Run a pinned AnyID merge on CPU without its CUDA-valued import default."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir',type=Path,required=True)
    parser.add_argument('--base-dir',type=Path,required=True)
    parser.add_argument('--lora-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new merged model directory; never replace source weights')
    revision=subprocess.run(['git','-C',str(args.upstream_dir),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if revision!='6c714f71af6e08d2af29aecd74a3a9fd9e8f94a3':
        raise ValueError('AnyID source revision mismatch')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    sys.path[:0]=[str(args.upstream_dir/'trainer/models/wan22'),str(args.upstream_dir)]
    # The pinned source evaluates current_device() in T5's function signature during import.
    # No model is constructed in this context; the upstream merge constructs only CPU DiT.
    current_device=torch.cuda.current_device
    torch.cuda.current_device=lambda:0
    try:
        import wan
    finally:
        torch.cuda.current_device=current_device
    entry=args.upstream_dir/'trainer/models/wan22/merge_lora.py'
    sys.argv=[str(entry),'--ckpt_dir',str(args.base_dir),'--lora_path',str(args.lora_dir),
              '--output_dir',str(args.output_dir),'--dtype','bf16']
    runpy.run_path(str(entry),run_name='__main__')


if __name__=='__main__':
    main()
