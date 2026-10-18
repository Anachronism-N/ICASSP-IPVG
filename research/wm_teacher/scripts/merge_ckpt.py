#!/usr/bin/env python3
"""
Weight-space interpolation (WiSE-FT style) of full-DiT checkpoints, on CPU:
  out = base + alpha * (ckpt - base)          (alpha in [0,1]; 0 = base, 1 = ckpt)
Usage: merge_ckpt.py CKPT ALPHA OUT.safetensors
"""
import glob
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/dockerdata/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors"


def main():
    ckpt, alpha, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    base = {}
    for f in sorted(glob.glob(BASE)):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                base[k] = h.get_tensor(k)
    merged = {}
    with safe_open(ckpt, "pt") as h:
        keys = list(h.keys())
        missing = [k for k in keys if k not in base]
        assert not missing, f"{len(missing)} ckpt keys not in base, e.g. {missing[:3]}"
        for k in keys:
            t = h.get_tensor(k)
            b = base[k].to(torch.float32)
            merged[k] = (b + alpha * (t.to(torch.float32) - b)).to(t.dtype).contiguous()
    print(f"{len(merged)} tensors, base has {len(base)}; alpha={alpha}")
    save_file(merged, out)


if __name__ == "__main__":
    main()
