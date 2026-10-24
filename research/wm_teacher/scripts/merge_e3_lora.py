#!/usr/bin/env python3
"""Validate/merge an E3 LoRA for scripts/gen_jobs.py.

gen_jobs.py build_pipe(ckpt): if any key contains 'lora' it calls pipe.load_lora(pipe.dit, ckpt, alpha=1)
(DiffSynth GeneralLoRALoader: W += B @ A), otherwise pipe.dit.load_state_dict(strict) of a full DiT.

* Default (CPU, no write): check that the exported e3_lora.safetensors maps through the native
  GeneralLoRALoader name rule onto exactly the 300 base target linears with matching shapes.
  Use that file directly as the job "ckpt" field (LoRA path; alpha==rank so scale is exactly 1).
* --out-full PATH: also write a merged full-DiT BF16 safetensors (W_fp32 + B@A, cast BF16; all 825
  base keys/shapes, no 'lora' keys -> gen_jobs full path) and verify keys/shapes/dtype against base.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_teacher_mechanism_probe as m1  # noqa: E402

TARGETS = ('q', 'k', 'v', 'o', 'ffn.0', 'ffn.2')
LORA_RANK = LORA_ALPHA = 32
ROOT = Path(__file__).resolve().parents[1]
GENERAL_LORA = ROOT / 'DiffSynth-Studio/diffsynth/utils/lora/general.py'
SHARD_RELS = tuple(sorted(k for k in m1.CACHE_PINS if '/diffusion_pytorch_model-' in k))
A_SUFFIX, B_SUFFIX = '.lora_A.default.weight', '.lora_B.default.weight'
require = m1.require


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        require(0 < n < 256 * 1024 * 1024, 'Bad safetensors header')
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    return {k: {'dtype': v['dtype'], 'shape': list(v['shape'])} for k, v in header.items()}


def base_header(shard_paths):
    out = {}
    for p in shard_paths:
        h = read_header(p)
        require(not set(h) & set(out), 'Duplicate key across shards')
        out.update(h)
    return out


def target_modules(header):
    mods = []
    for key, v in header.items():
        if key.endswith('.weight') and len(v['shape']) == 2:
            mod = key[:-len('.weight')]
            if any(mod == s or mod.endswith('.' + s) for s in TARGETS):
                mods.append(mod)
    return sorted(mods)


def expected_lora(header, rank=LORA_RANK):
    out = {}
    for mod in target_modules(header):
        o, i = header[mod + '.weight']['shape']
        out[mod + A_SUFFIX] = [rank, i]
        out[mod + B_SUFFIX] = [o, rank]
    return out


def load_general_loader(path=GENERAL_LORA):
    import importlib.util
    spec = importlib.util.spec_from_file_location('_e3_general_lora', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GeneralLoRALoader


def check_lora(lora, header, loader_path=GENERAL_LORA, rank=LORA_RANK):
    """lora: {name: tensor}. Exact key/shape coverage plus native GeneralLoRALoader name mapping."""
    import torch
    expected = expected_lora(header, rank)
    require(set(lora) == set(expected), 'LoRA keys differ from base targets (missing %d, extra %d)'
            % (len(set(expected) - set(lora)), len(set(lora) - set(expected))))
    for k, shape in expected.items():
        t = lora[k]
        require(list(t.shape) == shape and t.dtype == torch.float32 and bool(torch.isfinite(t).all()), 'Bad LoRA tensor: ' + k)
    converted = load_general_loader(loader_path)().convert_state_dict(lora)
    names = {k[:-len('.lora_B.weight')] for k in converted if k.endswith('.lora_B.weight')}
    require(names == set(target_modules(header)) and len(converted) == len(lora), 'Native loader name mapping differs')
    return {'lora_tensors': len(lora), 'target_layers': len(names), 'parameters': sum(t.numel() for t in lora.values()),
            'native_loader_mapping': 'passed', 'scale': LORA_ALPHA / rank}


def read_lora(path):
    import torch
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        return load_file(str(path), device='cpu')
    value = torch.load(path, map_location='cpu', weights_only=True)
    return dict(value['adapters'])


def merge_full(lora, shard_paths, out_path, scale=LORA_ALPHA / LORA_RANK):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    out_path = Path(out_path)
    require(not out_path.exists() and out_path.parent.is_dir(), 'New --out-full path required')
    merged = {}
    with torch.no_grad():
        for shard in shard_paths:
            with safe_open(str(shard), 'pt', device='cpu') as f:
                for key in f.keys():
                    w = f.get_tensor(key).to(torch.float32)
                    mod = key[:-len('.weight')] if key.endswith('.weight') else None
                    if mod is not None and mod + A_SUFFIX in lora:
                        w = w + scale * (lora[mod + B_SUFFIX].float() @ lora[mod + A_SUFFIX].float())
                    merged[key] = w.to(torch.bfloat16).contiguous()
    tmp = out_path.with_name(out_path.name + '.' + uuid.uuid4().hex + '.tmp')
    save_file(merged, str(tmp), metadata={'format': 'pt', 'e3_merged_lora_scale': str(scale)})
    os.chmod(tmp, 0o600)
    os.link(tmp, out_path)
    tmp.unlink()
    return {'path': str(out_path), 'sha256': m1.sha256(out_path), 'bytes': out_path.stat().st_size, 'keys': len(merged)}


def verify_full(out_path, header, lora, shard_paths, probe_layers=4, scale=LORA_ALPHA / LORA_RANK):
    import torch
    from safetensors import safe_open
    merged = read_header(out_path)
    require(set(merged) == set(header), 'Merged key set differs from base')
    require(all(merged[k]['shape'] == header[k]['shape'] for k in header), 'Merged shapes differ from base')
    require(all(v['dtype'] == 'BF16' for v in merged.values()), 'Merged dtype must be BF16 (gen_jobs pipe dtype)')
    require(not any('lora' in k for k in merged), 'Merged file must not look like a LoRA to gen_jobs')
    mods = target_modules(header)
    probes = [mods[i * (len(mods) - 1) // max(1, probe_layers - 1)] for i in range(probe_layers)] if mods else []
    worst = 0.0
    with safe_open(str(out_path), 'pt', device='cpu') as fm:
        for mod in probes:
            key = mod + '.weight'
            base = None
            for shard in shard_paths:
                with safe_open(str(shard), 'pt', device='cpu') as fb:
                    if key in fb.keys():
                        base = fb.get_tensor(key).float()
            delta = scale * (lora[mod + B_SUFFIX].float() @ lora[mod + A_SUFFIX].float())
            expect = (base + delta).to(torch.bfloat16)
            got = fm.get_tensor(key)
            require(torch.equal(got, expect), 'Merged weight differs from W + B@A: ' + key)
            worst = max(worst, float((got.float() - base - delta).abs().max()))
    return {'keys_match_base': True, 'shapes_match_base': True, 'dtype': 'BF16', 'probe_layers': probes,
            'probe_exact_bf16_equal': True, 'probe_max_abs_bf16_rounding': worst}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lora', type=Path, required=True, help='export/e3_lora.safetensors or private checkpoint .pt')
    p.add_argument('--models-dir', type=Path, default=m1.MODELS_DEFAULT)
    p.add_argument('--out-full', type=Path)
    args = p.parse_args(argv)
    try:
        shards = [args.models_dir / r for r in SHARD_RELS]
        header = base_header(shards)
        require(len(header) == 825, 'Unexpected base key count')
        lora = read_lora(args.lora)
        report = {'lora': str(args.lora), 'lora_sha256': m1.sha256(args.lora), 'check': check_lora(lora, header)}
        report['gen_jobs_ckpt_lora'] = str(args.lora) if args.lora.suffix == '.safetensors' else None
        if args.out_full:
            report['merged'] = merge_full(lora, shards, args.out_full)
            report['merged_verify'] = verify_full(args.out_full, header, lora, shards)
            report['gen_jobs_ckpt_full'] = str(args.out_full)
        print(json.dumps(report, indent=1))
        return 0
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
