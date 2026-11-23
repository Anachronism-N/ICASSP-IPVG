#!/usr/bin/env python3
"""VBench 'Qwen prompt-optimized' inputs with the official Wan2.2 local prompt extender (commit 81e67351).

Uses wan.utils.prompt_extend.QwenPromptExpander exactly as generate.py --use_prompt_extend does for ti2v-5B with
an input image: local_qwen, default VL model Qwen2.5-VL-7B-Instruct, default target language zh. One extension per
unique (image, raw prompt) pair, reused for its five samples (documented deviation: generate.py would re-extend per
call). Failed extensions fall back to the raw prompt as generate.py does. dashscope is stubbed (import-only).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types

WAN_REPO = os.environ.get('WAN22_OFFICIAL_REPO', '/dockerdata/wm_eval_20261121/official/Wan2.2')


def import_official_wan():
    """Expose the official repo as package 'wan' without running wan/__init__.py (which imports the unused
    speech2video/animate stacks and their extra deps). Modules actually used are the official files unchanged."""
    import importlib.machinery
    import types
    if 'wan' not in sys.modules:
        pkg = types.ModuleType('wan')
        pkg.__path__ = [os.path.join(WAN_REPO, 'wan')]
        pkg.__spec__ = importlib.machinery.ModuleSpec('wan', None, is_package=True)
        sys.modules['wan'] = pkg


def pairs(preflight, images_dir):
    recs = json.loads(Path(preflight).read_text())['coverage_records']
    seen, out = set(), []
    for r in recs:
        k = (r['image_key'], r['raw_prompt'])
        if k not in seen:
            seen.add(k); out.append(dict(image=str(Path(images_dir) / r['image_key']), raw_prompt=r['raw_prompt']))
    return out


def worker(a):
    sys.modules.setdefault('dashscope', types.ModuleType('dashscope'))  # DashScope expander unused
    import_official_wan()
    import torch
    from PIL import Image
    from wan.utils.prompt_extend import QwenPromptExpander
    assert torch.cuda.device_count() == 1
    items = pairs(a.preflight, a.images)[a.shard::a.nshards]
    out = Path(a.out_dir) / f'shard{a.shard}.jsonl'
    done = {json.loads(l)['raw_prompt'] + '|' + json.loads(l)['image'] for l in out.read_text().splitlines()} if out.exists() else set()
    ex = QwenPromptExpander(model_name=a.model, task='ti2v-5B', is_vl=True, device=0)
    with open(out, 'a') as f:
        for it in items:
            if it['raw_prompt'] + '|' + it['image'] in done:
                continue
            o = ex(it['raw_prompt'], image=Image.open(it['image']).convert('RGB'), tar_lang='zh', seed=a.seed)
            f.write(json.dumps(dict(it, status=bool(o.status), prompt=o.prompt if o.status else it['raw_prompt'],
                                    fallback=not o.status), ensure_ascii=False) + '\n')
            f.flush()


def spawn(a):
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    procs = []
    for k, u in enumerate(uuids):
        log = open(Path(a.out_dir) / f'shard{k}.log', 'ab')
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--preflight', a.preflight,
                                       '--images', a.images, '--model', a.model, '--out-dir', a.out_dir, '--seed', str(a.seed),
                                       '--shard', str(k), '--nshards', str(len(uuids))],
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=u), stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    n = sum(len(Path(a.out_dir, f'shard{k}.jsonl').read_text().splitlines()) for k in range(len(uuids))
            if Path(a.out_dir, f'shard{k}.jsonl').exists())
    print(json.dumps(dict(returncodes=codes, extended=n, expected=len(pairs(a.preflight, a.images)))), flush=True)
    raise SystemExit(0 if not any(codes) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=('worker', 'spawn'))
    ap.add_argument('--preflight', required=True); ap.add_argument('--images', required=True)
    ap.add_argument('--model', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=20261122)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    worker(a) if a.cmd == 'worker' else spawn(a)


if __name__ == '__main__':
    main()
