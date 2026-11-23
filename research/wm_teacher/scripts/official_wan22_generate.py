#!/usr/bin/env python3
"""Official-protocol TI2V generation with the official Wan2.2 code (commit 81e67351, the VBench-team commit).

Calls wan.WanTI2V.generate exactly as generate.py does for task ti2v-5B, size 1280*704 (output size follows the
input image aspect via best_output_size, e.g. 1248x704 for 16:9), official sampler defaults (UniPC, shift 5,
guide 5, 50 steps, official negative prompt). frame_num and save fps come from each job (protocol-specific).
One process per GPU; atomic host-aware claims on the shared filesystem; outputs never overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WAN_REPO = os.environ.get('WAN22_OFFICIAL_REPO', '/dockerdata/wm_eval_20261121/official/Wan2.2')
JOB_KEYS = {'id', 'image', 'prompt', 'out', 'seed', 'frame_num', 'fps', 'ckpt_dir'}


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


def load_jobs(path):
    jobs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    assert jobs and all(set(j) == JOB_KEYS for j in jobs), 'Exact job schema required'
    assert len({j['out'] for j in jobs}) == len(jobs) and len({j['id'] for j in jobs}) == len(jobs), 'Unique outputs/ids'
    assert all(type(j['seed']) is int and j['seed'] >= 0 and (j['frame_num'] - 1) % 4 == 0 for j in jobs), 'Seed/frames invalid'
    return jobs


def claim(out):
    try:
        with open(out + '.claim', 'x') as f:
            f.write('%s:%d' % (os.uname().nodename, os.getpid()))
        return True
    except FileExistsError:
        return False


def clear_stale_claims(jobs):
    cleared = []
    for j in jobs:
        c = j['out'] + '.claim'
        if os.path.exists(c) and not os.path.exists(j['out']):
            host, _, pid = Path(c).read_text().strip().rpartition(':')
            if host == os.uname().nodename and (not pid.isdigit() or not Path('/proc', pid).exists()):
                os.remove(c); cleared.append(j['id'])
    return cleared


def worker(a):
    import_official_wan()
    import torch
    from PIL import Image
    from wan.textimage2video import WanTI2V
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    assert torch.cuda.device_count() == 1, 'One GPU per worker'
    cfg = WAN_CONFIGS['ti2v-5B']
    jobs = load_jobs(a.jobs)
    model, model_dir, failures = None, None, 0
    for j in jobs:
        if os.path.exists(j['out']) or os.path.exists(j['out'] + '.claim'):
            continue
        os.makedirs(os.path.dirname(j['out']), exist_ok=True)
        if not claim(j['out']):
            continue
        if model_dir != j['ckpt_dir']:
            model = None; torch.cuda.empty_cache()
            model = WanTI2V(config=cfg, checkpoint_dir=j['ckpt_dir'], device_id=0, rank=0, t5_fsdp=False,
                                dit_fsdp=False, use_sp=False, t5_cpu=False, convert_model_dtype=False)
            model_dir = j['ckpt_dir']
        t0 = time.time()
        try:
            img = Image.open(j['image']).convert('RGB')
            video = model.generate(j['prompt'], img=img, size=SIZE_CONFIGS['1280*704'], max_area=MAX_AREA_CONFIGS['1280*704'],
                                   frame_num=j['frame_num'], shift=cfg.sample_shift, sample_solver='unipc',
                                   sampling_steps=cfg.sample_steps, guide_scale=cfg.sample_guide_scale, seed=j['seed'],
                                   offload_model=False)
            tmp = j['out'] + '.tmp.mp4'
            save_video(tensor=video[None], save_file=tmp, fps=j['fps'], nrow=1, normalize=True, value_range=(-1, 1))
            os.replace(tmp, j['out'])
            rec = dict(id=j['id'], seed=j['seed'], seconds=time.time() - t0, ok=True, shape=list(video.shape))
        except Exception as e:
            failures += 1
            rec = dict(id=j['id'], seed=j['seed'], ok=False, error=type(e).__name__ + ': ' + str(e)[:200])
        with open(j['out'] + '.json', 'w') as f:
            json.dump(rec, f)
        print(json.dumps(rec), flush=True)
    if failures:
        raise SystemExit(f'{failures} generation failures')


def spawn(a):
    print(json.dumps(dict(stale_claims_cleared=clear_stale_claims(load_jobs(a.jobs)))), flush=True)
    uuids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    procs = []
    for k, u in enumerate(uuids):
        log = open(Path(a.log_dir) / f'gen{k}.log', 'xb')
        procs.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--jobs', a.jobs],
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=u), stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    missing = [j['out'] for j in load_jobs(a.jobs) if not os.path.exists(j['out'])]
    print(json.dumps(dict(returncodes=codes, missing=len(missing))), flush=True)
    raise SystemExit(0 if not any(codes) else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('worker'); s.add_argument('--jobs', required=True)
    s = sub.add_parser('spawn'); s.add_argument('--jobs', required=True); s.add_argument('--log-dir', required=True)
    a = ap.parse_args()
    worker(a) if a.cmd == 'worker' else spawn(a)


if __name__ == '__main__':
    main()
