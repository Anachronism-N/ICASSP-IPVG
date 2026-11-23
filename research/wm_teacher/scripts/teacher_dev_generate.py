#!/usr/bin/env python3
"""Paired dev-set TI2V generation on the new cluster (base and LoRA candidates, identical sampler).

Same sampler/defaults as scripts/gen_jobs.py (Wan2.2-TI2V-5B official: 1280x704, 121 frames, 24 fps,
50 steps, CFG 5, shift 5, tiled VAE, fixed negative prompt), but loads the locally pinned DiT shards and
converted T5/VAE used on this cluster. One process per GPU ('spawn' splits CUDA_VISIBLE_DEVICES);
outputs are claimed atomically and never overwritten. Not a quality claim by itself.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / 'DiffSynth-Studio'
BASE = Path('/dockerdata/wm_models_teacher_base_20261103/Wan-AI/Wan2.2-TI2V-5B')
ENC = Path('/dockerdata/wm_models_teacher_encoders_originalpins_20261104')
W, H, NUM_FRAMES, FPS = 1280, 704, 121, 24
SAMPLER = dict(cfg_scale=5.0, sigma_shift=5.0, num_inference_steps=50, tiled=True)
JOB_KEYS = {'id', 'model', 'ckpt', 'image', 'prompt', 'out', 'seed'}


def load_jobs(path):
    jobs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert jobs and all(set(j) == JOB_KEYS for j in jobs), 'Exact job schema required'
    assert len({j['out'] for j in jobs}) == len(jobs) and len({j['id'] for j in jobs}) == len(jobs), 'Unique outputs/ids'
    assert all(type(j['seed']) is int and (j['ckpt'] == 'NONE' or Path(j['ckpt']).is_file()) for j in jobs), 'Seed/checkpoint invalid'
    return jobs


def negative_prompt():
    sys.path.insert(0, str(ROOT / 'scripts'))
    src = (ROOT / 'scripts/gen_jobs.py').read_text()
    start = src.index('NEG = (')
    ns = {}
    exec(src[start:src.index(')', start) + 1], ns)
    return ns['NEG']


def load_image(path):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    s = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    l, t = (img.width - W) // 2, (img.height - H) // 2
    return img.crop((l, t, l + W, t + H))


def claim(out):
    try:
        with open(out + '.claim', 'x') as f:
            f.write('%s:%d' % (os.uname().nodename, os.getpid()))
        return True
    except FileExistsError:
        return False


def build_pipe(ckpt):
    import torch
    sys.path.insert(0, str(FRAMEWORK))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    conv = ENC / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors'
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda', model_configs=[
        ModelConfig(path=[str(BASE / f'diffusion_pytorch_model-0000{i}-of-00003.safetensors') for i in (1, 2, 3)], skip_download=True),
        ModelConfig(path=str(conv / 'models_t5_umt5-xxl-enc-bf16.safetensors'), skip_download=True),
        ModelConfig(path=str(conv / 'Wan2.2_VAE.safetensors'), skip_download=True)],
        tokenizer_config=ModelConfig(path=str(ENC / 'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'), skip_download=True),
        redirect_common_files=False)
    if ckpt != 'NONE':
        pipe.load_lora(pipe.dit, ckpt, alpha=1)
    return pipe


def worker(a):
    import torch
    sys.path.insert(0, str(FRAMEWORK))
    from diffsynth.utils.data import save_video
    assert torch.cuda.device_count() == 1, 'One GPU per worker'
    jobs = load_jobs(a.jobs)
    neg = negative_prompt()
    groups = {}
    for j in jobs:
        groups.setdefault((j['model'], j['ckpt']), []).append(j)
    failures = 0
    for (model, ckpt), group in groups.items():
        if all(os.path.exists(j['out']) for j in group):
            continue
        pipe = None
        for j in group:
            if os.path.exists(j['out']) or os.path.exists(j['out'] + '.claim'):
                continue
            os.makedirs(os.path.dirname(j['out']), exist_ok=True)
            if not claim(j['out']):
                continue
            if pipe is None:
                pipe = build_pipe(ckpt)
            t0 = time.time()
            try:
                video = pipe(prompt=j['prompt'], negative_prompt=neg, seed=j['seed'], height=H, width=W,
                             num_frames=NUM_FRAMES, input_image=load_image(j['image']), **SAMPLER)
                tmp = j['out'] + '.tmp.mp4'
                save_video(video, tmp, fps=FPS, quality=5)
                os.replace(tmp, j['out'])
                rec = dict(id=j['id'], model=model, seed=j['seed'], seconds=time.time() - t0, ok=True)
            except Exception as e:  # recorded, never silently dropped
                failures += 1
                rec = dict(id=j['id'], model=model, seed=j['seed'], ok=False, error=type(e).__name__ + ': ' + str(e)[:200])
            with open(j['out'] + '.json', 'w') as f:
                json.dump(rec, f)
            print(json.dumps(rec), flush=True)
        del pipe
        torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f'{failures} generation failures')


def clear_stale_claims(jobs):
    """Remove claims of dead owner processes whose output never appeared (killed earlier attempts)."""
    cleared = []
    for j in jobs:
        c = j['out'] + '.claim'
        if os.path.exists(c) and not os.path.exists(j['out']):
            owner = Path(c).read_text().strip()
            host, _, pid = owner.rpartition(':') if ':' in owner else (os.uname().nodename, '', owner)
            if host != os.uname().nodename:
                continue  # another node's claim: never judged from here
            pid = int(pid or 0)
            if pid <= 1 or not Path('/proc', str(pid)).exists():
                os.remove(c); cleared.append(j['id'])
    return cleared


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
    raise SystemExit(0 if not any(codes) and not missing else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('worker'); s.add_argument('--jobs', required=True)
    s = sub.add_parser('spawn'); s.add_argument('--jobs', required=True); s.add_argument('--log-dir', required=True)
    a = ap.parse_args()
    worker(a) if a.cmd == 'worker' else spawn(a)


if __name__ == '__main__':
    main()
