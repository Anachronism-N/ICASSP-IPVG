#!/usr/bin/env python3
"""Offline E3 input cache: VAE x0 + first-frame latent once per video, T5 context per caption version.

Manifest JSONL, one line per clip:
  {"sample_id": "...", "video": "/abs/clip.mp4", "captions": {"orig": "...", "fixed": "..."}, ...extra}
Every line must carry the same caption-version set. Preprocessing is M1's
(run_teacher_mechanism_probe): first 121 frames, cover-resize LANCZOS + center crop
1280x704, VAE tiled (30,52)/(15,26); first-frame latent as WanVideoUnit_ImageEmbedderFused;
context as WanVideoUnit_PromptEmbedder (positive). Only T5+VAE+tokenizer are loaded.

One GPU per process (set CUDA_VISIBLE_DEVICES); --shard k --num-shards N takes lines i%N==k.
Resume-safe: a sample is done iff records/<id>.json exists and its files re-hash; orphan
tensors without a record (crash mid-sample) are recomputed. --finalize (CPU, no torch)
merges all records into cache_index.json in manifest order. Files are private 0600,
torch.save dicts readable with weights_only=True.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_teacher_mechanism_probe as m1  # noqa: E402

PROTOCOL = 'e3-input-cache-v1'
W, H, FRAMES = m1.W, m1.H, m1.FRAMES
TILE_SIZE, TILE_STRIDE = (30, 52), (15, 26)
X0_SHAPE = (1, 48, 31, 44, 80)
FIRST_SHAPE = (1, 48, 1, 44, 80)
CTX_SHAPE = (1, 512, 4096)
ID_RE = re.compile(r'^[A-Za-z0-9_.-]{1,128}$')
VERSION_RE = re.compile(r'^[a-z0-9_]{1,32}$')
ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DEFAULT = ROOT / 'DiffSynth-Studio'
ENCODER_RELS = tuple(k for k in m1.CACHE_PINS if '/diffusion_pytorch_model-' not in k)
require = m1.require


def read_manifest(path):
    rows, versions = [], None
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict) and ID_RE.match(str(row.get('sample_id', ''))), 'Bad sample_id at line %d' % n)
            require(isinstance(row.get('video'), str) and os.path.isabs(row['video']), 'Video must be absolute path at line %d' % n)
            caps = row.get('captions')
            require(isinstance(caps, dict) and caps and all(VERSION_RE.match(k) and isinstance(v, str) and v.strip()
                                                            for k, v in caps.items()), 'Bad captions at line %d' % n)
            require(versions is None or set(caps) == versions, 'Caption version set differs at line %d' % n)
            versions = set(caps)
            rows.append(row)
    require(rows, 'Empty manifest')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate sample_id')
    return rows, sorted(versions)


def make_binding(manifest_path, rows, versions, allow_longer, framework):
    return {'protocol': PROTOCOL, 'manifest': {'path': str(Path(manifest_path).resolve()),
                                               'sha256': m1.sha256(manifest_path), 'count': len(rows)},
            'versions': versions,
            'preprocessing': {'frames': FRAMES, 'width': W, 'height': H, 'frame_policy': 'first121' if allow_longer else 'exact121',
                              'resize': 'M1 cover_center_crop (PIL LANCZOS cover + center crop)',
                              'vae': {'tiled': True, 'tile_size': list(TILE_SIZE), 'tile_stride': list(TILE_STRIDE)},
                              'first_frame': 'WanVideoUnit_ImageEmbedderFused equivalent: vae.encode([preprocess_image(f0).transpose(0,1)])',
                              'context': 'WanVideoUnit_PromptEmbedder positive, seq_len 512, bf16'},
            'encoder_pins': {r: m1.CACHE_PINS[r] for r in ENCODER_RELS},
            'code': {'cacher': m1.sha256(__file__), 'm1_helper': m1.sha256(m1.__file__),
                     'framework_wan_video': m1.sha256(Path(framework) / 'diffsynth/pipelines/wan_video.py')}}


def bind(root, binding):
    path = root / 'binding.json'
    if path.exists():
        require(m1.read_json(path) == binding, 'Existing cache was built from a different manifest/config/code')
        return
    try:
        m1.atomic_json(path, binding)
    except (ValueError, FileExistsError):  # another shard won the race
        require(m1.read_json(path) == binding, 'Concurrent cache binding differs')


def decode_frames(path, allow_longer):
    if not allow_longer:
        return m1.decode_gt(path)
    import imageio.v2 as imageio
    from PIL import Image
    reader = imageio.get_reader(str(path))
    frames = []
    try:
        for frame in reader:
            frames.append(m1.cover_center_crop(Image.fromarray(frame)))
            if len(frames) == FRAMES:
                break
    finally:
        reader.close()
    require(len(frames) == FRAMES, 'Video has fewer than 121 decodable frames: ' + str(path))
    return frames


def encode_video(pipe, frames):
    import torch
    with torch.no_grad():
        pipe.load_models_to_device(['vae'])
        x0 = pipe.vae.encode(pipe.preprocess_video(frames), device=pipe.device, tiled=True,
                             tile_size=TILE_SIZE, tile_stride=TILE_STRIDE)
        image = pipe.preprocess_image(frames[0].resize((W, H))).transpose(0, 1)
        first = pipe.vae.encode([image], device=pipe.device, tiled=True, tile_size=TILE_SIZE, tile_stride=TILE_STRIDE)
    x0 = x0.to(torch.bfloat16).detach().cpu().contiguous()
    first = first.to(torch.bfloat16).detach().cpu().contiguous()
    require(tuple(x0.shape) == X0_SHAPE and tuple(first.shape) == FIRST_SHAPE, 'Unexpected latent shapes')
    return x0, first


def encode_prompt(pipe, prompt):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
    with torch.no_grad():
        ctx = WanVideoUnit_PromptEmbedder().process(pipe, prompt, True)['context']
    ctx = ctx.to(torch.bfloat16).detach().cpu().contiguous()
    require(tuple(ctx.shape) == CTX_SHAPE, 'Unexpected context shape')
    return ctx


def paths(root, sid, versions):
    return root / 'x0' / (sid + '.pt'), {v: root / 'ctx' / v / (sid + '.pt') for v in versions}, root / 'records' / (sid + '.json')


def record_ok(root, record):
    for rec in [record['x0']] + list(record['contexts'].values()):
        p = m1.private_path(root, rec['path'])
        if not (p.is_file() and p.stat().st_size == rec['bytes'] and m1.sha256(p) == rec['sha256']):
            return False
    return True


def relative(root, rec):
    return dict(rec, path=str(Path(rec['path']).relative_to(root)))


def process_sample(root, row, versions, pipe, allow_longer, encode_video_fn=encode_video, encode_prompt_fn=encode_prompt,
                   decode_fn=decode_frames):
    sid = row['sample_id']
    x0_path, ctx_paths, rec_path = paths(root, sid, versions)
    if rec_path.exists():
        require(record_ok(root, m1.read_json(rec_path)), 'Completed record no longer matches files: ' + sid)
        return 'reused'
    for p in [x0_path] + list(ctx_paths.values()):  # own orphans from an interrupted attempt
        if p.exists():
            p.unlink()
    video_sha = m1.sha256(row['video'])
    frames = decode_fn(row['video'], allow_longer)
    x0, first = encode_video_fn(pipe, frames)
    del frames
    x0_rec = m1.save_tensors(x0_path, {'x0': x0, 'first_frame_latents': first})
    contexts = {}
    for v in versions:
        ctx_paths[v].parent.mkdir(mode=0o700, exist_ok=True)
        prompt = row['captions'][v]
        ctx = encode_prompt_fn(pipe, prompt)
        contexts[v] = dict(relative(root, m1.save_tensors(ctx_paths[v], {'context': ctx})), prompt_sha256=m1.text_hash(prompt))
    extra = {k: v for k, v in row.items() if k not in ('sample_id', 'video', 'captions')}
    record = {'sample_id': sid, 'video': {'path': row['video'], 'sha256': video_sha, 'bytes': Path(row['video']).stat().st_size},
              'x0': relative(root, x0_rec), 'contexts': contexts, 'extra': extra}
    m1.atomic_json(rec_path, record)
    return 'computed'


def prepare_root(root):
    root.mkdir(mode=0o700, exist_ok=True)
    require(root.resolve() == root, 'Cache root must not contain symlinks')
    for sub in ('x0', 'ctx', 'records', 'logs'):
        (root / sub).mkdir(mode=0o700, exist_ok=True)


def build_encoders(models_dir, framework, hash_models=True):
    import torch
    sys.path.insert(0, str(Path(framework).resolve()))
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    files = {}
    for rel in ENCODER_RELS:
        p = Path(models_dir) / rel
        require(p.is_file(), 'Missing encoder asset: ' + str(p))
        if hash_models:
            m1.file_record(p, m1.CACHE_PINS[rel])
        files[rel] = str(p)
    t5 = next(v for k, v in files.items() if 'models_t5_' in k)
    vae = next(v for k, v in files.items() if '/Wan2.2_VAE.' in k)
    tok = str(Path(next(v for k, v in files.items() if k.endswith('/spiece.model'))).parent)
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[ModelConfig(path=t5, skip_download=True), ModelConfig(path=vae, skip_download=True)],
        tokenizer_config=ModelConfig(path=tok, skip_download=True), redirect_common_files=False)
    require(pipe.dit is None and pipe.vae is not None and pipe.text_encoder is not None and pipe.tokenizer is not None,
            'Cache pipeline must hold exactly T5+VAE+tokenizer')
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


def run_shard(args, rows, versions, pipe_factory=build_encoders, **fns):
    root = Path(args.out_dir).absolute()
    prepare_root(root)
    bind(root, make_binding(args.manifest, rows, versions, args.allow_longer, args.framework))
    mine = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    stats = {'shard': args.shard, 'num_shards': args.num_shards, 'assigned': len(mine), 'computed': 0, 'reused': 0,
             'failed': [], 'seconds': []}
    pending = [r for r in mine if not paths(root, r['sample_id'], versions)[2].exists()]
    pipe = pipe_factory(args.models_dir, args.framework, not args.skip_model_hash) if pending else None
    for row in mine:
        t0 = time.perf_counter()
        try:
            stats[process_sample(root, row, versions, pipe, args.allow_longer, **fns)] += 1
        except Exception as error:
            stats['failed'].append({'sample_id': row['sample_id'], 'error': type(error).__name__ + ': ' + str(error)[:300]})
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        stats['seconds'].append(time.perf_counter() - t0)
    m1.atomic_json(root / 'logs' / ('shard_%03d_of_%03d_%d.json' % (args.shard, args.num_shards, int(time.time() * 1000))), stats)
    return stats


def finalize(args, rows, versions):
    root = Path(args.out_dir).absolute()
    binding = m1.read_json(root / 'binding.json')
    require(binding == make_binding(args.manifest, rows, versions, binding['preprocessing']['frame_policy'] == 'first121',
                                    args.framework), 'Binding differs from manifest/code')
    samples = []
    for row in rows:
        rec_path = paths(root, row['sample_id'], versions)[2]
        require(rec_path.exists(), 'Missing record: ' + row['sample_id'])
        rec = m1.read_json(rec_path)
        require(rec['sample_id'] == row['sample_id'] and set(rec['contexts']) == set(versions)
                and all(rec['contexts'][v]['prompt_sha256'] == m1.text_hash(row['captions'][v]) for v in versions),
                'Record differs from manifest: ' + row['sample_id'])
        require(record_ok(root, rec), 'Record files changed: ' + row['sample_id'])
        samples.append(rec)
    index = {'protocol': PROTOCOL, 'complete': True, 'binding': binding, 'versions': versions, 'samples': samples,
             'x0_bytes_total': sum(s['x0']['bytes'] for s in samples),
             'context_bytes_total': sum(c['bytes'] for s in samples for c in s['contexts'].values())}
    m1.atomic_json(root / 'cache_index.json', index)
    return {'status': 'finalized', 'samples': len(samples), 'versions': versions, 'index': str(root / 'cache_index.json'),
            'index_sha256': m1.sha256(root / 'cache_index.json'),
            'bytes_total': index['x0_bytes_total'] + index['context_bytes_total']}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--models-dir', type=Path, default=m1.MODELS_DEFAULT)
    p.add_argument('--framework', type=Path, default=FRAMEWORK_DEFAULT)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--allow-longer', action='store_true', help='take frames 0..120 of longer clips (DiffSynth LoadVideo style)')
    p.add_argument('--skip-model-hash', action='store_true')
    p.add_argument('--finalize', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require(0 <= args.shard < args.num_shards <= 256, 'Bad shard spec')
        rows, versions = read_manifest(args.manifest)
        if args.finalize:
            print(json.dumps(finalize(args, rows, versions)))
            return 0
        require(os.environ.get('CUDA_VISIBLE_DEVICES', '').count(',') == 0 and os.environ.get('CUDA_VISIBLE_DEVICES', '') != '',
                'Set CUDA_VISIBLE_DEVICES to exactly one GPU per process')
        stats = run_shard(args, rows, versions)
        print(json.dumps({k: v for k, v in stats.items() if k != 'seconds'}))
        return int(bool(stats['failed']))
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
