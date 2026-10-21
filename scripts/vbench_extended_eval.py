"""Offline, pinned native VBench dimensions; never an official combined score."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time

from ipvg.common import load_json, sha256_file, write_json

# Make the sibling offline guard importable both from CLI and importlib tests.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from mm2026_quality_eval import offline_only

VBENCH_REVISION = 'fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490'
DIMENSIONS = ('dynamic_degree', 'subject_consistency', 'background_consistency',
              'aesthetic_quality', 'overall_consistency')
ASSETS = {
    'raft': 'raft_model/models/raft-things.pth',
    'dino': 'dino_model/dino_vitbase16_pretrain.pth',
    'clip_b32': 'clip_model/ViT-B-32.pt',
    'clip_l14': 'clip_model/ViT-L-14.pt',
    'aesthetic': 'aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth',
    'viclip': 'ViCLIP/ViClip-InternVid-10M-FLT.pth',
    'vocab': 'ViCLIP/bpe_simple_vocab_16e6.txt.gz',
}
PROTOCOL = {
    'version': 'vbench_extended_native_v1', 'vbench_revision': VBENCH_REVISION,
    'dimensions': {
        'dynamic_degree': 'Native RAFT FP32 boolean; interval=max(1,round(fps/8)); adaptive movement thresholds, not smoothness or semantic correctness',
        'subject_consistency': 'DINO ViT-B/16 all decoded frames; mean nonnegative first/previous-frame cosines per video',
        'background_consistency': 'CLIP ViT-B/32 all decoded frames; mean nonnegative first/previous-frame cosines per video',
        'aesthetic_quality': 'CLIP ViT-L/14 and LAION linear head; native all-frame mean prediction/10, no extra normalization or clipping',
        'overall_consistency': 'ViCLIP InternVid-10M-FLT; native 8 uniform-middle frames and original prompt; raw unit cosine including negatives',
    },
    'aggregation': 'Equal video weight within each dimension; upstream aggregate retained separately (subject/background are frame-transition weighted)',
    'precision': 'FP32 DINO/RAFT/ViCLIP/aesthetic head; native clip.load CUDA precision for CLIP image encoders; actual parameter dtypes recorded',
    'prompt_policy': 'Original manifest prompt, no prompt refinement; SHA256 only in output. Native ViCLIP tokenizer context=32, truncate=True; token overflow recorded',
    'temporal_flickering': 'Excluded: assumes static content, not a moving-video quality metric',
    'official_final_score': None, 'human_score': None,
    'warning': 'Native named dimensions on custom frozen media, not the official VBench benchmark or challenge final score',
}


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Require lowercase SHA256')
    return value


def _relative(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('Lock file names must be relative POSIX paths')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
        raise ValueError('Lock file names must be relative POSIX paths')
    # Existing cache symlinks may point into the approved shared model cache.
    return Path(root) / name


def validate_local(vbench, cache, model_lock):
    """Check revision, source integrity and every declared asset before model imports."""
    vbench, cache = Path(vbench).resolve(), Path(cache).resolve()
    lock = load_json(model_lock)
    if not isinstance(lock, dict) or lock.get('vbench_revision') != VBENCH_REVISION:
        raise ValueError('Model lock VBench revision differs')
    revision = subprocess.run(['git', '-C', str(vbench), 'rev-parse', 'HEAD'],
                              check=True, capture_output=True, text=True).stdout.strip()
    if revision != VBENCH_REVISION:
        raise ValueError('VBench revision differs')
    subprocess.run(['git', '-C', str(vbench), 'diff', '--quiet', 'HEAD', '--', 'vbench'], check=True)
    extra = subprocess.run(['git', '-C', str(vbench), 'ls-files', '--others', '--exclude-standard',
                            '--', 'vbench/**/*.py', 'vbench/*.py'],
                           check=True, capture_output=True, text=True).stdout.strip()
    if extra:
        raise ValueError('Untracked VBench Python source')
    files = lock.get('files')
    if not isinstance(files, dict) or not set(ASSETS.values()).issubset(files):
        raise ValueError('Model lock must cover all seven local assets including ViCLIP vocabulary')
    for name, digest in files.items():
        path = _relative(cache, name)
        if not path.is_file() or sha256_file(path) != _sha(digest):
            raise ValueError('Local asset missing or SHA256 mismatch: ' + name)
    repo_value = lock.get('dino_repo')
    if not isinstance(repo_value, str) or not Path(repo_value).is_absolute():
        raise ValueError('dino_repo must be an explicit absolute local path')
    repo = Path(repo_value).resolve()
    sources = lock.get('dino_source')
    present = {p.relative_to(repo).as_posix() for p in repo.rglob('*.py')}
    if not isinstance(sources, dict) or set(sources) != present or not {'hubconf.py', 'vision_transformer.py'}.issubset(present):
        raise ValueError('DINO source hashes must cover every local Python source')
    for name, digest in sources.items():
        if sha256_file(_relative(repo, name)) != _sha(digest):
            raise ValueError('DINO source SHA256 mismatch: ' + name)
    return lock, repo


def validate_manifest(request, probe=None):
    if probe is None:
        from ipvg.evaluate import probe_video
        probe = probe_video
    rows = request.get('samples') if isinstance(request, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Require nonempty samples')
    ids, paths, validated = set(), set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each sample must be an object')
        sid = row.get('sample_id')
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', sid) or sid in ids:
            raise ValueError('Require unique safe sample_id')
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('Require original nonempty string prompt')
        if not isinstance(row.get('group'), str) or not row['group'].strip():
            raise ValueError('Require nonempty string group')
        raw_path = row.get('video_path')
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ValueError('Require absolute video_path')
        path = Path(raw_path).resolve()
        digest = _sha(row.get('video_sha256'))
        if path.suffix != '.mp4' or not path.is_file() or sha256_file(path) != digest:
            raise ValueError('Frozen MP4 media missing or SHA256 mismatch')
        if str(path) in paths:
            raise ValueError('Video paths must be unique for native per-video coverage')
        media = probe(path)
        if not media.get('ok') or media.get('width', 0) < 32 or media.get('height', 0) < 32 or media.get('decoded_frames', 0) < 2:
            raise ValueError('Require valid media dimensions>=32 and decoded frames>=2')
        fps = media.get('fps')
        if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Require finite positive media fps')
        interval = max(1, round(fps / 8))
        sampled = len(range(0, media['decoded_frames'], interval))
        if sampled < 2:
            raise ValueError('Native RAFT sampling requires at least two frames')
        ids.add(sid)
        paths.add(str(path))
        validated.append(dict(row, video_path=str(path), media=media,
                              raft_sampled_frames=sampled, raft_interval=interval))
    return validated


@contextmanager
def dimension_info(rows, output):
    """Private, unique local scratch adapter with prompts anchored to the manifest."""
    parent = Path(output).parent
    if not parent.is_dir():
        raise FileNotFoundError('Output parent must already exist')
    with tempfile.TemporaryDirectory(prefix='.vbench-info-', dir=parent) as scratch:
        info = Path(scratch) / 'dimensions.json'
        payload = [{'prompt_en': r['prompt'], 'dimension': list(DIMENSIONS),
                    'video_list': [r['video_path']]} for r in rows]
        info.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        info.chmod(0o600)
        yield info


def finite_score(value, dimension):
    if isinstance(value, bool):
        if dimension != 'dynamic_degree':
            raise ValueError('Boolean only permitted for native dynamic_degree')
        return float(value)
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Require finite native score for ' + dimension)
    return float(value)


def collect_dimension(rows, dimension, aggregate, detail):
    expected = {r['video_path'] for r in rows}
    scores = {}
    for result in detail:
        path = result.get('video_path')
        if path not in expected or path in scores:
            raise ValueError('Unknown or duplicate native video result for ' + dimension)
        scores[path] = finite_score(result.get('video_results'), dimension)
    if set(scores) != expected:
        raise ValueError('Incomplete native per-dimension coverage for ' + dimension)
    return scores, {
        'mean': math.fsum(scores.values()) / len(rows),
        'native_aggregate': finite_score(aggregate, dimension),
        'coverage': {'expected': len(rows), 'scored': len(scores), 'fraction': 1.0},
    }


@contextmanager
def model_offline(torch):
    def denied(*args, **kwargs):
        raise RuntimeError('URL state loading forbidden in offline evaluation; explicit local weights required')
    original = torch.hub.load_state_dict_from_url
    from torch.utils import model_zoo
    original_zoo = model_zoo.load_url
    try:
        with offline_only():
            torch.hub.load_state_dict_from_url = denied
            model_zoo.load_url = denied
            yield
    finally:
        torch.hub.load_state_dict_from_url = original
        model_zoo.load_url = original_zoo


def load_dino(torch, repo, weight, device):
    loaded = sys.modules.get('vision_transformer')
    if loaded is not None and Path(getattr(loaded, '__file__', '')).resolve() != (Path(repo) / 'vision_transformer.py').resolve():
        raise RuntimeError('Refuse preloaded foreign DINO vision_transformer')
    model = torch.hub.load(str(repo), 'dino_vitb16', source='local', pretrained=False)
    model.load_state_dict(torch.load(str(weight), map_location='cpu', weights_only=True), strict=True)
    return model.float().to(device).eval()


def prompt_tokenization(rows, tokenizer, context=32):
    return {r['video_path']: {'context_length': context,
                             'tokens_with_special': len(tokenizer.encode(r['prompt'])) + 2,
                             'truncated': len(tokenizer.encode(r['prompt'])) + 2 > context,
                             'truncate': True} for r in rows}


def model_precision(model):
    precision = {'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()})}
    if hasattr(model, 'dtype'):
        precision['encode_image_dtype'] = str(model.dtype)
    return precision


def evaluate_native(info, rows, cache, dino_repo, torch, device='cuda'):
    paths = [r['video_path'] for r in rows]
    assets = {name: str(Path(cache) / relative) for name, relative in ASSETS.items()}
    results, dtypes, tokens = {}, {}, {}
    import clip
    from vbench.utils import load_dimension_info

    for dimension in DIMENSIONS:
        module = importlib.import_module('vbench.' + dimension)
        with torch.inference_mode():
            if dimension == 'dynamic_degree':
                args = module.edict({'model': assets['raft'], 'small': False,
                                     'mixed_precision': False, 'alternate_corr': False})
                model = module.DynamicDegree(args, device)
                result = module.dynamic_degree(model, paths)
                dtypes[dimension] = model_precision(model.model)
            elif dimension == 'subject_consistency':
                model = load_dino(torch, dino_repo, assets['dino'], device)
                result = module.subject_consistency(model, paths, device, False)
                dtypes[dimension] = model_precision(model)
            elif dimension == 'background_consistency':
                model, preprocess = clip.load(assets['clip_b32'], device=device, jit=False)
                model.eval()
                result = module.background_consistency(model, preprocess, paths, device, False)
                dtypes[dimension] = model_precision(model)
            elif dimension == 'aesthetic_quality':
                model, _ = clip.load(assets['clip_l14'], device=device, jit=False)
                model.eval()
                head = torch.nn.Linear(768, 1)
                head.load_state_dict(torch.load(assets['aesthetic'], map_location='cpu', weights_only=True), strict=True)
                head = head.float().to(device).eval()
                result = module.laion_aesthetic(head, model, paths, device)
                dtypes[dimension] = {'clip': model_precision(model),
                                     'linear': model_precision(head)}
                del head
            else:
                tokenizer = module.SimpleTokenizer(assets['vocab'])
                tokens = prompt_tokenization(rows, tokenizer)
                model = module.ViCLIP(tokenizer=tokenizer, pretrain=assets['viclip']).float().to(device).eval()
                _, video_dict = load_dimension_info(str(info), dimension=dimension, lang='en')
                # Upstream's mutable-default cache can otherwise retain features from a previous model.
                module.get_text_features.__defaults__[0].clear()
                try:
                    result = module.overall_consistency(model, video_dict, tokenizer, device, sample='middle')
                finally:
                    module.get_text_features.__defaults__[0].clear()
                dtypes[dimension] = model_precision(model)
        results[dimension] = collect_dimension(rows, dimension, *result)
        for row in rows:
            if sha256_file(row['video_path']) != row['video_sha256']:
                raise ValueError('Frozen media changed during evaluation')
        del model
        torch.cuda.empty_cache()
    return results, dtypes, tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'vbench', 'cache', 'model-lock', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose fresh VBench output')
    if not args.output.parent.is_dir():
        raise FileNotFoundError('Output parent must already exist')
    manifest_sha, lock_sha = sha256_file(args.manifest), sha256_file(args.model_lock)
    lock, dino_repo = validate_local(args.vbench, args.cache, args.model_lock)
    rows = validate_manifest(load_json(args.manifest))
    if any(name == 'vbench' or name.startswith('vbench.') for name in sys.modules):
        raise RuntimeError('VBench must not be imported before local cache validation')
    sys.path.insert(0, str(args.vbench.resolve()))
    os.environ.update(VBENCH_CACHE_DIR=str(args.cache.resolve()), HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    import torch
    import numpy.testing  # Initialize NumPy local capability probing before Popen lockdown.
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Require exactly one visible CUDA device')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with dimension_info(rows, args.output) as info, model_offline(torch):
        dimensions, dtypes, tokens = evaluate_native(info, rows, args.cache, dino_repo, torch)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    records = []
    for row in rows:
        if sha256_file(row['video_path']) != row['video_sha256']:
            raise ValueError('Frozen media changed after evaluation')
        scores = {d: dimensions[d][0][row['video_path']] for d in DIMENSIONS}
        records.append({'sample_id': row['sample_id'], 'group': row['group'], 'scores': scores,
                        'video_sha256': row['video_sha256'],
                        'prompt_sha256': hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest(),
                        'media': row['media'], 'raft_interval': row['raft_interval'],
                        'raft_sampled_frames': row['raft_sampled_frames'],
                        'overall_tokenization': tokens[row['video_path']], 'score_kind': 'local_proxy'})
    if sha256_file(args.manifest) != manifest_sha or sha256_file(args.model_lock) != lock_sha:
        raise ValueError('Manifest or model lock changed during evaluation')
    write_json(args.output, {
        'protocol': PROTOCOL, 'manifest_sha256': manifest_sha, 'model_lock_sha256': lock_sha,
        'evaluator_sha256': sha256_file(__file__),
        'weight_hashes': lock['files'], 'dino_source_hashes': lock['dino_source'],
        'actual_parameter_dtypes': dtypes, 'records': records,
        'dimensions': {d: dimensions[d][1] for d in DIMENSIONS},
        'resource': {'gpus': 1, 'elapsed_seconds': elapsed, 'allocated_gpu_hours': elapsed / 3600,
                     'peak_allocated_vram_gb': torch.cuda.max_memory_allocated() / 1e9,
                     'peak_reserved_vram_gb': torch.cuda.max_memory_reserved() / 1e9},
        'status': 'completed', 'official_score': None,
    })
    print('Offline native VBench dimensions scored', len(records), 'videos')


if __name__ == '__main__':
    main()
