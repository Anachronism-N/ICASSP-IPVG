#!/usr/bin/env python3
"""M1: bounded GT-forward-process diagnostics, not generation or rollout replay.

The coordinator imports only stdlib. --preflight-only is read-only and never
queries CUDA. Execution reserves both GPU 0/1 for the entire wall clock, including
preflight, freezing, loading, one shared preparation, waiting, CPU analysis and
cleanup. No resume, downloads, training, scoring, retries or video outputs.
"""
import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

P1_DEFAULT = Path('/srv/wm_tmp/teacher_p1_cfg_20261019_run01')
MODELS_DEFAULT = Path('/dockerdata/models')
CASE_ORDER = tuple(s + '_dev01' for s in
                   ('direct', 'robomind', 'hoigen1m', 'matrix', 'charades', 'easycom'))
MODELS = ('base', 'm_full1_a07')
BRANCHES = ('positive', 'negative')
INDICES = (5, 25, 45)
A07_SHA256 = '057d9a1e682e2f237594f5ef99f994f07699188c97b4196016fd371ab40402a6'
W, H, FRAMES = 1280, 704, 121
# These are the verified local converted assets used by P1, not shared weights.
CACHE_PINS = {
    'Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors':
        '720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1',
    'Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors':
        '09ec5ef720d8396f6cfa51fbdcbdb2327e37722afd6e89fd38f1e7e5e782c283',
    'Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors':
        '6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49',
    'DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors':
        'd92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23',
    'DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors':
        '0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996',
    'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json':
        '7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d',
    'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model':
        'e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458',
    'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json':
        'ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05',
    'Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json':
        '6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_deadline(deadline):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError('M1 work deadline reached; cleanup reservation cannot be used for work')


def sha256(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            check_deadline(deadline)
            h.update(block)
    check_deadline(deadline)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path):
    require(Path(path).stat().st_size <= 16 * 1024 * 1024, 'Oversized JSON manifest')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  allow_nan=False) + '\n').encode())


def atomic_bytes(path, data):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'No overwrite: ' + str(path))
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    # Hard-link publication is atomic and refuses a concurrently created target.
    os.link(tmp, path)
    tmp.unlink()


def private_path(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'Snapshot path escapes run')
    target = root / p
    require(target.resolve().is_relative_to(root.resolve()), 'Snapshot path escapes run')
    return target


def file_record(path, expected=None, deadline=None, local=False, allow_empty=False):
    path = Path(path).absolute()
    require(path.is_file() and (allow_empty or path.stat().st_size > 0), 'Missing/empty file: ' + str(path))
    if local:
        require(path == path.resolve(), 'Local model cache must not symlink to shared storage')
    actual = sha256(path, deadline)
    require(expected is None or actual == expected, 'SHA256 mismatch: ' + str(path))
    stat = path.stat()
    return {'path': str(path), 'sha256': actual, 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}


def validate_record(record, deadline=None, hash_content=True):
    p = Path(record['path'])
    require(p.is_file() and p.stat().st_size == record['bytes'], 'Frozen file changed: ' + str(p))
    require(p.stat().st_mtime_ns == record['mtime_ns'] and p.stat().st_ino == record['inode'],
            'Frozen file identity changed: ' + str(p))
    if hash_content:
        require(sha256(p, deadline) == record['sha256'], 'Frozen file SHA256 changed: ' + str(p))


def budget(gpus, timeout, maximum, grace):
    require(gpus == '0,1', 'M1 fixes exactly --gpus 0,1')
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in (timeout, maximum, grace)), 'Budget values must be finite and positive')
    require(timeout <= 3600 and maximum <= 2, 'M1 hard cap: 3600 seconds, 2 allocated GPUh')
    require(2 * timeout / 3600 <= maximum, 'Two-card wall reservation exceeds GPUh budget')
    require(grace < timeout / 2, 'Cleanup grace must be less than half the wall budget')
    return {'gpus': ['0', '1'], 'timeout_seconds': timeout, 'cleanup_grace_seconds': grace,
            'reserved_allocated_gpu_hours': 2 * timeout / 3600, 'hard_cap_gpu_hours': 2.0}


def original_neg(source):
    tree = ast.parse(Path(source).read_text(encoding='utf-8'))
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NEG'
                                                       for t in node.targets)]
    require(len(values) == 1 and isinstance(values[0], str) and values[0], 'P1 original NEG missing')
    return values[0]


def validate_cases(jobs, manifest):
    require(isinstance(jobs, list) and len(jobs) == 24, 'P1 must contain its original 24 jobs')
    require(jobs == manifest.get('jobs'), 'P1 frozen jobs differ from its manifest')
    cases, ids = defaultdict(list), set()
    for job in jobs:
        require(job['id'] not in ids, 'Duplicate P1 job ID')
        ids.add(job['id'])
        cases[job['case_id']].append(job)
        require(job['model'] in MODELS and job['cfg'] in (3, 5), 'Unexpected P1 arm')
        require(type(job['seed']) is int and job['seed'] == 0, 'P1 seed must remain zero')
        require(job['steps'] == 50 and job['shift'] == 5, 'P1 scheduler differs')
        require((job['width'], job['height'], job['num_frames']) == (W, H, FRAMES), 'P1 shape differs')
        require(job.get('cfg_zero_star') is False and job.get('cfg_zero_steps') == 0, 'P1 Zero differs')
        require(job['prompt_review']['prompt_sha256'] == text_hash(job['prompt']), 'P1 corrected label hash differs')
        if job['model'] == 'base':
            require(job['ckpt'] == 'NONE', 'P1 base checkpoint differs')
        else:
            require(job['ckpt_sha256'] == A07_SHA256, 'P1 A07 SHA differs')
    require(tuple(cases) == CASE_ORDER, 'P1 six-case order differs')
    result = []
    same_keys = ('canonical_source', 'image', 'image_sha256', 'prompt', 'prompt_review', 'seed',
                 'shift', 'steps', 'width', 'height', 'num_frames')
    for case_id in CASE_ORDER:
        rows = cases[case_id]
        require(len(rows) == 4 and {(r['model'], r['cfg']) for r in rows}
                == {(m, s) for m in MODELS for s in (3, 5)}, 'P1 case coverage differs')
        require(all(all(r[k] == rows[0][k] for k in same_keys) for r in rows), 'P1 case state differs by arm')
        result.append({k: rows[0][k] for k in same_keys} | {'case_id': case_id})
    return result


def validate_cache(models_dir, ckpt, deadline=None):
    models_dir = Path(models_dir).absolute()
    require(models_dir == models_dir.resolve(), 'Model root must be real local cache, not a symlink')
    records = {rel: file_record(models_dir / rel, pin, deadline, local=True)
               for rel, pin in CACHE_PINS.items()}
    key = text_hash(str(Path(ckpt).resolve())) + '.safetensors'
    records['a07'] = file_record(models_dir / 'wm_checkpoints' / key, A07_SHA256, deadline, local=True)
    return records


def preflight(p1_run, out_dir, models_dir=MODELS_DEFAULT, deadline=None):
    p1_run, out_dir = Path(p1_run).resolve(), Path(out_dir).absolute()
    require(not out_dir.exists() and not out_dir.is_symlink(), '--out-dir must be a new directory; no resume')
    require(out_dir.resolve() != p1_run and not out_dir.resolve().is_relative_to(p1_run), 'Cannot write inside P1 run')
    mpath = p1_run / 'probe_manifest.json'
    m = read_json(mpath)
    require(m.get('protocol') == 'internal-dev-teacher-cfg-v1', 'Wrong P1 protocol')
    require(m.get('checkpoint_sha256', {}).get('m_full1_a07') == A07_SHA256, 'P1 checkpoint pin differs')
    records = {'probe_manifest': file_record(mpath, deadline=deadline)}
    records['jobs.frozen'] = file_record(p1_run / 'jobs.frozen.jsonl', m['jobs_frozen_sha256'], deadline)
    records['jobs.input'] = file_record(p1_run / 'jobs.input.jsonl', m['jobs_input_sha256'], deadline)
    jobs = [json.loads(x) for x in (p1_run / 'jobs.frozen.jsonl').read_text().splitlines() if x.strip()]
    cases = validate_cases(jobs, m)
    source = {}
    require({'gen_jobs.py', 'run_teacher_probe.py'} <= m['source_sha256'].keys(), 'P1 source snapshot incomplete')
    for rel, pin in m['source_sha256'].items():
        source[rel] = file_record(private_path(p1_run / 'code', rel), pin, deadline)
    framework = {}
    expected = m['framework_source_sha256']
    require({'diffsynth/__init__.py', 'diffsynth/pipelines/wan_video.py',
             'diffsynth/diffusion/flow_match.py'} <= expected.keys(), 'P1 framework snapshot incomplete')
    actual = {p.relative_to(p1_run / 'framework').as_posix()
              for p in (p1_run / 'framework').rglob('*') if p.is_file() and
              '__pycache__' not in p.parts and not p.name.endswith('.pyc')}
    require(actual == set(expected), 'P1 framework file coverage differs')
    for rel, pin in expected.items():
        framework[rel] = file_record(private_path(p1_run / 'framework', rel), pin, deadline, allow_empty=True)
    neg = original_neg(p1_run / 'code/gen_jobs.py')
    summary = read_json(p1_run / 'probe_summary.json')
    require(summary.get('status') == 'completed' and summary.get('completed_outputs') == 24
            and not summary.get('failures') and summary.get('manifest_sha256') == m['jobs_frozen_sha256'],
            'P1 completion audit failed')
    records['probe_summary'] = file_record(p1_run / 'probe_summary.json', deadline=deadline)
    outputs = {r['id']: r for r in summary['outputs']}
    require(len(outputs) == 24 and set(outputs) == {j['id'] for j in jobs}, 'P1 output coverage differs')
    for j in jobs:
        o = outputs[j['id']]
        require(Path(o['out']).resolve() == Path(j['out']).resolve()
                and Path(j['out']).resolve().is_relative_to(p1_run / 'videos'), 'P1 output path differs')
        require((o['decode']['frames'], o['decode']['width'], o['decode']['height']) == (FRAMES, W, H),
                'P1 decoded shape differs')
        records['output:' + j['id']] = file_record(j['out'], o['sha256'], deadline)
    csv_cache = {}
    for c in cases:
        records['input:' + c['case_id']] = file_record(c['image'], c['image_sha256'], deadline)
        require(Path(c['image']).suffix.lower() == '.mp4', 'M1 requires the real P1 GT MP4, not an image')
        review = c['prompt_review']
        for label, path in (('original', review['csv_path']), ('preserved', review['preserved_csv'])):
            require(m['original_csv_sha256'].get(review['csv_path']) == review['csv_sha256'], 'P1 CSV pin differs')
            records['csv:' + label + ':' + review['csv_sha256']] = file_record(path, review['csv_sha256'], deadline)
        if review['csv_path'] not in csv_cache:
            with Path(review['csv_path']).open(encoding='utf-8-sig', newline='') as f:
                csv_cache[review['csv_path']] = list(csv.DictReader(f))
        rows = csv_cache[review['csv_path']]
        csv_root = Path(review['csv_path']).resolve().parents[2]
        # Match lexical full paths first: resolving every FT2 row incurs hundreds
        # of thousands of shared-filesystem lookups. Resolve only the unique match.
        input_path = str(Path(c['image']).resolve())
        matches = [(i, row) for i, row in enumerate(rows)
                   if os.path.abspath(os.path.join(str(csv_root), row['video'])) == input_path]
        require(len(matches) == 1, 'P1 GT must match exactly one original CSV row by full path')
        row_index, row = matches[0]
        require(Path(os.path.join(str(csv_root), row['video'])).resolve() == Path(input_path),
                'Unique CSV source points to a different real file')
        if 'csv_row' in review:
            require(row_index == review['csv_row'] and row[review['video_column']] == review['original_video']
                    and text_hash(row[review['prompt_column']]) == review['original_prompt_sha256'],
                    'P1 original CSV row differs')
        c['original_csv_binding'] = {'row_index': row_index, 'video': row['video'],
                                     'original_prompt_sha256': text_hash(row['prompt'])}
    checkpoints = {j['ckpt'] for j in jobs if j['model'] == 'm_full1_a07'}
    require(len(checkpoints) == 1, 'P1 A07 identity differs')
    caches = validate_cache(models_dir, checkpoints.pop(), deadline)
    return {'protocol': 'teacher-mechanism-m1-v1', 'p1_run': str(p1_run), 'cases': cases,
            'negative_prompt': neg, 'negative_prompt_sha256': text_hash(neg), 'p1_files': records,
            'p1_source': source, 'framework': framework, 'model_cache': caches,
            'indices': list(INDICES), 'shape': [FRAMES, W, H], 'expected_inputs': 18,
            'expected_predictions': 72, 'cfg_merge': False, 'tea_cache': False,
            'training': False, 'rollout': False, 'video_output': False,
            'scope': 'training_visible_GT_forward_process_diagnostic_only'}


def freeze_run(run, manifest, allocation, deadline, directory_created=False):
    if not directory_created:
        run.mkdir(mode=0o700)
    for name in ('code', 'framework', 'p1', 'private', 'logs'):
        (run / name).mkdir(mode=0o700)
    snapshot_hashes = {}
    own = Path(__file__).resolve()
    sources = [('code/' + own.name, file_record(own, deadline=deadline))]
    sources += [('code/p1_' + rel, r) for rel, r in manifest['p1_source'].items()]
    sources += [('framework/' + rel, r) for rel, r in manifest['framework'].items()]
    sources += [('p1/' + key + ('.jsonl' if key.startswith('jobs.') else '.json'),
                 manifest['p1_files'][key]) for key in ('probe_manifest', 'probe_summary', 'jobs.frozen', 'jobs.input')]
    for rel, rec in sources:
        check_deadline(deadline)
        data = Path(rec['path']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == rec['sha256'], 'Source changed while freezing')
        target = private_path(run, rel)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_bytes(target, data)
        snapshot_hashes[rel] = rec['sha256']
    manifest = dict(manifest, allocation=allocation, snapshot_sha256=snapshot_hashes,
                    runner_source=sources[0][1], frozen_framework_root=str(run / 'framework'))
    atomic_json(run / 'mechanism_manifest.json', manifest)
    return manifest


def assert_idle(deadline):
    check_deadline(deadline)
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       check=True, capture_output=True, text=True, timeout=min(15, deadline - time.perf_counter()))
    memory = dict((a.strip(), int(b.strip())) for a, b in (s.split(',') for s in r.stdout.splitlines()))
    require(all(g in memory and memory[g] < 100 for g in ('0', '1')),
            'GPU 0/1 missing or uses >=100 MiB; never stop other jobs')
    check_deadline(deadline)


def verify_snapshot(run, manifest, deadline):
    for rel, expected in manifest['snapshot_sha256'].items():
        require(sha256(private_path(run, rel), deadline) == expected, 'Frozen runtime changed: ' + rel)


def verify_final_sources(run, manifest, deadline):
    verify_snapshot(run, manifest, deadline)
    groups = ('p1_files', 'p1_source', 'framework', 'model_cache')
    for group in groups:
        for record in manifest[group].values():
            validate_record(record, deadline, hash_content=True)
    validate_record(manifest['runner_source'], deadline, hash_content=True)
    return {'status': 'passed', 'sha256_reverified': True,
            'counts': {group: len(manifest[group]) for group in groups},
            'runner_source_reverified': True, 'snapshot_files': len(manifest['snapshot_sha256'])}


def tensor_hash(tensor):
    import torch
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(json.dumps({'shape': list(t.shape), 'dtype': str(t.dtype)}, sort_keys=True).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tensor_record(tensor):
    import torch
    require(tensor.device.type == 'cpu', 'Persistent tensors must be CPU only')
    require(bool(torch.isfinite(tensor).all()), 'Nonfinite tensor')
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': tensor_hash(tensor), 'finite': True}


def save_tensors(path, tensors):
    import torch
    path = Path(path)
    require(not path.exists(), 'No tensor overwrite')
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('xb') as f:
        os.chmod(tmp, 0o600)
        torch.save(tensors, f)
    os.link(tmp, path)
    tmp.unlink()
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size,
            'tensors': {k: tensor_record(v) for k, v in tensors.items()}}


def load_tensors(record):
    import torch
    require(sha256(record['path']) == record['sha256'], 'Private tensor file hash changed')
    tensors = torch.load(record['path'], map_location='cpu', weights_only=True)
    require(set(tensors) == set(record['tensors']), 'Tensor key coverage changed')
    for k, t in tensors.items():
        require(tensor_record(t) == record['tensors'][k], 'Private tensor content changed: ' + k)
    return tensors


def frozen_framework(run):
    framework = str((run / 'framework').resolve())
    sys.path.insert(0, framework)
    import diffsynth
    require(Path(diffsynth.__file__).resolve().is_relative_to(Path(framework)), 'Imported nonfrozen framework')
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    return WanVideoPipeline, ModelConfig


def build_pipe(run, manifest, model, deadline):
    import torch
    Pipeline, Config = frozen_framework(run)
    records = manifest['model_cache']
    for record in records.values():
        validate_record(record, deadline, hash_content=False)
        require(Path(record['path']).absolute() == Path(record['path']).resolve(), 'Cache now links shared weights')
    shards = [r['path'] for rel, r in records.items() if '/diffusion_pytorch_model-' in rel]
    t5 = next(r['path'] for rel, r in records.items() if 'models_t5_' in rel)
    vae = next(r['path'] for rel, r in records.items() if '/Wan2.2_VAE.' in rel)
    tokenizer = str(Path(next(r['path'] for rel, r in records.items() if rel.endswith('/spiece.model'))).parent)
    # Explicit paths eliminate ModelConfig's converted-file/download/fallback search.
    pipe = Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[Config(path=t5, skip_download=True), Config(path=sorted(shards), skip_download=True),
                       Config(path=vae, skip_download=True)],
        tokenizer_config=Config(path=tokenizer, skip_download=True))
    if model == 'm_full1_a07':
        from safetensors.torch import load_file
        weights = load_file(records['a07']['path'], device='cpu')
        pipe.dit.load_state_dict(weights, strict=True)
        del weights
    require(pipe.dit2 is None and pipe.dit.fuse_vae_embedding_in_latents, 'M1 requires native TI2V5B single DiT fused image')
    require(not pipe.dit.require_clip_embedding and not pipe.dit.require_vae_embedding, 'Unexpected TI2V auxiliary conditioning')
    for name in ('dit', 'vae', 'text_encoder'):
        getattr(pipe, name).eval()
    pipe.scheduler.set_timesteps(50, denoising_strength=1.0, shift=5, training=False)
    return pipe


def cover_center_crop(image):
    from PIL import Image
    image = image.convert('RGB')
    scale = max(W / image.width, H / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left, top = (image.width - W) // 2, (image.height - H) // 2
    return image.crop((left, top, left + W, top + H))


def decode_gt(path):
    import imageio.v2 as imageio
    from PIL import Image
    reader = imageio.get_reader(str(path))
    try:
        frames = [cover_center_crop(Image.fromarray(frame)) for frame in reader]
    finally:
        reader.close()
    require(len(frames) == FRAMES, 'GT must fully decode to exactly 121 frames')
    return frames


def make_shared_states(scheduler, x0, first_frame):
    import torch
    require(x0.device.type == first_frame.device.type == 'cpu', 'Shared construction must use CPU tensors')
    noise_fp32 = torch.randn(x0.shape, generator=torch.Generator(device='cpu').manual_seed(0),
                             device='cpu', dtype=torch.float32)
    noise = noise_fp32.to(torch.bfloat16)
    target = scheduler.training_target(x0, noise, scheduler.timesteps[0]).cpu()
    common = {'x0': x0, 'first_frame_latents': first_frame, 'noise_fp32': noise_fp32,
              'noise': noise, 'target': target}
    states = []
    for index in INDICES:
        raw_t = scheduler.timesteps[index]
        timestep = raw_t.unsqueeze(0).to(device='cpu', dtype=torch.bfloat16)
        selected = int(torch.argmin((scheduler.timesteps - timestep.cpu()).abs()))
        xt = scheduler.add_noise(x0, noise, timestep=timestep).cpu()
        xt[:, :, 0:1] = first_frame
        sigma = float(scheduler.sigmas[index])
        sigma_actual = float(scheduler.sigmas[selected])
        next_sigma = float(scheduler.sigmas[index + 1]) if index + 1 < len(scheduler.sigmas) else 0.0
        metadata = {'index': index, 'raw_timestep': float(raw_t), 'bf16_timestep': float(timestep.item()),
                    'raw_sigma': sigma, 'add_noise_index': selected, 'add_noise_sigma': sigma_actual,
                    'next_raw_sigma': next_sigma, 'step_delta_sigma': next_sigma - sigma}
        states.append((metadata, {'latents': xt, 'timestep': timestep}))
    return common, states


def prepare_cache(run, manifest, pipe, deadline):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder, WanVideoUnit_ImageEmbedderFused
    prepared = {'commons': [], 'states': [], 'scheduler':
                {'sigmas': pipe.scheduler.sigmas.tolist(), 'timesteps': pipe.scheduler.timesteps.tolist(),
                 'steps': 50, 'shift': 5}, 'preparation_gpu': '0', 'complete': False}
    for case in manifest['cases']:
        check_deadline(deadline)
        validate_record(manifest['p1_files']['input:' + case['case_id']], deadline)
        frames = decode_gt(case['image'])
        pipe.load_models_to_device(['vae'])
        x0 = pipe.vae.encode(pipe.preprocess_video(frames), device=pipe.device,
                             tiled=True, tile_size=(30, 52), tile_stride=(15, 26))
        x0 = x0.to(dtype=torch.bfloat16).detach().cpu()
        image = WanVideoUnit_ImageEmbedderFused().process(pipe, frames[0],
            x0.clone().to(pipe.device), H, W, True, (30, 52), (15, 26))
        first = image['first_frame_latents'].to(torch.bfloat16).detach().cpu()
        common, states = make_shared_states(pipe.scheduler, x0, first)
        embedder = WanVideoUnit_PromptEmbedder()
        common['positive'] = embedder.process(pipe, case['prompt'], True)['context'].detach().cpu()
        common['negative'] = embedder.process(pipe, manifest['negative_prompt'], False)['context'].detach().cpu()
        require(tensor_hash(common['positive']) != tensor_hash(common['negative']), 'Real NEG context cannot be pos copy')
        cid = case['case_id']
        record = save_tensors(run / 'private' / (cid + '_common.pt'), common)
        prepared['commons'].append({'case_id': cid, 'cache': record})
        for meta, tensors in states:
            record = save_tensors(run / 'private' / (cid + '_t' + str(meta['index']) + '.pt'), tensors)
            prepared['states'].append({'case_id': cid, 'scheduler': meta, 'cache': record,
                                       'x_t_sha256': record['tensors']['latents']['sha256']})
        del frames, image, common, states, x0, first
        check_deadline(deadline)
    require(len(prepared['states']) == 18, 'Prepared input coverage differs')
    prepared['complete'] = True
    atomic_json(run / 'prepared_index.json', prepared)
    return prepared


def cloned_inputs(state, common, device):
    # Even model_fn implementations that mutate tensors cannot contaminate another branch.
    return {'latents': state['latents'].to(device).clone(),
            'timestep': state['timestep'].to(device).clone(),
            'first_frame_latents': common['first_frame_latents'].to(device).clone(),
            'fuse_vae_embedding_in_latents': True, 'cfg_merge': False, 'tea_cache': None,
            'use_gradient_checkpointing': False, 'use_gradient_checkpointing_offload': False}


def predict(run, manifest, pipe, prepared, model, report, deadline):
    import torch
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    commons = {c['case_id']: c['cache'] for c in prepared['commons']}
    common, current_case = None, None
    for entry in prepared['states']:
        check_deadline(deadline)
        cid, index = entry['case_id'], entry['scheduler']['index']
        if cid != current_case:
            common = load_tensors(commons[cid])
            current_case = cid
        state = load_tensors(entry['cache'])
        expected_hash = entry['x_t_sha256']
        for branch in BRANCHES:
            check_deadline(deadline)
            t0 = time.perf_counter()
            inputs = cloned_inputs(state, common, pipe.device)
            inputs['context'] = common[branch].to(pipe.device).clone()
            require(tensor_hash(inputs['latents']) == expected_hash, 'Worker model_fn input x_t differs')
            output = pipe.model_fn(**models, **inputs).detach().cpu()
            torch.cuda.synchronize()
            require(output.shape == state['latents'].shape, 'Prediction shape differs from same x_t')
            record = save_tensors(run / 'private' / (model + '_' + branch + '_' + cid + '_t' + str(index) + '.pt'),
                                  {'prediction': output})
            report['predictions'].append({'model': model, 'branch': branch, 'case_id': cid, 'index': index,
                'x_t_sha256': expected_hash, 'context_sha256': tensor_hash(common[branch]),
                'timestep_sha256': tensor_hash(state['timestep']), 'cache': record,
                'seconds': time.perf_counter() - t0})
            require(tensor_hash(state['latents']) == expected_hash, 'Cached x_t mutated by branch')
            del inputs, output
        del state
    require(len(report['predictions']) == 36, 'Worker prediction coverage differs')


def validate_coverage(prepared, predictions):
    expected_states = {(c, i) for c in CASE_ORDER for i in INDICES}
    states = {(e['case_id'], e['scheduler']['index']): e for e in prepared['states']}
    require(prepared.get('complete') is True and len(prepared['states']) == 18 and set(states) == expected_states,
            'Expected exactly 18 shared inputs')
    require(len(prepared['commons']) == 6 and {c['case_id'] for c in prepared['commons']} == set(CASE_ORDER),
            'Expected exactly six once-prepared common caches')
    expected = {(c, i, m, b) for c, i in expected_states for m in MODELS for b in BRANCHES}
    seen = set()
    for p in predictions:
        key = (p['case_id'], p['index'], p['model'], p['branch'])
        require(key not in seen and key in expected, 'Duplicate/unexpected single-branch prediction')
        seen.add(key)
        state = states[(p['case_id'], p['index'])]
        require(p['x_t_sha256'] == state['x_t_sha256'], 'Cross-model/branch x_t hash differs')
        require(p['timestep_sha256'] == state['cache']['tensors']['timestep']['sha256'], 'Branch timestep differs')
        common = next(c['cache'] for c in prepared['commons'] if c['case_id'] == p['case_id'])
        require(p['context_sha256'] == common['tensors'][p['branch']]['sha256'], 'Branch context differs')
        require(p['cache']['tensors']['prediction']['finite'] is True, 'Nonfinite prediction audit')
    require(len(predictions) == 72 and seen == expected, 'Expected exactly 72 single-branch predictions')


def rms(t):
    return float(t.float().square().mean().sqrt())


def angle(a, b):
    import torch
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if float(na) == 0.0 or float(nb) == 0.0:
        return {'cosine': None, 'degrees': None, 'reason': 'zero_norm'}
    cosine = float((a.dot(b) / (na * nb)).clamp(-1, 1))
    return {'cosine': cosine, 'degrees': math.degrees(math.acos(cosine)), 'reason': None}


def future_stats(t):
    import torch
    require(t.device.type == 'cpu' and t.ndim == 5 and t.shape[2] > 1, 'Expected full CPU BCTHW tensor')
    t = t.float()[:, :, 1:]
    require(bool(torch.isfinite(t).all()), 'Nonfinite CPU analysis')
    mean = t.mean(dim=2, keepdim=True)
    return {'future_rms': rms(t), 'per_latent_frame_rms': [rms(t[:, :, i]) for i in range(t.shape[2])],
            'latent_frame_indices': list(range(1, t.shape[2] + 1)), 'temporal_mean_rms': rms(mean),
            'temporal_residual_rms': rms(t - mean),
            'adjacent_latent_difference_rms': rms(t[:, :, 1:] - t[:, :, :-1]) if t.shape[2] > 1 else None}


def relative_rms(numerator, denominator, multiplier=1.0):
    a = abs(multiplier) * rms(numerator[:, :, 1:])
    b = rms(denominator[:, :, 1:])
    return {'ratio': a / b if b > 0 else None, 'numerator_rms': a,
            'denominator_rms': b, 'reason': None if b > 0 else 'zero_denominator_norm'}


def zero_guidance(p, n, scale):
    p, n = p.float(), n.float()
    axes = tuple(range(1, p.ndim))
    # Production reduction includes FIRST latent. Only statistics exclude it.
    k = (p * n).sum(dim=axes, keepdim=True) / (n.square().sum(dim=axes, keepdim=True) + 1e-8)
    kn = k * n
    return kn + scale * (p - kn), k


def formula_residual(actual, expected):
    value = float((actual - expected).abs().max())
    tolerance = 2e-5 * max(1.0, float(actual.abs().max()), float(expected.abs().max()))
    require(math.isfinite(value) and value <= tolerance, 'FP32 formula identity residual exceeds tolerance')
    return {'max_abs': value, 'tolerance': tolerance, 'passes': True}


def analyze_pair(bp, bn, ap, an, target, delta_sigma, xt):
    import torch
    p, n, P, N, target = [t.float().cpu() for t in (bp, bn, ap, an, target)]
    require(all(t.shape == p.shape and bool(torch.isfinite(t).all()) for t in (n, P, N, target)),
            'Analysis tensor shapes/finiteness differ')
    dp, dn, dd = P - p, N - n, (P - N) - (p - n)
    require(xt.device.type == 'cpu' and xt.shape == p.shape, 'Same-state x_t is required for relative updates')
    result = {'x_t': future_stats(xt), 'models': {},
              'deltas': {'positive': future_stats(dp), 'negative': future_stats(dn),
              'p_minus_n': future_stats(dd), 'positive_relative_base': relative_rms(dp, p),
              'negative_relative_base': relative_rms(dn, n),
              'p_minus_n_relative_base': relative_rms(dd, p - n)}, 'cross': {}}
    for name, pos, neg in (('base', p, n), ('m_full1_a07', P, N)):
        diff = pos - neg
        fpos, fneg = pos[:, :, 1:], neg[:, :, 1:]
        item = {'positive': future_stats(pos), 'negative': future_stats(neg), 'p_minus_n': future_stats(diff),
                'positive_target_raw_mse': float((fpos - target[:, :, 1:]).square().mean()),
                'negative_relative_positive': relative_rms(neg, pos),
                'p_minus_n_relative_positive': relative_rms(diff, pos),
                'angles': {'p_n': angle(fpos, fneg), 'p_diff': angle(fpos, diff[:, :, 1:]),
                           'n_diff': angle(fneg, diff[:, :, 1:])}, 'cfg': {}}
        for scale in (3, 5):
            g = neg + scale * diff
            z, k = zero_guidance(pos, neg, scale)
            item['cfg'][str(scale)] = {'guided': future_stats(g), 'one_step_future_update_rms':
                abs(delta_sigma) * rms(g[:, :, 1:]),
                'one_step_update_relative_x_t': relative_rms(g, xt, delta_sigma),
                'guided_relative_positive': relative_rms(g, pos),
                'positive_term': future_stats(scale * pos),
                'negative_term': future_stats((1 - scale) * neg), 'positive_negative_angle':
                angle((scale * pos)[:, :, 1:], ((1 - scale) * neg)[:, :, 1:]),
                'formula': formula_residual(g, scale * pos + (1 - scale) * neg),
                'angles': {'p_g': angle(fpos, g[:, :, 1:]), 'n_g': angle(fneg, g[:, :, 1:]),
                           'diff_g': angle(diff[:, :, 1:], g[:, :, 1:])},
                'zero_projection': {'k_full_production_reduction': k.reshape(-1).tolist(),
                    'guided_fp32': future_stats(z), 'guided_production_bf16': future_stats(z.to(torch.bfloat16)),
                    'production_cast_delta_rms': rms((z.to(torch.bfloat16).float() - z)[:, :, 1:]),
                    'one_step_future_update_rms': abs(delta_sigma) * rms(z[:, :, 1:]),
                    'one_step_update_relative_x_t': relative_rms(z, xt, delta_sigma),
                    'normal_projected_angle': angle(g[:, :, 1:], z[:, :, 1:]), 'early_zero_steps': 0}}
        result['models'][name] = item
    result['deltas']['positive_negative_angle'] = angle(dp[:, :, 1:], dn[:, :, 1:])
    result['cross_model_angles'] = {'positive': angle(p[:, :, 1:], P[:, :, 1:]),
        'negative': angle(n[:, :, 1:], N[:, :, 1:]),
        'p_minus_n': angle((p - n)[:, :, 1:], (P - N)[:, :, 1:])}
    for scale in (3, 5):
        BB = scale * p + (1 - scale) * n
        AB = scale * P + (1 - scale) * n
        BA = scale * p + (1 - scale) * N
        AA = scale * P + (1 - scale) * N
        u, v = scale * dp, (1 - scale) * dn
        total = AA - BB
        uf, vf, tf = u[:, :, 1:], v[:, :, 1:], total[:, :, 1:]
        cross = float(2 * (uf * vf).mean())
        squared = float(uf.square().mean() + vf.square().mean() + 2 * (uf * vf).mean())
        result['cross'][str(scale)] = {
            'BB': future_stats(BB), 'AB': future_stats(AB), 'BA': future_stats(BA), 'AA': future_stats(AA),
            'delta_guided': future_stats(total), 'base_a07_guided_angle': angle(BB[:, :, 1:], AA[:, :, 1:]),
            'delta_positive_contribution': future_stats(u),
            'delta_negative_contribution': future_stats(v),
            'delta_guided_relative_base': relative_rms(total, BB),
            'delta_positive_contribution_relative_base_guided': relative_rms(u, BB),
            'delta_negative_contribution_relative_base_guided': relative_rms(v, BB),
            'delta_one_step_update_relative_x_t': relative_rms(total, xt, delta_sigma),
            'contribution_angle': angle(uf, vf),
            'cross_term_mean_square': cross, 'decomposed_total_mean_square': squared,
            'actual_total_mean_square': float(tf.square().mean()),
            'AB_minus_BB_identity': formula_residual(AB - BB, u),
            'BA_minus_BB_identity': formula_residual(BA - BB, v),
            'AA_minus_BB_identity': formula_residual(total, u + v),
            'delta_diff_identity': formula_residual(dd, dp - dn),
            'mean_square_identity': formula_residual(tf.square().mean(),
                uf.square().mean() + vf.square().mean() + 2 * (uf * vf).mean())}
    return result


def analyze_run(run, manifest, deadline):
    prepared = read_json(run / 'prepared_index.json')
    reports = [read_json(run / ('worker_' + m + '.json')) for m in MODELS]
    require(all(r['status'] == 'completed' for r in reports), 'Analysis requires both completed workers')
    predictions = [p for report in reports for p in report['predictions']]
    validate_coverage(prepared, predictions)
    lookup = {(p['case_id'], p['index'], p['model'], p['branch']): p for p in predictions}
    commons = {c['case_id']: c['cache'] for c in prepared['commons']}
    output = {'protocol': manifest['protocol'], 'status': 'completed', 'inputs': 18, 'predictions': 72,
        'first_latent_excluded_from_statistics': True, 'zero_k_includes_first_latent': True,
        'math_dtype': 'torch.float32 CPU', 'target': 'native scheduler.training_target(eps_bf16-x0_bf16)',
        'interpretation': 'GT forward process only; latent temporal differences are not optical flow or action success; '
                          'NEG MSE is not negative-branch quality; cross guidance is not mixed-video evidence', 'cases': []}
    for state in prepared['states']:
        check_deadline(deadline)
        cid, index = state['case_id'], state['scheduler']['index']
        common = load_tensors(commons[cid])
        tensors = load_tensors(state['cache'])
        require(tensor_hash(tensors['latents']) == state['x_t_sha256'], 'Analysis x_t hash differs')
        require(tensor_hash(tensors['latents'][:, :, :1]) == tensor_hash(common['first_frame_latents']),
                'Shared first latent was not overwritten')
        values = [load_tensors(lookup[(cid, index, m, b)]['cache'])['prediction']
                  for m in MODELS for b in BRANCHES]
        metrics = analyze_pair(*values, common['target'], state['scheduler']['step_delta_sigma'], tensors['latents'])
        output['cases'].append({'case_id': cid, 'scheduler': state['scheduler'],
                                'x_t_sha256': state['x_t_sha256'], 'metrics': metrics})
        del common, tensors, values, metrics
    check_deadline(deadline)
    atomic_json(run / 'mechanism_analysis.json', output)
    return output


def worker(run, role, manifest_sha, deadline):
    require(os.environ.get('M1_PARENT_PID') == str(os.getppid()), '--worker is internal to coordinator')
    require(sha256(run / 'mechanism_manifest.json', deadline) == manifest_sha, 'Worker manifest SHA differs')
    manifest = read_json(run / 'mechanism_manifest.json')
    verify_snapshot(run, manifest, deadline)
    expected_gpu = {'base': '0', 'm_full1_a07': '1', 'analysis': ''}[role]
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == expected_gpu, 'Worker GPU assignment differs')
    started = time.perf_counter()
    report = {'model': role, 'status': 'running', 'gpu': expected_gpu, 'pid': os.getpid(),
              'predictions': [], 'preparation_count': 0, 'torch_version': None}
    pipe = torch = None
    failed = False
    try:
        import torch
        report['torch_version'] = torch.__version__
        if role == 'analysis':
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            result = analyze_run(run, manifest, deadline)
            report['analysis_cases'] = len(result['cases'])
        else:
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                t0 = time.perf_counter()
                pipe = build_pipe(run, manifest, role, deadline)
                torch.cuda.synchronize()
                report['load_seconds'] = time.perf_counter() - t0
                if role == 'base':
                    t0 = time.perf_counter()
                    prepared = prepare_cache(run, manifest, pipe, deadline)
                    report['preparation_count'] = 1
                    report['preparation_seconds'] = time.perf_counter() - t0
                else:
                    t0 = time.perf_counter()
                    while not (run / 'prepared_index.json').exists():
                        check_deadline(deadline)
                        time.sleep(min(0.1, max(0.0, deadline - time.perf_counter())))
                    prepared = read_json(run / 'prepared_index.json')
                    report['cache_wait_seconds'] = time.perf_counter() - t0
                report['prepared_index_sha256'] = sha256(run / 'prepared_index.json', deadline)
                predict(run, manifest, pipe, prepared, role, report, deadline)
                report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    except BaseException as error:
        failed = True
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        if pipe is not None:
            del pipe
        if torch is not None and role != 'analysis':
            try:
                report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                torch.cuda.empty_cache()
            except Exception as error:
                failed = True
                report['cleanup_error'] = str(error)
        report['status'] = 'failed' if failed else 'completed'
        report['worker_seconds'] = time.perf_counter() - started
        atomic_json(run / ('worker_' + role + '.json'), report)
    return int(failed)


def enable_subreaper():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(36, 1, 0, 0, 0) == 0, 'Cannot enable owned-descendant cleanup subreaper')


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def reap_group(pgid):
    while True:
        try:
            pid, _ = os.waitpid(-pgid, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break


def stop_workers(workers, hard_deadline, grace):
    # All PGIDs below were created by this runner using start_new_session=True.
    # Signal groups even if their Python leader exited, then reap adopted FFmpeg children.
    for info in workers:
        try:
            os.killpg(info['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    term_until = min(hard_deadline, time.perf_counter() + max(0.0, grace - 1.0))
    while time.perf_counter() < term_until:
        active = False
        for info in workers:
            info['process'].poll()
            reap_group(info['process'].pid)
            active |= group_alive(info['process'].pid)
        if not active:
            break
        time.sleep(min(0.02, max(0.0, term_until - time.perf_counter())))
    for info in workers:
        try:
            os.killpg(info['process'].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while time.perf_counter() < hard_deadline:
        for info in workers:
            info['process'].poll()
            reap_group(info['process'].pid)
        if not any(group_alive(info['process'].pid) for info in workers):
            break
        time.sleep(min(0.01, max(0.0, hard_deadline - time.perf_counter())))
    cleanup = []
    for info in workers:
        proc = info['process']
        try:
            proc.wait(timeout=max(0.0, hard_deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            pass
        reap_group(proc.pid)
        cleanup.append({'role': info['role'], 'pgid': proc.pid, 'returncode': proc.returncode,
                        'owned_group_gone': not group_alive(proc.pid)})
        info['log'].close()
    return cleanup


def spawn_worker(run, role, manifest_sha, work_deadline, workers):
    check_deadline(work_deadline)
    gpu = {'base': '0', 'm_full1_a07': '1', 'analysis': ''}[role]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, M1_PARENT_PID=str(os.getpid()),
        DIFFSYNTH_SKIP_DOWNLOAD='true', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8')
    env['PYTHONPATH'] = str(run / 'framework')
    log = (run / 'logs' / (role + '.log')).open('x')
    try:
        proc = subprocess.Popen([sys.executable, str(run / 'code/run_teacher_mechanism_probe.py'),
            '--out-dir', str(run), '--worker', role, '--manifest-sha', manifest_sha,
            '--work-deadline', str(work_deadline)], cwd=str(run / 'framework'), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        log.close()
        raise
    workers.append({'role': role, 'process': proc, 'log': log, 'started': time.perf_counter()})


def wait_workers(workers, work_deadline):
    while True:
        check_deadline(work_deadline)
        codes = [info['process'].poll() for info in workers]
        require(all(code in (None, 0) for code in codes), 'Worker failed; no retries or budget expansion')
        if all(code == 0 for code in codes):
            for info in workers:
                info['process'].wait(timeout=max(0, work_deadline - time.perf_counter()))
            return
        time.sleep(min(0.1, max(0.0, work_deadline - time.perf_counter())))


def allocated_ledger(started, ended, allocation, failed, cleanup):
    elapsed = max(0.0, ended - started)
    return {'accounting': 'Both fixed cards allocated for whole coordinator wall, including setup, cache preparation, '
                          'waiting, CPU analysis, failure and owned-child cleanup; not sum of CUDA kernel time',
        'allocated_cards': 2, 'wall_seconds': elapsed, 'allocated_gpu_hours': 2 * elapsed / 3600,
        'reserved_allocated_gpu_hours': allocation['reserved_allocated_gpu_hours'],
        'hard_wall_seconds': allocation['timeout_seconds'], 'failed': failed,
        'within_reservation': elapsed <= allocation['timeout_seconds'],
        'owned_cleanup_complete': all(c['owned_group_gone'] for c in cleanup)}


def execute(args, allocation):
    started = time.perf_counter()
    hard_deadline = started + allocation['timeout_seconds']
    work_deadline = hard_deadline - allocation['cleanup_grace_seconds']
    run = Path(args.out_dir).absolute()
    workers, cleanup, manifest, failure = [], [], None, None
    failure_traceback, final_verification = None, None
    created = False
    old_handlers = {}
    def interrupted(signum, frame):
        raise InterruptedError('Coordinator received signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, work_deadline - time.perf_counter()))
    try:
        manifest = preflight(args.p1_run, run, args.models_dir, work_deadline)
        assert_idle(work_deadline)
        enable_subreaper()
        run.mkdir(mode=0o700)
        created = True
        manifest = freeze_run(run, manifest, allocation, work_deadline, directory_created=True)
        msha = sha256(run / 'mechanism_manifest.json', work_deadline)
        spawn_worker(run, 'base', msha, work_deadline, workers)
        spawn_worker(run, 'm_full1_a07', msha, work_deadline, workers)
        wait_workers(workers, work_deadline)
        reports = [read_json(run / ('worker_' + m + '.json')) for m in MODELS]
        require(all(r['status'] == 'completed' for r in reports) and reports[0]['preparation_count'] == 1
                and reports[1]['preparation_count'] == 0, 'Shared preprocessing must happen exactly once on GPU0')
        require(reports[0]['prepared_index_sha256'] == reports[1]['prepared_index_sha256'], 'Model workers used different cache index')
        validate_coverage(read_json(run / 'prepared_index.json'), [p for r in reports for p in r['predictions']])
        spawn_worker(run, 'analysis', msha, work_deadline, workers)
        wait_workers(workers, work_deadline)
        require(read_json(run / 'mechanism_analysis.json')['status'] == 'completed', 'CPU analysis incomplete')
        final_verification = verify_final_sources(run, manifest, work_deadline)
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        failure_traceback = traceback.format_exc()
    finally:
        # A real timer also bounds blocking preflight/setup calls, not just poll loops.
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Do not allow another signal to interrupt deterministic owned-group cleanup.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        bookkeeping_reserve = min(2.0, allocation['cleanup_grace_seconds'] / 4)
        cleanup = stop_workers(workers, hard_deadline - bookkeeping_reserve,
                               allocation['cleanup_grace_seconds'] - bookkeeping_reserve)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    ended = time.perf_counter()
    ledger = allocated_ledger(started, ended, allocation, failure is not None, cleanup)
    if not ledger['within_reservation'] or not ledger['owned_cleanup_complete']:
        failure = failure or 'Hard-wall reservation or owned-child cleanup failed'
        ledger['failed'] = True
    summary = {'protocol': 'teacher-mechanism-m1-v1', 'status': 'failed' if failure else 'completed',
               'error': failure, 'traceback': failure_traceback, 'final_source_verification': final_verification,
               'ledger': ledger, 'cleanup': cleanup,
               'inputs': 18 if failure is None else None, 'predictions': 72 if failure is None else None,
               'worker_reports': {}, 'analysis_sha256': None}
    if created:
        for info in workers:
            path = run / ('worker_' + info['role'] + '.json')
            if path.exists():
                summary['worker_reports'][info['role']] = read_json(path)
        if (run / 'mechanism_analysis.json').exists():
            summary['analysis_sha256'] = sha256(run / 'mechanism_analysis.json')
        # Final CPU report reads/hashing are also allocated on BOTH cards. Charge an
        # extra bookkeeping second conservatively for publication and process exit.
        measured_end = time.perf_counter()
        summary['ledger'] = allocated_ledger(started, measured_end + 1.0, allocation,
                                             failure is not None, cleanup)
        summary['ledger']['measured_wall_seconds_before_summary_publication'] = measured_end - started
        summary['ledger']['conservative_publication_exit_allowance_seconds'] = 1.0
        summary['ledger']['publication_exit_allowance_explanation'] = (
            'The final JSON write/stdout/process exit are not GPU kernel time but remain two-card allocation. '
            'Charge one additional wall second conservatively, inside the original reservation; '
            'this is not a budget extension or a claim that the measured write took exactly one second.')
        if not summary['ledger']['within_reservation']:
            summary['status'] = 'failed'
            summary['error'] = failure or 'Final bookkeeping exceeded allocated hard wall'
            summary['ledger']['failed'] = True
        atomic_json(run / 'mechanism_summary.json', summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--p1-run', type=Path, default=P1_DEFAULT)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--models-dir', type=Path, default=MODELS_DEFAULT)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--timeout-seconds', type=float, default=3600)
    p.add_argument('--max-gpu-hours', type=float, default=2)
    p.add_argument('--cleanup-grace-seconds', type=float, default=15)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--worker', choices=(*MODELS, 'analysis'), help=argparse.SUPPRESS)
    p.add_argument('--manifest-sha', help=argparse.SUPPRESS)
    p.add_argument('--work-deadline', type=float, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.worker:
            require(args.manifest_sha is not None and args.work_deadline is not None, 'Internal worker binding required')
            return worker(args.out_dir.absolute(), args.worker, args.manifest_sha, args.work_deadline)
        allocation = budget(args.gpus, args.timeout_seconds, args.max_gpu_hours, args.cleanup_grace_seconds)
        if args.preflight_only:
            result = preflight(args.p1_run, args.out_dir, args.models_dir)
            print(json.dumps({'status': 'preflight_passed_no_gpu', 'allocation': allocation,
                              'inputs': result['expected_inputs'], 'predictions': result['expected_predictions'],
                              'case_order': [c['case_id'] for c in result['cases']],
                              'p1_files_verified': len(result['p1_files']),
                              'framework_files_verified': len(result['framework']),
                              'negative_prompt_sha256': result['negative_prompt_sha256'],
                              'model_cache_verified': {k: v['sha256'] for k, v in result['model_cache'].items()},
                              'out_dir_created': False, 'cuda_queried': False, 'torch_imported': 'torch' in sys.modules},
                             sort_keys=True, allow_nan=False))
            return 0
        result = execute(args, allocation)
        print(json.dumps({'status': result['status'], 'error': result['error'], 'ledger': result['ledger'],
                          'out_dir': str(args.out_dir)}, sort_keys=True, allow_nan=False))
        return int(result['status'] != 'completed')
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
