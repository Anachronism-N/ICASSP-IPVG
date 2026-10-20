#!/usr/bin/env python3
"""Bounded base-DiT LoRA technical smoke using corrected-label M1 caches only.

Not FT3, Student, distillation, AR, label-quality or generalization acceptance.
Top-level imports use only stdlib; --help/bad budgets never import torch.
The coordinator owns one worker; no retries, resume or downloads. CPU preflight
lazily uses CPU torch for real caches, never queries/initializes CUDA.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import run_teacher_mechanism_probe as m1

M1_DEFAULT = Path('/dockerdata/wm_tmp/teacher_m1_20261020_run01')
PROTOCOL = 'teacher-lora-training-technical-smoke-v1'
TARGETS = 'q,k,v,o,ffn.0,ffn.2'
SEED = 20261020
LABEL_SCOPE = 'P1_agent_corrected_textcache_not_original_caption_not_human_gold'


def budget(gpu, steps, timeout, maximum, grace):
    m1.require(gpu == '0', 'Exactly --gpu 0 is allowed')
    m1.require(type(steps) is int and 1 <= steps <= 4, 'Max optimizer steps must be 1..4')
    m1.require(all(type(v) in (float, int) and math.isfinite(v) and v > 0
                   for v in (timeout, maximum, grace)), 'Finite positive budget required')
    m1.require(timeout <= 7200 and maximum <= 2, 'Hard caps: 7200 seconds and 2 allocated GPUh')
    m1.require(timeout / 3600 <= maximum, 'One-card whole wall exceeds GPUh reservation')
    m1.require(grace < timeout / 2, 'Cleanup grace must be less than half wall')
    return {'gpu': '0', 'max_steps': steps, 'timeout_seconds': timeout,
            'max_gpu_hours': maximum, 'cleanup_grace_seconds': grace,
            'reserved_allocated_gpu_hours': timeout / 3600}


def safe_out(out, mrun):
    out = Path(out).absolute()
    m1.require(not out.exists() and not out.is_symlink(), 'New --out-dir required; no resume/overwrite')
    m1.require(out == out.resolve(), 'Output ancestors must not be symlinks')
    m1.require(not out.is_relative_to(Path(mrun).resolve()), 'Cannot write inside M1')
    m1.require(out.parent.is_dir(), 'Output parent must already exist')
    return out


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_common(tensors):
    import torch
    shape = (1, 48, 31, 44, 80)
    expected = {'x0': shape, 'noise': shape, 'noise_fp32': shape, 'target': shape,
                'first_frame_latents': (1, 48, 1, 44, 80),
                'positive': (1, 512, 4096), 'negative': (1, 512, 4096)}
    m1.require(set(tensors) == set(expected), 'Common cache exact key coverage differs')
    for key, value in tensors.items():
        dtype = torch.float32 if key == 'noise_fp32' else torch.bfloat16
        m1.require(value.device.type == 'cpu' and value.dtype == dtype
                   and tuple(value.shape) == expected[key] and bool(torch.isfinite(value).all()),
                   'Common tensor shape/dtype/finite differs: ' + key)
    m1.require(torch.equal(tensors['noise'], tensors['noise_fp32'].to(torch.bfloat16)), 'Noise downcast differs')
    m1.require(torch.equal(tensors['target'], tensors['noise'] - tensors['x0']), 'M1 native target differs')
    m1.require(m1.tensor_hash(tensors['positive']) != m1.tensor_hash(tensors['negative']), 'NEG copied positive')


def checked_tensors(record, run, deadline):
    m1.check_deadline(deadline)
    m1.require(Path(record['path']).resolve().is_relative_to(run / 'private'), 'M1 tensor path escapes private cache')
    m1.require(m1.sha256(record['path'], deadline) == record['sha256'], 'M1 tensor file hash differs')
    import torch
    tensors = torch.load(record['path'], map_location='cpu', weights_only=True)
    m1.require(isinstance(tensors, dict) and set(tensors) == set(record['tensors']), 'M1 tensor keys differ')
    for key, tensor in tensors.items():
        m1.check_deadline(deadline)
        m1.require(isinstance(tensor, torch.Tensor) and m1.tensor_record(tensor) == record['tensors'][key],
                   'M1 tensor metadata/hash differs: ' + key)
    return tensors


def preflight(mrun, out, models_dir, deadline=None):
    mrun = Path(mrun).resolve()
    safe_out(out, mrun)
    manifest = m1.read_json(mrun / 'mechanism_manifest.json')
    m1.require(manifest.get('protocol') == 'teacher-mechanism-m1-v1'
               and manifest.get('shape') == [121, 1280, 704]
               and manifest.get('scope') == 'training_visible_GT_forward_process_diagnostic_only', 'Wrong M1 scope')
    # Full original P1 provenance, CSV/MP4/source/frozen framework and all ten model pins.
    current = m1.preflight(manifest['p1_run'], out, models_dir, deadline)
    for key in ('cases', 'p1_files', 'p1_source', 'framework', 'model_cache', 'negative_prompt_sha256'):
        m1.require(current[key] == manifest[key], 'M1 original source binding differs: ' + key)
    m1.verify_snapshot(mrun, manifest, deadline)
    m1.validate_record(manifest['runner_source'], deadline)
    required = {'diffsynth/diffusion/training_module.py', 'diffsynth/diffusion/loss.py'}
    m1.require(required <= set(manifest['framework']), 'Frozen native training source missing')
    files = {name: m1.file_record(mrun / name, deadline=deadline)
             for name in ('mechanism_manifest.json', 'mechanism_summary.json', 'prepared_index.json', 'worker_base.json')}
    summary = m1.read_json(mrun / 'mechanism_summary.json')
    m1.require(summary.get('status') == 'completed'
               and summary.get('final_source_verification', {}).get('status') == 'passed', 'M1 not completed/verified')
    prepared = m1.read_json(mrun / 'prepared_index.json')
    worker = m1.read_json(mrun / 'worker_base.json')
    m1.require(worker.get('status') == 'completed' and worker.get('preparation_count') == 1
               and worker.get('prepared_index_sha256') == files['prepared_index.json']['sha256'], 'M1 preparation binding differs')
    m1.require(prepared.get('complete') is True and prepared.get('preparation_gpu') == '0'
               and [c['case_id'] for c in prepared['commons']] == list(m1.CASE_ORDER), 'M1 six commons order differs')
    m1.require(len(prepared['states']) == 18
               and {(s['case_id'], s['scheduler']['index']) for s in prepared['states']}
               == {(c, i) for c in m1.CASE_ORDER for i in m1.INDICES}, 'M1 state coverage differs')
    scheduler = source_module(mrun / 'framework/diffsynth/diffusion/flow_match.py', '_smoke_cpu_scheduler').FlowMatchScheduler('Wan')
    scheduler.set_timesteps(50, shift=5, training=False)
    m1.require(prepared['scheduler'] == {'steps': 50, 'shift': 5,
               'sigmas': scheduler.sigmas.tolist(), 'timesteps': scheduler.timesteps.tolist()}, 'M1 scheduler metadata differs')
    import torch
    for entry in prepared['commons']:
        common = checked_tensors(entry['cache'], mrun, deadline)
        validate_common(common)
        # Verify the seed-0 FP32 construction too, not merely arbitrary internally matching downcasts.
        expected_noise = torch.randn(common['x0'].shape, generator=torch.Generator().manual_seed(0), dtype=torch.float32)
        m1.require(torch.equal(common['noise_fp32'], expected_noise), 'M1 noise seed/source differs')
        _, states = m1.make_shared_states(scheduler, common['x0'], common['first_frame_latents'])
        actual = [s for s in prepared['states'] if s['case_id'] == entry['case_id']]
        for record, (meta, expected) in zip(actual, states):
            loaded = checked_tensors(record['cache'], mrun, deadline)
            m1.require(record['scheduler'] == meta and record['x_t_sha256'] == m1.tensor_hash(loaded['latents']), 'M1 downcast metadata differs')
            m1.require(all(torch.equal(loaded[k], expected[k]) for k in expected), 'M1 native cached state math differs')
        del common, expected_noise, states
    return {'protocol': PROTOCOL, 'm1_run': str(mrun), 'm1_manifest': manifest,
            'm1_files': files, 'prepared': prepared, 'label_scope': LABEL_SCOPE,
            'case_order': list(m1.CASE_ORDER), 'step_case_order': list(m1.CASE_ORDER[:4]),
            'source': {'runner': m1.file_record(__file__, deadline=deadline),
                       'helper': m1.file_record(m1.__file__, deadline=deadline)},
            'scope': 'technical_gradient_optimizer_checkpoint_only_no_quality_generalization',
            'full_ft3': False, 'student': False, 'distillation': False, 'autoregressive': False}


def freeze_run(run, manifest, allocation, deadline):
    for name in ('code', 'framework', 'private', 'logs'):
        (run / name).mkdir(mode=0o700)
    sources = [('code/' + Path(rec['path']).name, rec) for rec in manifest['source'].values()]
    old = manifest['m1_manifest']
    sources += [('framework/' + rel, dict(rec, path=str(Path(manifest['m1_run']) / 'framework' / rel)))
                for rel, rec in old['framework'].items()]
    hashes = {}
    for rel, record in sources:
        m1.check_deadline(deadline)
        data = Path(record['path']).read_bytes()
        m1.require(hashlib.sha256(data).hexdigest() == record['sha256'], 'Snapshot source changed')
        dest = m1.private_path(run, rel)
        dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        m1.atomic_bytes(dest, data)
        hashes[rel] = record['sha256']
    result = dict(manifest, allocation=allocation, snapshot_sha256=hashes, owner_pid=os.getpid())
    m1.atomic_json(run / 'training_manifest.json', result)
    return result


def trainable_adapters(model):
    params = dict(model.named_parameters())
    adapters = {k: p for k, p in params.items() if p.requires_grad}
    m1.require(adapters, 'No trainable adapters')
    for key, param in params.items():
        is_adapter = '.lora_A.default.weight' in key or '.lora_B.default.weight' in key
        m1.require(param.requires_grad == is_adapter, 'Non-adapter trainable or frozen adapter: ' + key)
        if is_adapter:
            import torch
            module = key.split('.lora_', 1)[0]
            m1.require(any(module == suffix or module.endswith('.' + suffix) for suffix in TARGETS.split(',')),
                       'Adapter outside fixed native targets: ' + key)
            m1.require(param.dtype == torch.float32, 'Adapters must be FP32')
    return adapters


def tree_cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tree_cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(tree_cpu(v) for v in value)
    return value


def state_hash(value):
    import torch
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            m1.require(bool(torch.isfinite(v).all()), 'Nonfinite checkpoint tensor')
            h.update(('tensor:' + str(tuple(v.shape)) + ':' + m1.tensor_hash(v.reshape(-1))).encode())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k)
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v:
                visit(x)
        else:
            m1.require(type(v) in (str, int, float, bool, type(None)), 'Unsupported checkpoint value')
            m1.require(not isinstance(v, float) or math.isfinite(v), 'Nonfinite checkpoint scalar')
            h.update((type(v).__name__ + ':' + repr(v) + ';').encode())
    visit(value)
    return h.hexdigest()


def rng_state(cuda=False):
    import torch
    return {'python': random.getstate(), 'torch_cpu': torch.get_rng_state().clone(),
            'torch_cuda': [v.cpu().clone() for v in torch.cuda.get_rng_state_all()] if cuda else []}


def restore_rng(state, cuda=False):
    import torch
    current = rng_state(cuda)
    m1.require(set(state) == set(current) and len(state['torch_cuda']) == len(current['torch_cuda']), 'RNG key/device schema differs')
    for saved, live in zip([state['torch_cpu']] + state['torch_cuda'], [current['torch_cpu']] + current['torch_cuda']):
        m1.require(isinstance(saved, torch.Tensor) and saved.dtype == torch.uint8
                   and saved.device.type == 'cpu' and saved.shape == live.shape, 'RNG tensor schema differs')
    check = random.Random()
    check.setstate(state['python'])
    random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu'])
    if cuda:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def checkpoint_state(model, optimizer, steps, cuda=False):
    adapters = trainable_adapters(model)
    return {'protocol': PROTOCOL, 'steps': steps, 'targets': TARGETS, 'rank': 32,
            'adapter_prefix': 'dit.', 'adapters': {'dit.' + k: tree_cpu(p) for k, p in adapters.items()},
            'adapter_names': ['dit.' + k for k in adapters], 'optimizer': tree_cpu(optimizer.state_dict()),
            'rng': rng_state(cuda)}


def validate_checkpoint(value, model, optimizer, steps):
    import torch
    required = {'protocol', 'steps', 'targets', 'rank', 'adapter_prefix', 'adapters', 'adapter_names', 'optimizer', 'rng'}
    m1.require(isinstance(value, dict) and set(value) == required, 'Checkpoint exact key schema differs')
    m1.require(value['protocol'] == PROTOCOL and value['steps'] == steps and value['targets'] == TARGETS
               and value['rank'] == 32 and value['adapter_prefix'] == 'dit.', 'Checkpoint training schema differs')
    params = trainable_adapters(model)
    names = ['dit.' + k for k in params]
    m1.require(value['adapter_names'] == names and set(value['adapters']) == set(names), 'Adapter keys/prefix/order differ')
    for name, live in zip(names, params.values()):
        saved = value['adapters'][name]
        m1.require(isinstance(saved, torch.Tensor) and saved.device.type == 'cpu' and saved.dtype == live.dtype
                   and saved.shape == live.shape and bool(torch.isfinite(saved).all()), 'Bad adapter shape/dtype/finite')
    opt = value['optimizer']
    live = optimizer.state_dict()
    m1.require(isinstance(opt, dict) and set(opt) == {'param_groups', 'state'}
               and opt['param_groups'] == live['param_groups'], 'Optimizer group hyperparameters/parameter order differ')
    ids = [i for g in live['param_groups'] for i in g['params']]
    m1.require(len(ids) == len(params) and len(set(ids)) == len(ids)
               and set(opt['state']) == set(ids), 'Optimizer state key coverage differs')
    for pid, param in zip(ids, params.values()):
        item = opt['state'][pid]
        m1.require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'}, 'AdamW state schema differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            t = item[key]
            m1.require(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and t.dtype == param.dtype
                       and t.shape == param.shape and bool(torch.isfinite(t).all()), 'Bad optimizer moment')
        st = item['step']
        m1.require(isinstance(st, torch.Tensor) and st.device.type == 'cpu' and st.dtype == torch.float32
                   and st.ndim == 0 and float(st) == steps, 'AdamW step count differs')
    state_hash(value)


def save_checkpoint(path, value):
    import torch
    # No full base state is ever serialized; CPU adapters + AdamW moments + RNG only.
    import io
    buf = io.BytesIO()
    torch.save(tree_cpu(value), buf)
    m1.atomic_bytes(path, buf.getvalue())
    return {'file_sha256': m1.sha256(path), 'state_sha256': state_hash(value), 'bytes': Path(path).stat().st_size}


def load_checkpoint(path, expected, model, optimizer, steps, cuda=False):
    import torch
    m1.require(m1.sha256(path) == expected['file_sha256'], 'Checkpoint file SHA256 differs')
    value = torch.load(path, map_location='cpu', weights_only=True)
    m1.require(state_hash(value) == expected['state_sha256'], 'Checkpoint state hash differs')
    validate_checkpoint(value, model, optimizer, steps)
    # Validate RNG on an isolated CPU schema before mutating adapters or optimizer.
    old_rng = rng_state(cuda)
    try:
        restore_rng(value['rng'], cuda)
    finally:
        restore_rng(old_rng, cuda)
    with torch.no_grad():
        for key, param in trainable_adapters(model).items():
            param.copy_(value['adapters']['dit.' + key].to(param.device))
    optimizer.load_state_dict(value['optimizer'])
    restore_rng(value['rng'], cuda)
    m1.require(state_hash(checkpoint_state(model, optimizer, steps, cuda)) == expected['state_sha256'], 'Exact adapter/optimizer/RNG restoration failed')
    return value


def frozen_state_hash(model, deadline=None):
    h = hashlib.sha256()
    for key, p in model.named_parameters():
        m1.check_deadline(deadline)
        if not p.requires_grad:
            m1.require(p.grad is None, 'Frozen base gradient unexpectedly populated')
            h.update(key.encode())
            h.update(m1.tensor_hash(p).encode())
    return h.hexdigest()


def optimizer_step(model, optimizer, loss):
    import torch
    m1.require(loss.numel() == 1 and bool(torch.isfinite(loss).all()), 'Nonfinite/scalar loss')
    adapters = trainable_adapters(model)
    loss.backward()
    nonzero = 0
    for key, p in model.named_parameters():
        if p.requires_grad:
            m1.require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Missing/nonfinite adapter gradient')
            nonzero += int(bool(torch.count_nonzero(p.grad)))
        else:
            m1.require(p.grad is None, 'Frozen base has gradient')
    m1.require(nonzero > 0, 'All adapter gradients zero (initial LoRA_A zeros are allowed)')
    norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), max_norm=1.0, error_if_nonfinite=True)
    before = state_hash({k: p for k, p in adapters.items()})
    optimizer.step()
    after = state_hash({k: p for k, p in adapters.items()})
    m1.require(before != after, 'Optimizer changed no adapter')
    return {'loss': float(loss.detach()), 'grad_norm_before_clip': float(norm),
            'nonzero_gradient_tensors': nonzero, 'adapter_before_sha256': before, 'adapter_after_sha256': after}


def cached_inputs(common, device):
    return {'input_latents': common['x0'].to(device).clone(),
            'context': common['positive'].to(device).clone(),
            'first_frame_latents': common['first_frame_latents'].to(device).clone(),
            'fuse_vae_embedding_in_latents': True, 'cfg_scale': 1, 'cfg_merge': False,
            'tea_cache': None, 'use_gradient_checkpointing': True,
            'use_gradient_checkpointing_offload': True}


def native_forward(pipe, common, loss_fn, capture=False):
    import torch
    original = pipe.model_fn
    observed = {}
    def wrapped(**inputs):
        prediction = original(**inputs)
        if capture:
            observed['prediction'] = prediction.detach().float().cpu().clone()
            observed['input_sha256'] = {k: m1.tensor_hash(inputs[k]) for k in ('latents', 'timestep', 'context')}
        return prediction
    pipe.model_fn = wrapped
    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = loss_fn(pipe, **cached_inputs(common, pipe.device))
        return loss, observed
    finally:
        pipe.model_fn = original


def verify_roundtrip(pipe, optimizer, common, loss_fn, path, steps, deadline, progress):
    import torch
    saved = checkpoint_state(pipe.dit, optimizer, steps, cuda=True)
    validate_checkpoint(saved, pipe.dit, optimizer, steps)
    record = save_checkpoint(path, saved)
    m1.check_deadline(deadline)
    # Both no-update forwards use the same probe minibatch and exact checkpoint RNG.
    progress['verification_forward_attempts'] += 1
    with torch.no_grad():
        before_loss, before = native_forward(pipe, common, loss_fn, capture=True)
        progress['verification_forwards'] += 1
        for p in trainable_adapters(pipe.dit).values():
            p.add_(0.125)
    with torch.no_grad():
        for item in optimizer.state.values():
            item['exp_avg'].add_(0.125)
            item['exp_avg_sq'].add_(0.125)
    random.seed(SEED + 1)
    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    perturbed = checkpoint_state(pipe.dit, optimizer, steps, cuda=True)
    for key in ('adapters', 'optimizer', 'rng'):
        m1.require(state_hash(perturbed[key]) != state_hash(saved[key]),
                   'Deliberate perturbation ineffective: ' + key)
    del perturbed
    load_checkpoint(path, record, pipe.dit, optimizer, steps, cuda=True)
    m1.check_deadline(deadline)
    progress['verification_forward_attempts'] += 1
    with torch.no_grad():
        after_loss, after = native_forward(pipe, common, loss_fn, capture=True)
        progress['verification_forwards'] += 1
    m1.require(before['input_sha256'] == after['input_sha256'], 'Roundtrip fixed probe RNG/inputs differ')
    torch.testing.assert_close(before['prediction'], after['prediction'], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(before_loss.float(), after_loss.float(), rtol=1e-5, atol=1e-6)
    restore_rng(saved['rng'], cuda=True)
    m1.require(state_hash(checkpoint_state(pipe.dit, optimizer, steps, cuda=True)) == record['state_sha256'], 'Post-probe exact state restore failed')
    return dict(record, status='passed', deliberate_adapter_perturbation=True,
                deliberate_optimizer_moment_perturbation=True, deliberate_independent_rng_reseed=True,
                exact_state_restored=True,
                optimizer_updates_during_verification=0, extra_native_loss_forwards=2,
                probe_loss=float(before_loss), prediction_max_abs_diff=float((before['prediction'] - after['prediction']).abs().max()),
                loss_abs_diff=abs(float(before_loss) - float(after_loss)), rtol=1e-5, atol=1e-6)


def build_pipe(run, manifest, deadline):
    import torch
    Pipeline, Config = m1.frozen_framework(run)
    for rec in manifest['m1_manifest']['model_cache'].values():
        m1.validate_record(rec, deadline, hash_content=False)
    shards = sorted(r['path'] for rel, r in manifest['m1_manifest']['model_cache'].items()
                    if '/diffusion_pytorch_model-' in rel)
    m1.require(len(shards) == 3, 'Exactly three local base shards required')
    pipe = Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda',
        model_configs=[Config(path=shards, skip_download=True)], tokenizer_config=None, redirect_common_files=False)
    m1.require(pipe.dit is not None and pipe.dit2 is None and pipe.dit.fuse_vae_embedding_in_latents
               and not pipe.dit.require_clip_embedding and not pipe.dit.require_vae_embedding, 'Wrong native TI2V DiT')
    m1.require(pipe.vae is None and pipe.text_encoder is None and pipe.tokenizer is None,
               'Cached smoke must not load VAE/T5/tokenizer')
    m1.require(all(getattr(pipe, n) is None for n in pipe.in_iteration_models if n != 'dit'), 'Unexpected iteration auxiliary model')
    from diffsynth.diffusion.training_module import DiffusionTrainingModule
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    training = DiffusionTrainingModule()
    training.switch_pipe_to_training_mode(pipe, trainable_models=None, lora_base_model='dit',
        lora_target_modules=TARGETS, lora_rank=32, task='sft')
    # Native injector initially uses pipe.torch_dtype; only adapters are upcast.
    for p in pipe.dit.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    pipe.dit.train()
    trainable_adapters(pipe.dit)
    m1.require(len(pipe.scheduler.timesteps) == 1000 and pipe.scheduler.training, 'Native training scheduler required')
    return pipe, FlowMatchSFTLoss


def worker(run, manifest_sha, deadline):
    started = time.perf_counter()
    report = {'status': 'failed', 'pid': os.getpid(), 'optimizer_steps': 0, 'steps': [],
              'label_scope': LABEL_SCOPE, 'seed': SEED, 'learning_rate': 1e-5,
              'rank': 32, 'targets': TARGETS, 'training_forwards': 0, 'verification_forwards': 0,
              'training_forward_attempts': 0, 'verification_forward_attempts': 0}
    pipe, torch = None, None
    try:
        m1.require(os.environ.get('TRAIN_SMOKE_PARENT_PID') == str(os.getppid()), 'Worker owner PID binding differs')
        m1.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Worker GPU binding differs')
        m1.require(m1.sha256(run / 'training_manifest.json', deadline) == manifest_sha, 'Worker manifest hash differs')
        manifest = m1.read_json(run / 'training_manifest.json')
        m1.require(manifest['owner_pid'] == os.getppid(), 'Manifest owner differs')
        m1.verify_snapshot(run, manifest, deadline)
        import torch as torch_module
        torch = torch_module
        m1.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible CUDA GPU required')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        report['torch_version'] = torch.__version__
        pipe, loss_fn = build_pipe(run, manifest, deadline)
        adapters = trainable_adapters(pipe.dit)
        report['trainable_parameter_count'] = sum(p.numel() for p in adapters.values())
        report['trainable_tensor_count'] = len(adapters)
        report['adapter_dtype'] = 'torch.float32'
        report['base_dtype'] = 'torch.bfloat16'
        report['autocast_dtype'] = 'torch.bfloat16'
        frozen_before = frozen_state_hash(pipe.dit, deadline)
        optimizer = torch.optim.AdamW(list(adapters.values()), lr=1e-5)
        commons = manifest['prepared']['commons']
        for index in range(manifest['allocation']['max_steps']):
            m1.check_deadline(deadline)
            t0 = time.perf_counter()
            entry = commons[index]
            common = checked_tensors(entry['cache'], Path(manifest['m1_run']), deadline)
            validate_common(common)
            optimizer.zero_grad(set_to_none=True)
            report['training_forward_attempts'] += 1
            loss, _ = native_forward(pipe, common, loss_fn)
            report['training_forwards'] += 1
            stats = optimizer_step(pipe.dit, optimizer, loss)
            report['optimizer_steps'] += 1
            torch.cuda.synchronize()
            stats.update(step=report['optimizer_steps'], case_id=entry['case_id'], seconds=time.perf_counter() - t0,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            report['steps'].append(stats)
            m1.atomic_json(run / 'private' / ('step_' + str(index + 1) + '.json'), stats)
            del loss, common
        m1.require(report['optimizer_steps'] == manifest['allocation']['max_steps'], 'Optimizer step count differs')
        optimizer.zero_grad(set_to_none=True)
        probe = checked_tensors(commons[0]['cache'], Path(manifest['m1_run']), deadline)
        report['checkpoint'] = verify_roundtrip(pipe, optimizer, probe, loss_fn,
            run / 'private/adapters_optimizer_rng.pt', report['optimizer_steps'], deadline, report)
        frozen_after = frozen_state_hash(pipe.dit, deadline)
        m1.require(frozen_before == frozen_after, 'Frozen base updated')
        report['frozen_base_sha256_before'] = frozen_before
        report['frozen_base_sha256_after'] = frozen_after
        report['vae_t5_loaded'] = False
        m1.check_deadline(deadline)
        report['status'] = 'completed'
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], file=sys.stderr, flush=True)
    finally:
        if pipe is not None:
            del pipe
        if torch is not None:
            try:
                report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                torch.cuda.empty_cache()
            except Exception as error:
                report['status'] = 'failed'
                report['cleanup_error'] = str(error)
        report['worker_seconds'] = time.perf_counter() - started
        m1.atomic_json(run / 'worker_training.json', report)
    return int(report['status'] != 'completed')


def assert_idle(deadline):
    m1.check_deadline(deadline)
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=min(15, deadline - time.perf_counter()))
    values = dict((a.strip(), int(b.strip())) for a, b in (s.split(',') for s in result.stdout.splitlines()))
    m1.require(values.get('0', 100) < 100, 'GPU0 missing/busy >=100MiB; do not stop other jobs')
    m1.check_deadline(deadline)


def spawn_worker(run, manifest_sha, deadline, workers):
    m1.check_deadline(deadline)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TRAIN_SMOKE_PARENT_PID=str(os.getpid()),
        DIFFSYNTH_SKIP_DOWNLOAD='true', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8',
        PYTHONPATH=str(run / 'framework'))
    log = (run / 'logs/training.log').open('x')
    os.chmod(run / 'logs/training.log', 0o600)
    try:
        proc = subprocess.Popen([sys.executable, str(run / 'code/run_teacher_training_smoke.py'),
            '--out-dir', str(run), '--worker', '--manifest-sha', manifest_sha, '--work-deadline', str(deadline)],
            cwd=run / 'framework', env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    except BaseException:
        log.close()
        raise
    workers.append({'role': 'training', 'process': proc, 'log': log})


def verify_sources(run, manifest, deadline):
    m1.verify_snapshot(run, manifest, deadline)
    m1.verify_final_sources(Path(manifest['m1_run']), manifest['m1_manifest'], deadline)
    for rec in list(manifest['source'].values()) + list(manifest['m1_files'].values()):
        m1.validate_record(rec, deadline)
    for entry in manifest['prepared']['commons'] + manifest['prepared']['states']:
        m1.require(m1.sha256(entry['cache']['path'], deadline) == entry['cache']['sha256'], 'M1 final cache hash differs')
    return {'status': 'passed', 'all_ten_model_cache_hashes_reverified': True,
            'commons_reverified': 6, 'states_reverified': 18, 'source_hashes_reverified': True}


def execute(args, allocation):
    started = time.perf_counter()
    hard = started + allocation['timeout_seconds']
    work = hard - allocation['cleanup_grace_seconds']
    run = Path(args.out_dir).absolute()
    workers, cleanup, created, failure, detail, report, final = [], [], False, None, None, None, None
    handlers = {}
    def interrupted(signum, frame):
        raise InterruptedError('Coordinator signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, work - time.perf_counter()))
    try:
        manifest = preflight(args.m1_run, run, args.models_dir, work)
        assert_idle(work)
        m1.enable_subreaper()
        run.mkdir(mode=0o700)
        created = True
        manifest = freeze_run(run, manifest, allocation, work)
        # Recheck immediately before dispatch, after source freeze.
        assert_idle(work)
        spawn_worker(run, m1.sha256(run / 'training_manifest.json', work), work, workers)
        m1.wait_workers(workers, work)
        report = m1.read_json(run / 'worker_training.json')
        m1.require(report['status'] == 'completed' and report['optimizer_steps'] == allocation['max_steps']
                   and report['checkpoint']['status'] == 'passed', 'Worker incomplete')
        final = verify_sources(run, manifest, work)
    except BaseException as error:
        failure = type(error).__name__ + ': ' + str(error)
        detail = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        margin = min(2.0, allocation['cleanup_grace_seconds'] / 4)
        cleanup = m1.stop_workers(workers, hard - margin, allocation['cleanup_grace_seconds'] - margin)
        for sig, old in handlers.items():
            signal.signal(sig, old)
    if created and report is None and (run / 'worker_training.json').exists():
        report = m1.read_json(run / 'worker_training.json')
    partial_steps = []
    if created:
        partial_steps = [m1.read_json(p) for p in sorted((run / 'private').glob('step_*.json'))]
    measured = time.perf_counter() - started
    charged = measured + min(1.0, allocation['cleanup_grace_seconds'] / 8)
    gone = all(c['owned_group_gone'] for c in cleanup)
    if charged > allocation['timeout_seconds'] or not gone:
        failure = failure or 'Hard allocation/cleanup exceeded'
    result = {'protocol': PROTOCOL, 'status': 'failed' if failure else 'completed', 'error': failure,
        'traceback': detail, 'label_scope': LABEL_SCOPE, 'worker_report': report, 'partial_step_reports': partial_steps,
        'final_source_verification': final, 'cleanup': cleanup,
        'ledger': {'allocated_cards': 1, 'gpu': '0', 'wall_seconds': charged,
            'measured_wall_seconds_before_publication': measured,
            'publication_exit_allowance_seconds': charged - measured,
            'allocated_gpu_hours': charged / 3600, 'reserved_allocated_gpu_hours': allocation['reserved_allocated_gpu_hours'],
            'hard_wall_seconds': allocation['timeout_seconds'], 'within_reservation': charged <= allocation['timeout_seconds'],
            'owned_cleanup_complete': gone,
            'accounting': 'Whole one-card coordinator wall: preflight/all hashes, setup, train, checkpoint/probe, CPU and owned cleanup plus conservative publication/exit margin'},
        'scope': 'technical_only_no_fullFT3_student_distillation_AR_quality_or_generalization'}
    if created:
        m1.atomic_json(run / 'training_summary.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m1-run', type=Path, default=M1_DEFAULT)
    p.add_argument('--models-dir', type=Path, default=m1.MODELS_DEFAULT)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--max-steps', type=int, default=4)
    p.add_argument('--timeout-seconds', type=float, default=7200)
    p.add_argument('--max-gpu-hours', type=float, default=2)
    p.add_argument('--cleanup-grace-seconds', type=float, default=30)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--manifest-sha', help=argparse.SUPPRESS)
    p.add_argument('--work-deadline', type=float, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.worker:
            m1.require(args.manifest_sha and args.work_deadline and math.isfinite(args.work_deadline), 'Internal worker binding required')
            return worker(args.out_dir.absolute(), args.manifest_sha, args.work_deadline)
        allocation = budget(args.gpu, args.max_steps, args.timeout_seconds, args.max_gpu_hours, args.cleanup_grace_seconds)
        if args.preflight_only:
            manifest = preflight(args.m1_run, args.out_dir, args.models_dir)
            print(json.dumps({'status': 'preflight_passed_no_gpu', 'allocation': allocation,
                'label_scope': LABEL_SCOPE, 'case_order': manifest['case_order'],
                'step_case_order': manifest['step_case_order'][:allocation['max_steps']],
                'commons_verified': 6, 'states_verified': 18, 'model_cache_verified': 10,
                'cuda_queried': False, 'out_dir_created': False}, sort_keys=True))
            return 0
        result = execute(args, allocation)
        print(json.dumps({'status': result['status'], 'error': result['error'], 'ledger': result['ledger'],
                          'optimizer_steps': result['worker_report'].get('optimizer_steps') if result['worker_report'] else len(result['partial_step_reports']),
                          'out_dir': str(args.out_dir)}, sort_keys=True))
        return int(result['status'] != 'completed')
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
