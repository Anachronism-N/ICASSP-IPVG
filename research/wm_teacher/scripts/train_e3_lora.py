#!/usr/bin/env python3
"""E3 cached-input LoRA trainer for the native Wan2.2-TI2V-5B base DiT.

Single-node torchrun DDP (up to 8 ranks, bs1/accum1 per rank) or a plain single process.
Inputs are caches written by scripts/cache_e3_inputs.py (VAE x0, first-frame latent,
T5 context per caption version); VAE/T5/tokenizer are never loaded here.

Identical across arms by construction: sample order (global --seed only), every
(t, noise) pair (derived from seed, step, sample_index), LoRA init, LR schedule.
Arms differ only in --caption-version and --anchor-lambda.

Loss: L = L_FM + lambda * w(t) * MSE(v_theta, stopgrad(v_base)); v_base is the same
DiT with all LoRA layers disabled (no grad), first latent frame removed in both terms.
L_FM is the native FlowMatchSFTLoss math with explicit deterministic t/noise.

Only rank0 writes. One final save (adapters + AdamW + RNG) plus a gen_jobs.py-loadable
LoRA safetensors; no intermediate checkpoints. Hard GPUh budget (world x wall).
Top-level imports are stdlib only; --help never imports torch.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_teacher_mechanism_probe as m1  # noqa: E402
import run_teacher_training_smoke as ts  # noqa: E402

PROTOCOL = 'e3-cached-lora-v1'
CACHE_PROTOCOL = 'e3-input-cache-v1'
TARGETS = ts.TARGETS
LORA_RANK = 32
LORA_ALPHA = 32
DEFAULT_SEED = 20261021
X0_SHAPE = (1, 48, 31, 44, 80)
FIRST_SHAPE = (1, 48, 1, 44, 80)
CTX_SHAPE = (1, 512, 4096)
ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_DEFAULT = ROOT / 'DiffSynth-Studio'
FRAMEWORK_FILES = ('diffsynth/diffusion/loss.py', 'diffsynth/diffusion/training_module.py',
                   'diffsynth/diffusion/flow_match.py', 'diffsynth/pipelines/wan_video.py',
                   'diffsynth/models/wan_video_dit.py', 'diffsynth/core/gradient/gradient_checkpoint.py',
                   'diffsynth/utils/lora/general.py')
SHARD_RELS = tuple(sorted(k for k in m1.CACHE_PINS if '/diffusion_pytorch_model-' in k))
MAX_GPU_HOURS_CAP = 256.0
EXPORT_NAME = 'e3_lora.safetensors'
CKPT_NAME = 'adapters_optimizer_rng.pt'
require = m1.require


# ----------------------------------------------------------------------------- determinism

def derive_seed(*parts):
    digest = hashlib.sha256(('e3|' + '|'.join(str(p) for p in parts)).encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


class DataOrder:
    """Global position g = step * world + rank -> dataset index; epochs reshuffle by seed only."""

    def __init__(self, seed, n):
        require(type(n) is int and n > 0, 'Empty dataset')
        self.seed, self.n, self._perms = seed, n, {}

    def permutation(self, epoch):
        if epoch not in self._perms:
            import torch
            g = torch.Generator().manual_seed(derive_seed('order', self.seed, epoch))
            self._perms[epoch] = torch.randperm(self.n, generator=g).tolist()
        return self._perms[epoch]

    def index(self, step, rank, world):
        epoch, pos = divmod(step * world + rank, self.n)
        return self.permutation(epoch)[pos]

    def plan(self, steps, world):
        return [[self.index(s, r, world) for r in range(world)] for s in range(steps)]


def plan_sha256(plan):
    return hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()


def draw_t_noise(seed, step, sample_index, shape, num_timesteps=1000):
    """(timestep id, FP32 CPU noise) fully determined by (seed, step, sample_index)."""
    import torch
    g = torch.Generator().manual_seed(derive_seed('t_noise', seed, step, sample_index))
    tid = int(torch.randint(0, num_timesteps, (1,), generator=g))
    noise = torch.randn(tuple(shape), generator=g, dtype=torch.float32)
    return tid, noise


def lr_factor(update, warmup, horizon):
    """update = 1-based optimizer update index; linear warmup then cosine to exactly 0 at horizon."""
    require(update >= 1 and warmup >= 0, 'Bad LR schedule index')
    if warmup and update <= warmup:
        return update / warmup
    require(horizon > warmup, 'Cosine horizon must exceed warmup')
    progress = min(1.0, (update - warmup) / (horizon - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------------- budget

def budget(max_gpu_hours, world, grace, save_reserve, first_step_estimate):
    vals = (max_gpu_hours, grace, save_reserve, first_step_estimate)
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in vals), 'Finite positive budget values required')
    require(type(world) is int and 1 <= world <= 8, 'Single node: world size 1..8')
    require(max_gpu_hours <= MAX_GPU_HOURS_CAP, 'max-gpu-hours exceeds hard cap')
    wall = max_gpu_hours * 3600.0 / world
    require(grace < wall / 4 and save_reserve < wall / 2, 'Grace/save reserve too large for the wall budget')
    return {'max_gpu_hours': float(max_gpu_hours), 'world_size': world, 'wall_seconds': wall,
            'cleanup_grace_seconds': float(grace), 'save_reserve_seconds': float(save_reserve),
            'first_step_estimate_seconds': float(first_step_estimate)}


def should_stop(elapsed, step_estimate, allocation):
    """True when one more step plus the final save cannot fit inside the wall budget."""
    return (elapsed + step_estimate + allocation['save_reserve_seconds'] + allocation['cleanup_grace_seconds']
            > allocation['wall_seconds'])


def recommend_steps(t_step, overhead_seconds, target_gpu_hours, target_world, safety=0.95):
    require(t_step > 0 and target_world >= 1, 'Bad recommendation inputs')
    wall = target_gpu_hours * 3600.0 / target_world
    return max(0, int(math.floor((wall * safety - overhead_seconds) / t_step)))


# ----------------------------------------------------------------------------- cache index

def load_cache_index(path, version, verify_files=False):
    path = Path(path).resolve()
    index = m1.read_json(path)
    require(index.get('protocol') == CACHE_PROTOCOL and index.get('complete') is True, 'Not a complete E3 cache index')
    require(version in index['versions'], 'Unknown caption version: ' + str(version))
    root = path.parent
    samples = index['samples']
    require(samples and len({s['sample_id'] for s in samples}) == len(samples), 'Empty/duplicate cache samples')
    out = []
    for s in samples:
        require(set(s['contexts']) == set(index['versions']), 'Sample lacks a caption version: ' + s['sample_id'])
        entry = {'sample_id': s['sample_id'],
                 'x0': dict(s['x0'], path=str(m1.private_path(root, s['x0']['path']))),
                 'context': dict(s['contexts'][version], path=str(m1.private_path(root, s['contexts'][version]['path'])))}
        for key in ('x0', 'context'):
            p = Path(entry[key]['path'])
            require(p.is_file() and p.stat().st_size == entry[key]['bytes'], 'Cache file missing/size differs: ' + str(p))
            if verify_files:
                require(m1.sha256(p) == entry[key]['sha256'], 'Cache file hash differs: ' + str(p))
        out.append(entry)
    identity = hashlib.sha256(json.dumps([[s['sample_id'], s['x0']['sha256']] for s in samples]).encode()).hexdigest()
    return {'index_sha256': m1.sha256(path), 'index_path': str(path), 'version': version,
            'samples': out, 'sample_identity_sha256': identity,
            'context_set_sha256': hashlib.sha256(json.dumps([e['context']['sha256'] for e in out]).encode()).hexdigest()}


def load_sample(entry, verify=True):
    import torch
    sample = {}
    for key in ('x0', 'context'):
        rec = entry[key]
        if verify:
            require(m1.sha256(rec['path']) == rec['sha256'], 'Cache file hash differs at load: ' + rec['path'])
        sample.update(torch.load(rec['path'], map_location='cpu', weights_only=True))
    validate_sample(sample)
    return sample


def validate_sample(sample):
    import torch
    expected = {'x0': X0_SHAPE, 'first_frame_latents': FIRST_SHAPE, 'context': CTX_SHAPE}
    require(set(sample) == set(expected), 'Cached sample key coverage differs')
    for key, shape in expected.items():
        t = sample[key]
        require(isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16 and tuple(t.shape) == shape
                and t.device.type == 'cpu', 'Cached tensor shape/dtype differs: ' + key)


def synthetic_sample(index):
    import torch
    g = torch.Generator().manual_seed(derive_seed('synthetic', index))
    x0 = torch.randn(X0_SHAPE, generator=g).to(torch.bfloat16)
    ctx = torch.randn(CTX_SHAPE, generator=g).to(torch.bfloat16)
    ctx[:, 96:] = 0  # padded T5 tokens are zeroed natively
    return {'x0': x0, 'first_frame_latents': x0[:, :, 0:1].clone(), 'context': ctx}


# ----------------------------------------------------------------------------- loss

@contextlib.contextmanager
def adapters_disabled(model):
    """peft disable_adapter equivalent for inject_adapter_in_model (no PeftModel wrapper).

    Only flips BaseTunerLayer._disable_adapters; never touches requires_grad, so the
    optimizer/all-reduce parameter set is unchanged."""
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    require(layers, 'No LoRA layers to disable')
    old = [m._disable_adapters for m in layers]
    require(not any(old), 'Adapters already disabled')
    try:
        for m in layers:
            m._disable_adapters = True
        yield len(layers)
    finally:
        for m, v in zip(layers, old):
            m._disable_adapters = v


def cached_inputs(sample, device, offload, checkpointing=True):
    return {'input_latents': sample['x0'].to(device).clone(),
            'context': sample['context'].to(device).clone(),
            'first_frame_latents': sample['first_frame_latents'].to(device).clone(),
            'fuse_vae_embedding_in_latents': True, 'cfg_scale': 1, 'cfg_merge': False,
            'tea_cache': None, 'use_gradient_checkpointing': checkpointing,
            'use_gradient_checkpointing_offload': bool(offload and checkpointing)}


def native_state(pipe, inputs, tid, noise_fp32):
    """Native FlowMatchSFTLoss state construction with explicit timestep id and noise."""
    import torch
    timestep = pipe.scheduler.timesteps[torch.tensor([tid])].to(dtype=pipe.torch_dtype, device=pipe.device)
    x0 = inputs['input_latents']
    noise = noise_fp32.to(device=x0.device, dtype=x0.dtype) * inputs.get('noise_scale', 1.0)
    latents = pipe.scheduler.add_noise(x0, noise, timestep)
    target = pipe.scheduler.training_target(x0, noise, timestep)
    latents[:, :, 0:1] = inputs['first_frame_latents']
    return timestep, latents, target


def predict(pipe, inputs, latents, timestep):
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    return pipe.model_fn(**models, **dict(inputs, latents=latents), timestep=timestep)


def e3_losses(pipe, inputs, tid, noise_fp32, anchor_lambda, anchor_weighting='training_weight', autocast=None):
    import torch
    autocast = autocast or contextlib.nullcontext
    timestep, latents, target = native_state(pipe, inputs, tid, noise_fp32)
    weight = pipe.scheduler.training_weight(timestep)
    v_base = None
    if anchor_lambda > 0:
        base_inputs = dict(inputs, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False)
        with torch.no_grad(), adapters_disabled(pipe.dit), autocast():
            v_base = predict(pipe, base_inputs, latents.clone(), timestep)[:, :, 1:].detach().float()
    with autocast():
        pred = predict(pipe, inputs, latents, timestep)[:, :, 1:]
        fm = torch.nn.functional.mse_loss(pred.float(), target[:, :, 1:].float()) * weight
        anchor = None
        if v_base is not None:
            aw = weight if anchor_weighting == 'training_weight' else 1.0
            anchor = torch.nn.functional.mse_loss(pred.float(), v_base) * aw
    total = fm if anchor is None else fm + anchor_lambda * anchor
    return {'total': total, 'fm': fm, 'anchor': anchor, 'timestep': float(timestep.float().cpu()[0]),
            'weight': float(weight)}


# ----------------------------------------------------------------------------- distributed helpers

class Comm:
    """Thin collective wrapper; world==1 is the DDP-free single-process path."""

    def __init__(self, dist=None, rank=0, world=1, device='cpu'):
        self.dist, self.rank, self.world, self.device = dist, rank, world, device

    def all_reduce_mean_(self, tensors):
        if self.world == 1 or not tensors:
            return
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        flat = _flatten_dense_tensors(tensors)
        self.dist.all_reduce(flat)
        flat.div_(self.world)
        for t, s in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
            t.copy_(s)

    def any_flag(self, flag):
        if self.world == 1:
            return bool(flag)
        import torch
        t = torch.tensor([1.0 if flag else 0.0], device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX)
        return bool(t.item() > 0)

    def gather(self, values):
        if self.world == 1:
            return [list(values)]
        import torch
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        out = [torch.zeros_like(t) for _ in range(self.world)]
        self.dist.all_gather(out, t)
        return [o.cpu().tolist() for o in out]

    def broadcast_(self, tensors, src=0):
        if self.world > 1:
            for t in tensors:
                self.dist.broadcast(t.data, src)

    def broadcast_float(self, value, src=0):
        if self.world == 1:
            return float(value)
        import torch
        t = torch.tensor([float(value)], dtype=torch.float64, device=self.device)
        self.dist.broadcast(t, src)
        return float(t.item())

    def barrier(self):
        if self.world > 1:
            self.dist.barrier()


def adapter_list(model):
    return list(ts.trainable_adapters(model).values())


def check_grads(model):
    import torch
    ok = True
    for _, p in model.named_parameters():
        if p.requires_grad:
            ok = ok and p.grad is not None and bool(torch.isfinite(p.grad).all())
        else:
            require(p.grad is None, 'Frozen base has gradient')
    return ok


def grad_norm(tensors):
    import torch
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t.float()) for t in tensors])))


def is_oom(error):
    import torch
    return isinstance(error, getattr(torch, 'OutOfMemoryError', ())) or 'out of memory' in str(error).lower()


def on_cuda(pipe):
    return str(pipe.device).startswith('cuda')


def train_step(pipe, params, optimizer, sample, step, sample_index, cfg, comm, autocast=None, tolerate_oom=False):
    """One synchronous optimizer update; gradients averaged over ranks (DDP semantics)."""
    import torch
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    tid, noise = draw_t_noise(cfg['seed'], step, sample_index, sample['x0'].shape)
    split = cfg['anchor_lambda'] > 0 and (step == 0 or (step + 1) % cfg['component_grad_every'] == 0)
    local_ok, oom, fm_grads, losses = True, False, None, None
    try:
        inputs = cached_inputs(sample, pipe.device, cfg['offload'])
        losses = e3_losses(pipe, inputs, tid, noise, cfg['anchor_lambda'], cfg['anchor_weighting'], autocast)
        local_ok = losses['total'].numel() == 1 and bool(torch.isfinite(losses['total'].detach()).all())
        if local_ok:
            if split:
                losses['fm'].backward(retain_graph=True)
                fm_grads = [p.grad.detach().clone() for p in params]
                (cfg['anchor_lambda'] * losses['anchor']).backward()
            else:
                losses['total'].backward()
            local_ok = check_grads(pipe.dit)
    except Exception as error:  # OOM must be agreed collectively before anyone leaves the step
        if not (tolerate_oom and is_oom(error)):
            raise
        local_ok, oom, fm_grads, losses = False, True, None, None
        optimizer.zero_grad(set_to_none=True)
        if on_cuda(pipe):
            torch.cuda.empty_cache()
    any_oom = comm.any_flag(oom)
    if comm.any_flag(not local_ok):
        optimizer.zero_grad(set_to_none=True)
        if tolerate_oom and any_oom:
            return {'status': 'oom', 'step': step + 1}
        raise FloatingPointError('Nonfinite loss/gradient on some rank at step ' + str(step + 1))
    grads = [p.grad for p in params]
    comm.all_reduce_mean_(grads)
    norm_fm = norm_anchor = None
    if fm_grads is not None:
        comm.all_reduce_mean_(fm_grads)
        norm_fm = grad_norm(fm_grads)
        norm_anchor = grad_norm([g - f for g, f in zip(grads, fm_grads)])
        del fm_grads
    lr = cfg['lr'] * lr_factor(step + 1, cfg['warmup'], cfg['horizon'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    total_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm=cfg['clip'], error_if_nonfinite=True))
    if cfg['anchor_lambda'] == 0:
        norm_fm, norm_anchor = total_norm, 0.0
    optimizer.step()
    if on_cuda(pipe):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() if on_cuda(pipe) else 0
    anchor = float(losses['anchor'].detach()) if losses['anchor'] is not None else 0.0
    rows = comm.gather([float(losses['fm'].detach()), anchor, float(losses['total'].detach()), tid, sample_index, seconds, peak])
    return {'status': 'ok', 'step': step + 1, 'lr': lr, 'grad_norm_total_before_clip': total_norm,
            'grad_norm_fm': norm_fm, 'grad_norm_anchor_weighted': norm_anchor,
            'loss_fm_mean': sum(r[0] for r in rows) / len(rows), 'loss_anchor_mean': sum(r[1] for r in rows) / len(rows),
            'loss_total_mean': sum(r[2] for r in rows) / len(rows),
            'per_rank': [{'loss_fm': r[0], 'loss_anchor': r[1], 'timestep_id': int(r[3]), 'sample_index': int(r[4])} for r in rows],
            'step_seconds_max': max(r[5] for r in rows), 'peak_allocated_bytes_max': int(max(r[6] for r in rows))}


# ----------------------------------------------------------------------------- checkpoint/export

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    require(value is None or type(value) in (str, int, float, bool), 'Non JSON-safe value')
    return value


def checkpoint_state(model, optimizer, steps, config, cuda=False):
    adapters = ts.trainable_adapters(model)
    return {'protocol': PROTOCOL, 'steps': steps, 'targets': TARGETS, 'rank': LORA_RANK, 'alpha': LORA_ALPHA,
            'adapters': {k: ts.tree_cpu(p) for k, p in adapters.items()}, 'adapter_names': list(adapters),
            'optimizer': ts.tree_cpu(optimizer.state_dict()), 'rng': ts.rng_state(cuda), 'config': json_safe(config)}


def validate_checkpoint(value, model, optimizer, steps):
    import torch
    keys = {'protocol', 'steps', 'targets', 'rank', 'alpha', 'adapters', 'adapter_names', 'optimizer', 'rng', 'config'}
    require(isinstance(value, dict) and set(value) == keys, 'Checkpoint key schema differs')
    require(value['protocol'] == PROTOCOL and value['steps'] == steps and value['targets'] == TARGETS
            and value['rank'] == LORA_RANK and value['alpha'] == LORA_ALPHA, 'Checkpoint training schema differs')
    params = ts.trainable_adapters(model)
    require(value['adapter_names'] == list(params) and set(value['adapters']) == set(params), 'Adapter names/order differ')
    for name, live in params.items():
        saved = value['adapters'][name]
        require(isinstance(saved, torch.Tensor) and saved.dtype == live.dtype and saved.shape == live.shape
                and bool(torch.isfinite(saved).all()), 'Bad adapter tensor: ' + name)
    live = optimizer.state_dict()
    opt = value['optimizer']
    require(set(opt) == {'state', 'param_groups'} and len(opt['param_groups']) == len(live['param_groups']), 'Optimizer schema differs')
    for a, b in zip(opt['param_groups'], live['param_groups']):
        require({k: v for k, v in a.items() if k != 'lr'} == {k: v for k, v in b.items() if k != 'lr'}, 'Optimizer hyperparameters differ')
    ids = [i for g in live['param_groups'] for i in g['params']]
    require(set(opt['state']) == (set(ids) if steps else set()), 'Optimizer state coverage differs')
    for pid, p in zip(ids, params.values()):
        item = opt['state'].get(pid)
        if item is None:
            continue
        require(set(item) == {'step', 'exp_avg', 'exp_avg_sq'} and float(item['step']) == steps, 'AdamW state differs')
        for k in ('exp_avg', 'exp_avg_sq'):
            require(item[k].shape == p.shape and bool(torch.isfinite(item[k]).all()), 'Bad AdamW moment')
    ts.state_hash(value)


def save_checkpoint(path, value):
    import torch
    buf = io.BytesIO()
    torch.save(ts.tree_cpu(value), buf)
    m1.atomic_bytes(path, buf.getvalue())
    return {'path': str(path), 'file_sha256': m1.sha256(path), 'state_sha256': ts.state_hash(value),
            'bytes': Path(path).stat().st_size}


def load_checkpoint(path, record, model, optimizer, steps, cuda=False):
    import torch
    require(m1.sha256(path) == record['file_sha256'], 'Checkpoint file hash differs')
    value = torch.load(path, map_location='cpu', weights_only=True)
    require(ts.state_hash(value) == record['state_sha256'], 'Checkpoint state hash differs')
    validate_checkpoint(value, model, optimizer, steps)
    with torch.no_grad():
        for key, p in ts.trainable_adapters(model).items():
            p.copy_(value['adapters'][key].to(p.device))
    optimizer.load_state_dict(value['optimizer'])
    ts.restore_rng(value['rng'], cuda)
    require(ts.state_hash(checkpoint_state(model, optimizer, steps, value['config'], cuda)) == record['state_sha256'],
            'Exact adapter/optimizer/RNG restoration failed')
    return value


def export_lora(path, adapters):
    """DiffSynth/gen_jobs.py LoRA format: <module>.lora_A.default.weight / .lora_B.default.weight; alpha==rank -> scale 1."""
    import torch
    from safetensors.torch import save
    require(adapters and all(k.endswith(('.lora_A.default.weight', '.lora_B.default.weight')) for k in adapters),
            'Not native LoRA adapter names')
    tensors = {k: v.detach().to('cpu', torch.float32).contiguous().clone() for k, v in adapters.items()}
    meta = {'protocol': PROTOCOL, 'rank': str(LORA_RANK), 'alpha': str(LORA_ALPHA), 'targets': TARGETS,
            'format': 'diffsynth_peft_default_lora_scale_1'}
    m1.atomic_bytes(path, save(tensors, metadata=meta))
    return {'path': str(path), 'file_sha256': m1.sha256(path), 'bytes': Path(path).stat().st_size,
            'tensor_count': len(tensors), 'parameter_count': sum(t.numel() for t in tensors.values())}


# ----------------------------------------------------------------------------- GPU pipeline

def framework_records(framework):
    return {rel: m1.file_record(Path(framework) / rel) for rel in FRAMEWORK_FILES}


def verify_base_shards(models_dir, hash_content=True):
    shards = []
    for rel in SHARD_RELS:
        p = Path(models_dir) / rel
        require(p.is_file(), 'Missing base shard: ' + str(p))
        if hash_content:
            m1.file_record(p, m1.CACHE_PINS[rel])
        shards.append(str(p))
    return shards


def build_pipe(framework, shards, seed, device=None):
    import torch
    sys.path.insert(0, str(Path(framework).resolve()))
    import diffsynth
    require(Path(diffsynth.__file__).resolve().is_relative_to(Path(framework).resolve()), 'Imported wrong diffsynth')
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.diffusion.training_module import DiffusionTrainingModule
    if device is None:
        device = 'cuda' if not torch.cuda.is_initialized() else f'cuda:{torch.cuda.current_device()}'
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=shards, skip_download=True)], tokenizer_config=None, redirect_common_files=False)
    require(pipe.dit is not None and pipe.dit2 is None and pipe.dit.fuse_vae_embedding_in_latents
            and not pipe.dit.require_clip_embedding and not pipe.dit.require_vae_embedding, 'Wrong native TI2V DiT')
    require(pipe.vae is None and pipe.text_encoder is None and pipe.tokenizer is None, 'Cached training must not load VAE/T5')
    torch.manual_seed(derive_seed('lora_init', seed))  # identical LoRA init on every rank and arm
    DiffusionTrainingModule().switch_pipe_to_training_mode(pipe, trainable_models=None, lora_base_model='dit',
        lora_target_modules=TARGETS, lora_rank=LORA_RANK, task='sft')
    for p in pipe.dit.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    pipe.dit.train()
    adapters = ts.trainable_adapters(pipe.dit)
    require(len(adapters) == 600 and sum(p.numel() for p in adapters.values()) == 80609280, 'Unexpected adapter set')
    require(len(pipe.scheduler.timesteps) == 1000 and pipe.scheduler.training, 'Native training scheduler required')
    return pipe


# ----------------------------------------------------------------------------- run loop

def progress_writer(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
    f = os.fdopen(fd, 'a', encoding='utf-8')
    def write(record):
        f.write(json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    return write, f


def run_loop(pipe, optimizer, params, samples_fn, order, cfg, comm, allocation, clock, log, autocast=None,
             steps=None, tolerate_oom=False, step_offset=0):
    """Runs up to steps updates (default cfg max_steps). Returns (completed, records, stop_reason)."""
    steps = cfg['max_steps'] if steps is None else steps
    records, estimate, stop, window = [], allocation['first_step_estimate_seconds'], None, []
    for k in range(steps):
        step = step_offset + k
        decision = comm.rank == 0 and should_stop(clock(), estimate, allocation)
        if comm.any_flag(decision):
            stop = 'budget_exhausted_before_max_steps'
            break
        t_iter = time.perf_counter()
        index = order.index(step, comm.rank, comm.world)
        stats = train_step(pipe, params, optimizer, samples_fn(index), step, index, cfg, comm, autocast, tolerate_oom)
        stats['elapsed_seconds'] = clock()
        if stats['status'] == 'oom':
            stop = 'oom'
            records.append(stats)
            break
        stats['iteration_seconds_rank0'] = time.perf_counter() - t_iter
        observed = max(stats['iteration_seconds_rank0'], stats['step_seconds_max'])
        estimate = observed if k == 0 else max(estimate, observed)
        window.append(stats['loss_total_mean'])
        records.append(stats)
        if comm.rank == 0 and log and (k == 0 or (step + 1) % cfg['log_every'] == 0 or k == steps - 1):
            log(dict(stats, window_mean_loss_total=sum(window) / len(window), window_steps=len(window)))
            window = []
    return len([r for r in records if r['status'] == 'ok']), records, stop


def setup_distributed():
    import torch
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    require(world == int(os.environ.get('LOCAL_WORLD_SIZE', str(world))), 'Single node only')
    require(torch.cuda.is_available() and torch.cuda.device_count() >= world, 'Need one visible GPU per rank')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world == 1:
        return Comm(None, 0, 1, device)
    import datetime
    import torch.distributed as dist
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=device)
    return Comm(dist, rank, world, device)


def validate_args(args):
    require(math.isfinite(args.anchor_lambda) and args.anchor_lambda >= 0, 'anchor-lambda must be finite >=0')
    require(args.lr > 0 and args.weight_decay >= 0 and args.clip > 0 and args.warmup >= 0, 'Bad optimizer settings')
    require(args.log_every >= 1 and args.component_grad_every >= 1 and args.load_concurrency >= 1, 'Bad cadence settings')
    if args.smoke:
        require(1 <= args.smoke_steps <= 20, 'smoke-steps 1..20')
        require(args.smoke_offload and set(args.smoke_offload) <= {'on', 'off'}, 'smoke-offload must be subset of on,off')
        require(args.target_gpu_hours > 0 and 1 <= args.target_world <= 8, 'Bad smoke recommendation target')
    else:
        require(args.max_steps is not None and args.max_steps > args.warmup, '--max-steps (> warmup) required')
        require(not args.synthetic_inputs, 'Synthetic inputs are smoke-only')
    require(args.synthetic_inputs or args.cache_index, '--cache-index required (or --smoke --synthetic-inputs)')
    require(args.synthetic_inputs or args.caption_version, '--caption-version required with a cache')


def make_cfg(args, horizon):
    return {'seed': args.seed, 'lr': args.lr, 'weight_decay': args.weight_decay, 'warmup': args.warmup,
            'horizon': horizon, 'clip': args.clip, 'anchor_lambda': float(args.anchor_lambda),
            'anchor_weighting': args.anchor_weighting, 'component_grad_every': args.component_grad_every,
            'log_every': args.log_every, 'offload': args.offload == 'on', 'max_steps': args.max_steps,
            'caption_version': args.caption_version, 'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
            'targets': TARGETS, 'per_rank_batch': 1, 'grad_accum': 1}


def device_type(comm):
    import torch
    return torch.device(comm.device).type


def reset_memory(comm):
    import torch
    if device_type(comm) == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def smoke_modes(args, pipe, optimizer, params, samples_fn, order, cfg, comm, allocation, clock, log, autocast):
    import torch
    modes, offset = {}, 0
    for mode in args.smoke_offload:
        reset_memory(comm)
        done, records, stop = run_loop(pipe, optimizer, params, samples_fn, order, dict(cfg, offload=(mode == 'on')),
                                       comm, allocation, clock, log, autocast, steps=args.smoke_steps,
                                       tolerate_oom=True, step_offset=offset)
        offset += done
        ok = [r for r in records if r['status'] == 'ok']
        secs = [max(r['iteration_seconds_rank0'], r['step_seconds_max']) for r in ok]
        steady = sorted(secs[1:] if len(secs) > 1 else secs)
        modes[mode] = {'completed_steps': done, 'stop': stop, 'iteration_seconds': secs,
                       't_step_median_excluding_first': steady[len(steady) // 2] if steady else None,
                       'peak_allocated_bytes_max': max([r['peak_allocated_bytes_max'] for r in ok] or [0]),
                       'peak_reserved_bytes_rank0': torch.cuda.max_memory_reserved() if device_type(comm) == 'cuda' else 0,
                       'losses_fm': [r['loss_fm_mean'] for r in ok]}
        if stop == 'budget_exhausted_before_max_steps':
            break
    return offset, modes


def execute(args):
    import torch
    started = time.time()
    t_start = time.monotonic()
    comm = setup_distributed()
    allocation = budget(args.max_gpu_hours, comm.world, args.cleanup_grace_seconds, args.save_reserve_seconds,
                        args.first_step_estimate_seconds)
    t0 = comm.broadcast_float(t_start)  # CLOCK_MONOTONIC is host-wide; all ranks charge from rank0 start
    clock = lambda: time.monotonic() - t0  # noqa: E731
    out = Path(args.out_dir).absolute()
    summary = {'protocol': PROTOCOL, 'status': 'failed', 'mode': 'smoke' if args.smoke else 'train',
               'allocation': allocation, 'started_unix': started}
    log_file, pipe, created = None, None, False

    def on_signal(signum, frame):
        raise TimeoutError('E3 hard wall budget reached or terminated (signal %d)' % signum)
    old = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
    for s in old:
        signal.signal(s, on_signal)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, allocation['wall_seconds'] - allocation['cleanup_grace_seconds'] - clock()))
    try:
        if comm.rank == 0:
            require(not out.exists() and not out.is_symlink() and out.parent.is_dir(), 'New --out-dir required (parent must exist)')
            out.mkdir(mode=0o700)
            created = True
            (out / 'private').mkdir(mode=0o700)
            (out / 'export').mkdir(mode=0o700)
        comm.barrier()
        if args.synthetic_inputs:
            data = {'samples': list(range(64)), 'sample_identity_sha256': 'synthetic', 'context_set_sha256': 'synthetic',
                    'index_sha256': None, 'version': 'synthetic'}
            samples_fn = synthetic_sample
        else:
            data = load_cache_index(args.cache_index, args.caption_version)
            samples_fn = lambda i: load_sample(data['samples'][i])  # noqa: E731
        order = DataOrder(args.seed, len(data['samples']))
        total_steps = args.smoke_steps * len(args.smoke_offload) if args.smoke else args.max_steps
        cfg = make_cfg(args, args.max_steps or 1000)
        plan = order.plan(total_steps, comm.world)
        shards = verify_base_shards(args.models_dir, hash_content=(comm.rank == 0 and not args.skip_model_hash))
        comm.barrier()
        if comm.rank == 0:
            manifest = {'protocol': PROTOCOL, 'config': cfg, 'allocation': allocation, 'world_size': comm.world,
                        'data': {k: v for k, v in data.items() if k != 'samples'}, 'num_samples': len(data['samples']),
                        'order_plan_sha256': plan_sha256(plan), 'order_plan_steps': total_steps,
                        't_noise_rule': 'Generator(derive_seed(t_noise,seed,step,sample_index)): randint(0,1000) then randn fp32 -> x0 dtype',
                        'source': {'trainer': m1.file_record(__file__), 'm1_helper': m1.file_record(m1.__file__),
                                   'smoke_helper': m1.file_record(ts.__file__)},
                        'framework': framework_records(args.framework),
                        'base_shards': {r: m1.CACHE_PINS[r] for r in SHARD_RELS},
                        'base_shards_hashed': not args.skip_model_hash, 'torch_version': str(torch.__version__),
                        'argv': sys.argv[1:]}
            m1.atomic_json(out / 'run_manifest.json', json_safe(manifest))
            log, log_file = progress_writer(out / 'progress.jsonl')
        else:
            log = None
        # Staggered load limits concurrent host-RAM peaks of FP32 shards.
        for wave in range(0, comm.world, args.load_concurrency):
            if wave <= comm.rank < wave + args.load_concurrency:
                pipe = build_pipe(args.framework, shards, args.seed, str(comm.device))
            comm.barrier()
        params = adapter_list(pipe.dit)
        comm.broadcast_(params)  # init is already seed-identical; broadcast makes it exact by construction
        frozen_before = ts.frozen_state_hash(pipe.dit) if comm.rank == 0 and not args.skip_frozen_hash else None
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
        cuda = device_type(comm) == 'cuda'
        autocast = lambda: torch.autocast(device_type=device_type(comm), dtype=torch.bfloat16)  # noqa: E731
        reset_memory(comm)
        summary['setup_seconds'] = setup_seconds = clock()
        if args.smoke:
            completed_steps, summary['smoke'] = smoke_modes(args, pipe, optimizer, params, samples_fn, order, cfg, comm,
                                                            allocation, clock, log, autocast)
        else:
            completed_steps, records, stop = run_loop(pipe, optimizer, params, samples_fn, order, cfg, comm, allocation,
                                                      clock, log, autocast)
            summary['stop_reason'] = stop
            summary['last_step'] = records[-1] if records else None
            summary['peak_allocated_bytes_max'] = max([r.get('peak_allocated_bytes_max', 0) for r in records] or [0])
            require(stop is None and completed_steps == args.max_steps, 'Stopped before --max-steps: ' + str(stop))
        summary['completed_steps'] = completed_steps
        comm.barrier()
        if comm.rank == 0:
            t_save = time.perf_counter()
            state = checkpoint_state(pipe.dit, optimizer, completed_steps, dict(cfg, completed_steps=completed_steps), cuda=cuda)
            validate_checkpoint(state, pipe.dit, optimizer, completed_steps)
            record = save_checkpoint(out / 'private' / CKPT_NAME, state)
            reread = torch.load(out / 'private' / CKPT_NAME, map_location='cpu', weights_only=True)
            require(ts.state_hash(reread) == record['state_sha256'], 'Checkpoint reread hash differs')
            del reread
            summary['checkpoint'] = record
            summary['export'] = export_lora(out / 'export' / EXPORT_NAME, state['adapters'])
            if frozen_before is not None:
                frozen_after = ts.frozen_state_hash(pipe.dit)
                require(frozen_after == frozen_before, 'Frozen base changed')
                summary['frozen_base_sha256'] = frozen_after
            summary['save_seconds'] = time.perf_counter() - t_save
            if args.smoke:
                overhead = setup_seconds + summary['save_seconds'] + allocation['cleanup_grace_seconds']
                summary['recommendation'] = {
                    mode: {'recommended_max_steps': recommend_steps(v['t_step_median_excluding_first'], overhead,
                                                                    args.target_gpu_hours, args.target_world),
                           't_step': v['t_step_median_excluding_first'], 'overhead_seconds': overhead,
                           'target_gpu_hours': args.target_gpu_hours, 'target_world': args.target_world,
                           'formula': 'floor((target_gpu_hours*3600/target_world*0.95 - overhead)/t_step)'}
                    for mode, v in summary['smoke'].items() if v['t_step_median_excluding_first']}
        comm.barrier()
        summary['status'] = 'completed'
    except BaseException as error:
        summary['error'] = type(error).__name__ + ': ' + str(error)
        summary['traceback'] = traceback.format_exc()
        print(summary['traceback'], file=sys.stderr, flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s, h in old.items():
            signal.signal(s, h)
        if log_file is not None:
            log_file.close()
        wall = clock() + 1.0  # publication allowance
        summary['ledger'] = {'cards': comm.world, 'wall_seconds': wall, 'gpu_hours': wall * comm.world / 3600.0,
                             'budget_gpu_hours': allocation['max_gpu_hours'], 'within_budget': wall <= allocation['wall_seconds'],
                             'accounting': 'world x wall from rank0 trainer start (hash/load/train/save/verify) + 1s; torchrun spawn excluded'}
        if not summary['ledger']['within_budget']:
            summary['status'] = 'failed'
        if comm.rank == 0 and created:
            m1.atomic_json(out / 'summary.json', json_safe(summary))
            print(json.dumps(json_safe({k: summary.get(k) for k in ('status', 'error', 'ledger', 'completed_steps', 'recommendation')})))
        if comm.dist is not None:
            with contextlib.suppress(Exception):
                comm.dist.destroy_process_group()
    return int(summary['status'] != 'completed')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-index', type=Path)
    p.add_argument('--caption-version')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--models-dir', type=Path, default=m1.MODELS_DEFAULT)
    p.add_argument('--framework', type=Path, default=FRAMEWORK_DEFAULT)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--anchor-lambda', type=float, default=0.0)
    p.add_argument('--anchor-weighting', choices=('training_weight', 'uniform'), default='training_weight')
    p.add_argument('--offload', choices=('on', 'off'), default='on')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--component-grad-every', type=int, default=10,
                   help='lambda>0: every K steps split backward (retain_graph) to log exact per-term grad norms')
    p.add_argument('--max-gpu-hours', type=float, required=True, help='hard cap, charged as world x wall')
    p.add_argument('--cleanup-grace-seconds', type=float, default=60)
    p.add_argument('--save-reserve-seconds', type=float, default=240)
    p.add_argument('--first-step-estimate-seconds', type=float, default=120)
    p.add_argument('--load-concurrency', type=int, default=4)
    p.add_argument('--skip-model-hash', action='store_true')
    p.add_argument('--skip-frozen-hash', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=4)
    p.add_argument('--smoke-offload', type=lambda s: s.split(','), default=['on', 'off'])
    p.add_argument('--synthetic-inputs', action='store_true', help='smoke only: random shape-exact inputs')
    p.add_argument('--target-gpu-hours', type=float, default=24.0)
    p.add_argument('--target-world', type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_args(args)
        budget(args.max_gpu_hours, int(os.environ.get('WORLD_SIZE', '1')), args.cleanup_grace_seconds,
               args.save_reserve_seconds, args.first_step_estimate_seconds)
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
    return execute(args)


if __name__ == '__main__':
    sys.exit(main())
