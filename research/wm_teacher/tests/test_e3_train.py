"""CPU-only tests for the E3 cached-input LoRA trainer, cacher and exporter. Never initializes CUDA."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
FRAMEWORK = ROOT / 'DiffSynth-Studio'
# Public snapshot guard: the trainer imports private helpers (run_teacher_training_smoke /
# run_teacher_mechanism_probe) that are not part of the public snapshot.
if not (SCRIPTS / 'cache_e3_inputs.py').is_file() or not (SCRIPTS / 'merge_e3_lora.py').is_file():
    raise unittest.SkipTest('e3_train tests require cache/merge helpers absent from the public snapshot')
sys.path.insert(0, str(SCRIPTS))
import train_e3_lora as tr  # noqa: E402
import cache_e3_inputs as ce  # noqa: E402
import merge_e3_lora as mg  # noqa: E402

TINY_X0 = (1, 4, 3, 2, 2)
TINY_FIRST = (1, 4, 1, 2, 2)
TINY_CTX = (1, 5, 6)
TINY_RANK = 2


def source_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def native_flowmatch_loss(torch_shim):
    """Exact FlowMatchSFTLoss source from the framework, executed with an RNG shim."""
    tree = ast.parse((FRAMEWORK / 'diffsynth/diffusion/loss.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'FlowMatchSFTLoss')
    fn.args.args[0].annotation = None
    ns = {'torch': torch_shim}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'loss.py', 'exec'), ns)
    return ns['FlowMatchSFTLoss']


def build_tiny_pipe(seed=0, dtype=None):
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    nn = torch.nn
    gc = source_module(FRAMEWORK / 'diffsynth/core/gradient/gradient_checkpoint.py', '_e3_gc')
    flow = source_module(FRAMEWORK / 'diffsynth/diffusion/flow_match.py', '_e3_flow')

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    class Block(nn.Module):
        def __init__(self, d, f):
            super().__init__()
            self.self_attn, self.cross_attn = Attn(d), Attn(d)
            self.ffn = nn.Sequential(nn.Linear(d, f), nn.GELU(), nn.Linear(f, d))

    class TinyDiT(nn.Module):
        def __init__(self, c=4, d=8, f=16, cd=6):
            super().__init__()
            self.patch, self.ctxproj, self.head = nn.Linear(c, d), nn.Linear(cd, d), nn.Linear(d, c)
            self.blocks = nn.ModuleList([Block(d, f) for _ in range(2)])

        def forward(self, latents, timestep, context):
            b, c, t, h, w = latents.shape
            x = self.patch(latents.permute(0, 2, 3, 4, 1).reshape(b, -1, c)) + (timestep.float() / 1000).view(-1, 1, 1)
            cx = self.ctxproj(context).mean(1, keepdim=True)
            for blk in self.blocks:
                a = blk.self_attn
                x = x + a.o(a.q(x) * torch.sigmoid(a.k(x)) + a.v(x))
                c2 = blk.cross_attn
                x = x + c2.o(c2.q(x) + c2.v(cx) * torch.sigmoid(c2.k(cx)))
                x = x + blk.ffn(x)
            return self.head(x).reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)

    torch.manual_seed(seed)
    base = TinyDiT()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    for p in base.parameters():
        p.requires_grad_(False)
    dtype = dtype or torch.bfloat16
    base = base.to(dtype)
    torch.manual_seed(seed + 1)
    model = inject_adapter_in_model(LoraConfig(r=TINY_RANK, lora_alpha=TINY_RANK, target_modules=list(tr.TARGETS.split(','))), base)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    def model_fn(dit, latents=None, timestep=None, context=None, use_gradient_checkpointing=False,
                 use_gradient_checkpointing_offload=False, **kwargs):
        return gc.gradient_checkpoint_forward(dit, use_gradient_checkpointing, use_gradient_checkpointing_offload,
                                              latents, timestep, context)

    scheduler = flow.FlowMatchScheduler('Wan')
    scheduler.set_timesteps(1000, training=True)
    pipe = types.SimpleNamespace(dit=model, scheduler=scheduler, torch_dtype=dtype, device='cpu',
                                 in_iteration_models=('dit',), model_fn=model_fn)
    pipe.base_state_fp32 = base_state
    return pipe


def tiny_sample(index, dtype=None):
    import torch
    dtype = dtype or torch.bfloat16
    g = torch.Generator().manual_seed(1000 + index)
    x0 = torch.randn(TINY_X0, generator=g).to(dtype)
    return {'x0': x0, 'first_frame_latents': x0[:, :, 0:1].clone(),
            'context': torch.randn(TINY_CTX, generator=g).to(dtype)}


def tiny_sample_fp32(index):
    import torch
    return tiny_sample(index, torch.float32)


def perturb_lora_b(model, scale=0.05, seed=7):
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k, p in model.named_parameters():
            if 'lora_B' in k:
                p.add_(torch.randn(p.shape, generator=g) * scale)


def cfg(**kw):
    base = {'seed': 11, 'lr': 5e-3, 'weight_decay': 0.01, 'warmup': 2, 'horizon': 6, 'clip': 1.0,
            'anchor_lambda': 0.0, 'anchor_weighting': 'training_weight', 'component_grad_every': 2,
            'log_every': 1, 'offload': True, 'max_steps': 6}
    base.update(kw)
    return base


def autocast():
    import torch
    return torch.autocast(device_type='cpu', dtype=torch.bfloat16)


def grads(model):
    return {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.requires_grad}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import peft  # noqa: F401  (warm import before any wall-clock test)
        torch.set_num_threads(2)
        cls.torch = torch


class TopologyTests(unittest.TestCase):
    def test_legacy_and_explicit_four_node_topology(self):
        for world in range(1, 9):
            for rank in range(world):
                env = {'WORLD_SIZE': str(world), 'LOCAL_WORLD_SIZE': str(world),
                       'RANK': str(rank), 'LOCAL_RANK': str(rank)}
                self.assertEqual(tr.validate_topology(env, visible_devices=world)['nnodes'], 1)
        for rank in range(32):
            env = {'WORLD_SIZE': '32', 'LOCAL_WORLD_SIZE': '8', 'RANK': str(rank),
                   'LOCAL_RANK': str(rank % 8), 'GROUP_RANK': str(rank // 8), 'NNODES': '4'}
            with self.assertRaises(ValueError):
                tr.validate_topology(env)
            topo = tr.validate_topology(env, teacher_four_node_admission=True, visible_devices=8)
            self.assertEqual((topo['node_rank'], topo['world_size']), (rank // 8, 32))
        self.assertEqual(tr.budget(6, 32, 60, 240, 120, teacher_four_node_admission=True)['wall_seconds'], 675)
        with self.assertRaises(ValueError):
            tr.budget(6, 32, 60, 240, 120)

    def test_rank_node_mapping_world_and_local_visibility_are_strict(self):
        good = {'WORLD_SIZE': '32', 'LOCAL_WORLD_SIZE': '8', 'RANK': '17',
                'LOCAL_RANK': '1', 'GROUP_RANK': '2', 'NNODES': '4'}
        for key, value in [('RANK', '-1'), ('RANK', '32'), ('RANK', '9'), ('LOCAL_RANK', '8'),
                           ('GROUP_RANK', '1'), ('NODE_RANK', '1'), ('NNODES', '2'),
                           ('LOCAL_WORLD_SIZE', '4'), ('WORLD_SIZE', '16')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                tr.validate_topology(dict(good, **{key: value}), teacher_four_node_admission=True, visible_devices=8)
        with self.assertRaises(ValueError):
            tr.validate_topology(good, teacher_four_node_admission=True, visible_devices=7)
        del good['GROUP_RANK']
        with self.assertRaises(ValueError):
            tr.validate_topology(good, teacher_four_node_admission=True)

    def test_clock_offsets_cancel_only_with_own_host_start(self):
        self.assertEqual(tr.local_duration(100, 115), tr.local_duration(900000100, 900000115))
        self.assertEqual(tr.Comm().max_duration(15), 15)
        for started, now in [(5, 4), (float('nan'), 5)]:
            with self.assertRaises(ValueError):
                tr.local_duration(started, now)


class DeterminismTests(Base):
    def test_t_noise_deterministic_and_independent_of_global_rng(self):
        torch = self.torch
        torch.manual_seed(1)
        a = tr.draw_t_noise(5, 3, 17, TINY_X0)
        torch.manual_seed(999)
        torch.rand(100)
        b = tr.draw_t_noise(5, 3, 17, TINY_X0)
        self.assertEqual(a[0], b[0])
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(0 <= a[0] < 1000 and a[1].dtype == torch.float32)
        for other in [(5, 4, 17), (5, 3, 18), (6, 3, 17)]:
            c = tr.draw_t_noise(*other, TINY_X0)
            self.assertFalse(torch.equal(a[1], c[1]))

    def test_data_order_epoch_permutations_and_world_partition(self):
        n, world, steps = 10, 3, 9
        plan = tr.DataOrder(4, n).plan(steps, world)
        self.assertEqual(plan, tr.DataOrder(4, n).plan(steps, world))
        self.assertNotEqual(plan, tr.DataOrder(5, n).plan(steps, world))
        flat = [i for row in plan for i in row]
        for e in range(len(flat) // n):
            self.assertEqual(sorted(flat[e * n:(e + 1) * n]), list(range(n)))
        order = tr.DataOrder(4, n)
        self.assertEqual([[order.index(s, r, world) for r in range(world)] for s in range(steps)], plan)

    def test_lr_schedule(self):
        self.assertEqual([tr.lr_factor(u, 2, 6) for u in (1, 2)], [0.5, 1.0])
        self.assertAlmostEqual(tr.lr_factor(4, 2, 6), 0.5)
        self.assertEqual(tr.lr_factor(6, 2, 6), 0.0)
        with self.assertRaises(ValueError):
            tr.lr_factor(6, 5, 5)


class LossTests(Base):
    def test_lambda0_equals_native_flowmatch_loss_and_grads(self):
        torch = self.torch
        pipe = build_tiny_pipe()
        perturb_lora_b(pipe.dit)
        sample = tiny_sample(0)
        tid, noise = tr.draw_t_noise(3, 0, 0, TINY_X0)
        inputs = tr.cached_inputs(sample, 'cpu', offload=False, checkpointing=False)
        with autocast():
            ours = tr.e3_losses(pipe, inputs, tid, noise, 0.0)
        self.assertIsNone(ours['anchor'])
        ours['total'].backward()
        g_ours = grads(pipe.dit)
        pipe.dit.zero_grad(set_to_none=True)

        class Shim:
            def __getattr__(self, name):
                return getattr(torch, name)
            def randint(self, low, high, size):
                self.bounds = (low, high)
                return torch.tensor([tid])
            def randn_like(self, x):
                return noise.to(dtype=x.dtype, device=x.device)
        shim = Shim()
        with autocast():
            native = native_flowmatch_loss(shim)(pipe, **tr.cached_inputs(sample, 'cpu', offload=False, checkpointing=False))
        self.assertEqual(shim.bounds, (0, 1000))
        native.backward()
        self.assertTrue(torch.equal(native.detach(), ours['total'].detach()))
        for k, g in grads(pipe.dit).items():
            self.assertTrue(torch.equal(g, g_ours[k]), k)

    def test_anchor_zero_at_init_and_gradients_only_to_adapters(self):
        torch = self.torch
        pipe = build_tiny_pipe()
        sample = tiny_sample(1)
        tid, noise = tr.draw_t_noise(3, 0, 1, TINY_X0)
        def run(lam):
            pipe.dit.zero_grad(set_to_none=True)
            out = tr.e3_losses(pipe, tr.cached_inputs(sample, 'cpu', False, False), tid, noise, lam, autocast=autocast)
            out['total'].backward()
            return out, grads(pipe.dit)
        out0, g0 = run(0.0)
        out1, g1 = run(0.5)
        self.assertEqual(float(out1['anchor']), 0.0)  # B==0 -> v_theta == v_base exactly
        self.assertTrue(all(torch.equal(g0[k], g1[k]) for k in g0))
        perturb_lora_b(pipe.dit)
        flags = {k: p.requires_grad for k, p in pipe.dit.named_parameters()}
        _, g_fm = run(0.0)
        out2, g2 = run(0.5)
        self.assertGreater(float(out2['anchor']), 0)
        self.assertEqual(flags, {k: p.requires_grad for k, p in pipe.dit.named_parameters()})
        for k, p in pipe.dit.named_parameters():
            if not p.requires_grad:
                self.assertIsNone(p.grad, k)
        self.assertTrue(any(not torch.equal(g_fm[k], g2[k]) for k in g2))
        # v_base really is the adapter-free base: compare with a disabled forward vs. B zeroed copy
        x = torch.randn(TINY_X0).to(torch.bfloat16)
        t = torch.tensor([500.0], dtype=torch.bfloat16)
        c = sample['context']
        # (separate autocast regions: autocast caches casted weights within one region)
        with torch.no_grad():
            with autocast(), tr.adapters_disabled(pipe.dit):
                vb = pipe.dit(x, t, c)
            with autocast():
                on = pipe.dit(x, t, c)
            saved = {k: p.clone() for k, p in pipe.dit.named_parameters() if 'lora_B' in k}
            for k, p in pipe.dit.named_parameters():
                if 'lora_B' in k:
                    p.zero_()
            with autocast():
                zero = pipe.dit(x, t, c)
            for k, p in pipe.dit.named_parameters():
                if k in saved:
                    p.copy_(saved[k])
            plain = strip_lora(build_tiny_pipe().dit)
            with autocast():
                ref = plain(x, t, c)
        self.assertTrue(torch.equal(vb, zero))
        self.assertTrue(torch.equal(vb, ref))
        self.assertFalse(torch.equal(vb, on))

    def test_split_backward_with_native_checkpoint_offload_matches_joint(self):
        # FP32 tiny model: split and joint backward are equal up to float summation order.
        torch = self.torch
        pipe = build_tiny_pipe(dtype=torch.float32)
        perturb_lora_b(pipe.dit)
        sample = tiny_sample(2, torch.float32)
        tid, noise = tr.draw_t_noise(3, 1, 2, TINY_X0)
        inputs = lambda: tr.cached_inputs(sample, 'cpu', offload=True, checkpointing=True)  # noqa: E731
        out = tr.e3_losses(pipe, inputs(), tid, noise, 0.7)
        self.assertGreater(float(out['anchor']), 0)
        out['total'].backward()
        joint = grads(pipe.dit)
        pipe.dit.zero_grad(set_to_none=True)
        out = tr.e3_losses(pipe, inputs(), tid, noise, 0.7)
        out['fm'].backward(retain_graph=True)
        fm_part = grads(pipe.dit)
        (0.7 * out['anchor']).backward()
        split = grads(pipe.dit)
        for k in joint:
            torch.testing.assert_close(split[k], joint[k], rtol=1e-5, atol=1e-8)
        pipe.dit.zero_grad(set_to_none=True)
        tr.e3_losses(pipe, inputs(), tid, noise, 0.0)['total'].backward()
        for k, g in grads(pipe.dit).items():
            torch.testing.assert_close(fm_part[k], g, rtol=1e-5, atol=1e-8)


class LoopTests(Base):
    def make(self, **kw):
        torch = self.torch
        pipe = build_tiny_pipe()
        params = tr.adapter_list(pipe.dit)
        opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.01)
        return pipe, params, opt, cfg(**kw)

    def allocation(self, hours=1.0, grace=1.0, save=1.0, first=0.01):
        return tr.budget(hours, 1, grace, save, first)

    def test_single_process_loop_schedule_and_frozen_base(self):
        torch = self.torch
        pipe, params, opt, c = self.make(anchor_lambda=0.3)
        frozen = tr.ts.frozen_state_hash(pipe.dit)
        before = tr.ts.state_hash({k: p for k, p in tr.ts.trainable_adapters(pipe.dit).items()})
        logged = []
        order = tr.DataOrder(c['seed'], 5)
        start = time.monotonic()
        done, recs, stop = tr.run_loop(pipe, opt, params, tiny_sample, order, c, tr.Comm(), self.allocation(),
                                       lambda: time.monotonic() - start, logged.append, autocast)
        self.assertEqual((done, stop), (6, None))
        self.assertEqual([r['lr'] for r in recs], [5e-3 * tr.lr_factor(u, 2, 6) for u in range(1, 7)])
        self.assertEqual([r['per_rank'][0]['sample_index'] for r in recs], [order.index(s, 0, 1) for s in range(6)])
        self.assertEqual([r['per_rank'][0]['timestep_id'] for r in recs],
                         [tr.draw_t_noise(c['seed'], s, order.index(s, 0, 1), TINY_X0)[0] for s in range(6)])
        exact = [r for r in recs if r['step'] in (1, 2, 4, 6)]
        self.assertTrue(all(r['grad_norm_fm'] is not None and r['grad_norm_anchor_weighted'] is not None for r in exact))
        self.assertTrue(all(r['grad_norm_fm'] is None for r in recs if r['step'] in (3, 5)))
        self.assertEqual(len(logged), 6)
        self.assertEqual(frozen, tr.ts.frozen_state_hash(pipe.dit))
        self.assertNotEqual(before, tr.ts.state_hash({k: p for k, p in tr.ts.trainable_adapters(pipe.dit).items()}))
        self.assertTrue(all(torch.isfinite(torch.tensor(r['loss_total_mean'])) for r in recs))

    def test_arms_share_plan_t_and_noise(self):
        # Arms differ only by context; order and (t, noise) never read the context.
        a = tr.DataOrder(11, 7).plan(5, 8)
        b = tr.DataOrder(11, 7).plan(5, 8)
        self.assertEqual(tr.plan_sha256(a), tr.plan_sha256(b))
        for s, row in enumerate(a):
            for idx in row:
                ta, na = tr.draw_t_noise(11, s, idx, TINY_X0)
                tb, nb = tr.draw_t_noise(11, s, idx, TINY_X0)
                self.assertTrue(ta == tb and self.torch.equal(na, nb))

    def test_budget_validation_stop_rule_and_recommendation(self):
        for bad in [(float('nan'), 1, 1, 1, 1), (300, 1, 1, 1, 1), (1, 9, 1, 1, 1), (1, 1, 0, 1, 1), (0.001, 1, 1, 1, 1)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tr.budget(*bad)
        a = tr.budget(8, 8, 60, 240, 120)
        self.assertEqual(a['wall_seconds'], 3600)
        self.assertFalse(tr.should_stop(3000, 299, a))
        self.assertTrue(tr.should_stop(3000, 301, a))
        self.assertEqual(tr.recommend_steps(35.0, 600, 24, 8), int((24 * 3600 / 8 * 0.95 - 600) // 35))

    def test_budget_exhaustion_stops_before_step_without_update(self):
        pipe, params, opt, c = self.make()
        alloc = tr.budget(0.01, 1, 1, 5, 100)  # wall 36s, first step estimated 100s
        before = tr.ts.state_hash({k: p for k, p in tr.ts.trainable_adapters(pipe.dit).items()})
        done, recs, stop = tr.run_loop(pipe, opt, params, tiny_sample, tr.DataOrder(1, 3), c, tr.Comm(), alloc,
                                       lambda: 0.0, None, autocast)
        self.assertEqual((done, recs, stop), (0, [], 'budget_exhausted_before_max_steps'))
        self.assertEqual(before, tr.ts.state_hash({k: p for k, p in tr.ts.trainable_adapters(pipe.dit).items()}))
        # Stop after observed steps once the measured step time no longer fits.
        t = [0.0]
        def clock():
            t[0] += 10.0
            return t[0]
        done, recs, stop = tr.run_loop(pipe, opt, params, tiny_sample, tr.DataOrder(1, 3), c, tr.Comm(),
                                       tr.budget(0.01, 1, 1, 5, 0.01), clock, None, autocast)
        self.assertEqual(stop, 'budget_exhausted_before_max_steps')
        self.assertTrue(0 < done < 6)

    def test_nonfinite_loss_raises(self):
        pipe, params, opt, c = self.make()
        bad = tiny_sample(0)
        bad['x0'] = bad['x0'].clone()
        bad['x0'][0, 0, 1, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            tr.train_step(pipe, params, opt, bad, 0, 0, c, tr.Comm(), autocast)

    def test_checkpoint_save_restore_exact_and_tamper_rejected(self):
        torch = self.torch
        pipe, params, opt, c = self.make()
        start = time.monotonic()
        tr.run_loop(pipe, opt, params, tiny_sample, tr.DataOrder(1, 4), dict(c, max_steps=3), tr.Comm(),
                    self.allocation(), lambda: time.monotonic() - start, None, autocast)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ck.pt'
            state = tr.checkpoint_state(pipe.dit, opt, 3, c)
            tr.validate_checkpoint(state, pipe.dit, opt, 3)
            rec = tr.save_checkpoint(path, state)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with torch.no_grad():
                for p in params:
                    p.add_(0.25)
                for item in opt.state.values():
                    item['exp_avg'].add_(1.0)
            torch.manual_seed(12345)
            self.assertNotEqual(tr.ts.state_hash(tr.checkpoint_state(pipe.dit, opt, 3, c)), rec['state_sha256'])
            tr.load_checkpoint(path, rec, pipe.dit, opt, 3)
            self.assertEqual(tr.ts.state_hash(tr.checkpoint_state(pipe.dit, opt, 3, c)), rec['state_sha256'])
            with self.assertRaises(ValueError):
                tr.load_checkpoint(path, rec, pipe.dit, opt, 4)
            bad = Path(tmp) / 'bad.pt'
            bad.write_bytes(path.read_bytes()[:-1] + b'x')
            with self.assertRaises(ValueError):
                tr.load_checkpoint(bad, rec, pipe.dit, opt, 3)

    def test_two_rank_gloo_equals_mean_gradient_single_process(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            worker = (
                "import sys,os,time,torch,json\n"
                "sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
                "import torch.distributed as dist\n"
                "import test_e3_train as T, train_e3_lora as tr\n"
                "torch.set_num_threads(1)\n"
                "dist.init_process_group('gloo', init_method='file://' + sys.argv[1], rank=int(sys.argv[2]), world_size=2)\n"
                "pipe = T.build_tiny_pipe(dtype=torch.float32); T.perturb_lora_b(pipe.dit, seed=int(sys.argv[2]) + 50)\n"
                "params = tr.adapter_list(pipe.dit); comm = tr.Comm(dist, int(sys.argv[2]), 2, 'cpu'); comm.broadcast_(params)\n"
                "opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.01)\n"
                "offset = int(sys.argv[2]) * 100000000; s = time.monotonic() + offset\n"
                "assert comm.max_duration(20 + int(sys.argv[2])) == 21\n"
                "done, recs, stop = tr.run_loop(pipe, opt, params, T.tiny_sample_fp32, tr.DataOrder(11, 5), T.cfg(max_steps=3, anchor_lambda=0.2), comm, tr.budget(1, 2, 1, 1, 0.01), lambda: tr.local_duration(s, time.monotonic() + offset), None)\n"
                "assert all(0 <= r['elapsed_seconds'] < 300 for r in recs)\n"
                "blocked, empty, reason = tr.run_loop(pipe, opt, params, T.tiny_sample_fp32, tr.DataOrder(11,5), T.cfg(max_steps=1), comm, tr.budget(1,2,1,1,0.01), lambda: 4000 * int(sys.argv[2]), None)\n"
                "assert (blocked, empty, reason) == (0, [], 'budget_exhausted_before_max_steps')\n"
                "torch.save({k: v.detach() for k, v in tr.ts.trainable_adapters(pipe.dit).items()}, sys.argv[3])\n"
                "json.dump([r['per_rank'] for r in recs], open(sys.argv[3] + '.json', 'w'))\n"
                "dist.destroy_process_group()\n") % (str(SCRIPTS), str(Path(__file__).resolve().parent))
            init = Path(tmp) / 'init'
            procs = [subprocess.Popen([sys.executable, '-c', worker, str(init), str(r), str(Path(tmp) / ('r%d.pt' % r))],
                                      env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stderr=subprocess.PIPE, text=True)
                     for r in range(2)]
            for p in procs:
                _, err = p.communicate(timeout=180)
                self.assertEqual(p.returncode, 0, err[-3000:])
            r0 = torch.load(Path(tmp) / 'r0.pt', weights_only=True)
            r1 = torch.load(Path(tmp) / 'r1.pt', weights_only=True)
            self.assertTrue(all(torch.equal(r0[k], r1[k]) for k in r0))
            per_rank = json.loads((Path(tmp) / 'r0.pt.json').read_text())
        # Single-process emulation: per step mean of both ranks' gradients, same optimizer.
        pipe = build_tiny_pipe(dtype=torch.float32)
        perturb_lora_b(pipe.dit, seed=50)  # rank0 init is broadcast
        params = tr.adapter_list(pipe.dit)
        opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.01)
        c = cfg(max_steps=3, anchor_lambda=0.2)
        order = tr.DataOrder(11, 5)
        for step in range(3):
            acc = None
            for rank in range(2):
                idx = order.index(step, rank, 2)
                self.assertEqual(per_rank[step][rank]['sample_index'], idx)
                tid, noise = tr.draw_t_noise(c['seed'], step, idx, TINY_X0)
                pipe.dit.zero_grad(set_to_none=True)
                tr.e3_losses(pipe, tr.cached_inputs(tiny_sample_fp32(idx), 'cpu', True), tid, noise, 0.2)['total'].backward()
                g = [p.grad.clone() for p in params]
                acc = g if acc is None else [a + b for a, b in zip(acc, g)]
            for p, a in zip(params, acc):
                p.grad = a / 2
            for group in opt.param_groups:
                group['lr'] = c['lr'] * tr.lr_factor(step + 1, c['warmup'], c['horizon'])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        for k, v in tr.ts.trainable_adapters(pipe.dit).items():
            torch.testing.assert_close(v.detach(), r0[k], rtol=1e-5, atol=1e-7)


def fake_encoders(calls):
    import torch
    def encode_video(pipe, frames):
        calls['video'] += 1
        g = torch.Generator().manual_seed(int(frames))
        x0 = torch.randn(TINY_X0, generator=g).to(torch.bfloat16)
        return x0, x0[:, :, 0:1].clone()
    def encode_prompt(pipe, prompt):
        calls['prompt'] += 1
        g = torch.Generator().manual_seed(len(prompt) * 7919 + sum(map(ord, prompt)))
        return torch.randn(TINY_CTX, generator=g).to(torch.bfloat16)
    def decode(path, allow_longer):
        return Path(path).stat().st_size  # stand-in "frames" token
    return {'encode_video_fn': encode_video, 'encode_prompt_fn': encode_prompt, 'decode_fn': decode}


def write_manifest(root, n=3, versions=('orig', 'fixed')):
    rows = []
    for i in range(n):
        video = root / ('v%d.mp4' % i)
        video.write_bytes(b'x' * (10 + i))
        rows.append({'sample_id': 's%d' % i, 'video': str(video),
                     'captions': {v: '%s caption %d' % (v, i) for v in versions}, 'motion': i})
    path = root / 'manifest.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


class CacheTests(Base):
    def args(self, root, manifest, shard=0, num=1):
        return types.SimpleNamespace(manifest=manifest, out_dir=root / 'cache', models_dir=root, framework=FRAMEWORK,
                                     shard=shard, num_shards=num, allow_longer=False, skip_model_hash=True)

    def test_cache_shards_resume_finalize_and_trainer_index(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_manifest(root)
            rows, versions = ce.read_manifest(manifest)
            self.assertEqual(versions, ['fixed', 'orig'])
            calls = {'video': 0, 'prompt': 0}
            factory = lambda *a: object()  # noqa: E731
            for shard in range(2):
                st = ce.run_shard(self.args(root, manifest, shard, 2), rows, versions, pipe_factory=factory, **fake_encoders(calls))
                self.assertFalse(st['failed'])
            self.assertEqual(calls, {'video': 3, 'prompt': 6})  # x0 once per video, one context per version
            cache = root / 'cache'
            for p in list(cache.rglob('*.pt')) + list((cache / 'records').glob('*.json')):
                self.assertEqual(p.stat().st_mode & 0o777, 0o600, p)
            loaded = torch.load(cache / 'x0/s0.pt', weights_only=True)
            self.assertEqual(set(loaded), {'x0', 'first_frame_latents'})
            # resume: nothing recomputed, pipeline never built
            never = lambda *a: self.fail('pipeline must not load on full resume')  # noqa: E731
            st = ce.run_shard(self.args(root, manifest, 0, 1), rows, versions, pipe_factory=never, **fake_encoders(calls))
            self.assertEqual((st['reused'], st['computed'], calls['video']), (3, 0, 3))
            # orphan after crash: record missing -> recompute that sample only
            (cache / 'records/s1.json').unlink()
            st = ce.run_shard(self.args(root, manifest, 0, 1), rows, versions, pipe_factory=factory, **fake_encoders(calls))
            self.assertEqual((st['reused'], st['computed'], calls['video']), (2, 1, 4))
            out = ce.finalize(self.args(root, manifest), rows, versions)
            self.assertEqual(out['samples'], 3)
            with patch.multiple(tr, X0_SHAPE=TINY_X0, FIRST_SHAPE=TINY_FIRST, CTX_SHAPE=TINY_CTX):
                a = tr.load_cache_index(cache / 'cache_index.json', 'orig', verify_files=True)
                b = tr.load_cache_index(cache / 'cache_index.json', 'fixed', verify_files=True)
                self.assertEqual(a['sample_identity_sha256'], b['sample_identity_sha256'])
                self.assertNotEqual(a['context_set_sha256'], b['context_set_sha256'])
                sa, sb = tr.load_sample(a['samples'][0]), tr.load_sample(b['samples'][0])
                self.assertTrue(torch.equal(sa['x0'], sb['x0']) and not torch.equal(sa['context'], sb['context']))
                with self.assertRaises(ValueError):
                    tr.load_cache_index(cache / 'cache_index.json', 'nope')
            # tamper -> finalize/load refuse
            ctx = cache / 'ctx/orig/s2.pt'
            os.chmod(ctx, 0o600)
            data = bytearray(ctx.read_bytes())
            data[-5] ^= 1
            ctx.write_bytes(bytes(data))
            with self.assertRaises(ValueError):
                ce.finalize(types.SimpleNamespace(**dict(vars(self.args(root, manifest)), out_dir=root / 'cache')), rows, versions)
            with self.assertRaises(ValueError):
                tr.load_sample(tr.load_cache_index(cache / 'cache_index.json', 'orig')['samples'][2])
            # binding refuses a different manifest in the same cache dir
            write_manifest(root, n=2)
            rows2, v2 = ce.read_manifest(manifest)
            with self.assertRaises(ValueError):
                ce.run_shard(self.args(root, manifest), rows2, v2, pipe_factory=factory, **fake_encoders(calls))

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = {'sample_id': 'a', 'video': '/x.mp4', 'captions': {'orig': 'o', 'fixed': 'f'}}
            for bad in ([good, dict(good)], [good, dict(good, sample_id='b', captions={'orig': 'o'})],
                        [dict(good, video='rel.mp4')], [dict(good, sample_id='a/b')], [dict(good, captions={'Orig': 'x'})]):
                p = root / 'm.jsonl'
                p.write_text(''.join(json.dumps(r) + '\n' for r in bad))
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    ce.read_manifest(p)

    def test_decode_first121_policy(self):
        from PIL import Image
        import numpy as np
        class Reader(list):
            def close(self):
                pass
        frames = [np.full((9, 16, 3), i % 255, dtype=np.uint8) for i in range(125)]
        with patch('imageio.v2.get_reader', return_value=Reader(frames)):
            out = ce.decode_frames('/x.mp4', allow_longer=True)
        self.assertEqual(len(out), 121)
        self.assertEqual(out[0].size, (1280, 704))
        self.assertIsInstance(out[0], Image.Image)
        with patch('imageio.v2.get_reader', return_value=Reader(frames[:120])), self.assertRaises(ValueError):
            ce.decode_frames('/x.mp4', allow_longer=True)


class ExportTests(Base):
    def tiny_base_files(self, pipe, root):
        import torch
        from safetensors.torch import save_file
        state = pipe.base_state_fp32
        keys = sorted(state)
        half = len(keys) // 2
        paths = [root / 'base-1.safetensors', root / 'base-2.safetensors']
        save_file({k: state[k].contiguous() for k in keys[:half]}, str(paths[0]))
        save_file({k: state[k].contiguous() for k in keys[half:]}, str(paths[1]))
        return paths

    def test_export_maps_through_native_loader_and_merge_matches(self):
        torch = self.torch
        pipe = build_tiny_pipe()
        perturb_lora_b(pipe.dit, scale=0.3)
        adapters = tr.ts.trainable_adapters(pipe.dit)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = tr.export_lora(root / 'e3_lora.safetensors', adapters)
            self.assertEqual(rec['tensor_count'], 2 * 2 * 10)
            shards = self.tiny_base_files(pipe, root)
            header = mg.base_header(shards)
            self.assertEqual(len(mg.target_modules(header)), 20)
            lora = mg.read_lora(root / 'e3_lora.safetensors')
            self.assertFalse(any('lora' in k for k in header))
            check = mg.check_lora(lora, header, rank=TINY_RANK)
            self.assertEqual(check['target_layers'], 20)
            # gen_jobs path: fresh base + native GeneralLoRALoader fuse (alpha=1) == trained peft model
            base = build_plain_base({k: v.to(torch.bfloat16).float() for k, v in pipe.base_state_fp32.items()})
            loader = mg.load_general_loader()(device='cpu', torch_dtype=torch.float32)
            loader.fuse_lora_to_base_model(base, lora, alpha=1)
            x = torch.randn(TINY_X0)
            t = torch.tensor([300.0])
            c = torch.randn(TINY_CTX)
            with torch.no_grad():
                ref = pipe.dit.float()(x, t, c)
                got = base(x, t, c)
            torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
            merged = mg.merge_full(lora, shards, root / 'merged.safetensors', scale=1.0)
            self.assertEqual(merged['keys'], len(header))
            v = mg.verify_full(root / 'merged.safetensors', header, lora, shards, scale=1.0)
            self.assertTrue(v['keys_match_base'] and v['shapes_match_base'] and v['probe_exact_bf16_equal'])
            with self.assertRaises(ValueError):
                mg.check_lora({k: t for k, t in list(lora.items())[1:]}, header, rank=TINY_RANK)
            with self.assertRaises(ValueError):
                mg.merge_full(lora, shards, root / 'merged.safetensors')


def strip_lora(model):
    """Replace each peft LoRA wrapper by its base layer (native module tree, no adapters)."""
    from peft.tuners.lora import LoraLayer
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoraLayer):
                setattr(module, child_name, child.base_layer)
    return model


def build_plain_base(state):
    """Un-injected TinyDiT with given FP32 base weights (what gen_jobs holds before load_lora)."""
    model = strip_lora(build_tiny_pipe().dit).float()
    model.load_state_dict(state, strict=True)
    return model


class EndToEndTests(Base):
    def patches(self, root):
        stack = [patch.object(tr, 'setup_distributed', return_value=tr.Comm(None, 0, 1, 'cpu')),
                 patch.object(tr, 'build_pipe', side_effect=lambda *a: build_tiny_pipe()),
                 patch.object(tr, 'verify_base_shards', return_value=[]),
                 patch.object(tr, 'framework_records', return_value={}),
                 patch.multiple(tr, X0_SHAPE=TINY_X0, FIRST_SHAPE=TINY_FIRST, CTX_SHAPE=TINY_CTX)]
        return stack

    def cache(self, root):
        manifest = write_manifest(root, n=4)
        rows, versions = ce.read_manifest(manifest)
        args = types.SimpleNamespace(manifest=manifest, out_dir=root / 'cache', models_dir=root, framework=FRAMEWORK,
                                     shard=0, num_shards=1, allow_longer=False, skip_model_hash=True)
        ce.run_shard(args, rows, versions, pipe_factory=lambda *a: object(), **fake_encoders({'video': 0, 'prompt': 0}))
        ce.finalize(args, rows, versions)
        return root / 'cache/cache_index.json'

    def run_main(self, root, argv):
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in self.patches(root):
                stack.enter_context(p)
            return tr.main(argv)

    def test_execute_train_completed_with_checkpoint_and_export(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.cache(root)
            outs = {}
            for arm, version, lam in (('a', 'orig', '0'), ('c', 'fixed', '0.5')):
                out = root / ('run_' + arm)
                rc = self.run_main(root, ['--cache-index', str(index), '--caption-version', version, '--out-dir', str(out),
                                          '--max-steps', '4', '--warmup', '1', '--anchor-lambda', lam, '--log-every', '2',
                                          '--component-grad-every', '2', '--max-gpu-hours', '1', '--seed', '9'])
                summary = json.loads((out / 'summary.json').read_text())
                self.assertEqual(rc, 0, summary.get('traceback'))
                self.assertEqual((summary['status'], summary['completed_steps']), ('completed', 4))
                self.assertTrue(summary['ledger']['within_budget'] and summary['ledger']['cards'] == 1)
                self.assertEqual((out / 'progress.jsonl').stat().st_mode & 0o777, 0o600)
                self.assertEqual(len((out / 'progress.jsonl').read_text().splitlines()), 3)  # steps 1, 2, 4
                ck = torch.load(out / 'private' / tr.CKPT_NAME, weights_only=True)
                self.assertEqual(tr.ts.state_hash(ck), summary['checkpoint']['state_sha256'])
                self.assertEqual(ck['steps'], 4)
                self.assertEqual(summary['export']['tensor_count'], 40)
                outs[arm] = json.loads((out / 'run_manifest.json').read_text())
            self.assertEqual(outs['a']['order_plan_sha256'], outs['c']['order_plan_sha256'])
            self.assertEqual(outs['a']['data']['sample_identity_sha256'], outs['c']['data']['sample_identity_sha256'])
            self.assertNotEqual(outs['a']['data']['context_set_sha256'], outs['c']['data']['context_set_sha256'])
            # restore final checkpoint into a fresh tiny model + optimizer: exact
            pipe = build_tiny_pipe()
            params = tr.adapter_list(pipe.dit)
            opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)
            for p in params:
                p.grad = torch.zeros_like(p)
            opt.step()  # materialize state slots; load overwrites everything
            for item in opt.state.values():
                item['step'] = torch.tensor(4.0)
            summary = json.loads((root / 'run_a/summary.json').read_text())
            tr.load_checkpoint(root / 'run_a/private' / tr.CKPT_NAME, summary['checkpoint'], pipe.dit, opt, 4)

    def test_execute_budget_exhausted_is_failed_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.cache(root)
            out = root / 'run'
            rc = self.run_main(root, ['--cache-index', str(index), '--caption-version', 'orig', '--out-dir', str(out),
                                      '--max-steps', '4', '--warmup', '1', '--max-gpu-hours', '0.01',
                                      '--cleanup-grace-seconds', '1', '--save-reserve-seconds', '5',
                                      '--first-step-estimate-seconds', '100'])
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual((rc, summary['status'], summary['stop_reason']), (1, 'failed', 'budget_exhausted_before_max_steps'))
            self.assertFalse((out / 'private' / tr.CKPT_NAME).exists())

    def test_execute_hard_wall_alarm_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.cache(root)
            out = root / 'run'
            def slow(*a, **k):
                time.sleep(30)
            with patch.object(tr, 'train_step', side_effect=slow):
                t0 = time.monotonic()
                rc = self.run_main(root, ['--cache-index', str(index), '--caption-version', 'orig', '--out-dir', str(out),
                                          '--max-steps', '4', '--warmup', '1', '--max-gpu-hours', '0.002',
                                          '--cleanup-grace-seconds', '1', '--save-reserve-seconds', '1',
                                          '--first-step-estimate-seconds', '0.1', '--skip-frozen-hash'])
                elapsed = time.monotonic() - t0
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual((rc, summary['status']), (1, 'failed'))
            self.assertIn('TimeoutError', summary['error'])
            self.assertLess(elapsed, 10)
            self.assertFalse((out / 'private' / tr.CKPT_NAME).exists())

    def test_existing_out_dir_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = self.cache(root)
            (root / 'run').mkdir()
            rc = self.run_main(root, ['--cache-index', str(index), '--caption-version', 'orig', '--out-dir', str(root / 'run'),
                                      '--max-steps', '4', '--warmup', '1', '--max-gpu-hours', '1'])
            self.assertEqual(rc, 1)
            self.assertFalse((root / 'run/summary.json').exists())


def resume_argv(out, steps=2):
    return ['--out-dir', str(out), '--max-gpu-hours', '1', '--max-steps', '4', '--warmup', '1', '--seed', '20261102',
            '--anchor-lambda', '0.5', '--log-every', '1', '--component-grad-every', '2',
            '--smoke', '--synthetic-inputs', '--smoke-offload', 'on', '--offload', 'on',
            '--smoke-steps', str(steps)]


def bound_json_record(path):
    raw = Path(path).read_bytes()
    return {'path': str(Path(path).absolute()), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def resume_proof(root, source):
    import prepare_teacher_training_admission as admission
    auth_path = root / 'authorization.json'
    tr.m1.atomic_json(auth_path, admission.AUTH_EXPECTED)
    manifest = tr.m1.read_json(source / 'run_manifest.json')
    summary = tr.m1.read_json(source / 'summary.json')
    return {'schema': tr.SMOKE_RESUME_SCHEMA, 'phase': 'single_gpu_training_resume',
            'authorization': bound_json_record(auth_path),
            'source_run_manifest': bound_json_record(source / 'run_manifest.json'),
            'source_summary': bound_json_record(source / 'summary.json'), 'checkpoint': summary['checkpoint'],
            'source_completed_steps': summary['completed_steps'], 'continuation_updates': 2, 'logical_horizon': 4,
            'config': manifest['config'], 'source': manifest['source'],
            'synthetic_inputs_allowed': True, 'formal_training_allowed': False}


class SmokeResumeTests(Base):
    def test_cli_python_seed_is_synthetic_smoke_only_before_cuda_setup(self):
        args = tr.parser().parse_args(resume_argv(Path('/tmp/unused_seed_test')))
        for smoke, synthetic in [(True, True), (True, False), (False, False)]:
            scoped = types.SimpleNamespace(**dict(vars(args), smoke=smoke, synthetic_inputs=synthetic))
            with self.subTest(smoke=smoke, synthetic=synthetic):
                with patch.object(tr.random, 'seed') as seed, patch.object(
                        tr, 'setup_distributed', side_effect=RuntimeError('stopped before CUDA setup')):
                    with self.assertRaisesRegex(RuntimeError, 'before CUDA setup'):
                        tr.execute(scoped)
                    if smoke and synthetic:
                        seed.assert_called_once_with(tr.derive_seed('smoke_python', 20261102))
                    else:
                        seed.assert_not_called()
        ident = tr.trainer_identity()
        self.assertEqual(ident['pid'], os.getpid())
        self.assertEqual(ident['start_ticks'], int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]))

    def fixture(self, root):
        source = root / 'prefix'
        runner = EndToEndTests()
        self.assertEqual(runner.run_main(root, resume_argv(source)), 0)
        return runner, source, resume_proof(root, source)

    def approved_args(self, root, proof, index=0):
        path = root / ('proof%d.json' % index)
        tr.m1.atomic_json(path, proof)
        return tr.parser().parse_args(resume_argv(root / ('resume%d' % index)) +
                                      ['--smoke-resume-proof', str(path), '--approved-smoke-resume-sha256',
                                       tr.m1.sha256(path)])

    def validate(self, runner, root, args):
        import contextlib
        with contextlib.ExitStack() as stack:
            for p in runner.patches(root):
                stack.enter_context(p)
            return tr.validate_smoke_resume(args)

    def test_fresh_process_execute_four_equals_prefix_two_restore_two(self):
        torch = self.torch
        worker = """import sys,torch,json,random
sys.path.insert(0, %r); sys.path.insert(0, %r)
import test_e3_train as T, train_e3_lora as tr
torch.set_num_threads(2)
trace = []
real_draw = tr.draw_t_noise
def draw(*a, **kw):
    tid, noise = real_draw(*a, **kw)
    trace.append([a[1], a[2], tid, tr.ts.state_hash(noise), random.random()])
    return tid, noise
tr.draw_t_noise = draw
rc=T.EndToEndTests().run_main(T.Path(sys.argv[1]),sys.argv[2:])
if rc == 0:
    out = tr.parser().parse_args(sys.argv[2:]).out_dir
    (out / 'actual_noise_trace.json').write_text(json.dumps(trace))
assert not torch.cuda.is_initialized()
sys.exit(rc)
""" % (str(SCRIPTS), str(Path(__file__).resolve().parent))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(argv):
                p = subprocess.run([sys.executable, '-c', worker, str(root), *argv],
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True, timeout=180)
                self.assertEqual(p.returncode, 0, p.stderr[-6000:])
            run(resume_argv(root / 'full', steps=4))
            run(resume_argv(root / 'prefix'))
            proof = resume_proof(root, root / 'prefix')
            args = self.approved_args(root, proof)
            run(resume_argv(Path(args.out_dir)) + ['--smoke-resume-proof', str(args.smoke_resume_proof),
                                                  '--approved-smoke-resume-sha256', args.approved_smoke_resume_sha256])
            full = torch.load(root / 'full/private' / tr.CKPT_NAME, weights_only=True)
            resumed = torch.load(args.out_dir / 'private' / tr.CKPT_NAME, weights_only=True)
            self.assertEqual(tr.ts.state_hash(full), tr.ts.state_hash(resumed))
            self.assertEqual(full['steps'], 4)
            expected_python = random.Random(tr.derive_seed('smoke_python', 20261102))
            for _ in range(4):
                expected_python.random()
            self.assertEqual(full['rng']['python'], expected_python.getstate())
            for key in ('adapters', 'optimizer', 'rng', 'config'):
                self.assertEqual(tr.ts.state_hash(full[key]), tr.ts.state_hash(resumed[key]), key)
            self.assertEqual(tr.m1.read_json(root / 'full/actual_noise_trace.json'),
                             tr.m1.read_json(root / 'prefix/actual_noise_trace.json') +
                             tr.m1.read_json(args.out_dir / 'actual_noise_trace.json'))
            summary = tr.m1.read_json(args.out_dir / 'summary.json')
            prefix_summary = tr.m1.read_json(root / 'prefix/summary.json')
            full_summary = tr.m1.read_json(root / 'full/summary.json')
            manifest = tr.m1.read_json(args.out_dir / 'run_manifest.json')
            for out, value in [(root / 'full', full_summary), (root / 'prefix', prefix_summary),
                               (args.out_dir, summary)]:
                ck = torch.load(out / 'private' / tr.CKPT_NAME, weights_only=True)
                self.assertNotEqual(value['adapter_initial_sha256'], value['adapter_final_sha256'])
                self.assertEqual(value['adapter_final_sha256'], tr.ts.state_hash(ck['adapters']))
                ident = value['trainer_identity']
                self.assertEqual(set(ident), {'pid', 'start_ticks'})
                self.assertTrue(type(ident['pid']) is int and ident['pid'] > 0
                                and type(ident['start_ticks']) is int and ident['start_ticks'] > 0)
                self.assertEqual(ident, tr.m1.read_json(out / 'run_manifest.json')['trainer_identity'])
                self.assertEqual(tr.m1.read_json(out / 'run_manifest.json')['smoke']['python_rng_rule'],
                                 tr.SMOKE_PYTHON_RNG_RULE)
            self.assertEqual(len({v['trainer_identity']['pid'] for v in [full_summary, prefix_summary, summary]}), 3)
            self.assertEqual(full_summary['adapter_initial_sha256'], prefix_summary['adapter_initial_sha256'])
            self.assertEqual(summary['adapter_initial_sha256'], prefix_summary['adapter_final_sha256'])
            self.assertEqual(summary['adapter_final_sha256'], full_summary['adapter_final_sha256'])
            self.assertEqual((summary['segment_executed_updates'], summary['logical_completed_steps'],
                              summary['completed_steps']), (2, 4, 4))
            self.assertEqual((manifest['logical_start_step'], manifest['logical_end_step_requested'],
                              manifest['segment_requested_updates']), (2, 4, 2))
            self.assertFalse(summary['resume']['allocation_authorized_by_this_proof'])
            self.assertEqual(summary['frozen_base_sha256'], tr.m1.read_json(root / 'full/summary.json')['frozen_base_sha256'])
            def rows(out):
                return [json.loads(line) for line in (out / 'progress.jsonl').read_text().splitlines()]
            a, b = rows(root / 'full'), rows(root / 'prefix') + rows(args.out_dir)
            self.assertEqual([r['step'] for r in rows(args.out_dir)], [3, 4])
            for ra, rb in zip(a, b):
                for key in ('step', 'lr', 'per_rank', 'loss_fm_mean', 'loss_anchor_mean', 'loss_total_mean',
                            'grad_norm_total_before_clip', 'grad_norm_fm', 'grad_norm_anchor_weighted'):
                    self.assertEqual(ra[key], rb[key], key)
            order = tr.DataOrder(full['config']['seed'], 64)
            for step in range(4):
                idx = order.index(step, 0, 1)
                tid, noise = tr.draw_t_noise(full['config']['seed'], step, idx, TINY_X0)
                self.assertEqual(a[step]['per_rank'][0]['sample_index'], idx)
                self.assertEqual(a[step]['per_rank'][0]['timestep_id'], tid)
                self.assertTrue(torch.equal(noise, tr.draw_t_noise(resumed['config']['seed'], step, idx, TINY_X0)[1]))
                self.assertEqual(a[step]['lr'], full['config']['lr'] * tr.lr_factor(step + 1, 1, 4))
            # The next sample/t/noise/LR is also unchanged across the checkpoint boundary.
            next_idx = order.index(4, 0, 1)
            self.assertEqual(next_idx, tr.DataOrder(resumed['config']['seed'], 64).index(4, 0, 1))
            self.assertEqual(tr.lr_factor(5, 1, 4), 0)
            self.assertTrue(torch.equal(tr.draw_t_noise(full['config']['seed'], 4, next_idx, TINY_X0)[1],
                                        tr.draw_t_noise(resumed['config']['seed'], 4, next_idx, TINY_X0)[1]))

    def test_proof_cli_scope_world_offload_and_config_drift_are_cpu_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, source, proof = self.fixture(root)
            args = self.approved_args(root, proof)
            self.assertEqual(self.validate(runner, root, args)['source_completed_steps'], 2)
            changes = [('smoke_resume_proof', None), ('approved_smoke_resume_sha256', None),
                       ('approved_smoke_resume_sha256', 'f' * 64), ('approved_smoke_resume_sha256', 'abc'),
                       ('smoke', False), ('synthetic_inputs', False), ('cache_index', root / 'fake_cache'),
                       ('offload', 'off'), ('smoke_offload', ['on', 'off']), ('smoke_offload', ['on', 'on']),
                       ('skip_model_hash', True), ('skip_frozen_hash', True), ('max_gpu_hours', 2.1),
                       ('smoke_steps', 3), ('max_steps', 5), ('seed', 42), ('lr', 0.001), ('warmup', 0),
                       ('weight_decay', 0.02), ('clip', 2.0), ('anchor_lambda', 0.0),
                       ('anchor_weighting', 'uniform'), ('component_grad_every', 3), ('log_every', 2),
                       ('caption_version', 'different'), ('models_dir', root / 'different_models'),
                       ('framework', root / 'different_framework')]
            for key, value in changes:
                with self.subTest(key=key), self.assertRaises((ValueError, OSError)):
                    self.validate(runner, root, types.SimpleNamespace(**dict(vars(args), **{key: value})))
            with patch.dict(os.environ, WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', RANK='0', LOCAL_RANK='0'):
                with self.assertRaises(ValueError):
                    self.validate(runner, root, args)
            # Main/execute must fail before even the CPU stand-in for CUDA setup is reached.
            with patch.object(tr, 'setup_distributed', side_effect=AssertionError('CUDA setup reached')):
                bad = dict(vars(args), approved_smoke_resume_sha256='f' * 64)
                with self.assertRaises(ValueError):
                    tr.execute(types.SimpleNamespace(**bad))

    def test_proof_manifest_summary_authorization_source_and_checkpoint_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, source, proof = self.fixture(root)
            args = self.approved_args(root, proof)
            paths = {'manifest': source / 'run_manifest.json', 'summary': source / 'summary.json',
                     'authorization': Path(proof['authorization']['path'])}
            originals = {key: path.read_bytes() for key, path in paths.items()}
            def reject_mutation(key, mutate, index):
                value = json.loads(originals[key])
                mutate(value)
                paths[key].write_bytes(json.dumps(value).encode())
                changed = copy.deepcopy(proof)
                rec_key = {'manifest': 'source_run_manifest', 'summary': 'source_summary',
                           'authorization': 'authorization'}[key]
                changed[rec_key] = bound_json_record(paths[key])
                try:
                    bad_args = self.approved_args(root, changed, index)
                    with self.assertRaises(ValueError):
                        self.validate(runner, root, bad_args)
                    if key == 'manifest' and type(value.get('num_samples')) is not int:
                        with patch.object(tr, 'setup_distributed') as setup:
                            with self.assertRaises(ValueError):
                                tr.execute(bad_args)
                            setup.assert_not_called()
                        self.assertFalse(bad_args.out_dir.exists())
                finally:
                    paths[key].write_bytes(originals[key])
            mutations = [
                ('manifest', lambda v: v['config'].update(seed=999)),
                ('manifest', lambda v: v['data'].update(version='cached')),
                ('manifest', lambda v: v['inputs'].update(models_dir='/different/models')),
                ('manifest', lambda v: v['data'].update(context_set_sha256='wrong')),
                ('manifest', lambda v: v.update(world_size=2)),
                ('manifest', lambda v: v.update(num_samples=64.0)),
                ('manifest', lambda v: v.update(num_samples=True)),
                ('manifest', lambda v: v['source']['trainer'].update(sha256='f' * 64)),
                ('manifest', lambda v: v['source'].pop('admission_helper')),
                ('manifest', lambda v: v['framework'].update(unbound='wrong')),
                ('manifest', lambda v: v['base_shards'].update(unbound='wrong')),
                ('manifest', lambda v: v.update(base_shards_hashed=False)),
                ('manifest', lambda v: v['smoke'].update(offload_modes=['off'])),
                ('manifest', lambda v: v['smoke'].update(synthetic_inputs=False)),
                ('manifest', lambda v: v['smoke'].update(python_seed=999)),
                ('manifest', lambda v: v['smoke'].pop('python_rng_rule')),
                ('manifest', lambda v: v.pop('smoke')),
                ('manifest', lambda v: v.update(torch_version='wrong')),
                ('manifest', lambda v: v.update(order_plan_sha256='f' * 64)),
                ('summary', lambda v: v.update(status='failed')),
                ('summary', lambda v: v.update(completed_steps=1)),
                ('summary', lambda v: v['smoke']['on'].update(stop='oom')),
                ('summary', lambda v: v['smoke']['on'].update(completed_steps=1)),
                ('summary', lambda v: v.pop('frozen_base_sha256')),
                ('authorization', lambda v: v.update(total_cap_GPUh=120)),
                ('authorization', lambda v: v['stage_caps_GPUh'].update(single_gpu_training_resume=True)),
                ('authorization', lambda v: v.update(phase_borrowing_allowed=True)),
                ('authorization', lambda v: v.update(formal_candidate_training_budget_approved=True)),
                ('authorization', lambda v: v.update(AR_allowed=True)),
            ]
            for i, (key, mutate) in enumerate(mutations, 1):
                with self.subTest(key=key, index=i):
                    reject_mutation(key, mutate, i)
            for i, (key, value) in enumerate([('schema', 'legacy'), ('phase', 'label_rebuild_and_cache'),
                                             ('formal_training_allowed', True), ('synthetic_inputs_allowed', 1),
                                             ('source_completed_steps', True), ('continuation_updates', 3),
                                             ('logical_horizon', 5), ('source', {})], 50):
                changed = copy.deepcopy(proof)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.validate(runner, root, self.approved_args(root, changed, i))
            for key, path in paths.items():
                raw = path.read_bytes()
                path.write_bytes(raw + b' ')
                try:
                    with self.subTest(raw_hash=key), self.assertRaises(ValueError):
                        self.validate(runner, root, args)
                finally:
                    path.write_bytes(raw)
            ck_path = Path(proof['checkpoint']['path'])
            raw = ck_path.read_bytes()
            ck_path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(ValueError):
                self.validate(runner, root, args)

    def test_saved_checkpoint_config_and_state_hash_rejected_before_update(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, source, proof = self.fixture(root)
            ck_path = Path(proof['checkpoint']['path'])
            value = torch.load(ck_path, weights_only=True)
            value['config']['seed'] += 1
            bad_path = source / 'private/bad.pt'
            record = tr.save_checkpoint(bad_path, value)
            # Low-level restoration checks the actual saved config, not just proof/manifest declarations.
            pipe = build_tiny_pipe()
            opt = torch.optim.AdamW(tr.adapter_list(pipe.dit), lr=5e-5, weight_decay=0.01)
            before = tr.ts.state_hash(tr.checkpoint_state(pipe.dit, opt, 0, {}))
            with self.assertRaisesRegex(ValueError, 'saved config'):
                tr.load_checkpoint(bad_path, record, pipe.dit, opt, 2, expected_config=proof['config'])
            self.assertEqual(before, tr.ts.state_hash(tr.checkpoint_state(pipe.dit, opt, 0, {})))
            # Actual execute revalidates checkpoint state/config after a valid raw-byte proof.
            ck_path.write_bytes(bad_path.read_bytes())
            record['path'] = str(ck_path)
            summary = tr.m1.read_json(source / 'summary.json')
            summary['checkpoint'] = record
            (source / 'summary.json').write_text(json.dumps(summary))
            proof['checkpoint'] = record
            proof['source_summary'] = bound_json_record(source / 'summary.json')
            args = self.approved_args(root, proof)
            with patch.object(tr, 'train_step', side_effect=AssertionError('Update before config validation')):
                self.assertEqual(runner.run_main(root, resume_argv(args.out_dir) +
                    ['--smoke-resume-proof', str(args.smoke_resume_proof), '--approved-smoke-resume-sha256',
                     args.approved_smoke_resume_sha256]), 1)
            failed = tr.m1.read_json(args.out_dir / 'summary.json')
            self.assertIn('saved config', failed['error'])
            self.assertFalse((args.out_dir / 'private' / tr.CKPT_NAME).exists())
            with self.assertRaisesRegex(ValueError, 'state hash'):
                tr.load_checkpoint(bad_path, dict(record, state_sha256='f' * 64), pipe.dit, opt, 2)

    def test_real_execute_smoke_oom_and_budget_shortfall_are_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = EndToEndTests()
            # run_loop catches a real torch OOM exception; no optimizer update or false completion.
            with patch.object(tr, 'e3_losses', side_effect=self.torch.OutOfMemoryError('CPU test out of memory')):
                self.assertEqual(runner.run_main(root, resume_argv(root / 'oom')), 1)
            oom = tr.m1.read_json(root / 'oom/summary.json')
            self.assertEqual((oom['status'], oom['completed_steps'], oom['smoke']['on']['stop']), ('failed', 0, 'oom'))
            self.assertEqual(oom['smoke']['on']['last_step']['status'], 'oom')
            self.assertEqual(oom['adapter_initial_sha256'], oom['adapter_final_sha256'])
            self.assertFalse((root / 'oom/private' / tr.CKPT_NAME).exists())
            argv = resume_argv(root / 'budget') + ['--max-gpu-hours', '0.01', '--cleanup-grace-seconds', '1',
                    '--save-reserve-seconds', '5', '--first-step-estimate-seconds', '100']
            self.assertEqual(runner.run_main(root, argv), 1)
            stopped = tr.m1.read_json(root / 'budget/summary.json')
            self.assertEqual(stopped['smoke']['on']['stop'], 'budget_exhausted_before_max_steps')
            self.assertEqual(stopped['status'], 'failed')
            self.assertFalse((root / 'budget/private' / tr.CKPT_NAME).exists())
            # Even nonzero but insufficient successful steps cannot become a completed smoke receipt.
            real_step = tr.train_step
            calls = [0]
            def one_then_oom(*a, **kw):
                calls[0] += 1
                if calls[0] == 2:
                    return {'status': 'oom', 'step': a[5] + 1}
                return real_step(*a, **kw)
            with patch.object(tr, 'train_step', side_effect=one_then_oom):
                self.assertEqual(runner.run_main(root, resume_argv(root / 'partial')), 1)
            partial = tr.m1.read_json(root / 'partial/summary.json')
            self.assertEqual((partial['completed_steps'], partial['smoke']['on']['stop']), (1, 'oom'))
            self.assertNotEqual(partial['adapter_initial_sha256'], partial['adapter_final_sha256'])
            self.assertFalse((root / 'partial/private' / tr.CKPT_NAME).exists())


def fake_fa3_modules(root, signature='q, k, v, deterministic=False'):
    import torch
    from einops import rearrange
    framework = root / 'framework'
    wan_path = framework / 'diffsynth/models/wan_video_dit.py'
    wan_path.parent.mkdir(parents=True)
    wan_path.write_text('# CPU mock native Wan module\n')
    interface_path = root / 'fa3/flash_attn_interface.py'
    interface_path.parent.mkdir()
    native_path = interface_path.parent / 'flash_attn_3/_C.abi3.so'
    native_path.parent.mkdir()
    native_path.write_bytes(b'CPU mock native identity; never a kernel')
    text = ('calls = []\ndef flash_attn_func(' + signature + '):\n'
            '    calls.append(locals().copy())\n    return q\n')
    interface_path.write_text(text)
    interface = types.ModuleType('flash_attn_interface')
    interface.__file__ = str(interface_path)
    interface.__spec__ = types.SimpleNamespace(origin=str(interface_path))
    exec(compile(text, str(interface_path), 'exec'), interface.__dict__)
    native = types.ModuleType('flash_attn_3._C')
    native.__file__ = str(native_path)
    native.__spec__ = types.SimpleNamespace(origin=str(native_path))
    wan = types.ModuleType('diffsynth.models.wan_video_dit')
    wan.__file__ = str(wan_path)
    wan.FLASH_ATTN_3_AVAILABLE = True
    wan.flash_attn_interface = interface
    tree = ast.parse((FRAMEWORK / 'diffsynth/models/wan_video_dit.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'flash_attention')
    wan.__dict__.update(torch=torch, rearrange=rearrange, F=torch.nn.functional)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(wan_path), 'exec'), wan.__dict__)
    return framework, wan, interface, native


class DeterministicFa3Tests(Base):
    def setUp(self):
        self.authorization_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.authorization_dir.cleanup)
        self.authorization_path = Path(self.authorization_dir.name) / 'synthetic_authorization.json'
        self.authorization = {
            'schema': 'teacher_deterministic_recovery_authorization_v1',
            'user_reply': 'Synthetic CPU test fixture; not a user GPU authorization',
            'in_response_to': 'CPU-only verification of the deterministic smoke authorization contract',
            'assistant_bounded_scope': {
                'new_GPUh_cap': 2, 'allocated_cards': 1, 'wall_cap_seconds': 7200, 'one_attempt': True,
                'reference_updates': 4, 'prefix_updates': 2, 'fresh_process_resume_updates': 2,
                'physical_optimizer_updates': 8,
                'selected_attention_backend': 'same native FA3; explicit deterministic backward',
                'backend_capability_probe': 'First reference update is the capability gate; no extra probe',
                'no_silent_backend_fallback': True,
            },
            'prior_failed_technical_GPUh_upper': 2, 'prior_costs_preserved': True,
            'original_training_stage_caps_GPUh': {
                'label_rebuild_and_cache': 24, 'single_gpu_training_resume': 2,
                'multi_node_technical_validation': 6,
            },
            'new_scope_additional_not_borrowed': True, 'labels_budget_borrowing_allowed': False,
            'formal_training_allowed': False, 'AR_allowed': False, 'student_distillation_allowed': False,
            'requires_new_source_runtime_publication_and_exact_resume_gate': True,
            'automatic_retry_allowed': False,
        }
        tr.m1.atomic_json(self.authorization_path, self.authorization)
        self.authorization_record = bound_json_record(self.authorization_path)
        self.enterContext(patch.multiple(tr, DET_FA3_AUTH=self.authorization_path,
                                        DET_FA3_AUTH_SHA256=self.authorization_record['sha256']))

    def args(self, root, framework):
        return tr.parser().parse_args(resume_argv(root / 'run') +
                                     ['--framework', str(framework), '--smoke-deterministic-fa3'])

    def modules_patch(self, wan, interface, native):
        import contextlib
        replacements = {'diffsynth.models.wan_video_dit': wan,
                        'flash_attn_interface': interface, 'flash_attn_3._C': native}
        @contextlib.contextmanager
        def scoped():
            old = {k: sys.modules.get(k) for k in replacements}
            try:
                sys.modules.update(replacements)
                yield
            finally:
                for k, v in old.items():
                    if v is None:
                        sys.modules.pop(k, None)
                    else:
                        sys.modules[k] = v
        return scoped()

    def test_scope_guard_and_sparse_old_config(self):
        self.assertEqual(tr.deterministic_authorization(), self.authorization_record)
        self.assertEqual(tr.read_bound_json(self.authorization_record), self.authorization)
        original = self.authorization_path.read_bytes()
        for key, value in (('prior_costs_preserved', 1), ('formal_training_allowed', 0),
                           ('prior_failed_technical_GPUh_upper', True)):
            changed = copy.deepcopy(self.authorization)
            changed[key] = value
            self.authorization_path.write_text(json.dumps(changed))
            try:
                with self.subTest(authorization_type_drift=key), self.assertRaisesRegex(ValueError, 'authorization'):
                    tr.deterministic_authorization()
            finally:
                self.authorization_path.write_bytes(original)
        args = tr.parser().parse_args(resume_argv(Path('/tmp/unused_det_scope')))
        old = tr.make_cfg(args, 4)
        self.assertNotIn('smoke_deterministic_fa3', old)
        args.smoke_deterministic_fa3 = True
        tr.validate_args(args)
        self.assertEqual(tr.make_cfg(args, 4), dict(old, smoke_deterministic_fa3=True))
        for kw in ({'smoke': False}, {'synthetic_inputs': False, 'cache_index': Path('/cached'),
                                      'caption_version': 'orig'}, {'cache_index': Path('/cached')}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                tr.validate_args(types.SimpleNamespace(**dict(vars(args), **kw)))
        with patch.dict(os.environ, WORLD_SIZE='2', LOCAL_WORLD_SIZE='2', RANK='0', LOCAL_RANK='0'):
            with self.assertRaises(ValueError):
                tr.validate_args(args)
        with patch.object(tr, 'deterministic_fa3_context') as context, patch.object(tr, '_execute', return_value=0):
            args.smoke_deterministic_fa3 = False
            self.assertEqual(tr.execute(args), 0)
            context.assert_not_called()

    def test_same_native_rearrange_self_cross_keyword_and_finally_restoration(self):
        torch = self.torch
        flags_before = tr.deterministic_flags()
        try:
            tr.set_deterministic_flags(dict(tr.DET_FA3_FLAGS, deterministic_algorithms=False,
                                           deterministic_warn_only=True, cudnn_allow_tf32=True,
                                           matmul_allow_tf32=True, cudnn_benchmark=True))
            saved = tr.deterministic_flags()
            for had_env in (False, True):
                for failure in (False, True):
                    with self.subTest(had_env=had_env, failure=failure), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        framework, wan, interface, native = fake_fa3_modules(root)
                        original = interface.flash_attn_func
                        args = self.args(root, framework)
                        with self.modules_patch(wan, interface, native), patch.dict(os.environ):
                            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
                            if had_env:
                                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
                            q, k, v = (torch.randn(1, 3, 8) for _ in range(3))
                            expected = wan.flash_attention(q, k, v, 2)
                            self.assertFalse(interface.calls[-1]['deterministic'])
                            try:
                                with tr.deterministic_fa3_context(args) as runtime:
                                    self.assertEqual(runtime['flags'], tr.DET_FA3_FLAGS)
                                    self.assertEqual(runtime['environment'], tr.DET_FA3_ENV)
                                    self.assertEqual(runtime['interface'], bound_json_record(interface.__file__))
                                    self.assertEqual(runtime['native'], bound_json_record(native.__file__))
                                    for _ in ('self', 'cross'):
                                        result = wan.flash_attention(q, k, v, 2)
                                        self.assertTrue(torch.equal(result, expected))
                                        self.assertIs(interface.calls[-1]['deterministic'], True)
                                        self.assertEqual(set(interface.calls[-1]), {'q', 'k', 'v', 'deterministic'})
                                    with self.assertRaises(ValueError):
                                        interface.flash_attn_func(q, k, v, deterministic=False)
                                    if failure:
                                        raise RuntimeError('simulated unsupported deterministic backward')
                            except RuntimeError:
                                self.assertTrue(failure)
                            self.assertIs(interface.flash_attn_func, original)
                            self.assertEqual(tr.deterministic_flags(), saved)
                            if had_env:
                                self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':16:8')
                            else:
                                self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', os.environ)
                            self.assertFalse(torch.cuda.is_initialized())
        finally:
            tr.set_deterministic_flags(flags_before)

    def test_unavailable_api_origin_and_flag_drift_fail_closed(self):
        for signature in ('q, k, v', 'q, k, v, **kwargs', 'q, k, v, deterministic=False, /'):
            with self.subTest(signature=signature), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                framework, wan, interface, native = fake_fa3_modules(root, signature)
                args = self.args(root, framework)
                before = tr.deterministic_flags()
                with self.modules_patch(wan, interface, native), patch.dict(os.environ):
                    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
                    with self.assertRaisesRegex(ValueError, 'keyword support'):
                        with tr.deterministic_fa3_context(args):
                            self.fail('Unsupported API accepted')
                    self.assertEqual(tr.deterministic_flags(), before)
                    self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', os.environ)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            framework, wan, interface, native = fake_fa3_modules(root)
            args = self.args(root, framework)
            original = interface.flash_attn_func
            before = tr.deterministic_flags()
            with self.modules_patch(wan, interface, native):
                wan.FLASH_ATTN_3_AVAILABLE = False
                with self.assertRaisesRegex(ValueError, 'fall back'):
                    tr.deterministic_fa3_runtime(args)
                wan.FLASH_ATTN_3_AVAILABLE = True
                native.__spec__.origin += '.different'
                with self.assertRaisesRegex(ValueError, 'origin'):
                    tr.deterministic_fa3_runtime(args)
                native.__spec__.origin = native.__file__
                with self.assertRaisesRegex(ValueError, 'drifted'):
                    with tr.deterministic_fa3_context(args):
                        self.torch.backends.cudnn.benchmark = True
                self.assertIs(interface.flash_attn_func, original)
                self.assertEqual(tr.deterministic_flags(), before)
                with patch.object(self.torch.cuda, 'is_initialized', return_value=True):
                    with self.assertRaisesRegex(ValueError, 'precede CUDA'):
                        with tr.deterministic_fa3_context(args):
                            self.fail('Already initialized CUDA accepted')

    def test_real_native_signature_import_cpu_only(self):
        # This checks the installed interface signature, not GPU kernel capability.
        import importlib.util
        if importlib.util.find_spec('flash_attn_interface') is None:
            self.skipTest('Native FA3 interface is not installed; synthetic contract tests still run')

        code = '''import inspect,os,sys,torch
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import train_e3_lora as tr
assert not torch.cuda.is_initialized()
environment=dict(os.environ); paths=list(sys.path); flags=tr.deterministic_flags()
assert 'diffsynth' not in sys.modules and 'deepspeed' not in sys.modules
with patch.multiple(tr, DET_FA3_AUTH=tr.Path(sys.argv[3]), DET_FA3_AUTH_SHA256=sys.argv[4]), \
     patch.object(torch.cuda,'is_available',return_value=True), \
     patch.object(torch.cuda,'device_count',return_value=1), \
     patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU helper attempted CUDA init')), \
     patch.object(torch._C,'_cuda_init',side_effect=AssertionError('CPU helper attempted native CUDA init')):
    value=tr.deterministic_fa3_runtime(type('Args',(),{'framework':sys.argv[2]})())
    assert value['backend']=='flash_attn_3' and value['deterministic'] is True
    import flash_attn_interface as f
    p=inspect.signature(f.flash_attn_func).parameters['deterministic']
    assert p.kind in (inspect.Parameter.KEYWORD_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD)
assert 'diffsynth' not in sys.modules and 'deepspeed' not in sys.modules
assert environment==dict(os.environ) and paths==sys.path and flags==tr.deterministic_flags()
assert not torch.cuda.is_initialized()
'''
        p = subprocess.run([sys.executable, '-B', '-c', code, str(SCRIPTS), str(FRAMEWORK),
                            str(self.authorization_path), self.authorization_record['sha256']],
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_execute_scope_before_setup_and_backward_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            framework, wan, interface, native = fake_fa3_modules(root)
            args = self.args(root, framework)
            runner = EndToEndTests()
            original = interface.flash_attn_func
            before = tr.deterministic_flags()
            with self.modules_patch(wan, interface, native):
                def setup():
                    self.assertEqual(tr.deterministic_flags(), tr.DET_FA3_FLAGS)
                    self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                    self.assertIsNot(interface.flash_attn_func, original)
                    return tr.Comm(None, 0, 1, 'cpu')
                with (patch.object(tr, 'train_step', side_effect=RuntimeError('deterministic backend unsupported')),
                      patch.object(tr, 'setup_distributed', side_effect=setup)):
                    import contextlib
                    with contextlib.ExitStack() as stack:
                        for p in runner.patches(root)[1:]:
                            stack.enter_context(p)
                        self.assertEqual(tr.execute(args), 1)
                summary = tr.m1.read_json(args.out_dir / 'summary.json')
                manifest = tr.m1.read_json(args.out_dir / 'run_manifest.json')
                self.assertIn('deterministic backend unsupported', summary['error'])
                self.assertFalse((args.out_dir / 'private' / tr.CKPT_NAME).exists())
                self.assertEqual(summary['deterministic_backend'], manifest['deterministic_backend'])
                self.assertEqual(tr.read_bound_json(manifest['deterministic_backend']), tr.deterministic_fa3_runtime(args))
                self.assertIs(interface.flash_attn_func, original)
                self.assertEqual(tr.deterministic_flags(), before)

    def test_deterministic_resume_backend_auth_contract_and_old_schema_isolation(self):
        import contextlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            framework, wan, interface, native = fake_fa3_modules(root)
            runner = EndToEndTests()
            source = root / 'prefix'
            argv = resume_argv(source) + ['--framework', str(framework), '--smoke-deterministic-fa3']
            with self.modules_patch(wan, interface, native):
                full_out = root / 'full_deterministic'
                self.assertEqual(runner.run_main(root, resume_argv(full_out, 4) +
                    ['--framework', str(framework), '--smoke-deterministic-fa3']), 0)
                self.assertEqual(runner.run_main(root, argv), 0)
                proof = resume_proof(root, source)
                manifest = tr.m1.read_json(source / 'run_manifest.json')
                proof.update(deterministic_backend=manifest['deterministic_backend'],
                             additional_deterministic_authorization=tr.deterministic_authorization())
                args = SmokeResumeTests.approved_args(self, root, proof)
                args.framework = framework
                args.smoke_deterministic_fa3 = True
                def validate(scoped):
                    with contextlib.ExitStack() as stack:
                        for p in runner.patches(root):
                            stack.enter_context(p)
                        return tr.validate_smoke_resume(scoped)
                self.assertEqual(validate(args)['deterministic_backend'], proof['deterministic_backend'])
                self.assertEqual(runner.run_main(root, resume_argv(args.out_dir) +
                    ['--framework', str(framework), '--smoke-deterministic-fa3', '--smoke-resume-proof',
                     str(args.smoke_resume_proof), '--approved-smoke-resume-sha256',
                     args.approved_smoke_resume_sha256]), 0)
                self.assertEqual(tr.m1.read_json(args.out_dir / 'summary.json')['completed_steps'], 4)
                full = self.torch.load(full_out / 'private' / tr.CKPT_NAME, map_location='cpu', weights_only=True)
                resumed = self.torch.load(args.out_dir / 'private' / tr.CKPT_NAME, map_location='cpu', weights_only=True)
                self.assertEqual(tr.ts.state_hash(full), tr.ts.state_hash(resumed))
                for key in ('additional_deterministic_authorization', 'deterministic_backend'):
                    changed = copy.deepcopy(proof)
                    changed.pop(key)
                    bad = SmokeResumeTests.approved_args(self, root, changed, 20 + len(key))
                    bad.framework, bad.smoke_deterministic_fa3 = framework, True
                    with self.subTest(missing=key), self.assertRaises(ValueError):
                        validate(bad)
                with self.assertRaises(ValueError):
                    validate(types.SimpleNamespace(**dict(vars(args), smoke_deterministic_fa3=False)))
                for asset in (Path(interface.__file__), Path(native.__file__)):
                    raw = asset.read_bytes()
                    asset.write_bytes(raw + b' actual changed bytes')
                    try:
                        with self.subTest(actual_asset=asset.name), self.assertRaisesRegex(ValueError, 'contract differs'):
                            validate(args)
                    finally:
                        asset.write_bytes(raw)
                backend_path = Path(proof['deterministic_backend']['path'])
                backend_original = backend_path.read_bytes()
                backend = tr.read_bound_json(proof['deterministic_backend'])
                manifest_original = (source / 'run_manifest.json').read_bytes()
                summary_original = (source / 'summary.json').read_bytes()
                changes = [('backend', 'flash_attn_2'), ('deterministic', False),
                           ('flags', dict(tr.DET_FA3_FLAGS, cudnn_benchmark=True)),
                           ('environment', {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}),
                           ('interface', dict(backend['interface'], sha256='f' * 64)),
                           ('native', dict(backend['native'], sha256='f' * 64)),
                           ('additional_deterministic_authorization', dict(proof['additional_deterministic_authorization'],
                                                                          sha256='f' * 64))]
                for i, (key, value) in enumerate(changes, 100):
                    altered = copy.deepcopy(backend)
                    altered[key] = value
                    backend_path.write_text(json.dumps(altered))
                    changed = copy.deepcopy(proof)
                    changed['deterministic_backend'] = bound_json_record(backend_path)
                    for path, raw in ((source / 'run_manifest.json', manifest_original),
                                      (source / 'summary.json', summary_original)):
                        doc = json.loads(raw)
                        doc['deterministic_backend'] = changed['deterministic_backend']
                        path.write_text(json.dumps(doc))
                    changed['source_run_manifest'] = bound_json_record(source / 'run_manifest.json')
                    changed['source_summary'] = bound_json_record(source / 'summary.json')
                    bad = SmokeResumeTests.approved_args(self, root, changed, i)
                    bad.framework, bad.smoke_deterministic_fa3 = framework, True
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'contract differs'):
                        validate(bad)
                backend_path.write_bytes(backend_original)
                (source / 'run_manifest.json').write_bytes(manifest_original)
                (source / 'summary.json').write_bytes(summary_original)
                changed = copy.deepcopy(proof)
                changed['additional_deterministic_authorization']['sha256'] = 'f' * 64
                bad = SmokeResumeTests.approved_args(self, root, changed, 200)
                bad.framework, bad.smoke_deterministic_fa3 = framework, True
                with self.assertRaisesRegex(ValueError, 'authorization'):
                    validate(bad)
                with patch.object(tr, 'setup_distributed') as setup:
                    with self.assertRaises(ValueError):
                        tr.execute(bad)
                    setup.assert_not_called()
            # An old 13-key nondeterministic proof remains old-only even with a new source snapshot.
            old_source = root / 'old_prefix'
            self.assertEqual(runner.run_main(root, resume_argv(old_source)), 0)
            auth_root = root / 'old_auth'
            auth_root.mkdir()
            old_proof = resume_proof(auth_root, old_source)
            self.assertEqual(len(old_proof), 13)
            bad = SmokeResumeTests.approved_args(self, root, old_proof, 300)
            bad.smoke_deterministic_fa3 = True
            with self.assertRaises(ValueError):
                with contextlib.ExitStack() as stack:
                    for p in runner.patches(root):
                        stack.enter_context(p)
                    tr.validate_smoke_resume(bad)


class CliTests(unittest.TestCase):
    def test_help_and_bad_args_do_not_import_torch(self):
        cases = [('train_e3_lora', ['--help'], 0), ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-gpu-hours', '1'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-steps', '50', '--cache-index', '/x', '--caption-version', 'a',
                                    '--max-gpu-hours', 'nan'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-steps', '50', '--cache-index', '/x', '--caption-version', 'a',
                                    '--max-gpu-hours', '999'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-gpu-hours', '6',
                                    '--smoke', '--synthetic-inputs', '--teacher-four-node-admission'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-gpu-hours', '1',
                                    '--smoke', '--synthetic-inputs', '--smoke-resume-proof', '/missing'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-gpu-hours', '1',
                                    '--smoke', '--synthetic-inputs', '--approved-smoke-resume-sha256', 'f' * 64], 1),
                 ('cache_e3_inputs', ['--help'], 0), ('merge_e3_lora', ['--help'], 0)]
        for mod, argv, expected in cases:
            code = ("import %s as r, sys\ntry: rc=r.main(sys.argv[1:])\nexcept SystemExit as e: rc=e.code\n"
                    "assert 'torch' not in sys.modules\nsys.exit(rc)") % mod
            p = subprocess.run([sys.executable, '-c', code, *argv], env=dict(os.environ, PYTHONPATH=str(SCRIPTS)),
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, expected, (mod, argv, p.stderr))
        self.assertFalse(Path('/tmp/never_e3').exists())


def tearDownModule():
    if 'torch' in sys.modules:
        import torch
        assert not torch.cuda.is_initialized(), 'CUDA was initialized by CPU tests'


if __name__ == '__main__':
    unittest.main()
