"""CPU-only tests for the E3 cached-input LoRA trainer, cacher and exporter. Never initializes CUDA."""
import ast
import json
import os
from pathlib import Path
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
                "s = time.monotonic()\n"
                "done, recs, stop = tr.run_loop(pipe, opt, params, T.tiny_sample_fp32, tr.DataOrder(11, 5), T.cfg(max_steps=3, anchor_lambda=0.2), comm, tr.budget(1, 2, 1, 1, 0.01), lambda: time.monotonic() - s, None)\n"
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


class CliTests(unittest.TestCase):
    def test_help_and_bad_args_do_not_import_torch(self):
        cases = [('train_e3_lora', ['--help'], 0), ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-gpu-hours', '1'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-steps', '50', '--cache-index', '/x', '--caption-version', 'a',
                                    '--max-gpu-hours', 'nan'], 1),
                 ('train_e3_lora', ['--out-dir', '/tmp/never_e3', '--max-steps', '50', '--cache-index', '/x', '--caption-version', 'a',
                                    '--max-gpu-hours', '999'], 1),
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
