"""CPU only. Tiny dense FP64 math is an independent oracle, NOT a CUDA test.

The 30-layer integration fixture keeps 5B attribute contracts but substitutes
small registered dimensions and a mock fused kernel; it has no real DiT weights,
RMSNorm/RoPE/projections, VAE or scheduler numerics. Metadata-only CUDA tensor
subclasses test layout/compile dispatch without allocating or executing on GPU.
Actual torch2.7.1/A100 correctness, memory and speed remain smoke blockers.
"""
import ast
import importlib.util
import math
import os
from pathlib import Path
import types
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teacher_dymos", ROOT / "scripts/teacher_dymos.py")
dymos = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dymos)
CALL = dict(prompt="positive", negative_prompt="negative", sigma_shift=5.0,
            cfg_scale=5.0, num_inference_steps=50, num_frames=121,
            height=704, width=1280, seed=0)
TINY = dict(CALL, num_frames=9, height=64, width=96)
TINY_FIXED = {name: TINY[name] for name in dymos._FIXED_CALL}


def oracle(q, k, v, heads, reference=0, gamma=0):
    """Independent tiny full softmax, explicit rectangular subtraction."""
    b, s, width = q.shape
    if s > 32:
        raise AssertionError("Dense oracle forbidden outside tiny tests")
    d = width // heads
    qh, kh, vh = (x.reshape(b, s, heads, d).transpose(1, 2).double() for x in (q, k, v))
    logits = qh @ kh.transpose(-2, -1) / math.sqrt(d)
    changed = logits.clone()
    if gamma:
        changed[:, :, reference:, :reference] -= gamma
    weights = changed.softmax(-1)
    output = (weights @ vh).transpose(1, 2).reshape(b, s, width).to(q.dtype)
    return output, logits, changed, weights


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 24
        self.original_calls = 0

    def forward(self, q, k, v):
        self.original_calls += 1
        # A recognizable cheap original-backend sentinel, not genuine attention.
        return q + v


class FixturePipe:
    def __init__(self, tiny=False, seed=0):
        self.dit = torch.nn.Module()
        self.dit.dim, self.dit.in_dim = 3072, 48
        self.dit.patch_size = (1, 2, 2)
        self.dit.patch_embedding = torch.nn.Conv3d(48, 3072, (1, 2, 2), (1, 2, 2), device="meta")
        self.dit.fuse_vae_embedding_in_latents = self.dit.seperated_timestep = True
        self.dit.require_vae_embedding = self.dit.require_clip_embedding = False
        self.dit.blocks = torch.nn.ModuleList()
        for _ in range(30):
            block = torch.nn.Module()
            block.self_attn = torch.nn.Module()
            block.self_attn.dim, block.self_attn.head_dim = 3072, 128
            block.self_attn.attn = TinyAttention()
            block.cross_attn = torch.nn.Module()
            block.cross_attn.attn = TinyAttention()
            self.dit.blocks.append(block)
        self.vae = types.SimpleNamespace(upsampling_factor=16)
        self.scheduler = types.SimpleNamespace(timesteps=torch.linspace(1000, 20, 50))
        self.call = dict(TINY if tiny else CALL, seed=seed)
        f, h, w = (self.call["num_frames"] - 1) // 4 + 1, self.call["height"] // 16, self.call["width"] // 16
        # Expanded production-shape fixtures consume O(1) storage until clone of z.
        self.x = torch.tensor(0.25, dtype=torch.float64).expand(1, 48, f, h, w)
        self.z = self.x[:, :, 0:1]
        s = f * (h // 2) * (w // 2)
        self.q = torch.tensor(0.1, dtype=torch.float64).expand(1, s, 3072)
        self.k = torch.tensor(0.2, dtype=torch.float64).expand_as(self.q)
        self.v = torch.tensor(0.3, dtype=torch.float64).expand_as(self.q)
        self.records = []
        self.fail = False
        self.layer_count = 30

    def kwargs(self, index=0, positive=True):
        values = dict(self.call)
        values.pop("negative_prompt" if positive else "prompt")
        values.update(dit=self.dit, latents=self.x, first_frame_latents=self.z,
                      fuse_vae_embedding_in_latents=True, timestep=self.scheduler.timesteps[index:index+1],
                      context=torch.ones(1, 4, 2, dtype=torch.float64))
        return values

    def model_fn(self, **kwargs):
        self.records.append(kwargs)
        if self.fail:
            raise RuntimeError("model failure")
        for block in list(self.dit.blocks)[:self.layer_count]:
            block.self_attn.attn(self.q, self.k, self.v)
            block.cross_attn.attn(self.q, self.k, self.v)
        return kwargs["latents"]

    def run(self):
        for index in range(50):
            self.model_fn(**self.kwargs(index, True))
            self.model_fn(**self.kwargs(index, False))
        return self.x


class MetadataOnlyCudaTensor(torch.Tensor):
    """CPU storage with fake .device metadata; never an actual CUDA tensor."""
    @property
    def device(self):
        return torch.device("cuda")


class TestMath(unittest.TestCase):
    def test_independent_fp64_logits_outputs_mass_and_odds(self):
        generator = torch.Generator().manual_seed(20261025)
        q, k, v = (torch.randn(1, 7, 8, dtype=torch.float64, generator=generator) for _ in range(3))
        r, gamma = 2, 0.6
        expected, original, changed, weights = oracle(q, k, v, 2, r, gamma)
        modify = dymos._score_mod(r, gamma)
        qi, ki = torch.arange(7)[:, None], torch.arange(7)[None, :]
        actual = modify(original, torch.tensor(0), torch.tensor(0), qi, ki)
        self.assertTrue(torch.equal(actual, changed))
        self.assertTrue(torch.equal(actual[:, :, :r, :], original[:, :, :r, :]))
        self.assertTrue(torch.equal(actual[:, :, :, r:], original[:, :, :, r:]))
        baseline = original.softmax(-1)
        for query in range(r, 7):
            before = baseline[:, :, query, :r].sum(-1)
            after = weights[:, :, query, :r].sum(-1)
            self.assertTrue(torch.all(after < before))
            ratio = (after / (1 - after)) / (before / (1 - before))
            torch.testing.assert_close(ratio, torch.full_like(ratio, math.exp(-gamma)), rtol=1e-12, atol=1e-12)
        result = (actual.softmax(-1) @ v.reshape(1, 7, 2, 4).transpose(1, 2)).transpose(1, 2).reshape_as(v)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(weights[:, :, :r], baseline[:, :, :r], rtol=0, atol=0)

    def test_uniform_mass_exact_formula(self):
        q, k = torch.zeros(1, 6, 4, dtype=torch.float64), torch.zeros(1, 6, 4, dtype=torch.float64)
        _, _, _, weights = oracle(q, k, q, 1, 2, 0.6)
        expected = 2 * math.exp(-0.6) / (2 * math.exp(-0.6) + 4)
        torch.testing.assert_close(weights[0, 0, 2:, :2].sum(-1), torch.full((4,), expected, dtype=torch.float64), rtol=1e-14, atol=1e-14)

    def test_layout_scaling_and_compile_contract_mock_only(self):
        # Dispatch simulation uses genuine CPU FP32 operations and fake device metadata.
        generator = torch.Generator().manual_seed(1)
        q, k, v = (torch.randn(1, 7, 8, generator=generator) for _ in range(3))
        expected = oracle(q, k, v, 2, 2, 0.6)[0]
        captured = []
        def fake_compiled(qh, kh, vh, *, score_mod, scale):
            captured.append((qh.shape, scale))
            logits = (qh.double() @ kh.double().transpose(-2, -1)) * scale
            qi, ki = torch.arange(7)[:, None], torch.arange(7)[None, :]
            logits = score_mod(logits, torch.tensor(0), torch.tensor(0), qi, ki)
            return (logits.softmax(-1) @ vh.double()).to(qh.dtype)
        with mock.patch("torch.compile", return_value=fake_compiled) as compile_fn:
            with mock.patch("torch.nn.attention.flex_attention.flex_attention") as flex:
                runner = dymos._FusedAttention()
                inputs = [x.as_subclass(MetadataOnlyCudaTensor) for x in (q, k, v)]
                actual = runner(*inputs, 2, 2, 0.6).as_subclass(torch.Tensor)
                runner(*inputs, 2, 2, 0.6)
        compile_fn.assert_called_once()
        target = compile_fn.call_args.args[0]
        self.assertEqual(target.__name__, "compiled_only")
        self.assertEqual(compile_fn.call_args.kwargs, {"backend": "inductor", "fullgraph": True, "dynamic": False})
        flex.assert_not_called()
        self.assertEqual(captured, [(torch.Size((1, 2, 7, 4)), 0.5)] * 2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_disabled_compile_cannot_reach_eager_flex_or_allocate_dense_logits(self):
        q = torch.ones(1, 7, 8).as_subclass(MetadataOnlyCudaTensor)
        with mock.patch("torch.compile", side_effect=lambda target, **kwargs: target):
            with mock.patch("torch.nn.attention.flex_attention.flex_attention", side_effect=AssertionError("eager forbidden")) as flex:
                with self.assertRaisesRegex(RuntimeError, "refuses eager.*disabled"):
                    dymos._FusedAttention()(q, q, q, 2, 2, 0.6)
                flex.assert_not_called()

    def test_active_real_cpu_is_rejected_without_compile_or_dense_fallback(self):
        q = torch.ones(1, 8, 4)
        with mock.patch("torch.compile", side_effect=AssertionError("must not compile CPU")):
            with self.assertRaisesRegex(RuntimeError, "compiled CUDA.*smoke pending"):
                dymos._FusedAttention()(q, q, q, 1, 2, 0.6)


class TestGeometryAndBoundary(unittest.TestCase):
    def test_actual_5b_latent_to_patch_geometry(self):
        pipe = FixturePipe()
        self.assertEqual(pipe.x.shape, (1, 48, 31, 44, 80))
        self.assertEqual(dymos._reference_span(pipe.dit, pipe.vae, pipe.x, CALL), (880, 27280))
        self.assertEqual(len(dymos._validate_boundary(pipe, CALL, "NONE")), 30)
        tiny = FixturePipe(tiny=True)
        self.assertEqual(dymos._reference_span(tiny.dit, tiny.vae, tiny.x, TINY), (6, 18))
        with self.assertRaisesRegex(ValueError, "registered frame/spatial"):
            dymos._reference_span(pipe.dit, pipe.vae, tiny.x, CALL)
        pipe.dit.patch_embedding = torch.nn.Conv3d(48, 3072, (1, 2, 2), (1, 2, 2), padding=(0, 1, 0), device="meta")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            dymos._reference_span(pipe.dit, pipe.vae, pipe.x, CALL)

    def test_bad_calls_and_combinations(self):
        cases = [dict(seed=2), dict(seed=True), dict(cfg_scale=1), dict(num_frames=81),
                 dict(num_inference_steps=49), dict(sigma_shift=3), dict(prompt=None),
                 dict(negative_prompt=None), dict(cfg_merge=True), dict(cfg_zero_star=True),
                 dict(tea_cache_l1_thresh=0.1), dict(use_unified_sequence_parallel=True),
                 dict(lora_path="loaded"), dict(control_video=object()), dict(end_image=object())]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                with dymos.teacher_dymos(FixturePipe(), call_kwargs=dict(CALL, **change), checkpoint="NONE"):
                    pass
        for options in (dict(checkpoint="lora"), dict(gamma=0.5), dict(gamma=True),
                        dict(gamma=float("nan")), dict(phi=0.5), dict(enabled=1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                with dymos.teacher_dymos(FixturePipe(), call_kwargs=CALL, **{"checkpoint": "NONE", **options}):
                    pass

    def test_reject_runtime_and_loaded_unsupported_models(self):
        for attribute, value in (("dit2", object()), ("tea_cache", object()), ("vace", object()), ("sp_size", 2)):
            pipe = FixturePipe()
            setattr(pipe, attribute, value)
            with self.subTest(attribute=attribute), self.assertRaises(ValueError):
                dymos._validate_boundary(pipe, CALL, "NONE")
        pipe = FixturePipe()
        pipe.dit.blocks[0].self_attn.attn.lora_A_weights = [object()]
        with self.assertRaisesRegex(ValueError, "loaded LoRA"):
            dymos._validate_boundary(pipe, CALL, "NONE")
        pipe = FixturePipe()
        pipe.dit.blocks = pipe.dit.blocks[:29]
        with self.assertRaisesRegex(ValueError, "30"):
            dymos._validate_boundary(pipe, CALL, "NONE")
        pipe = FixturePipe()
        pipe.dit.blocks[1].self_attn.attn = pipe.dit.blocks[0].self_attn.attn
        with self.assertRaisesRegex(ValueError, "shared"):
            dymos._validate_boundary(pipe, CALL, "NONE")
        pipe = FixturePipe()
        pipe.dit.blocks[0].cross_attn.attn = pipe.dit.blocks[0].self_attn.attn
        with self.assertRaisesRegex(ValueError, "shared"):
            dymos._validate_boundary(pipe, CALL, "NONE")
        pipe = FixturePipe()
        pipe.dit.wantodance_enable_global = True
        with self.assertRaisesRegex(ValueError, "WanToDance"):
            dymos._validate_boundary(pipe, CALL, "NONE")


class TestHooks(unittest.TestCase):
    def test_seed0_seed1_30_layers_positive_early_only_no_extra_forwards(self):
        for seed in (0, 1):
            pipe = FixturePipe(tiny=True, seed=seed)
            originals = [block.self_attn.attn.forward for block in pipe.dit.blocks]
            cross = [block.cross_attn.attn.forward for block in pipe.dit.blocks]
            scheduler = pipe.scheduler
            seen = []
            def fake_fused(q, k, v, heads, reference, gamma):
                seen.append((len(pipe.records), heads, reference, gamma, q.shape))
                return q + v
            with mock.patch.dict(dymos._FIXED_CALL, TINY_FIXED), mock.patch.object(dymos._FusedAttention, "__call__", side_effect=fake_fused):
                with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE") as stats:
                    result = pipe.run()
            self.assertIs(result, pipe.x)
            self.assertIs(pipe.scheduler, scheduler)
            self.assertEqual(len(pipe.records), 100)
            self.assertEqual(stats["completed_steps"], 50)
            self.assertEqual(stats["active_steps"], 10)
            self.assertEqual(stats["biased_attention_calls"], 300)
            self.assertEqual(stats["original_attention_calls"], 2700)
            self.assertEqual((stats["reference_tokens"], stats["sequence_tokens"]), (6, 18))
            self.assertEqual([entry[0] for entry in seen], [index * 2 + 1 for index in range(10) for _ in range(30)])
            self.assertTrue(all(entry[1:] == (24, 6, 0.6, torch.Size((1, 18, 3072))) for entry in seen))
            for block, original, cross_original in zip(pipe.dit.blocks, originals, cross):
                self.assertNotIn("forward", vars(block.self_attn.attn))
                self.assertEqual(block.self_attn.attn.forward, original)
                self.assertEqual(block.cross_attn.attn.forward, cross_original)
                self.assertEqual(block.cross_attn.attn.original_calls, 100)
                self.assertEqual(block.self_attn.attn.original_calls, 90)
            self.assertNotIn("model_fn", vars(pipe))
            self.assertNotIn(id(pipe), dymos._ACTIVE_PIPELINES)
            self.assertTrue(all(record["latents"] is pipe.x and record["first_frame_latents"] is pipe.z for record in pipe.records))

    def test_off_gamma0_no_wrappers_compilation_or_attention_mutation_bitwise(self):
        for seed in (0, 1):
            for options in (dict(enabled=False), dict(gamma=0.0)):
                pipe = FixturePipe(tiny=True, seed=seed)
                original = pipe.model_fn
                q = torch.tensor([[[float("nan"), -0.0, 0.0, float("inf")]]])
                attn = pipe.dit.blocks[0].self_attn.attn
                with mock.patch.dict(dymos._FIXED_CALL, TINY_FIXED), mock.patch("torch.compile", side_effect=AssertionError("identity compile")), mock.patch.object(dymos._FusedAttention, "__call__", side_effect=AssertionError("identity fused")):
                    expected = attn(q, q, q)
                    with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE", **options) as stats:
                        self.assertEqual(pipe.model_fn, original)
                        self.assertNotIn("forward", vars(attn))
                        actual = attn(q, q, q)
                        output = pipe.run()
                self.assertTrue(dymos._same_bits(expected, actual))
                self.assertIs(output, pipe.x)
                self.assertEqual(len(pipe.records), 100)
                self.assertTrue(stats["bypassed"])

    def test_negative_inactive_dispatch_original_even_nonstandard_inputs(self):
        pipe = FixturePipe(tiny=True)
        hook = dymos._DyMoSHook(pipe, pipe.call, 0.6, [block.self_attn.attn for block in pipe.dit.blocks])
        attn = pipe.dit.blocks[0].self_attn.attn
        original = mock.Mock(return_value=object())
        wrapper = hook.attention_forward(0, attn, original)
        q, k, v = object(), object(), object()
        with mock.patch("torch.compile", side_effect=AssertionError("must not compile")):
            self.assertIs(wrapper(q, k, v), original.return_value)
        original.assert_called_once_with(q, k, v)

    def test_exception_restores_exact_owned_and_inherited_methods_future_run_clean(self):
        pipe = FixturePipe(tiny=True)
        pipe.model_fn = pipe.model_fn
        owned_model = pipe.model_fn
        pipe.dit.blocks[0].self_attn.attn.forward = pipe.dit.blocks[0].self_attn.attn.forward
        owned_attention = pipe.dit.blocks[0].self_attn.attn.forward
        pipe.fail = True
        with mock.patch.dict(dymos._FIXED_CALL, TINY_FIXED):
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE"):
                    pipe.model_fn(**pipe.kwargs())
            self.assertIs(pipe.model_fn, owned_model)
            self.assertIs(pipe.dit.blocks[0].self_attn.attn.forward, owned_attention)
            self.assertNotIn(id(pipe), dymos._ACTIVE_PIPELINES)
            pipe.fail = False
            pipe.run()
            self.assertEqual(len(pipe.records), 101)
            with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE", enabled=False):
                pass

    def test_cpu_kernel_exception_restores_and_clears_branch_state(self):
        pipe = FixturePipe()
        original = pipe.model_fn
        with self.assertRaisesRegex(RuntimeError, "GPU smoke pending"):
            with dymos.teacher_dymos(pipe, call_kwargs=CALL, checkpoint="NONE"):
                pipe.model_fn(**pipe.kwargs())
        self.assertEqual(pipe.model_fn, original)
        self.assertNotIn("model_fn", vars(pipe))
        self.assertTrue(all("forward" not in vars(block.self_attn.attn) for block in pipe.dit.blocks))
        self.assertNotIn(id(pipe), dymos._ACTIVE_PIPELINES)

    def test_bad_branch_runtime_timestep_shape_and_missing_layer_restore(self):
        mutations = [dict(prompt="other"), dict(negative_prompt="negative"), dict(seed=1),
                     dict(cfg_merge=True), dict(tea_cache=object()), dict(dit=object()),
                     dict(fuse_vae_embedding_in_latents=False), dict(context=torch.ones(2, 4, 2)),
                     dict(timestep=torch.tensor([999.])), dict(timestep=torch.tensor(1000.)),
                     dict(first_frame_latents=torch.zeros(1, 48, 1, 4, 6)), dict(latents=torch.zeros(1, 48, 2, 4, 6))]
        for mutation in mutations:
            pipe = FixturePipe(tiny=True)
            with mock.patch.dict(dymos._FIXED_CALL, TINY_FIXED), self.subTest(mutation=list(mutation)), self.assertRaises(ValueError):
                with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE"):
                    pipe.model_fn(**dict(pipe.kwargs(), **mutation))
            self.assertNotIn(id(pipe), dymos._ACTIVE_PIPELINES)
        pipe = FixturePipe(tiny=True)
        pipe.layer_count = 29
        with mock.patch.dict(dymos._FIXED_CALL, TINY_FIXED), mock.patch.object(dymos._FusedAttention, "__call__", side_effect=lambda q, *args: q):
            with self.assertRaisesRegex(ValueError, "all 30"):
                with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE"):
                    pipe.model_fn(**pipe.kwargs())

    def test_qkv_layout_bad_shape_and_repeated_layer_rejected(self):
        for bad in ("shape", "dtype", "repeat"):
            pipe = FixturePipe(tiny=True)
            hook = dymos._DyMoSHook(pipe, pipe.call, 0.6, [block.self_attn.attn for block in pipe.dit.blocks])
            hook.active, hook.reference_tokens, hook.sequence_tokens = True, 6, 18
            attn = pipe.dit.blocks[0].self_attn.attn
            wrapper = hook.attention_forward(0, attn, attn.forward)
            q, k, v = pipe.q, pipe.k, pipe.v
            if bad == "shape":
                q = q[:, :1]
            elif bad == "dtype":
                k = k.float()
            else:
                hook.seen_layers.add(0)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                wrapper(q, k, v)

    def test_incomplete_nested_and_successive_contexts(self):
        pipe = FixturePipe()
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            with dymos.teacher_dymos(pipe, call_kwargs=CALL, checkpoint="NONE"):
                pass
        with dymos.teacher_dymos(pipe, call_kwargs=CALL, checkpoint="NONE", gamma=0):
            with self.assertRaisesRegex(ValueError, "nested/concurrent"):
                with dymos.teacher_dymos(pipe, call_kwargs=CALL, checkpoint="NONE", enabled=False):
                    pass
        with dymos.teacher_dymos(pipe, call_kwargs=CALL, checkpoint="NONE", enabled=False):
            pass
        self.assertNotIn(id(pipe), dymos._ACTIVE_PIPELINES)

    def test_production_code_has_no_dense_logits_or_global_patch(self):
        source = (ROOT / "scripts/teacher_dymos.py").read_text()
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, ast.MatMult) for node in ast.walk(tree)))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for forbidden in ("matmul", "bmm", "einsum", "scaled_dot_product_attention", "softmax", "outer"):
            self.assertNotIn(forbidden, calls)
        self.assertNotIn("flash_attention =", source)
        self.assertNotIn("scheduler.step =", source)


WAN_SOURCE = ROOT / "DiffSynth-Studio/diffsynth/pipelines/wan_video.py"
DIT_SOURCE = ROOT / "DiffSynth-Studio/diffsynth/models/wan_video_dit.py"


class TestExtractedUpstreamCPU(unittest.TestCase):
    @unittest.skipUnless(DIT_SOURCE.exists(), "optional upstream source not shipped in public tree")
    def test_actual_selfattention_rmsnorm_rope_and_output_projection_unchanged(self):
        tree = ast.parse(DIT_SOURCE.read_text())
        names = {"RMSNorm", "AttentionModule", "SelfAttention", "rope_apply"}
        nodes = [node for node in tree.body if getattr(node, "name", None) in names]
        self.assertEqual(len(nodes), 4)
        def tiny_rearrange(x, pattern, n):
            self.assertEqual(pattern, "b s (n d) -> b s n d")
            return x.reshape(x.shape[0], x.shape[1], n, x.shape[2] // n)
        scope = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional,
                 "rearrange": tiny_rearrange,
                 "flash_attention": lambda q, k, v, num_heads: oracle(q, k, v, num_heads)[0]}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DIT_SOURCE), "exec"), scope)
        torch.manual_seed(41)
        model = scope["SelfAttention"](8, 2).double()
        x = torch.randn(1, 7, 8, dtype=torch.float64)
        angles = torch.arange(7, dtype=torch.float64).reshape(7, 1, 1) * 0.21
        freqs = torch.polar(torch.ones_like(angles), angles)
        q = scope["rope_apply"](model.norm_q(model.q(x)), freqs, 2)
        k = scope["rope_apply"](model.norm_k(model.k(x)), freqs, 2)
        v = model.v(x)
        baseline = model.o(oracle(q, k, v, 2)[0])
        torch.testing.assert_close(model(x, freqs), baseline, rtol=0, atol=0)
        pipe = types.SimpleNamespace(model_fn=lambda **kw: None, dit=types.SimpleNamespace(dim=8))
        hook = dymos._DyMoSHook(pipe, TINY, 0.6, [model.attn])
        hook.active, hook.reference_tokens, hook.sequence_tokens = True, 2, 7
        original = model.attn.forward
        def fake_fused(a, b, c, heads, reference, gamma):
            torch.testing.assert_close(a, q, rtol=0, atol=0)
            torch.testing.assert_close(b, k, rtol=0, atol=0)
            torch.testing.assert_close(c, v, rtol=0, atol=0)
            return oracle(a, b, c, heads, reference, gamma)[0]
        model.attn.forward = hook.attention_forward(0, model.attn, original)
        try:
            with mock.patch.object(dymos._FusedAttention, "__call__", side_effect=fake_fused):
                actual = model(x, freqs)
        finally:
            vars(model.attn).pop("forward")
        torch.testing.assert_close(actual, model.o(oracle(q, k, v, 2, 2, 0.6)[0]), rtol=0, atol=0)
        self.assertEqual(model.attn.forward, original)

    @unittest.skipUnless(WAN_SOURCE.exists(), "optional upstream source not shipped in public tree")
    def test_real_wan_call_cfg_branch_keys_and_original_first_latent_lock(self):
        tree = ast.parse(WAN_SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WanVideoPipeline")
        call = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
        call.decorator_list, call.returns = [], None
        for arg in call.args.args + call.args.kwonlyargs:
            arg.annotation = None
        scope = {"torch": torch, "tqdm": lambda values: values}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])), str(WAN_SOURCE), "exec"), scope)
        pipe = FixturePipe(tiny=True)
        pipe.torch_dtype, pipe.device = torch.float64, "cpu"
        pipe.dit2 = pipe.vace2 = None
        pipe.in_iteration_models = ("dit",)
        pipe.units, pipe.post_units = [object()], []
        pipe.load_models_to_device = lambda names: None
        def unit_runner(unit, p, shared, positive, negative):
            shared.update(latents=p.x.clone(), first_frame_latents=p.z, fuse_vae_embedding_in_latents=True)
            positive["context"] = torch.ones(1, 4, 2, dtype=torch.float64)
            negative["context"] = torch.zeros(1, 4, 2, dtype=torch.float64)
            return shared, positive, negative
        pipe.unit_runner = unit_runner
        pipe.scheduler.set_timesteps = lambda n, **kw: None
        stepped = []
        def step(noise, timestep, sample):
            stepped.append((sample, noise))
            return sample + noise * 0.01
        pipe.scheduler.step = step
        pipe.vae.decode = lambda x, **kw: x
        with mock.patch.dict(dymos._FIXED_CALL, TINY_FIXED), mock.patch.object(dymos._FusedAttention, "__call__", side_effect=lambda q, k, v, *args: q + v):
            with dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE") as stats:
                result = scope["__call__"](pipe, **pipe.call, output_type="floatpoint")
        self.assertEqual(stats["biased_attention_calls"], 300)
        self.assertEqual(stats["model_forwards"], 100)
        self.assertEqual(len(stepped), 50)
        self.assertIs(pipe.scheduler.step, step)
        self.assertTrue(dymos._same_bits(result[:, :, :1], pipe.z))
        for i in range(50):
            positive, negative = pipe.records[2 * i:2 * i + 2]
            self.assertIn("prompt", positive)
            self.assertNotIn("negative_prompt", positive)
            self.assertIn("negative_prompt", negative)
            self.assertNotIn("prompt", negative)
            self.assertIs(positive["latents"], negative["latents"])
            self.assertTrue(dymos._same_bits(positive["latents"][:, :, :1], pipe.z))
            self.assertIs(positive["latents"], stepped[i][0])
            torch.testing.assert_close(stepped[i][1], negative["latents"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
