"""CPU-only tests of the isolated ALG hooks, including the real Wan call loop."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teacher_alg", ROOT / "scripts/teacher_alg.py")
alg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alg)
CALL = dict(prompt="positive", negative_prompt="negative", sigma_shift=5.0,
            cfg_scale=5.0, num_inference_steps=50, num_frames=121,
            width=1280, height=704, seed=0)


def checker(shape=(1, 2, 1, 10, 12), dtype=torch.float32):
    grid = (torch.arange(shape[-2])[:, None] + torch.arange(shape[-1])[None, :]) % 2
    return (grid * 2 - 1).to(dtype).expand(shape).clone()


def prediction(x, positive):
    energy = x[:, :, 0:1].square().mean()
    return x * (2 if positive else 0.5) + energy * (3 if positive else 1) + (7 if positive else 1)


class FakeScheduler:
    def __init__(self):
        self.records = []
        self.set_timesteps(50)

    def set_timesteps(self, num_inference_steps, **kwargs):
        self.timesteps = torch.linspace(1000, 20, num_inference_steps)

    def step(self, model_output, timestep, sample, **kwargs):
        self.records.append(dict(noise=model_output.clone(), noise_object=model_output,
                                 timestep=timestep, sample=sample, kwargs=kwargs))
        return sample - model_output * 0.001


class FakePipeline:
    def __init__(self, dtype=torch.float32):
        self.dit = types.SimpleNamespace(fuse_vae_embedding_in_latents=True)
        self.scheduler = FakeScheduler()
        self.model_fn = self.predict
        self.records = []
        self.z = checker(dtype=dtype)
        self.initial = torch.arange(1 * 2 * 3 * 10 * 12, dtype=torch.float32).reshape(1, 2, 3, 10, 12)
        self.initial = (self.initial / 1000).to(dtype)
        self.initial[:, :, 0:1] = self.z
        self.dtype = dtype

    def predict(self, **kwargs):
        self.records.append(dict(kwargs=kwargs, x=kwargs["latents"].clone(),
                                 z=kwargs["first_frame_latents"].clone()))
        return prediction(kwargs["latents"], "prompt" in kwargs)

    def inputs(self, x, i, positive, **changes):
        kwargs = dict(CALL, dit=self.dit, latents=x, first_frame_latents=self.z,
                      fuse_vae_embedding_in_latents=True, cfg_merge=False,
                      context=torch.ones(1, 2, 3, dtype=self.dtype),
                      timestep=self.scheduler.timesteps[i].reshape(1).to(self.dtype))
        kwargs.pop("negative_prompt" if positive else "prompt")
        kwargs.update(changes)
        return kwargs

    def iteration(self, x, i):
        c = self.model_fn(**self.inputs(x, i, True))
        u = self.model_fn(**self.inputs(x, i, False))
        cfg = u + CALL["cfg_scale"] * (c - u)
        result = self.scheduler.step(cfg, self.scheduler.timesteps[i], x)
        result[:, :, 0:1] = self.z
        return result

    def run(self):
        x = self.initial.clone()
        for i in range(50):
            x = self.iteration(x, i)
        return x


class LowpassTests(unittest.TestCase):
    def test_fused_48_channels_44_by_80_rounds_to_18_by_32(self):
        z = checker((1, 48, 1, 44, 80))
        before = z.clone()
        with mock.patch.object(alg.F, "interpolate", wraps=F.interpolate) as interpolate:
            result = alg.lowpass_first_latent(z)
        self.assertEqual(interpolate.call_args_list[0].kwargs,
                         dict(size=(18, 32), mode="bilinear", align_corners=False, antialias=True))
        self.assertEqual(interpolate.call_args_list[1].kwargs,
                         dict(size=(44, 80), mode="bilinear", align_corners=False, antialias=True))
        self.assertEqual(result.shape, z.shape)
        self.assertLess(result.square().mean().item(), z.square().mean().item() / 10)
        self.assertTrue(alg._same_bits(z, before))

    def test_constant_and_dtypes(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                z = torch.full((1, 48, 1, 44, 80), 2, dtype=dtype)
                out = alg.lowpass_first_latent(z)
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(out.device, z.device)
                torch.testing.assert_close(out, z)
                self.assertIs(alg.lowpass_first_latent(z, 1), z)

    def test_reject_bad_shapes_factor_and_nonfinite(self):
        for z in (torch.ones(1, 2, 3, 4), torch.ones(2, 2, 1, 4, 4),
                  torch.ones(1, 2, 2, 4, 4), torch.full((1, 2, 1, 4, 4), float("nan"))):
            with self.subTest(shape=z.shape), self.assertRaises(ValueError):
                alg.lowpass_first_latent(z)
        for factor in (0, -1, 2, float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                alg.lowpass_first_latent(checker(), factor)


class HookTests(unittest.TestCase):
    def test_preregistered_seed1_full_run_and_identity_baseline_routes(self):
        with mock.patch.dict(CALL, seed=1):
            baseline = FakePipeline()
            expected = baseline.run()
            for options, count in (({}, 110), (dict(enabled=False), 100), (dict(factor=1.0), 100)):
                p = FakePipeline()
                original_model, original_step = p.model_fn, p.scheduler.step
                with self.subTest(options=options):
                    with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE", **options) as stats:
                        result = p.run()
                    self.assertEqual(len(p.records), count)
                    self.assertTrue(all(record["kwargs"]["seed"] == 1 for record in p.records))
                    self.assertIs(p.model_fn, original_model)
                    self.assertEqual(p.scheduler.step, original_step)
                    self.assertTrue(alg._same_bits(result[:, :, 0:1], p.z))
                    if options:
                        self.assertTrue(alg._same_bits(result, expected))
                    else:
                        self.assertEqual(stats["completed_steps"], 50)
                        self.assertEqual(stats["active_steps"], 10)
                        self.assertEqual(stats["model_forwards"], 110)

    def test_exact_guidance_branch_latents_active_window_and_original_scheduler_sample(self):
        p = FakePipeline()
        z, initial = p.z.clone(), p.initial.clone()
        model_fn, scheduler_step = p.model_fn, p.scheduler.step
        with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE") as stats:
            out = p.run()
        self.assertIs(p.model_fn, model_fn)
        self.assertNotIn("step", vars(p.scheduler))
        self.assertEqual(p.scheduler.step, scheduler_step)
        self.assertTrue(alg._same_bits(p.z, z))
        self.assertTrue(alg._same_bits(p.initial, initial))
        self.assertTrue(alg._same_bits(out[:, :, 0:1], z))
        self.assertEqual(stats["completed_steps"], 50)
        self.assertEqual(stats["active_steps"], 10)
        self.assertEqual(stats["model_forwards"], 110)
        self.assertEqual(stats["first_frame_sha256"], hashlib.sha256(z.view(torch.uint8).numpy().tobytes()).hexdigest())
        lp = alg.lowpass_first_latent(z)
        offset = 0
        for i, step in enumerate(p.scheduler.records):
            original = step["sample"]
            self.assertTrue(alg._same_bits(original[:, :, 0:1], z))
            self.assertEqual(len(p.records[offset:offset + (3 if i < 10 else 2)]), 3 if i < 10 else 2)
            if i < 10:
                c, uo, ul = p.records[offset:offset + 3]
                self.assertIn("prompt", c["kwargs"])
                self.assertIn("negative_prompt", uo["kwargs"])
                self.assertIn("negative_prompt", ul["kwargs"])
                self.assertTrue(alg._same_bits(c["x"][:, :, 0:1], lp))
                self.assertTrue(alg._same_bits(ul["x"][:, :, 0:1], lp))
                self.assertTrue(alg._same_bits(uo["x"], original))
                self.assertIs(uo["kwargs"]["latents"], original)
                self.assertIsNot(c["kwargs"]["latents"], original)
                self.assertIsNot(ul["kwargs"]["latents"], original)
                self.assertTrue(alg._same_bits(c["x"][:, :, 1:], original[:, :, 1:]))
                self.assertTrue(alg._same_bits(ul["x"][:, :, 1:], original[:, :, 1:]))
                expected = prediction(original, False) + 5 * (prediction(c["x"], True) - prediction(ul["x"], False))
                ordinary = prediction(ul["x"], False) + 5 * (prediction(c["x"], True) - prediction(ul["x"], False))
                self.assertFalse(torch.equal(expected, ordinary))
            else:
                c, u = p.records[offset:offset + 2]
                self.assertIs(c["kwargs"]["latents"], original)
                self.assertIs(u["kwargs"]["latents"], original)
                expected = prediction(original, False) + 5 * (prediction(original, True) - prediction(original, False))
            for record in p.records[offset:offset + (3 if i < 10 else 2)]:
                self.assertTrue(alg._same_bits(record["z"], z))
                self.assertIs(record["kwargs"]["first_frame_latents"], p.z)
            self.assertTrue(alg._same_bits(step["noise"], expected))
            offset += 3 if i < 10 else 2
        self.assertEqual(offset, 110)

    def test_disabled_and_identity_are_bitwise_baseline_without_any_wrappers(self):
        for options in (dict(enabled=False), dict(factor=1.0)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(options=options, dtype=dtype):
                    baseline, p = FakePipeline(dtype), FakePipeline(dtype)
                    expected = baseline.run()
                    model_fn, step = p.model_fn, p.scheduler.step
                    with mock.patch.object(alg, "lowpass_first_latent", side_effect=AssertionError("must not filter")):
                        with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE", **options) as stats:
                            self.assertIs(p.model_fn, model_fn)
                            self.assertEqual(p.scheduler.step, step)
                            actual = p.run()
                    self.assertTrue(alg._same_bits(actual, expected))
                    self.assertEqual(len(p.records), 100)
                    self.assertTrue(stats["bypassed"])
                    for a, b in zip(p.records, baseline.records):
                        self.assertEqual(set(a["kwargs"]), set(b["kwargs"]))
                        self.assertTrue(alg._same_bits(a["x"], b["x"]))
                    for a, b in zip(p.scheduler.records, baseline.scheduler.records):
                        self.assertTrue(alg._same_bits(a["noise"], b["noise"]))

    def test_half_bfloat16_active_route(self):
        for dtype in (torch.float16, torch.bfloat16):
            p = FakePipeline(dtype)
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                result = p.run()
            self.assertEqual(result.dtype, dtype)
            self.assertTrue(torch.isfinite(result).all().item())
            self.assertEqual(len(p.records), 110)

    def test_model_exception_restores_owned_attributes_and_allows_baseline_afterwards(self):
        p = FakePipeline()
        original = p.model_fn
        def fail(**kwargs):
            raise RuntimeError("forward failed")
        p.model_fn = fail
        owned_step = p.scheduler.step
        p.scheduler.step = owned_step
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                p.run()
        self.assertIs(p.model_fn, fail)
        self.assertIs(p.scheduler.step, owned_step)
        self.assertNotIn(id(p), alg._ACTIVE_PIPELINES)
        p.model_fn = original
        self.assertEqual(p.run().shape, p.initial.shape)
        with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
            p.run()

    def test_scheduler_exception_and_nonfinite_output_restore(self):
        for failure in (RuntimeError("step failed"), "nonfinite"):
            p = FakePipeline()
            original = p.model_fn
            def fail(model_output, timestep, sample, **kwargs):
                if failure == "nonfinite":
                    return torch.full_like(sample, float("inf"))
                raise failure
            p.scheduler.step = fail
            with self.assertRaises((RuntimeError, ValueError)):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                    p.run()
            self.assertIs(p.model_fn, original)
            self.assertIs(p.scheduler.step, fail)
            self.assertNotIn(id(p), alg._ACTIVE_PIPELINES)

    def test_runtime_missing_z_bad_batch_unmerged_context_and_nonfinite_rejected(self):
        variants = [dict(first_frame_latents=None), dict(fuse_vae_embedding_in_latents=False),
                    dict(context=torch.ones(2, 2, 3)), dict(context=torch.full((1, 2, 3), float("nan"))),
                    dict(latents=torch.ones(2, 2, 3, 10, 12)), dict(cfg_merge=True),
                    dict(latents=torch.full((1, 2, 3, 10, 12), float("nan"))),
                    dict(first_frame_latents=torch.zeros_like(checker())), dict(seed=1)]
        for changes in variants:
            p = FakePipeline()
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                    p.model_fn(**p.inputs(p.initial.clone(), 0, True, **changes))
            self.assertEqual(len(p.records), 0)
            self.assertNotIn("step", vars(p.scheduler))

    def test_sequence_identity_and_extra_calls_rejected(self):
        def wrong_negative(p, x):
            p.model_fn(**p.inputs(x, 0, False))
        def extra_positive(p, x):
            p.model_fn(**p.inputs(x, 0, True))
            p.model_fn(**p.inputs(x, 0, True))
        def extra_forward(p, x):
            p.model_fn(**p.inputs(x, 0, True))
            p.model_fn(**p.inputs(x, 0, False))
            p.model_fn(**p.inputs(x, 0, True))
        def early_scheduler(p, x):
            p.scheduler.step(torch.zeros_like(x), p.scheduler.timesteps[0], x)
        def different_sample(p, x):
            p.model_fn(**p.inputs(x, 0, True))
            p.model_fn(**p.inputs(x, 0, False))
            p.scheduler.step(torch.zeros_like(x), p.scheduler.timesteps[0], x.clone())
        def mutated_z(p, x):
            p.model_fn(**p.inputs(x, 0, True))
            p.z.add_(1)
            x[:, :, 0:1] = p.z
            p.model_fn(**p.inputs(x, 0, False))
        def timestep(p, x):
            p.model_fn(**p.inputs(x, 1, True))
        def missing_prompt(p, x):
            kwargs = p.inputs(x, 0, True)
            kwargs.pop("prompt")
            p.model_fn(**kwargs)
        def both_prompts(p, x):
            p.model_fn(**p.inputs(x, 0, True, negative_prompt="negative"))
        for operation in (wrong_negative, extra_positive, extra_forward, early_scheduler,
                          different_sample, mutated_z, timestep, missing_prompt, both_prompts):
            p = FakePipeline()
            with self.subTest(operation=operation.__name__), self.assertRaises(ValueError):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                    operation(p, p.initial.clone())
            self.assertNotIn(id(p), alg._ACTIVE_PIPELINES)

    def test_incomplete_context_and_nested_context_rejected_and_restored(self):
        p = FakePipeline()
        original = p.model_fn
        with self.assertRaisesRegex(ValueError, "complete 50-step"):
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                p.iteration(p.initial.clone(), 0)
        self.assertIs(p.model_fn, original)
        with self.assertRaisesRegex(ValueError, "nested/concurrent"):
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE", enabled=False):
                    pass
        self.assertIs(p.model_fn, original)

    def test_nonfinite_forward_and_guidance_rejected(self):
        for overflow in (False, True):
            p = FakePipeline()
            def predict(**kwargs):
                if not overflow:
                    return torch.full_like(kwargs["latents"], float("nan"))
                return torch.full_like(kwargs["latents"], 3e38 if "prompt" in kwargs else -3e38)
            p.model_fn = predict
            with self.subTest(overflow=overflow), self.assertRaisesRegex(ValueError, "nonfinite"):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                    p.run()
            self.assertEqual(len(p.scheduler.records), 0)


class BoundaryTests(unittest.TestCase):
    def test_seed2_and_noninteger_seeds_rejected_before_forward(self):
        for seed in (2, -1, None, "1", True, 1.0):
            p = FakePipeline()
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                with alg.teacher_alg(p, call_kwargs=dict(CALL, seed=seed), checkpoint="NONE"):
                    self.fail("must reject unregistered seed")
            self.assertEqual(p.records, [])
            self.assertNotIn("step", vars(p.scheduler))

    def test_all_registered_unsupported_combinations_fail_before_forward(self):
        for name in alg._FALSE_OPTIONS + alg._ABSENT_OPTIONS:
            p = FakePipeline()
            call = dict(CALL, **{name: True})
            with self.subTest(option=name), self.assertRaisesRegex(ValueError, name):
                with alg.teacher_alg(p, call_kwargs=call, checkpoint="NONE"):
                    self.fail("must not enter context")
            self.assertEqual(p.records, [])
        for name in alg._FIXED_CALL:
            call = dict(CALL)
            call[name] += 1
            with self.subTest(fixed=name), self.assertRaisesRegex(ValueError, name):
                with alg.teacher_alg(FakePipeline(), call_kwargs=call, checkpoint="NONE"):
                    pass

    def test_model_side_dit2_hotlora_and_sequenceparallel_rejected(self):
        p = FakePipeline()
        for name, value in (("dit2", object()), ("vace", object()), ("audio_encoder", object()),
                            ("use_unified_sequence_parallel", True), ("sp_size", 2)):
            setattr(p, name, value)
            with self.subTest(option=name), self.assertRaises(ValueError):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                    pass
            delattr(p, name)
        p.dit.modules = lambda: [types.SimpleNamespace(lora_A_weights=[torch.ones(1)])]
        with self.assertRaisesRegex(ValueError, "LoRA"):
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                pass

    def test_nonbaseline_nonfused_and_unregistered_factor_rejected(self):
        p = FakePipeline()
        with self.assertRaisesRegex(ValueError, "checkpoint=NONE"):
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="finetuned"):
                pass
        for factor in (0.3, 0, float("nan")):
            with self.assertRaisesRegex(ValueError, "factor"):
                with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE", factor=factor):
                    pass
        p.dit.fuse_vae_embedding_in_latents = False
        with self.assertRaisesRegex(ValueError, "non-fused 14B"):
            with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
                pass


class ActualWanLoopTests(unittest.TestCase):
    def test_real_flowmatch_scheduler_steps_original_sample(self):
        path = ROOT / "DiffSynth-Studio/diffsynth/diffusion/flow_match.py"
        spec = importlib.util.spec_from_file_location("alg_real_flowmatch", path)
        flow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow)
        p = FakePipeline()
        scheduler = flow.FlowMatchScheduler("Wan")
        scheduler.set_timesteps(50, shift=5)
        records = []
        original = scheduler.step
        def recording_step(model_output, timestep, sample, **kwargs):
            result = original(model_output, timestep, sample, **kwargs)
            records.append((sample, model_output.clone(), result.clone()))
            return result
        scheduler.step = recording_step
        p.scheduler = scheduler
        with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE"):
            result = p.run()
        self.assertIs(scheduler.step, recording_step)
        self.assertEqual(len(records), 50)
        for i, (sample, noise, stepped) in enumerate(records):
            sigma = scheduler.sigmas[i]
            next_sigma = scheduler.sigmas[i + 1] if i < 49 else 0
            self.assertTrue(alg._same_bits(stepped, sample + noise * (next_sigma - sigma)))
            self.assertTrue(alg._same_bits(sample[:, :, 0:1], p.z))
        self.assertTrue(alg._same_bits(result[:, :, 0:1], p.z))

    def test_actual_call_loop_and_prompt_unitrunner_preserve_branch_keys(self):
        wan_path = ROOT / "DiffSynth-Studio/diffsynth/pipelines/wan_video.py"
        base_path = ROOT / "DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py"
        wan_tree = ast.parse(wan_path.read_text())
        cls = next(n for n in wan_tree.body if isinstance(n, ast.ClassDef) and n.name == "WanVideoPipeline")
        call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
        call.decorator_list = []
        call.returns = None
        for arg in call.args.args + call.args.kwonlyargs:
            arg.annotation = None
        scope = {"torch": torch, "tqdm": lambda seq: seq}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])), str(wan_path), "exec"), scope)
        base_tree = ast.parse(base_path.read_text())
        runner = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == "PipelineUnitRunner")
        runner_scope = {"PipelineUnit": object, "BasePipeline": object}
        exec(compile(ast.Module(body=[runner], type_ignores=[]), str(base_path), "exec"), runner_scope)
        class PromptUnit:
            take_over = False
            seperate_cfg = True
            input_params_posi = {"prompt": "prompt"}
            input_params_nega = {"prompt": "negative_prompt"}
            input_params = ()
            def process(self, pipe, prompt):
                return {"context": torch.ones(1, 2, 3)}
        class FusedUnit:
            take_over = False
            seperate_cfg = False
            input_params = ()
            def process(self, pipe):
                return dict(latents=pipe.initial.clone(), first_frame_latents=pipe.z,
                            fuse_vae_embedding_in_latents=True)
        p = FakePipeline()
        p.torch_dtype, p.device = torch.float32, "cpu"
        p.dit2 = p.vace2 = None
        p.in_iteration_models = ("dit",)
        p.units = [PromptUnit(), FusedUnit()]
        p.post_units = []
        p.unit_runner = runner_scope["PipelineUnitRunner"]()
        p.load_models_to_device = lambda names: None
        p.vae = types.SimpleNamespace(decode=lambda x, **kwargs: x)
        original_z = p.z.clone()
        with alg.teacher_alg(p, call_kwargs=CALL, checkpoint="NONE") as stats:
            result = scope["__call__"](p, **CALL, output_type="floatpoint")
        self.assertEqual(stats["active_steps"], 10)
        self.assertEqual(len(p.records), 110)
        self.assertTrue(alg._same_bits(result[:, :, 0:1], original_z))
        self.assertTrue(alg._same_bits(p.z, original_z))
        self.assertIn("prompt", p.records[0]["kwargs"])
        self.assertIn("negative_prompt", p.records[1]["kwargs"])
        self.assertNotIn("negative_prompt", p.records[0]["kwargs"])
        self.assertNotIn("prompt", p.records[1]["kwargs"])


class MixedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "DiffSynth-Studio/diffsynth/pipelines/wan_video.py"
        tree = ast.parse(path.read_text())
        scope = {"torch": torch, "tqdm": lambda values: values}
        for class_name, method_name in (("WanVideoPipeline", "__call__"),
                                        ("WanVideoUnit_ImageEmbedderFused", "process")):
            klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
            method.decorator_list, method.returns = [], None
            for arg in method.args.args + method.args.kwonlyargs:
                arg.annotation = None
            exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                         str(path), "exec"), scope)
        cls.call_upstream = staticmethod(scope["__call__"])
        cls.fuse_upstream = staticmethod(scope["process"])

    def fixture(self):
        pipe = FakePipeline(torch.bfloat16)
        pipe.z = torch.arange(pipe.z.numel(), dtype=torch.float32).reshape_as(pipe.z) / 79 + 0.001
        pipe.initial[:, :, :1] = pipe.z
        return pipe

    def test_real_fused_unit_preserves_raw_fp32_and_casts_only_assignment(self):
        pipe = self.fixture()
        before = pipe.z.clone()
        pipe.preprocess_image = mock.Mock(return_value=torch.ones(1, 3, 10, 12))
        pipe.load_models_to_device = mock.Mock()
        pipe.vae = types.SimpleNamespace(encode=mock.Mock(return_value=pipe.z))
        pipe.device = "cpu"
        image = types.SimpleNamespace(resize=mock.Mock(return_value=object()))
        x = torch.zeros_like(pipe.initial)
        result = self.fuse_upstream(types.SimpleNamespace(onload_model_names=("vae",)),
                                    pipe, image, x, 10, 12, False, (10, 12), (5, 6))
        effective = pipe.z.to(dtype=x.dtype, device=x.device)
        self.assertIs(result["first_frame_latents"], pipe.z)
        self.assertIs(result["latents"], x)
        self.assertEqual(pipe.z.dtype, torch.float32)
        self.assertEqual(x.dtype, torch.bfloat16)
        self.assertTrue(alg._same_bits(pipe.z, before))
        self.assertTrue(alg._same_bits(x[:, :, :1], effective))
        self.assertFalse(torch.equal(pipe.z, effective.float()))
        self.assertTrue(torch.equal(x[:, :, 1:], torch.zeros_like(x[:, :, 1:])))
        image.resize.assert_called_once_with((12, 10))
        pipe.vae.encode.assert_called_once()

    def test_mixed_dtype_full_run_lowpass_effective_and_raw_hashes(self):
        pipe = self.fixture()
        before = pipe.z.clone()
        effective = pipe.z.to(pipe.initial)
        with mock.patch.object(alg, "lowpass_first_latent", wraps=alg.lowpass_first_latent) as lowpass:
            with alg.teacher_alg(pipe, call_kwargs=CALL, checkpoint="NONE") as stats:
                result = pipe.run()
        self.assertTrue(alg._same_bits(lowpass.call_args.args[0], effective))
        lp = alg.lowpass_first_latent(effective)
        self.assertTrue(alg._same_bits(pipe.records[0]["x"][:, :, :1], lp))
        self.assertTrue(alg._same_bits(pipe.records[2]["x"][:, :, :1], lp))
        self.assertTrue(alg._same_bits(pipe.records[1]["x"][:, :, :1], effective))
        self.assertTrue(alg._same_bits(pipe.z, before))
        self.assertTrue(alg._same_bits(result[:, :, :1], effective))
        self.assertEqual(stats["model_forwards"], 110)
        self.assertEqual(stats["active_steps"], 10)
        self.assertEqual(stats["raw_first_frame_dtype"], "torch.float32")
        self.assertEqual(stats["effective_first_frame_dtype"], "torch.bfloat16")
        self.assertEqual(stats["raw_first_frame_sha256"], stats["first_frame_sha256"])
        self.assertEqual(stats["raw_first_frame_sha256"], hashlib.sha256(before.view(torch.uint8).numpy().tobytes()).hexdigest())
        self.assertEqual(stats["effective_first_frame_sha256"], hashlib.sha256(effective.view(torch.uint8).numpy().tobytes()).hexdigest())
        self.assertNotEqual(stats["raw_first_frame_sha256"], stats["effective_first_frame_sha256"])
        self.assertTrue(all(alg._same_bits(record["sample"][:, :, :1], effective) for record in pipe.scheduler.records))
        self.assertTrue(all(record["kwargs"]["first_frame_latents"] is pipe.z for record in pipe.records))

    def test_mixed_dtype_real_call_initializes_fresh_scheduler_after_context_registration(self):
        pipe = self.fixture()
        flow_path = ROOT / "DiffSynth-Studio/diffsynth/diffusion/flow_match.py"
        spec = importlib.util.spec_from_file_location("alg_fresh_flow", flow_path)
        flow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow)
        pipe.scheduler = flow.FlowMatchScheduler("Wan")
        self.assertFalse(hasattr(pipe.scheduler, "timesteps"))
        original_set = pipe.scheduler.set_timesteps
        initialized = []
        def set_timesteps(n, **kwargs):
            self.assertIn(id(pipe), alg._ACTIVE_PIPELINES)
            self.assertEqual(pipe.model_fn.__self__.__class__, alg._ALGHook)
            initialized.append((n, kwargs))
            original_set(n, **kwargs)
        pipe.scheduler.set_timesteps = set_timesteps
        pipe.torch_dtype, pipe.device = torch.bfloat16, "cpu"
        pipe.dit2 = pipe.vace2 = None
        pipe.in_iteration_models = ("dit",)
        unit = types.SimpleNamespace(onload_model_names=("vae",))
        pipe.units, pipe.post_units = [unit], []
        pipe.load_models_to_device = lambda names: None
        pipe.preprocess_image = lambda image: torch.ones(1, 3, 10, 12)
        pipe.vae = types.SimpleNamespace(encode=lambda *a, **kw: pipe.z, decode=lambda x, **kw: x)
        image = types.SimpleNamespace(resize=lambda size: object())
        def unit_runner(unit, p, shared, positive, negative):
            shared.update(self.fuse_upstream(unit, p, image, p.initial.clone(),
                                            shared["height"], shared["width"], False, (10, 12), (5, 6)))
            positive["context"] = negative["context"] = torch.ones(1, 2, 3)
            return shared, positive, negative
        pipe.unit_runner = unit_runner
        before = pipe.z.clone()
        original_model, original_step = pipe.model_fn, pipe.scheduler.step
        with alg.teacher_alg(pipe, call_kwargs=CALL, checkpoint="NONE") as stats:
            result = self.call_upstream(pipe, **CALL, input_image=image, output_type="floatpoint")
        self.assertEqual(initialized, [(50, dict(denoising_strength=1.0, shift=5.0))])
        self.assertEqual(stats["completed_steps"], 50)
        self.assertEqual(stats["model_forwards"], 110)
        self.assertTrue(alg._same_bits(pipe.z, before))
        self.assertTrue(alg._same_bits(result[:, :, :1], before.to(result)))
        self.assertIs(pipe.model_fn, original_model)
        self.assertEqual(pipe.scheduler.step, original_step)
        self.assertIs(pipe.scheduler.set_timesteps, set_timesteps)

    def test_mixed_dtype_raw_mutation_hidden_by_cast_and_effective_first_frame_corruption_rejected(self):
        for raw_mutation in (True, False):
            pipe = self.fixture()
            original_model, original_step = pipe.model_fn, pipe.scheduler.step
            x = pipe.initial.clone()
            with self.subTest(raw_mutation=raw_mutation), self.assertRaises(ValueError):
                with alg.teacher_alg(pipe, call_kwargs=CALL, checkpoint="NONE"):
                    pipe.model_fn(**pipe.inputs(x, 0, True))
                    if raw_mutation:
                        effective = pipe.z.to(x).clone()
                        pipe.z[0, 0, 0, 1, 1] += 1e-6
                        self.assertTrue(alg._same_bits(pipe.z.to(x), effective))
                    else:
                        x[0, 0, 0, 1, 1] += 1
                    pipe.model_fn(**pipe.inputs(x, 0, False))
            self.assertIs(pipe.model_fn, original_model)
            self.assertEqual(pipe.scheduler.step, original_step)
            self.assertNotIn(id(pipe), alg._ACTIVE_PIPELINES)

    def test_last_scheduler_mutation_of_raw_reference_is_rejected_and_restored(self):
        pipe = self.fixture()
        original_model, step = pipe.model_fn, pipe.scheduler.step
        def mutate(prediction, timestep, sample):
            result = step(prediction, timestep, sample)
            if len(pipe.scheduler.records) == 50:
                pipe.z[0, 0, 0, 1, 1] += 1e-6
            return result
        pipe.scheduler.step = mutate
        with self.assertRaisesRegex(ValueError, "changed during scheduler.step"):
            with alg.teacher_alg(pipe, call_kwargs=CALL, checkpoint="NONE"):
                pipe.run()
        self.assertIs(pipe.model_fn, original_model)
        self.assertIs(pipe.scheduler.step, mutate)
        self.assertNotIn(id(pipe), alg._ACTIVE_PIPELINES)


if __name__ == "__main__":
    unittest.main()
