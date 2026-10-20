import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_teacher_training_smoke as runner


class CoordinatorTests(unittest.TestCase):
    def test_budget_hard_limits(self):
        self.assertEqual(runner.budget('0', 4, 7200, 2, 30)['reserved_allocated_gpu_hours'], 2)
        for values in [('1', 4, 7200, 2, 30), ('0', 5, 7200, 2, 30),
                       ('0', 0, 7200, 2, 30), ('0', 4, 7201, 2, 30),
                       ('0', 4, 7200, 2.1, 30), ('0', 4, 7200, 1, 30),
                       ('0', 4, float('nan'), 2, 30), ('0', 4, 60, 2, 40)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                runner.budget(*values)

    def test_help_and_invalid_budget_do_not_import_torch(self):
        code = "import run_teacher_training_smoke as r, sys\ntry: rc=r.main(sys.argv[1:])\nexcept SystemExit as e: rc=e.code\nassert 'torch' not in sys.modules\nsys.exit(rc)"
        env = dict(os.environ, PYTHONPATH=str(SCRIPTS), CUDA_VISIBLE_DEVICES='')
        for arguments, expected in [(['--help'], 0), (['--out-dir', '/tmp/never-created-smoke', '--max-steps', '5'], 1),
                                    (['--out-dir', '/tmp/never-created-smoke', '--timeout-seconds', 'nan'], 1)]:
            p = subprocess.run([sys.executable, '-c', code, *arguments], env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(p.returncode, expected, p.stderr)

    def test_paths_and_noreplace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(runner.safe_out(root / 'new', root / 'm1'), root / 'new')
            (root / 'exists').mkdir()
            (root / 'link').symlink_to(root, target_is_directory=True)
            for path in (root / 'exists', root / 'link', root / 'link/child', root / 'm1/child'):
                with self.assertRaises(ValueError):
                    runner.safe_out(path, root / 'm1')
            runner.m1.atomic_json(root / 'file.json', {'x': 1})
            self.assertEqual((root / 'file.json').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(ValueError):
                runner.m1.atomic_json(root / 'file.json', {'x': 2})

    def test_busy_gpu_never_killed(self):
        result = types.SimpleNamespace(stdout='0, 100\n1, 0\n')
        with patch.object(runner.subprocess, 'run', return_value=result), self.assertRaises(ValueError):
            runner.assert_idle(time.perf_counter() + 10)

    def test_actual_cpu_child_timeout_and_owned_cleanup(self):
        runner.m1.enable_subreaper()
        with tempfile.TemporaryDirectory() as tmp:
            log = (Path(tmp) / 'cpu.log').open('x')
            child_code = "import subprocess,sys,time,signal; subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); time.sleep(30)"
            process = subprocess.Popen([sys.executable, '-c', child_code], start_new_session=True, stdout=log, stderr=log)
            workers = [{'role': 'cpu-test', 'process': process, 'log': log}]
            with self.assertRaises(TimeoutError):
                runner.m1.wait_workers(workers, time.perf_counter() + 0.25)
            cleanup = runner.m1.stop_workers(workers, time.perf_counter() + 2, 0.5)
            self.assertTrue(cleanup[0]['owned_group_gone'])
            self.assertIsNotNone(process.returncode)

    def test_partial_progress_retained_and_all_wall_charged(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            args = types.SimpleNamespace(out_dir=out, m1_run=Path(tmp) / 'm1', models_dir=Path(tmp))
            allocation = runner.budget('0', 4, 5, 2, 1)
            def freeze(run, manifest, allocation, deadline):
                (run / 'private').mkdir()
                runner.m1.atomic_json(run / 'training_manifest.json', {})
                runner.m1.atomic_json(run / 'private/step_1.json', {'step': 1, 'loss': 0.4})
                return {}
            with patch.object(runner, 'preflight', return_value={}), patch.object(runner, 'assert_idle'), \
                    patch.object(runner, 'freeze_run', side_effect=freeze), \
                    patch.object(runner, 'spawn_worker', side_effect=RuntimeError('CPU simulated failure')):
                result = runner.execute(args, allocation)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['partial_step_reports'], [{'step': 1, 'loss': 0.4}])
            self.assertEqual(result['ledger']['allocated_cards'], 1)
            self.assertGreater(result['ledger']['wall_seconds'], result['ledger']['measured_wall_seconds_before_publication'])
            self.assertTrue((out / 'training_summary.json').exists())


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)

    def tiny_model(self):
        torch = self.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(3, 3, bias=False)
                self.q.weight.requires_grad_(False)
                self.q.lora_A = torch.nn.ModuleDict({'default': torch.nn.Linear(3, 2, bias=False)})
                self.q.lora_B = torch.nn.ModuleDict({'default': torch.nn.Linear(2, 3, bias=False)})
                torch.nn.init.zeros_(self.q.lora_B['default'].weight)
            def forward(self, x):
                return torch.nn.functional.linear(x, self.q.weight) + self.q.lora_B['default'](self.q.lora_A['default'](x))
        return Tiny()

    def trained(self):
        torch = self.torch
        torch.manual_seed(runner.SEED)
        random.seed(runner.SEED)
        model = self.tiny_model()
        optimizer = torch.optim.AdamW(list(runner.trainable_adapters(model).values()), lr=1e-5)
        frozen = runner.frozen_state_hash(model)
        loss = model(torch.ones(2, 3)).square().mean()
        stats = runner.optimizer_step(model, optimizer, loss)
        self.assertGreater(stats['nonzero_gradient_tensors'], 0)
        self.assertEqual(runner.frozen_state_hash(model), frozen)
        return model, optimizer

    def test_trainable_filter_base_unchanged_and_initial_A_zero_gradient(self):
        model, optimizer = self.trained()
        self.assertEqual(len(runner.trainable_adapters(model)), 2)
        self.assertEqual(int(self.torch.count_nonzero(model.q.lora_A['default'].weight.grad)), 0)
        model.q.weight.requires_grad_(True)
        with self.assertRaises(ValueError):
            runner.trainable_adapters(model)

    def test_nonfinite_or_zero_gradient_rejected(self):
        torch = self.torch
        model = self.tiny_model()
        opt = torch.optim.AdamW(list(runner.trainable_adapters(model).values()), lr=1e-5)
        with self.assertRaises(ValueError):
            runner.optimizer_step(model, opt, model(torch.ones(2, 3)).sum() * float('nan'))
        opt.zero_grad(set_to_none=True)
        with self.assertRaises(ValueError):
            runner.optimizer_step(model, opt, model(torch.ones(2, 3)).sum() * 0)

    def test_adapter_optimizer_rng_save_restore_and_prediction(self):
        torch = self.torch
        model, opt = self.trained()
        state = runner.checkpoint_state(model, opt, 1)
        runner.validate_checkpoint(state, model, opt, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter.pt'
            record = runner.save_checkpoint(path, state)
            before = model(torch.randn(2, 3)).detach().clone()
            before_python = random.random()
            with torch.no_grad():
                for p in runner.trainable_adapters(model).values():
                    p.add_(0.1)
                for item in opt.state.values():
                    item['exp_avg'].add_(0.2)
            runner.load_checkpoint(path, record, model, opt, 1)
            self.assertEqual(runner.state_hash(runner.checkpoint_state(model, opt, 1)), record['state_sha256'])
            torch.testing.assert_close(before, model(torch.randn(2, 3)), rtol=0, atol=0)
            self.assertEqual(random.random(), before_python)
            self.assertNotIn('q.weight', state['adapters'])
            self.assertLess(record['bytes'], 50000)
            with self.assertRaises(ValueError):
                runner.save_checkpoint(path, state)

    def test_actual_roundtrip_perturbs_optimizer_rng_without_update_cpu(self):
        torch = self.torch
        model, optimizer = self.trained()
        pipe = types.SimpleNamespace(dit=model)
        get_rng, set_rng = runner.rng_state, runner.restore_rng
        saved = runner.state_hash(runner.checkpoint_state(model, optimizer, 1))
        def forward(pipe, common, loss_fn, capture):
            inputs = torch.randn(2, 3)
            prediction = pipe.dit(inputs).detach()
            return prediction.square().mean(), {'prediction': prediction,
                'input_sha256': {'latents': runner.m1.tensor_hash(inputs)}}
        progress = {'verification_forward_attempts': 0, 'verification_forwards': 0}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(runner, 'rng_state', side_effect=lambda cuda=False: get_rng(False)), \
                patch.object(runner, 'restore_rng', side_effect=lambda state, cuda=False: set_rng(state, False)), \
                patch.object(torch.cuda, 'manual_seed_all'), \
                patch.object(runner, 'native_forward', side_effect=forward), \
                patch.object(optimizer, 'step', wraps=optimizer.step) as update:
            result = runner.verify_roundtrip(pipe, optimizer, {}, None, Path(tmp) / 'adapters.pt',
                1, time.perf_counter() + 10, progress)
            self.assertTrue(result['deliberate_optimizer_moment_perturbation'])
            self.assertTrue(result['deliberate_independent_rng_reseed'])
            self.assertEqual(result['prediction_max_abs_diff'], 0)
            self.assertEqual(result['loss_abs_diff'], 0)
            update.assert_not_called()
        self.assertEqual(progress['verification_forwards'], 2)
        self.assertEqual(runner.state_hash(runner.checkpoint_state(model, optimizer, 1)), saved)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bad_checkpoint_keys_shape_finite_optimizer_and_hash(self):
        torch = self.torch
        model, opt = self.trained()
        good = runner.checkpoint_state(model, opt, 1)
        variants = []
        bad = copy.deepcopy(good); del bad['adapters'][bad['adapter_names'][0]]; variants.append(bad)
        bad = copy.deepcopy(good); bad['adapters'][bad['adapter_names'][0]] = torch.zeros(1); variants.append(bad)
        bad = copy.deepcopy(good); bad['adapters'][bad['adapter_names'][0]].fill_(float('nan')); variants.append(bad)
        bad = copy.deepcopy(good); bad['optimizer']['param_groups'][0]['lr'] = 2e-5; variants.append(bad)
        bad = copy.deepcopy(good); bad['optimizer']['state'].pop(0); variants.append(bad)
        bad = copy.deepcopy(good); bad['steps'] = 2; variants.append(bad)
        for value in variants:
            with self.assertRaises(ValueError):
                runner.validate_checkpoint(value, model, opt, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pt'
            record = runner.save_checkpoint(path, good)
            with self.assertRaises(ValueError):
                runner.load_checkpoint(path, dict(record, state_sha256='0' * 64), model, opt, 1)
            with path.open('ab') as f:
                f.write(b'changed')
            with self.assertRaises(ValueError):
                runner.load_checkpoint(path, record, model, opt, 1)

    def test_cache_load_weights_only_cpu_hash_and_label_truth(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'private').mkdir()
            tensors = {'x0': torch.zeros(1, 2), 'positive': torch.ones(1, 2), 'first_frame_latents': torch.zeros(1, 1)}
            record = runner.m1.save_tensors(root / 'private/cache.pt', tensors)
            loaded = runner.checked_tensors(record, root, None)
            inputs = runner.cached_inputs(loaded, 'cpu')
            self.assertTrue(torch.equal(inputs['context'], loaded['positive']))
            self.assertEqual(inputs['cfg_scale'], 1)
            self.assertTrue(inputs['use_gradient_checkpointing_offload'])
            self.assertIn('corrected', runner.LABEL_SCOPE)
            self.assertIn('not_original', runner.LABEL_SCOPE)
            with self.assertRaises(ValueError):
                runner.checked_tensors(dict(record, sha256='0' * 64), root, None)
            with self.assertRaises(ValueError):
                runner.validate_common(loaded)

    def test_real_tiny_peft_injection_if_installed(self):
        if importlib.util.find_spec('peft') is None:
            self.skipTest('PEFT not installed; native-math LoRA tested above')
        from peft import LoraConfig, inject_adapter_in_model
        torch = self.torch
        model = torch.nn.Module()
        model.q = torch.nn.Linear(3, 3)
        model.requires_grad_(False)
        model = inject_adapter_in_model(LoraConfig(r=32, lora_alpha=32, target_modules=['q']), model)
        self.assertEqual(len(runner.trainable_adapters(model)), 2)
        frozen = runner.frozen_state_hash(model)
        opt = torch.optim.AdamW(list(runner.trainable_adapters(model).values()), lr=1e-5)
        runner.optimizer_step(model, opt, model.q(torch.ones(2, 3)).square().mean())
        self.assertEqual(runner.frozen_state_hash(model), frozen)

    def test_native_switch_injector_fp32_adapters_bf16_base_cpu(self):
        if importlib.util.find_spec('peft') is None:
            self.skipTest('PEFT unavailable')
        from peft import LoraConfig, inject_adapter_in_model
        torch = self.torch
        framework = SCRIPTS.parent / 'DiffSynth-Studio/diffsynth/diffusion'
        tree = ast.parse((framework / 'training_module.py').read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DiffusionTrainingModule')
        methods = {'switch_pipe_to_training_mode', 'add_lora_to_model', 'parse_lora_target_modules'}
        original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        namespace = {'torch': torch, 'LoraConfig': LoraConfig, 'inject_adapter_in_model': inject_adapter_in_model}
        exec(compile(ast.Module(body=[original], type_ignores=[]), str(framework / 'training_module.py'), 'exec'), namespace)
        class Dit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for name in ('q', 'k', 'v', 'o'):
                    setattr(self, name, torch.nn.Linear(3, 3))
                self.ffn = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.GELU(), torch.nn.Linear(3, 3))
            def forward(self, x):
                return self.q(x) + self.k(x) + self.v(x) + self.o(x) + self.ffn(x)
        class Pipe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = Dit().to(torch.bfloat16)
                self.torch_dtype = torch.bfloat16
                self.scheduler = runner.source_module(framework / 'flow_match.py', '_switch_scheduler').FlowMatchScheduler('Wan')
            def freeze_except(self, names):
                self.assert_frozen_names = names
                self.eval()
                self.requires_grad_(False)
        pipe = Pipe()
        training = namespace['DiffusionTrainingModule']()
        training.switch_pipe_to_training_mode(pipe, lora_base_model='dit', lora_target_modules=runner.TARGETS, lora_rank=32)
        self.assertEqual(pipe.assert_frozen_names, [])
        self.assertEqual(len(pipe.scheduler.timesteps), 1000)
        self.assertTrue(pipe.scheduler.training)
        for p in pipe.dit.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.assertEqual(len(runner.trainable_adapters(pipe.dit)), 12)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in pipe.dit.parameters() if not p.requires_grad))
        frozen = runner.frozen_state_hash(pipe.dit)
        opt = torch.optim.AdamW(list(runner.trainable_adapters(pipe.dit).values()), lr=1e-5)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            loss = pipe.dit(torch.ones(2, 3)).float().square().mean()
        runner.optimizer_step(pipe.dit, opt, loss)
        self.assertEqual(runner.frozen_state_hash(pipe.dit), frozen)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_frozen_native_loss_math_cpu(self):
        torch = self.torch
        framework = SCRIPTS.parent / 'DiffSynth-Studio/diffsynth/diffusion'
        scheduler = runner.source_module(framework / 'flow_match.py', '_test_scheduler').FlowMatchScheduler('Wan')
        scheduler.set_timesteps(1000, training=True)
        tree = ast.parse((framework / 'loss.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'FlowMatchSFTLoss')
        fn.args.args[0].annotation = None
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(framework / 'loss.py'), 'exec'), namespace)
        observed = {}
        def predict(**inputs):
            observed.update(inputs)
            return inputs['latents'] * 0
        pipe = types.SimpleNamespace(scheduler=scheduler, torch_dtype=torch.float32, device='cpu', in_iteration_models=(), model_fn=predict)
        x0 = torch.ones(1, 2, 3, 2, 2)
        first = torch.full((1, 2, 1, 2, 2), 7.0)
        saved = torch.get_rng_state()
        timestep_id = torch.randint(0, 1000, (1,))
        timestep = scheduler.timesteps[timestep_id]
        noise = torch.randn_like(x0)
        target = noise - x0
        expected = target[:, :, 1:].square().mean() * scheduler.training_weight(timestep)
        torch.set_rng_state(saved)
        loss = namespace['FlowMatchSFTLoss'](pipe, input_latents=x0, first_frame_latents=first)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(observed['latents'][:, :, :1], first))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
