"""CPU-only M1 tests: never build or run a GPU pipeline."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_teacher_mechanism_probe.py'
spec = importlib.util.spec_from_file_location('teacher_m1', SCRIPT)
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def json_put(path, value):
    return put(path, json.dumps(value, sort_keys=True))


def fixture(root):
    p1, models, out = root / 'p1', root / 'models', root / 'new_run'
    p1.mkdir()
    csv = put(root / 'original.csv', 'video,prompt\n' + ''.join(
        f'{root / (cid + ".mp4")},original {cid}\n' for cid in m1.CASE_ORDER))
    csv_sha = m1.sha256(csv)
    preserved = put(p1 / 'original_csv' / (csv_sha + '.csv'), csv.read_text())
    code = {'gen_jobs.py': put(p1 / 'code/gen_jobs.py', "NEG = ('original nonempty ' 'NEG')\n"),
            'run_teacher_probe.py': put(p1 / 'code/run_teacher_probe.py', '# frozen P1\n')}
    framework = {'diffsynth/__init__.py': "ORIGIN = 'frozen-package'\n",
                 'diffsynth/pipelines/wan_video.py': '# frozen wan\n',
                 'diffsynth/diffusion/flow_match.py': '# frozen scheduler\n',
                 'diffsynth/utils/__init__.py': ''}
    framework_paths = {rel: put(p1 / 'framework' / rel, text) for rel, text in framework.items()}
    a07_original = root / 'shared_merge.safetensors'
    jobs, outputs = [], []
    for i, cid in enumerate(m1.CASE_ORDER):
        gt = put(root / (cid + '.mp4'), 'GT input ' + cid)
        prompt = 'corrected label ' + cid
        review = {'csv_path': str(csv), 'preserved_csv': str(preserved), 'csv_sha256': csv_sha,
                  'csv_row': i, 'video_column': 'video', 'original_video': str(gt),
                  'prompt_column': 'prompt', 'original_prompt_sha256': m1.text_hash('original ' + cid),
                  'prompt_sha256': m1.text_hash(prompt)}
        for model in m1.MODELS:
            for scale in (5, 3):
                ident = model + '_cfg' + str(scale) + '_' + cid
                video = put(p1 / 'videos' / model / str(scale) / (cid + '.mp4'), 'P1 output ' + ident)
                jobs.append({'id': ident, 'case_id': cid, 'model': model, 'cfg': scale,
                    'ckpt': 'NONE' if model == 'base' else str(a07_original),
                    'ckpt_sha256': 'NONE' if model == 'base' else 'TEST_A07',
                    'canonical_source': 'source:' + cid, 'image': str(gt), 'image_sha256': m1.sha256(gt),
                    'prompt': prompt, 'prompt_review': review, 'seed': 0, 'steps': 50, 'shift': 5,
                    'width': 1280, 'height': 704, 'num_frames': 121,
                    'cfg_zero_star': False, 'cfg_zero_steps': 0, 'out': str(video)})
                outputs.append({'id': ident, 'out': str(video), 'sha256': m1.sha256(video),
                                'decode': {'frames': 121, 'width': 1280, 'height': 704}})
    cache = {}
    for name in ('base.safetensors', 't5.safetensors', 'vae.safetensors', 'spiece.model'):
        path = put(models / name, 'LOCAL ' + name)
        cache[name] = m1.sha256(path)
    a07 = put(models / 'wm_checkpoints' / (m1.text_hash(str(a07_original.resolve())) + '.safetensors'), 'A07 local')
    a07_sha = m1.sha256(a07)
    for j in jobs:
        if j['model'] == 'm_full1_a07':
            j['ckpt_sha256'] = a07_sha
    frozen = put(p1 / 'jobs.frozen.jsonl', ''.join(json.dumps(j) + '\n' for j in jobs))
    original = put(p1 / 'jobs.input.jsonl', frozen.read_text())
    manifest = {'protocol': 'internal-dev-teacher-cfg-v1', 'jobs': jobs,
        'jobs_frozen_sha256': m1.sha256(frozen), 'jobs_input_sha256': m1.sha256(original),
        'checkpoint_sha256': {'m_full1_a07': a07_sha},
        'source_sha256': {rel: m1.sha256(path) for rel, path in code.items()},
        'framework_source_sha256': {rel: m1.sha256(path) for rel, path in framework_paths.items()},
        'original_csv_sha256': {str(csv): csv_sha}}
    json_put(p1 / 'probe_manifest.json', manifest)
    json_put(p1 / 'probe_summary.json', {'status': 'completed', 'completed_outputs': 24, 'failures': [],
        'manifest_sha256': m1.sha256(frozen), 'outputs': outputs})
    return p1, models, out, cache, a07_sha


class CliAndPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.p1, self.models, self.out, self.pins, self.a07_sha = fixture(self.root)
        self.patch_pins = mock.patch.object(m1, 'CACHE_PINS', self.pins)
        self.patch_a07 = mock.patch.object(m1, 'A07_SHA256', self.a07_sha)
        self.patch_pins.start()
        self.patch_a07.start()

    def tearDown(self):
        self.patch_pins.stop()
        self.patch_a07.stop()
        self.tmp.cleanup()

    def preflight(self):
        return m1.preflight(self.p1, self.out, self.models)

    def test_actual_cli_help_bad_budget_lazy_torch(self):
        wrapper = ("import builtins,runpy; old=builtins.__import__; "
                   "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(AssertionError('torch imported')) "
                   "if name.split('.')[0]=='torch' else old(name,*a,**k); "
                   f"runpy.run_path({str(SCRIPT)!r},run_name='__main__')")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([sys.executable, '-c', wrapper, '--out-dir', str(self.out), '--help'],
                           capture_output=True, text=True, env=env, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('--preflight-only', r.stdout)
        r = subprocess.run([sys.executable, '-c', wrapper, '--out-dir', str(self.out),
                            '--timeout-seconds', '3601', '--preflight-only'],
                           capture_output=True, text=True, env=env, timeout=20)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('hard cap', r.stderr)
        self.assertFalse(self.out.exists())

    def test_actual_cli_preflight_fixture_no_torch_no_cuda_no_directory(self):
        wrapper = (
            "import builtins,runpy;old=builtins.__import__;"
            "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(AssertionError('torch imported')) "
            "if name.split('.')[0]=='torch' else old(name,*a,**k);"
            f"ns=runpy.run_path({str(SCRIPT)!r},run_name='m1_fixture');"
            f"ns['main'].__globals__['CACHE_PINS']={self.pins!r};"
            f"ns['main'].__globals__['A07_SHA256']={self.a07_sha!r};"
            "raise SystemExit(ns['main']())")
        r = subprocess.run([sys.executable, '-c', wrapper, '--p1-run', str(self.p1),
            '--models-dir', str(self.models), '--out-dir', str(self.out), '--preflight-only'],
            capture_output=True, text=True, timeout=20,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertFalse(result['torch_imported'])
        self.assertFalse(result['cuda_queried'])
        self.assertFalse(self.out.exists())
        self.assertEqual(result['predictions'], 72)

    def test_budget_allocated_two_cards_cleanup_inside_wall(self):
        b = m1.budget('0,1', 3600, 2, 15)
        self.assertEqual(b['reserved_allocated_gpu_hours'], 2)
        for args in [('0', 3600, 2, 15), ('0,1', 3601, 2, 15), ('0,1', 3600, 1, 15),
                     ('0,1', float('nan'), 2, 15), ('0,1', 100, 2, 60), ('0,1', 3600, 3, 15)]:
            with self.assertRaises(ValueError):
                m1.budget(*args)

    def test_original_inputs_labels_neg_preflight_readonly(self):
        data = self.preflight()
        self.assertEqual([c['case_id'] for c in data['cases']], list(m1.CASE_ORDER))
        self.assertEqual(data['negative_prompt'], 'original nonempty NEG')
        self.assertEqual(data['expected_predictions'], 72)
        self.assertEqual(len([k for k in data['p1_files'] if k.startswith('output:')]), 24)
        self.assertFalse(self.out.exists())

    def test_actual_p1_review_without_csv_row_binds_unique_original_full_path(self):
        manifest = json.loads((self.p1 / 'probe_manifest.json').read_text())
        for j in manifest['jobs']:
            review = j['prompt_review']
            for key in ('csv_row', 'video_column', 'original_video', 'prompt_column', 'original_prompt_sha256'):
                review.pop(key, None)
        frozen = put(self.p1 / 'jobs.frozen.jsonl', ''.join(json.dumps(j) + '\n' for j in manifest['jobs']))
        manifest['jobs_frozen_sha256'] = m1.sha256(frozen)
        json_put(self.p1 / 'probe_manifest.json', manifest)
        summary = json.loads((self.p1 / 'probe_summary.json').read_text())
        summary['manifest_sha256'] = m1.sha256(frozen)
        json_put(self.p1 / 'probe_summary.json', summary)
        result = self.preflight()
        self.assertEqual([c['original_csv_binding']['row_index'] for c in result['cases']], list(range(6)))
        self.assertFalse(self.out.exists())

    def test_new_directory_and_bad_source_hash(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'new directory'):
            self.preflight()
        self.out.rmdir()
        put(self.p1 / 'code/gen_jobs.py', 'NEG = "changed"\n')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.preflight()
        self.assertFalse(self.out.exists())

    def test_bad_gt_output_csv_hashes_fail_before_model_load(self):
        data = self.preflight()
        for prefix in ('input:', 'output:', 'csv:original:'):
            key = next(k for k in data['p1_files'] if k.startswith(prefix))
            p = Path(data['p1_files'][key]['path'])
            old = p.read_bytes()
            p.write_bytes(old + b'corruption')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                self.preflight()
            p.write_bytes(old)

    def test_shared_cache_symlink_and_bad_local_hash_rejected(self):
        rel = next(iter(self.pins))
        local = self.models / rel
        with local.open('ab') as f:
            f.write(b'bad hash')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.preflight()
        shared = put(self.root / 'shared_weight', 'shared')
        local.unlink()
        local.symlink_to(shared)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.preflight()

    def test_case_order_seed_caption_coverage_cannot_drift(self):
        manifest = json.loads((self.p1 / 'probe_manifest.json').read_text())
        for mutate in ('seed', 'prompt', 'case_order', 'coverage'):
            jobs = copy.deepcopy(manifest['jobs'])
            if mutate == 'seed':
                jobs[0]['seed'] = 1
            elif mutate == 'prompt':
                jobs[0]['prompt'] += ' changed'
            elif mutate == 'case_order':
                jobs = jobs[4:8] + jobs[:4] + jobs[8:]
            else:
                jobs[-1] = jobs[-2]
            with self.assertRaises(ValueError):
                m1.validate_cases(jobs, dict(manifest, jobs=jobs))

    def test_frozen_framework_preferred_over_live_pythonpath(self):
        data = self.preflight()
        frozen = m1.freeze_run(self.out, data, m1.budget('0,1', 3600, 2, 15), time.perf_counter() + 30)
        m1.verify_snapshot(self.out, frozen, time.perf_counter() + 30)
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'DiffSynth-Studio'), PYTHONDONTWRITEBYTECODE='1')
        code = (f"import sys;sys.path.insert(0,{str(self.out / 'framework')!r});"
                "import diffsynth;print(diffsynth.ORIGIN);print(diffsynth.__file__)")
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=15)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('frozen-package', r.stdout)
        self.assertIn(str(self.out / 'framework'), r.stdout)
        put(self.out / 'framework/diffsynth/pipelines/wan_video.py', '# altered')
        with self.assertRaisesRegex(ValueError, 'runtime changed'):
            m1.verify_snapshot(self.out, frozen, time.perf_counter() + 30)
        with self.assertRaisesRegex(ValueError, 'No overwrite'):
            m1.atomic_json(self.out / 'mechanism_manifest.json', {})


    def test_final_verification_rehashes_source_framework_gt_and_local_cache(self):
        data = self.preflight()
        frozen = m1.freeze_run(self.out, data, m1.budget('0,1', 3600, 2, 15), time.perf_counter() + 30)
        verified = m1.verify_final_sources(self.out, frozen, time.perf_counter() + 30)
        self.assertEqual(verified['status'], 'passed')
        self.assertTrue(verified['sha256_reverified'])
        for group, key in (('p1_source', 'gen_jobs.py'),
                           ('framework', 'diffsynth/pipelines/wan_video.py'),
                           ('p1_files', 'input:' + m1.CASE_ORDER[0]),
                           ('model_cache', next(iter(self.pins)))):
            path = Path(frozen[group][key]['path'])
            old = path.read_bytes()
            stat = path.stat()
            changed = bytearray(old)
            changed[0] ^= 1
            path.write_bytes(changed)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
                m1.verify_final_sources(self.out, frozen, time.perf_counter() + 30)
            path.write_bytes(old)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


class CpuMathAndCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(2)
        scheduler_path = ROOT / 'DiffSynth-Studio/diffsynth/diffusion/flow_match.py'
        spec = importlib.util.spec_from_file_location('native_m1_flow_match', scheduler_path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        cls.Scheduler = native.FlowMatchScheduler

    def test_native_scheduler_bf16_times_and_shared_seed0_noise(self):
        t = self.torch
        scheduler = self.Scheduler('Wan')
        scheduler.set_timesteps(50, shift=5, training=False)
        x0 = t.linspace(-1, 1, 24).reshape(1, 2, 3, 2, 2).to(t.bfloat16)
        first = t.full_like(x0[:, :, :1], 7)
        a, states = m1.make_shared_states(scheduler, x0, first)
        b, again = m1.make_shared_states(scheduler, x0, first)
        self.assertEqual(m1.tensor_hash(a['noise_fp32']), m1.tensor_hash(b['noise_fp32']))
        self.assertEqual(m1.tensor_hash(a['noise']), m1.tensor_hash(a['noise_fp32'].to(t.bfloat16)))
        self.assertEqual(m1.tensor_hash(a['target']), m1.tensor_hash(a['noise'] - x0))
        self.assertEqual([s[0]['index'] for s in states], [5, 25, 45])
        self.assertEqual([s[0]['bf16_timestep'] for s in states], [980.0, 832.0, 358.0])
        for (meta, state), (_, repeat) in zip(states, again):
            self.assertEqual(m1.tensor_hash(state['latents']), m1.tensor_hash(repeat['latents']))
            self.assertEqual(m1.tensor_hash(state['latents'][:, :, :1]), m1.tensor_hash(first))
            selected = int(t.argmin((scheduler.timesteps - state['timestep']).abs()))
            sigma = scheduler.sigmas[selected]
            expected = (1 - sigma) * x0 + sigma * a['noise']
            expected[:, :, :1] = first
            self.assertTrue(t.equal(state['latents'], expected))
            self.assertEqual(meta['add_noise_sigma'], float(sigma))
            self.assertNotEqual(meta['raw_sigma'], meta['index'] / 50)
        self.assertFalse(t.cuda.is_initialized())

    def test_branch_clones_cannot_mutate_shared_state(self):
        t = self.torch
        state = {'latents': t.ones(1, 1, 3, 1, 1, dtype=t.bfloat16), 'timestep': t.tensor([980.], dtype=t.bfloat16)}
        common = {'first_frame_latents': t.ones(1, 1, 1, 1, 1, dtype=t.bfloat16)}
        a, b = m1.cloned_inputs(state, common, 'cpu'), m1.cloned_inputs(state, common, 'cpu')
        a['latents'].zero_()
        a['first_frame_latents'].zero_()
        self.assertTrue(t.equal(b['latents'], state['latents']))
        self.assertFalse(t.equal(a['latents'], state['latents']))
        self.assertFalse(b['cfg_merge'])
        self.assertIsNone(b['tea_cache'])

    def test_zero_full_production_reduction_native_equality(self):
        t = self.torch
        p = t.tensor([100., 2., 3.]).reshape(1, 1, 3, 1, 1)
        n = t.tensor([10., 1., 1.]).reshape_as(p)
        z, k = m1.zero_guidance(p, n, 5)
        expected = (p * n).sum() / (n.square().sum() + 1e-8)
        future_only = (p[:, :, 1:] * n[:, :, 1:]).sum() / n[:, :, 1:].square().sum()
        self.assertAlmostEqual(float(k), float(expected), places=6)
        self.assertNotAlmostEqual(float(k), float(future_only), places=2)
        tree = ast.parse((ROOT / 'DiffSynth-Studio/diffsynth/pipelines/wan_video.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WanVideoPipeline')
        fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'zero_star_guidance'))
        fn.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        namespace = {'torch': t}
        exec(compile(module, '<native production Zero>', 'exec'), namespace)
        actual = namespace['zero_star_guidance'](p.to(t.bfloat16), n.to(t.bfloat16), 5)
        bf16, _ = m1.zero_guidance(p.to(t.bfloat16), n.to(t.bfloat16), 5)
        self.assertTrue(t.equal(actual, bf16.to(t.bfloat16)))
        self.assertEqual(m1.future_stats(z)['latent_frame_indices'], [1, 2])
        self.assertFalse(t.cuda.is_initialized())

    def test_zero_signed_k_and_zero_negative_no_fabricated_angle(self):
        t = self.torch
        p = t.tensor([1., 2., 3.]).reshape(1, 1, 3, 1, 1)
        n = -p
        projected, k = m1.zero_guidance(p, n, 5)
        self.assertLess(float(k), 0)
        self.assertTrue(t.allclose(projected, p))
        empty, k0 = m1.zero_guidance(p, t.zeros_like(n), 5)
        self.assertEqual(float(k0), 0)
        self.assertTrue(t.equal(empty, 5 * p))
        self.assertIsNone(m1.angle(n * 0, p)['degrees'])

    def test_cross_formulas_cross_term_mse_zero_norm_na(self):
        t = self.torch
        shape = (1, 1, 4, 1, 1)
        bp = t.tensor([100., 1., 2., 3.]).reshape(shape)
        bn = t.tensor([-100., -1., 2., -3.]).reshape(shape)
        ap, an = bp + .25, bn - .5
        result = m1.analyze_pair(bp, bn, ap, an, t.zeros(shape), -.02, t.ones(shape))
        self.assertAlmostEqual(result['models']['base']['positive_target_raw_mse'], 14 / 3, places=5)
        for scale in (3, 5):
            cross = result['cross'][str(scale)]
            for key in ('AB_minus_BB_identity', 'BA_minus_BB_identity', 'AA_minus_BB_identity',
                        'delta_diff_identity', 'mean_square_identity'):
                self.assertTrue(cross[key]['passes'])
            self.assertNotEqual(cross['cross_term_mean_square'], 0)
            self.assertAlmostEqual(cross['actual_total_mean_square'], cross['decomposed_total_mean_square'], places=5)
            guided = result['models']['base']['cfg'][str(scale)]
            self.assertAlmostEqual(guided['one_step_future_update_rms'], .02 * guided['guided']['future_rms'], places=7)
            self.assertAlmostEqual(guided['one_step_update_relative_x_t']['ratio'],
                                   guided['one_step_future_update_rms'], places=7)
            self.assertAlmostEqual(cross['delta_guided_relative_base']['ratio'],
                                   cross['delta_guided']['future_rms'] / cross['BB']['future_rms'], places=7)
        ratio = m1.relative_rms(bp, t.zeros(shape))
        self.assertIsNone(ratio['ratio'])
        self.assertEqual(ratio['reason'], 'zero_denominator_norm')
        self.assertEqual(m1.angle(t.zeros(3), t.ones(3)), {'cosine': None, 'degrees': None, 'reason': 'zero_norm'})
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            m1.future_stats(t.full(shape, float('nan')))
        json.dumps(result, allow_nan=False)

    def test_tensor_finite_private_roundtrip_bad_hash(self):
        t = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'
            value = t.arange(4, dtype=t.float32).reshape(1, 1, 4, 1, 1).to(t.bfloat16)
            record = m1.save_tensors(path, {'latents': value})
            self.assertTrue(t.equal(m1.load_tensors(record)['latents'], value))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with path.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                m1.load_tensors(record)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            m1.tensor_record(t.tensor([float('inf')]))

    def test_18_inputs_72_predictions_shared_hash_coverage(self):
        states, commons, predictions = [], [], []
        for cid in m1.CASE_ORDER:
            commons.append({'case_id': cid, 'cache': {'tensors': {b: {'sha256': cid + b} for b in m1.BRANCHES}}})
            for i in m1.INDICES:
                states.append({'case_id': cid, 'scheduler': {'index': i}, 'x_t_sha256': cid + str(i),
                               'cache': {'tensors': {'timestep': {'sha256': 't' + str(i)}}}})
                for model in m1.MODELS:
                    for branch in m1.BRANCHES:
                        predictions.append({'case_id': cid, 'index': i, 'model': model, 'branch': branch,
                            'x_t_sha256': cid + str(i), 'timestep_sha256': 't' + str(i),
                            'context_sha256': cid + branch, 'cache': {'tensors': {'prediction': {'finite': True}}}})
        prepared = {'complete': True, 'states': states, 'commons': commons}
        m1.validate_coverage(prepared, predictions)
        for field in ('missing', 'duplicate', 'x_t_sha256', 'context_sha256', 'timestep_sha256', 'finite'):
            broken = copy.deepcopy(predictions)
            if field == 'missing':
                broken.pop()
            elif field == 'duplicate':
                broken[-1] = broken[-2]
            elif field == 'finite':
                broken[0]['cache']['tensors']['prediction']['finite'] = False
            else:
                broken[0][field] += 'changed'
            with self.assertRaises(ValueError):
                m1.validate_coverage(prepared, broken)


    def test_cpu_full_18_state_72_tensor_analysis_roundtrip(self):
        t = self.torch
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'private').mkdir()
            prepared = {'complete': True, 'commons': [], 'states': []}
            reports = {m: {'status': 'completed', 'predictions': []} for m in m1.MODELS}
            scheduler = self.Scheduler('Wan')
            scheduler.set_timesteps(50, shift=5, training=False)
            for case_no, cid in enumerate(m1.CASE_ORDER):
                x0 = t.arange(16).reshape(1, 1, 4, 2, 2).to(t.bfloat16) / 10
                common, states = m1.make_shared_states(scheduler, x0, x0[:, :, :1].clone())
                common['positive'] = t.full((1, 2, 3), case_no + 1., dtype=t.bfloat16)
                common['negative'] = -common['positive']
                rec = m1.save_tensors(run / 'private' / (cid + '_common.pt'), common)
                prepared['commons'].append({'case_id': cid, 'cache': rec})
                for meta, state in states:
                    index = meta['index']
                    rec = m1.save_tensors(run / 'private' / (cid + '_t' + str(index) + '.pt'), state)
                    xhash = rec['tensors']['latents']['sha256']
                    prepared['states'].append({'case_id': cid, 'scheduler': meta, 'cache': rec, 'x_t_sha256': xhash})
                    for model in m1.MODELS:
                        for branch in m1.BRANCHES:
                            value = state['latents'].float() + (1 if branch == 'positive' else -1)
                            if model == 'm_full1_a07':
                                value += .2
                            prec = m1.save_tensors(run / 'private' / (cid + str(index) + model + branch + '.pt'),
                                                  {'prediction': value.to(t.bfloat16)})
                            reports[model]['predictions'].append({'case_id': cid, 'index': index, 'model': model,
                                'branch': branch, 'cache': prec, 'x_t_sha256': xhash,
                                'context_sha256': m1.tensor_hash(common[branch]),
                                'timestep_sha256': m1.tensor_hash(state['timestep'])})
            json_put(run / 'prepared_index.json', prepared)
            for model, report in reports.items():
                json_put(run / ('worker_' + model + '.json'), report)
            result = m1.analyze_run(run, {'protocol': 'teacher-mechanism-m1-v1'}, time.perf_counter() + 30)
            self.assertEqual(result['inputs'], 18)
            self.assertEqual(result['predictions'], 72)
            self.assertEqual(len(result['cases']), 18)
            self.assertTrue(result['zero_k_includes_first_latent'])
            self.assertTrue(result['first_latent_excluded_from_statistics'])
            self.assertTrue((run / 'mechanism_analysis.json').is_file())
            self.assertFalse(list(run.rglob('*.mp4')))
            self.assertFalse(t.cuda.is_initialized())


class OwnedProcessAndFailureTests(unittest.TestCase):
    def test_failure_setup_wait_cpu_analysis_charges_both_cards(self):
        allocation = m1.budget('0,1', 3600, 2, 15)
        ledger = m1.allocated_ledger(10, 1810, allocation, True, [])
        self.assertEqual(ledger['allocated_gpu_hours'], 1)
        self.assertEqual(ledger['allocated_cards'], 2)
        self.assertTrue(ledger['failed'])
        self.assertIn('CPU analysis', ledger['accounting'])
        self.assertFalse(m1.allocated_ledger(10, 3611, allocation, True, [])['within_reservation'])

    def test_preflight_failure_no_directory_and_failure_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not_created'
            args = m1.parser().parse_args(['--out-dir', str(out), '--timeout-seconds', '2',
                                          '--cleanup-grace-seconds', '.2'])
            with mock.patch.object(m1, 'preflight', side_effect=ValueError('bad input SHA')):
                result = m1.execute(args, m1.budget('0,1', 2, 2, .2))
            self.assertEqual(result['status'], 'failed')
            self.assertIn('bad input SHA', result['error'])
            self.assertIn('ValueError: bad input SHA', result['traceback'])
            self.assertIsNone(result['final_source_verification'])
            self.assertGreaterEqual(result['ledger']['allocated_gpu_hours'], 0)
            self.assertFalse(out.exists())

    def test_deadline_interrupts_blocking_preflight_and_charges_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not_created'
            args = m1.parser().parse_args(['--out-dir', str(out), '--timeout-seconds', '.3',
                                          '--cleanup-grace-seconds', '.05'])
            def blocking_preflight(*args, **kwargs):
                time.sleep(5)
                raise AssertionError('Deadline timer did not interrupt blocking setup')
            start = time.perf_counter()
            with mock.patch.object(m1, 'preflight', side_effect=blocking_preflight):
                result = m1.execute(args, m1.budget('0,1', .3, 2, .05))
            self.assertLess(time.perf_counter() - start, 1)
            self.assertEqual(result['status'], 'failed')
            self.assertIn('InterruptedError', result['error'])
            self.assertGreater(result['ledger']['allocated_gpu_hours'], 0)
            self.assertFalse(out.exists())

    def test_owned_child_after_leader_exit_not_unrelated(self):
        m1.enable_subreaper()
        with tempfile.TemporaryDirectory() as tmp:
            log = (Path(tmp) / 'owned.log').open('x')
            leaf_code = 'import time;time.sleep(120)'
            leader_code = ('import subprocess,sys;'
                           f'p=subprocess.Popen([sys.executable,"-c",{leaf_code!r}]);print(p.pid,flush=True)')
            leader = subprocess.Popen([sys.executable, '-c', leader_code], stdout=log, stderr=log, start_new_session=True)
            other = subprocess.Popen([sys.executable, '-c', leaf_code], start_new_session=True)
            try:
                leader.wait(timeout=10)
                self.assertTrue(m1.group_alive(leader.pid))
                cleanup = m1.stop_workers([{'role': 'owned', 'process': leader, 'log': log}], time.perf_counter() + 3, .2)
                self.assertTrue(cleanup[0]['owned_group_gone'])
                self.assertFalse(m1.group_alive(leader.pid))
                self.assertIsNone(other.poll())
            finally:
                if other.poll() is None:
                    os.killpg(other.pid, signal.SIGKILL)
                other.wait(timeout=10)
                if not log.closed:
                    log.close()
                if m1.group_alive(leader.pid):
                    os.killpg(leader.pid, signal.SIGKILL)
                    m1.reap_group(leader.pid)

    def test_timeout_kills_sigterm_resistant_owned_worker_waits(self):
        m1.enable_subreaper()
        with tempfile.TemporaryDirectory() as tmp:
            log = (Path(tmp) / 'worker.log').open('x')
            code = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(120)'
            p = subprocess.Popen([sys.executable, '-c', code], stdout=log, stderr=log, start_new_session=True)
            try:
                end = time.perf_counter() + 5
                while not (Path(tmp) / 'worker.log').read_text().strip():
                    self.assertLess(time.perf_counter(), end)
                    time.sleep(.01)
                info = {'role': 'resistant', 'process': p, 'log': log}
                with self.assertRaises(TimeoutError):
                    m1.wait_workers([info], time.perf_counter() + .05)
                cleanup = m1.stop_workers([info], time.perf_counter() + 2, .2)
                self.assertIsNotNone(p.returncode)
                self.assertTrue(cleanup[0]['owned_group_gone'])
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
                if not log.closed:
                    log.close()

    def test_worker_failure_never_retried_or_completed(self):
        p = mock.Mock()
        p.poll.return_value = 1
        with self.assertRaisesRegex(ValueError, 'no retries'):
            m1.wait_workers([{'process': p}], time.perf_counter() + 1)
        p.wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
