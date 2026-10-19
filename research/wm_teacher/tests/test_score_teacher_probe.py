import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import types
from contextlib import nullcontext
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('score_teacher_probe', Path(__file__).resolve().parents[1] / 'scripts/score_teacher_probe.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / 'run'
        (self.run / 'videos').mkdir(parents=True)
        self.csv = self.root / 'original.csv'
        self.csv.write_text('video,caption\noriginal.mp4,original caption\n')
        preserved = self.run / 'original.csv'
        preserved.write_bytes(self.csv.read_bytes())
        self.prompt = ' '.join('visible' + str(i) for i in range(140))
        self.jobs, self.rows, self.outputs = [], [], []
        for case in range(6):
            gt = self.root / ('gt' + str(case) + '.mp4')
            gt.write_bytes(('original GT video ' + str(case)).encode())
            for model in score.MODELS:
                for cfg in (3, 5):
                    identifier = str(case) + '_' + model + '_' + str(cfg)
                    out = self.run / 'videos' / (identifier + '.mp4')
                    out.write_bytes(('fixed generated video ' + identifier).encode())
                    job = {'id': identifier, 'case_id': 'case' + str(case), 'canonical_source': 'parent' + str(case),
                           'model': model, 'cfg': cfg, 'seed': 0, 'shift': 5, 'steps': 50,
                           'ckpt': 'NONE' if model == 'base' else 'fixed_a07.safetensors',
                           'ckpt_sha256': 'NONE' if model == 'base' else score.MERGE_SHA256,
                           'image': str(gt), 'image_sha256': score.sha256(gt), 'prompt': self.prompt,
                           'out': str(out), 'cfg_zero_star': False, 'cfg_zero_steps': 0,
                           'width': 1280, 'height': 704, 'num_frames': 121, 'fps': 24, 'quality': 5,
                           'prompt_review': {'csv_path': str(self.csv), 'csv_sha256': score.sha256(self.csv),
                                             'preserved_csv': str(preserved), 'reviewed_by': 'assistant-frame-review',
                                             'reviewed_at': '2026-10-19T00:00:00+00:00',
                                             'prompt_sha256': hashlib.sha256(self.prompt.encode()).hexdigest()}}
                    self.jobs.append(job)
                    self.outputs.append({'id': identifier, 'out': str(out), 'sha256': score.sha256(out),
                                         'decode': {'frames': 121, 'width': 1280, 'height': 704, 'fps': 24}})
                    candidate = cfg == 3
                    gm, vm = 2., 2.
                    self.rows.append({'id': identifier, 'case_id': job['case_id'], 'canonical_source': job['canonical_source'],
                                      'model': model, 'cfg': cfg, 'subject_consistency': .8 + (.004 if candidate else 0),
                                      'imaging_quality': .6 + (.006 if candidate else 0), 'dreamsim': .2,
                                      'dreamsim_last': .25, 'motion_ratio': 0., 'gt_motion': gm,
                                      'gen_motion': vm, 'motion_amount_ratio': vm / gm, 'dur': 5.040, 'seconds': .1})
        self.summary = {'status': 'completed', 'expected_jobs': 24, 'completed_outputs': 24, 'failures': [],
                        'outputs': self.outputs, 'workers': [{'gpu': str(i), 'seconds': 1800., 'gpu_hours': .5, 'returncode': 0} for i in range(4)],
                        'gpu_hours': 2., 'reserved_generation_gpu_hours': 4., 'effective_generation_cap_gpu_hours': 4.}
        self.freeze()

    def freeze(self):
        frozen = self.run / 'jobs.frozen.jsonl'
        frozen.write_text(''.join(json.dumps(row) + '\n' for row in self.jobs))
        self.summary['manifest_sha256'] = score.sha256(frozen)
        score.atomic_json(self.run / 'probe_summary.json', self.summary)
        code = self.run / 'code'
        code.mkdir(exist_ok=True)
        sources = {}
        for name in ('run_teacher_probe.py', 'gen_jobs.py'):
            (code / name).write_text('frozen test source ' + name)
            sources[name] = score.sha256(code / name)
        framework = self.run / 'framework'
        (framework / 'diffsynth/models').mkdir(parents=True, exist_ok=True)
        framework_sources = {}
        for relative in ('diffsynth/__init__.py', 'diffsynth/models/wan_video_dit.py'):
            (framework / relative).write_text('frozen framework test source ' + relative)
            framework_sources[relative] = score.sha256(framework / relative)
        inputs = self.run / 'jobs.input.jsonl'
        inputs.write_bytes(frozen.read_bytes())
        score.atomic_json(self.run / 'probe_manifest.json', {'protocol': 'internal-dev-teacher-cfg-v1',
                          'jobs_frozen_sha256': self.summary['manifest_sha256'], 'jobs': self.jobs,
                          'source_sha256': sources, 'jobs_input_sha256': score.sha256(inputs),
                          'framework_root': str(framework), 'framework_source_sha256': framework_sources,
                          'gpus': ['0', '1', '2', '3'], 'timeout_seconds': 3600,
                          'reserved_generation_gpu_hours': 4., 'effective_generation_cap_gpu_hours': 4.,
                          'zero_disabled': True, 'checkpoint_sha256': {'m_full1_a07': score.MERGE_SHA256},
                          'label_protocol': 'agent_authored_conservative_dense_internal_dev_v1',
                          'use_scope': 'training_visible_internal_diagnostic_only', 'teacher_human_review': 'pending'})

    def test_valid_run_hashes_and_complete_decode_every_output(self):
        with mock.patch.object(score, 'decode_video') as decode:
            jobs, actual, hashes = score.validate_run(self.run)
        self.assertEqual(decode.call_count, 24)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(actual, 2)
        self.assertEqual(jobs[0]['gt'], jobs[0]['image'])
        score.verify_hashes(hashes)
        Path(jobs[0]['out']).write_bytes(b'changed video')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.verify_hashes(hashes)

    def test_final_hash_deadline_does_not_escape_budget(self):
        with self.assertRaisesRegex(ValueError, 'hard timeout'):
            score.verify_hashes({str(self.csv): score.sha256(self.csv)}, score.time.perf_counter() - 1)

    def test_source_snapshot_and_generation_reservation_tampering_rejected(self):
        (self.run / 'code/gen_jobs.py').write_bytes(b'modified sampler')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.validate_run(self.run, decode=False)
        self.freeze()
        manifest = score.read_json(self.run / 'probe_manifest.json')
        manifest['timeout_seconds'] = 3599
        score.atomic_json(self.run / 'probe_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'reservation'):
            score.validate_run(self.run, decode=False)

    def test_framework_mapping_hashes_safe_paths_and_final_recheck(self):
        _, _, hashes = score.validate_run(self.run, decode=False)
        architecture = self.run / 'framework/diffsynth/models/wan_video_dit.py'
        self.assertIn(str(architecture), hashes)
        architecture.write_bytes(b'mutated runtime architecture')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.validate_run(self.run, decode=False)
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.verify_hashes(hashes)
        self.freeze()
        original = score.read_json(self.run / 'probe_manifest.json')
        for relative in ('../outside.py', '/tmp/outside.py', 'diffsynth/../outside.py', 'diffsynth//x.py',
                         'diffsynth/./x.py', 'diffsynth/__pycache__/x.pyc', 'diffsynth/.git/config'):
            manifest = copy.deepcopy(original)
            manifest['framework_source_sha256'][relative] = score.sha256(self.csv)
            score.atomic_json(self.run / 'probe_manifest.json', manifest)
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                score.validate_run(self.run, decode=False)
        manifest = copy.deepcopy(original)
        manifest['framework_root'] = str(self.root)
        score.atomic_json(self.run / 'probe_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'framework root'):
            score.validate_run(self.run, decode=False)
        manifest = copy.deepcopy(original)
        manifest['framework_source_sha256'] = {}
        score.atomic_json(self.run / 'probe_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'mapping missing'):
            score.validate_run(self.run, decode=False)

    def test_framework_symlink_escape_is_rejected(self):
        manifest = score.read_json(self.run / 'probe_manifest.json')
        link = self.run / 'framework/diffsynth/outside.py'
        link.symlink_to(self.csv)
        manifest['framework_source_sha256']['diffsynth/outside.py'] = score.sha256(self.csv)
        score.atomic_json(self.run / 'probe_manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'symlink/path'):
            score.validate_run(self.run, decode=False)

    def test_original_imaging_units(self):
        self.assertEqual(score.normalized_imaging(.625, 62.5), .625)
        for aggregate, raw in ((62.5, 62.5), (.625, .625), (.7, 62.5), (True, 100), (.5, float('nan'))):
            with self.subTest(aggregate=aggregate, raw=raw), self.assertRaises(ValueError):
                score.normalized_imaging(aggregate, raw)

    def test_metrics_pass_only_ready_for_pending_review_never_teacher(self):
        result = score.aggregate(self.jobs, self.rows)
        for model in score.MODELS:
            entry = result['models'][model]
            self.assertEqual(entry['machine_metrics'], 'PASS')
            self.assertEqual(entry['decision'], 'ready_for_review')
            self.assertFalse(entry['promotion_allowed'])
            self.assertFalse(entry['teacher_passed'])
            self.assertEqual(entry['human_semantic_action_review'], 'pending')
        self.assertEqual(result['bootstrap']['replicates'], 5000)
        self.assertEqual(result['bootstrap']['rng_seed'], 20261019)
        self.assertFalse(result['bootstrap']['across_seed_uncertainty'])
        self.assertFalse(result['bootstrap']['human_score_variance_included'])
        self.assertEqual(result, score.aggregate(self.jobs, list(reversed(self.rows))))
        self.assertAlmostEqual(result['models']['base']['metrics']['subject_consistency']['cfg3_minus_cfg5'], .004)

    def test_main_win_guard_failure_cannot_promote(self):
        for metric, delta in (('dreamsim', .006), ('dreamsim_last', .011), ('motion_ratio', .051)):
            rows = copy.deepcopy(self.rows)
            for row in rows:
                if row['model'] == 'm_full1_a07' and row['cfg'] == 3:
                    row[metric] += delta
                    if metric == 'motion_ratio':
                        row['motion_amount_ratio'] = math.exp(row[metric])
                        row['gen_motion'] = row['gt_motion'] * row['motion_amount_ratio']
            entry = score.aggregate(self.jobs, rows)['models']['m_full1_a07']
            self.assertTrue(entry['primary_passed'])
            self.assertFalse(entry['guards'][metric])
            self.assertEqual(entry['machine_metrics'], 'FAIL')
            self.assertFalse(entry['promotion_allowed'])

    def test_primary_needs_four_sources_and_original_direction(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['cfg'] == 3 and row['case_id'] in ('case3', 'case4', 'case5'):
                row['subject_consistency'] = .799
                row['imaging_quality'] = .599
            elif row['cfg'] == 3:
                row['subject_consistency'] = .820
                row['imaging_quality'] = .640
        result = score.aggregate(self.jobs, rows)
        for model in score.MODELS:
            self.assertGreater(result['models'][model]['metrics']['subject_consistency']['cfg3_minus_cfg5'], .003)
            self.assertFalse(result['models'][model]['primary_passed'])
        self.assertEqual(result['models']['base']['metrics']['dreamsim']['direction'], 'lower')
        self.assertEqual(result['models']['base']['metrics']['imaging_quality']['direction'], 'higher')

    def test_anti_static_guard_even_when_mean_error_guard_passes(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['model'] == 'm_full1_a07' and row['case_id'] == 'case0':
                row['gen_motion'] = 8. if row['cfg'] == 5 else 5.
                row['motion_amount_ratio'] = row['gen_motion'] / row['gt_motion']
                row['motion_ratio'] = abs(math.log(row['motion_amount_ratio']))
        entry = score.aggregate(self.jobs, rows)['models']['m_full1_a07']
        self.assertTrue(entry['guards']['motion_ratio'])
        self.assertFalse(entry['guards']['anti_static_motion'])
        self.assertEqual(entry['machine_metrics'], 'FAIL')
        for row in rows:
            if row['case_id'] == 'case0':
                row['gt_motion'] = .5
                row['motion_amount_ratio'] = row['gen_motion'] / .5
                row['motion_ratio'] = abs(math.log(row['motion_amount_ratio']))
        self.assertTrue(score.aggregate(self.jobs, rows)['models']['m_full1_a07']['guards']['anti_static_motion'])

    def test_difference_in_differences_is_paired_noncausal(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['model'] == 'm_full1_a07' and row['cfg'] == 3:
                row['subject_consistency'] += .002
        did = score.aggregate(self.jobs, rows)['difference_in_differences']['subject_consistency']
        self.assertAlmostEqual(did['a07_delta_minus_base_delta'], .002)
        self.assertFalse(did['causal_claim'])
        self.assertEqual(len(did['per_source_delta']), 6)

    def test_duplicate_missing_cells_nonfinite_bool_and_raw_quality_rejected(self):
        cases = [self.rows[:-1], self.rows[:-1] + [copy.deepcopy(self.rows[0])]]
        for metric, value in (('dreamsim', float('nan')), ('dreamsim_last', float('inf')),
                              ('subject_consistency', True), ('imaging_quality', 62.5), ('motion_ratio', -1)):
            rows = copy.deepcopy(self.rows)
            rows[0][metric] = value
            cases.append(rows)
        missing_arm = copy.deepcopy(self.rows)
        missing_arm[0]['cfg'] = 5
        cases.append(missing_arm)
        for rows in cases:
            with self.subTest(rows=rows[0]), self.assertRaises(ValueError):
                score.aggregate(self.jobs, rows)

    def test_budget_limits_and_charge_consistency(self):
        self.assertEqual(score.budget('0,1,2,3', 7200, 8, 4), (['0', '1', '2', '3'], 8, 8))
        for values in (('0,1,2,3', 7201, 8, 4), ('0,1,2,3', 7200, 7.9, 4), ('0', 1, 8, 4.001),
                       ('0,0', 1, 8, 0), ('00', 1, 8, 0), ('0', True, 8, 0), ('0', 1, float('inf'), 0), ('0,1,2,3,4', 1, 8, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                score.budget(*values)
        resources = copy.deepcopy(self.summary['workers'])
        for actual, reserve, effective in ((2.1, 4, 4), (2, 1.9, 4), (2, 4, 1.9)):
            with self.assertRaises(ValueError):
                score.validate_charge(resources, actual, reserve, effective, 'test')
        resources[0]['seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'charge mismatch'):
            score.validate_charge(resources, 2, 4, 4, 'test')

    def test_changed_data_manifest_output_or_missing_gt_rejected(self):
        self.csv.write_bytes(b'changed original caption')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.validate_run(self.run, decode=False)
        self.csv.write_text('video,caption\noriginal.mp4,original caption\n')
        output = Path(self.jobs[0]['out'])
        original = output.read_bytes()
        output.write_bytes(b'mutated output')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.validate_run(self.run, decode=False)
        output.write_bytes(original)
        (self.run / 'jobs.frozen.jsonl').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'SHA256 changed'):
            score.validate_run(self.run, decode=False)
        self.freeze()
        image = self.root / 'input.png'
        image.write_bytes(b'still image')
        self.jobs[0].update(image=str(image), image_sha256=score.sha256(image))
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Original GT video required'):
            score.validate_run(self.run, decode=False)

    def test_protocol_matrix_pairing_seed_and_generation_budget_rejected(self):
        original = copy.deepcopy(self.jobs)
        changes = [('cfg', 5), ('seed', 1), ('canonical_source', 'parent1'), ('width', 1279), ('cfg_zero_star', True), ('prompt', self.prompt + ' changed')]
        for key, value in changes:
            self.jobs = copy.deepcopy(original)
            self.jobs[0][key] = value
            self.freeze()
            with self.subTest(key=key), self.assertRaises(ValueError):
                score.validate_run(self.run, decode=False)
        self.jobs = original
        self.summary['gpu_hours'] = 4.1
        self.freeze()
        with self.assertRaises(ValueError):
            score.validate_run(self.run, decode=False)

    def test_existing_scoring_never_overwritten_and_invalid_budget_no_gpu(self):
        scoring = self.run / 'scoring'
        scoring.mkdir()
        sentinel = scoring / 'score_summary.json'
        sentinel.write_bytes(b'old-score-do-not-overwrite')
        with mock.patch.object(score, 'assert_idle') as idle, mock.patch.object(score, 'validate_run') as validate:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                score.main(['--run-dir', str(self.run)])
            idle.assert_not_called()
            validate.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b'old-score-do-not-overwrite')
        with mock.patch.object(score, 'validate_run') as validate, mock.patch.object(score, 'assert_idle') as idle:
            with self.assertRaises(ValueError):
                score.main(['--run-dir', str(self.root / 'missing'), '--timeout-seconds', '7201'])
            validate.assert_not_called()
            idle.assert_not_called()

    def test_completed_status_describes_24_scores_not_quality_pass(self):
        jobs, actual, hashes = score.validate_run(self.run, decode=False)
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row['cfg'] == 3:
                row['dreamsim'] += .1
        def launch(scoring, kind, assignments, deadline):
            resource = {'index': 0, 'gpu': '' if kind == 'cpu' else '0', 'pid': 12345, 'returncode': 0,
                        'seconds': 1., 'gpu_hours': 0 if kind == 'cpu' else 1 / 3600}
            score.atomic_json(scoring / (kind + '_worker_0.json'), {'status': 'completed', 'kind': kind, 'gpu': resource['gpu'], 'rows': rows})
            return [resource], None
        with mock.patch.object(score, 'validate_run', return_value=(jobs, actual, hashes)), mock.patch.object(score, 'metric_assets', return_value={}),                 mock.patch.object(score, 'assert_idle'), mock.patch.object(score, 'launch_workers', side_effect=launch):
            rc = score.main(['--run-dir', str(self.run), '--gpus', '0'])
        self.assertEqual(rc, 0)
        summary = score.read_json(self.run / 'scoring/score_summary.json')
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['completed_rows'], 24)
        self.assertFalse(summary['teacher_passed'])
        self.assertEqual(summary['diagnostics']['models']['base']['machine_metrics'], 'FAIL')

    def test_failed_cpu_attempt_charges_reserved_gpus_setup_and_no_retry(self):
        jobs, actual, hashes = score.validate_run(self.run, decode=False)
        with mock.patch.object(score, 'validate_run', return_value=(jobs, actual, hashes)), \
                mock.patch.object(score, 'metric_assets', return_value={}), mock.patch.object(score, 'assert_idle'), \
                mock.patch.object(score, 'launch_workers', return_value=([{'index': 0, 'gpu': '', 'returncode': 1, 'seconds': 1., 'gpu_hours': 0}], 'CPU failed')) as launch, \
                mock.patch.object(score.traceback, 'print_exc'):
            self.assertEqual(score.main(['--run-dir', str(self.run), '--gpus', '0,1']), 1)
        self.assertEqual(launch.call_count, 1)
        result = score.read_json(self.run / 'scoring/score_summary.json')
        self.assertEqual(result['status'], 'failed')
        self.assertGreater(result['actual_scoring_gpu_hours'], 0)
        self.assertAlmostEqual(result['actual_scoring_gpu_hours'], 2 * result['wall_seconds'] / 3600)
        self.assertTrue(result['allocation_includes_cpu_setup_and_final_verification'])
        self.assertEqual(result['gpu_process_gpu_hours'], 0)

    def test_worker_nonzero_charges_all_occupied_wall_and_no_retry(self):
        scoring = self.run / 'scoring'
        scoring.mkdir()
        proc = mock.Mock(pid=54321, returncode=1)
        proc.poll.return_value = 1
        proc.wait.return_value = 1
        assignment = {'gpu': '0', 'jobs': [], 'asset_sha256': {}}
        with mock.patch.object(score.subprocess, 'Popen', return_value=proc) as popen, mock.patch.object(score.os, 'killpg') as kill:
            resources, error = score.launch_workers(scoring, 'gpu', [assignment], score.time.perf_counter() + 10)
        self.assertTrue(error)
        self.assertEqual(popen.call_count, 1)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertGreaterEqual(resources[0]['seconds'], 0)
        self.assertEqual(resources[0]['gpu_hours'], resources[0]['seconds'] / 3600)
        proc.wait.assert_called()

    def test_cpu_worker_uses_original_shared_sampling_motion_and_preprocess(self):
        scoring = self.run / 'scoring'
        scoring.mkdir()
        jobs = [dict(job, gt=job['image'], gt_sha256=job['image_sha256'], output_sha256=score.sha256(job['out'])) for job in self.jobs[:4]]
        assignment = scoring / 'cpu_assignment_0.json'
        score.atomic_json(assignment, {'gpu': '', 'jobs': jobs, 'asset_sha256': {}})
        frame = mock.Mock()
        frame.resize.return_value = frame
        gt_ref = types.ModuleType('gt_ref_metrics')
        gt_ref.read = mock.Mock(side_effect=lambda path: ([frame] * 121, 24.))
        gt_ref.sample = mock.Mock(return_value=[frame] * 12)
        gt_ref.motion = mock.Mock(return_value=2.)
        torch = types.ModuleType('torch')
        torch.set_num_threads = mock.Mock()
        torch.cat = mock.Mock(return_value='preprocessed-batch')
        torch.no_grad = nullcontext
        model = mock.Mock()
        model.return_value.flatten.return_value.tolist.return_value = [.2] * 12
        preprocess = mock.Mock(return_value='preprocessed-frame')
        dream = types.ModuleType('dreamsim')
        dream.dreamsim = mock.Mock(return_value=(model, preprocess))
        with mock.patch.dict(sys.modules, {'torch': torch, 'dreamsim': dream, 'gt_ref_metrics': gt_ref}), \
                mock.patch.dict(score.os.environ, {'TEACHER_SCORE_PARENT_PID': str(score.os.getppid()), 'CUDA_VISIBLE_DEVICES': ''}), \
                mock.patch.object(score, 'offline_loaders', side_effect=nullcontext):
            self.assertEqual(score.worker(scoring, 'cpu', 0, score.sha256(assignment)), 0)
        self.assertEqual(gt_ref.read.call_count, 5)  # one independent case cache + four outputs
        self.assertEqual(gt_ref.sample.call_count, 8)
        self.assertAlmostEqual(gt_ref.sample.call_args.args[2], 121 / 24 - .001)
        self.assertEqual(gt_ref.motion.call_count, 8)
        self.assertEqual(preprocess.call_count, 96)
        frame.resize.assert_called_with((256, 144))
        dream.dreamsim.assert_called_once_with(pretrained=True, device='cpu', cache_dir=str(score.DREAM_CACHE))
        report = score.read_json(scoring / 'cpu_worker_0.json')
        self.assertEqual(report['status'], 'completed')
        self.assertAlmostEqual(report['rows'][0]['dreamsim'], .2)
        self.assertEqual(report['rows'][0]['motion_ratio'], 0)
        self.assertEqual(report['rows'][0]['gt_motion'], 2.)
        torch.set_num_threads.assert_called_with(8)

    def test_gpu_worker_calls_exact_original_apis_and_normalizes_musiq(self):
        scoring = self.run / 'scoring'
        scoring.mkdir()
        job = dict(self.jobs[0], output_sha256=score.sha256(self.jobs[0]['out']))
        assignment = scoring / 'gpu_assignment_0.json'
        score.atomic_json(assignment, {'gpu': '2', 'jobs': [job], 'asset_sha256': {}})
        torch = types.ModuleType('torch')
        torch.set_num_threads = mock.Mock()
        torch.no_grad = nullcontext
        torch.cuda = mock.Mock()
        torch.cuda.max_memory_allocated.return_value = 100
        torch.cuda.max_memory_reserved.return_value = 200
        model = mock.Mock()
        model.eval.return_value.to.return_value = model
        torch.hub = types.SimpleNamespace(load=mock.Mock(return_value=model))
        subject = types.ModuleType('vbench.subject_consistency')
        subject.subject_consistency = mock.Mock(return_value=(.8, [{'video_path': job['out'], 'video_results': .8}]))
        imaging = types.ModuleType('vbench.imaging_quality')
        imaging.technical_quality = mock.Mock(return_value=(.625, [{'video_path': job['out'], 'video_results': 62.5}]))
        musiq = types.ModuleType('pyiqa.archs.musiq_arch')
        musiq.MUSIQ = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {'torch': torch, 'vbench.subject_consistency': subject, 'vbench.imaging_quality': imaging,
                                          'pyiqa.archs.musiq_arch': musiq}), \
                mock.patch.dict(score.os.environ, {'TEACHER_SCORE_PARENT_PID': str(score.os.getppid()), 'CUDA_VISIBLE_DEVICES': '2'}), \
                mock.patch.object(score, 'offline_loaders', side_effect=nullcontext):
            self.assertEqual(score.worker(scoring, 'gpu', 0, score.sha256(assignment)), 0)
        subject.subject_consistency.assert_called_once_with(model, [job['out']], 'cuda', False)
        imaging.technical_quality.assert_called_once_with(model, [job['out']], 'cuda')
        musiq.MUSIQ.assert_called_once_with(pretrained_model_path=str(score.MUSIQ_WEIGHT))
        row = score.read_json(scoring / 'gpu_worker_0.json')['rows'][0]
        self.assertEqual(row['imaging_quality'], .625)
        self.assertEqual(row['imaging_quality_raw'], 62.5)

    def test_timeout_kills_only_owned_group_and_waits(self):
        proc = mock.Mock(pid=65432, returncode=-signal.SIGKILL)
        proc.poll.return_value = None
        proc.wait.side_effect = [score.subprocess.TimeoutExpired('owned-worker', 0), 0]
        workers = [{'process': proc, 'started': 0}]
        with mock.patch.object(score.os, 'killpg') as kill:
            score.stop_workers(workers, .001)
        self.assertEqual(kill.call_args_list, [mock.call(65432, signal.SIGTERM), mock.call(65432, signal.SIGKILL)])
        self.assertEqual(proc.wait.call_count, 2)


if __name__ == '__main__':
    unittest.main()
