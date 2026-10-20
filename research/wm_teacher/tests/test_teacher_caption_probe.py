import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('caption_probe_test', ROOT / 'scripts/run_teacher_caption_probe.py')
c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c1)


class CaptionProbeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'new_run'
        csv_path = self.root / 'training_metadata/ft2/ft2_train.csv'
        csv_path.parent.mkdir(parents=True)
        originals = ['  original caption ' + str(i) + (chr(10) + 'second line' if i == 0 else '') + '  '
                     for i in range(6)]
        with csv_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['video', 'prompt'])
            for i, cid in enumerate(c1.CASE_ORDER):
                writer.writerow(['dataset/' + cid + '.mp4', originals[i]])
        with csv_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            lines = [(reader.line_num, row) for row in reader]
        self.cases, selected, self.controls, self.originals = [], [], [], {}
        for i, cid in enumerate(c1.CASE_ORDER):
            image = self.root / 'dataset' / (cid + '.mp4')
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(('GT ' + cid).encode())
            corrected = 'corrected caption ' + cid
            review = {'csv_path': str(csv_path), 'csv_sha256': c1.sha256(csv_path),
                      'prompt_sha256': c1.m1.text_hash(corrected), 'reviewed_by': 'test_agent',
                      'reviewed_at': '2026-10-20T00:00:00+00:00'}
            case = {'case_id': cid, 'image': str(image), 'image_sha256': c1.sha256(image),
                    'canonical_source': 'source_' + str(i), 'prompt': corrected, 'prompt_review': review}
            self.cases.append(case)
            selected.append({'case_id': cid, 'csv_line': lines[i][0], 'video': str(image),
                             'video_sha256': c1.sha256(image), 'canonical_source': case['canonical_source'],
                             'agent_review': review, 'original_prompt': originals[i],
                             'original_prompt_sha256': c1.m1.text_hash(originals[i]),
                             'corrected_prompt': corrected, 'corrected_prompt_sha256': c1.m1.text_hash(corrected),
                             'original_preserved': True})
            self.originals[cid] = {'prompt': originals[i], 'prompt_sha256': c1.m1.text_hash(originals[i]),
                                   'csv_path': str(csv_path), 'csv_sha256': c1.sha256(csv_path),
                                   'csv_row': i, 'csv_line': lines[i][0], 'video_relative': 'dataset/' + cid + '.mp4'}
            for model in c1.MODELS:
                identifier = model + '_cfg5_' + cid
                job = dict(case, id=identifier, model=model, cfg=5, seed=0, steps=50, shift=5,
                           ckpt='NONE' if model == 'base' else '/fixed/a07.safetensors',
                           fps=24, quality=5, width=1280, height=704, num_frames=121,
                           cfg_zero_star=False, cfg_zero_steps=0, gt=str(image), gt_sha256=c1.sha256(image),
                           output_sha256='a' * 64, out=str(self.root / (identifier + '.mp4')))
                row = self.metric_row(job)
                self.controls.append({'job': job, 'row': row, 'row_sha256': 'b' * 64, 'video_sha256': 'a' * 64})
        self.provenance = {'protocol': 'P1_CFG_ONLY_CORRECTED_AGENT_DENSE_DEVLABEL_V1', 'selected_cases': selected}
        self.jobs = c1.make_jobs(self.controls, self.originals, self.run)
        self.rows = [self.metric_row(job) for job in self.jobs]

    @staticmethod
    def metric_row(job):
        return {k: job[k] for k in ('id', 'case_id', 'canonical_source', 'model', 'cfg')} | {
            'subject_consistency': .9, 'imaging_quality': .7, 'imaging_quality_raw': 70.,
            'dreamsim': .2, 'dreamsim_last': .3, 'gen_motion': 2., 'gt_motion': 2.,
            'motion_amount_ratio': 1., 'motion_ratio': 0., 'dur': 5.,
            'cpu_seconds': .01, 'gpu_seconds': .02, 'seconds': .01,
            'peak_allocated_bytes': 10, 'peak_reserved_bytes': 20}

    def test_budget_caps_and_invalid_values(self):
        value = c1.budget('0,1,2,3', 1800, 3600, 5)
        self.assertEqual(value['generation_gpu_hours'] + value['scoring_gpu_hours'], 6)
        for args in [('0,1', 1800, 3600, 5), ('0,1,2,3', 1801, 3600, 5),
                     ('0,1,2,3', 1800, 3601, 5), ('0,1,2,3', float('nan'), 20, 1),
                     ('0,1,2,3', True, 20, 1), ('0,1,2,3', 20, 20, 10)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                c1.budget(*args)

    def test_actual_cli_help_imports_no_torch(self):
        script = ROOT / 'scripts/run_teacher_caption_probe.py'
        code = ("import builtins,runpy,sys;sys.path.insert(0," + repr(str(ROOT / 'scripts')) + ");"
                "old=builtins.__import__;"
                "builtins.__import__=lambda n,*a,**k: (_ for _ in ()).throw(AssertionError('torch imported')) "
                "if n.split('.')[0]=='torch' else old(n,*a,**k);"
                "runpy.run_path(" + repr(str(script)) + ",run_name='__main__')")
        result = subprocess.run([sys.executable, '-c', code, '--help'], capture_output=True, text=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--preflight-only', result.stdout)

    def test_original_caption_preserves_parsed_whitespace_and_physical_line(self):
        actual = c1.original_captions(self.cases, self.provenance, self.root)
        self.assertEqual(actual, self.originals)
        first = actual[c1.CASE_ORDER[0]]
        self.assertTrue(first['prompt'].startswith('  '))
        self.assertIn(chr(10) + 'second line', first['prompt'])
        self.assertGreater(first['csv_line'], first['csv_row'] + 2)

    def test_original_provenance_tamper_or_wrong_case_rejected(self):
        for key, value in [('original_prompt', 'wrong'), ('original_prompt_sha256', 'c' * 64),
                           ('corrected_prompt', 'wrong'), ('csv_line', 99), ('original_preserved', False)]:
            bad = copy.deepcopy(self.provenance)
            bad['selected_cases'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c1.original_captions(self.cases, bad, self.root)
        bad = copy.deepcopy(self.provenance)
        bad['selected_cases'] = bad['selected_cases'][::-1]
        with self.assertRaises(ValueError):
            c1.original_captions(self.cases, bad, self.root)

    def test_csv_paths_must_be_full_and_safe(self):
        for value in ['../escape.mp4', '/absolute.mp4', 'dataset//bad.mp4', 'dataset/./bad.mp4',
                      'https://example.com/video', 'a' + chr(92) + 'b.mp4']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                c1.relative_video(self.root, value)
        self.assertEqual(c1.relative_video(self.root, 'dataset/video.mp4'), str(self.root / 'dataset/video.mp4'))

    def test_new_jobs_change_only_caption_and_bind_their_own_hash(self):
        self.assertEqual(len(self.jobs), 12)
        for job, control in zip(self.jobs, self.controls):
            old = control['job']
            self.assertNotIn('prompt_review', job)
            self.assertEqual(c1.m1.text_hash(job['prompt']), job['caption_provenance']['parsed_original_prompt_sha256'])
            self.assertEqual(job['caption_provenance']['corrected_control_review'], old['prompt_review'])
            for key in set(old) - {'id', 'prompt', 'out', 'output_sha256', 'prompt_review'}:
                self.assertEqual(job[key], old[key])
        self.assertTrue(all(c['job']['prompt'].startswith('corrected') for c in self.controls))

    def test_static_assignments_cover_all_twelve_once(self):
        groups = c1.assignments(self.jobs)
        self.assertEqual([g['gpu'] for g in groups], ['0', '1', '2', '3'])
        self.assertEqual([len(g['jobs']) for g in groups], [3, 3, 3, 3])
        self.assertEqual({j['id'] for g in groups for j in g['jobs']}, {j['id'] for j in self.jobs})
        self.assertTrue(all(j['model'] == 'base' for g in groups[:2] for j in g['jobs']))

    def test_metric_units_motion_formula_and_matrix(self):
        c1.validate_rows(self.jobs, self.rows)
        for key, value in [('imaging_quality_raw', 71), ('motion_ratio', .1), ('cfg', True),
                           ('subject_consistency', float('nan')), ('dreamsim', True)]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c1.validate_rows(self.jobs, rows)
        with self.assertRaises(ValueError):
            c1.validate_rows(self.jobs, self.rows[:-1])
        with self.assertRaises(ValueError):
            c1.validate_rows(self.jobs, [self.rows[0]] + self.rows[:-1])

    def test_diagnostics_corrected_minus_original_and_no_teacher_promotion(self):
        original = copy.deepcopy(self.rows)
        for row in original:
            row['subject_consistency'] = .89
            row['dreamsim'] = .21
        controls_before = copy.deepcopy(self.controls)
        result = c1.diagnostics(self.jobs, original, self.controls)
        self.assertEqual(result['new_original_rows'], 12)
        self.assertEqual(result['reused_corrected_rows'], 12)
        self.assertEqual(len(result['paired_rows']), 24)
        for model in c1.MODELS:
            self.assertAlmostEqual(result['model_mean_corrected_minus_original'][model]['subject_consistency'], .01)
            self.assertAlmostEqual(result['model_mean_corrected_minus_original'][model]['dreamsim'], -.01)
        self.assertEqual(result['a07_change_minus_base_change']['dreamsim'], 0)
        self.assertFalse(result['teacher_passed'])
        self.assertFalse(result['training_benefit'])
        self.assertEqual(self.controls, controls_before)
        bad = copy.deepcopy(original)
        bad[0]['gt_motion'] = 4
        bad[0]['motion_amount_ratio'] = .5
        bad[0]['motion_ratio'] = abs(math.log(.5))
        with self.assertRaises(ValueError):
            c1.diagnostics(self.jobs, bad, self.controls)

    def test_generation_reports_require_every_completed_job_and_zero_exits(self):
        (self.run / 'generation').mkdir(parents=True)
        resources = []
        for i, group in enumerate(c1.assignments(self.jobs)):
            resources.append({'index': i, 'gpu': group['gpu'], 'returncode': 0})
            c1.generation.atomic_json(self.run / 'generation' / ('worker_' + str(i) + '.json'), {
                'status': 'completed', 'gpu': group['gpu'], 'jobs': [dict(id=j['id'], out=j['out'],
                status='completed', seconds=1., peak_allocated_bytes=2, peak_reserved_bytes=3) for j in group['jobs']]})
        c1.generation_reports(self.run, self.jobs, resources)
        bad = copy.deepcopy(resources)
        bad[0]['returncode'] = 1
        with self.assertRaises(ValueError):
            c1.generation_reports(self.run, self.jobs, bad)
        path = self.run / 'generation/worker_0.json'
        report = c1.read_json(path)
        report['jobs'].pop()
        c1.generation.atomic_json(path, report)
        with self.assertRaises(ValueError):
            c1.generation_reports(self.run, self.jobs, resources)

    def test_full_decode_verification_binds_output_ids_and_fixed_shape(self):
        (self.run / 'generation').mkdir(parents=True)
        outputs = [dict(id=j['id'], out=j['out'], sha256='a' * 64,
                        decode={'frames':121,'width':1280,'height':704,'fps':24}) for j in self.jobs]
        path = self.run / 'generation/verify_gen_worker_0.json'
        c1.generation.atomic_json(path, {'status':'completed','outputs':outputs})
        self.assertEqual(len(c1.decoded_report(self.run,'verify_gen',self.jobs)),12)
        outputs[0]['decode']['frames'] = 120
        c1.generation.atomic_json(path, {'status':'completed','outputs':outputs})
        with self.assertRaises(ValueError):
            c1.decoded_report(self.run,'verify_gen',self.jobs)

    def test_cpu_gpu_report_merge_requires_all_rows(self):
        (self.run / 'scoring').mkdir(parents=True)
        cpu = [{'index':0,'gpu':'','returncode':0}]
        gpu = []
        c1.generation.atomic_json(self.run / 'scoring/cpu_worker_0.json', {
            'status':'completed','kind':'cpu','gpu':'','rows':self.rows})
        for i, group in enumerate(c1.assignments(self.jobs)):
            gpu.append({'index':i,'gpu':group['gpu'],'returncode':0})
            ids={j['id'] for j in group['jobs']}
            c1.generation.atomic_json(self.run / 'scoring' / ('gpu_worker_'+str(i)+'.json'), {
                'status':'completed','kind':'gpu','gpu':group['gpu'],'rows':[r for r in self.rows if r['id'] in ids]})
        merged=c1.merge_score_reports(self.run,self.jobs,{'cpu':cpu,'gpu':gpu})
        self.assertEqual(len(merged),12)
        gpu[0]['returncode']=1
        with self.assertRaises(ValueError):
            c1.merge_score_reports(self.run,self.jobs,{'cpu':cpu,'gpu':gpu})

    def test_preflight_rejects_existing_directory_before_assets_or_gpu(self):
        path=Path('/srv/wm_tmp/c1_test_existing_'+str(os.getpid()))
        with mock.patch.object(Path,'exists',return_value=True), mock.patch.object(c1.m1,'preflight') as assets:
            with self.assertRaises(ValueError):
                c1.preflight(path)
            assets.assert_not_called()
        with self.assertRaises(ValueError):
            c1.preflight(self.root / 'outside_local_storage')

    def test_phase_failure_and_cpu_setup_are_charged_and_total_cap_recorded(self):
        self.run.mkdir()
        c1.write_json(self.run / 'generation_summary.json', {'allocated_gpu_hours':1.5})
        def fail(deadline, workers):
            time.sleep(.01)
            raise RuntimeError('setup failed')
        result=c1.phase(self.run,'score',3,1,fail)
        self.assertEqual(result['status'],'failed')
        self.assertGreaterEqual(result['allocated_gpu_hours'],4/3600)
        self.assertAlmostEqual(result['actual_experiment_gpu_hours'],1.5+result['allocated_gpu_hours'])
        self.assertIn('setup failed',result['failure_traceback'])
        self.assertFalse(result['teacher_passed'])
        self.assertEqual(c1.read_json(self.run/'score_summary.json'),result)

    def test_real_phase_deadline_bounds_blocking_operation(self):
        self.run.mkdir()
        def block(deadline, workers):
            time.sleep(20)
        t0=time.perf_counter()
        result=c1.phase(self.run,'generation',3,1,block)
        self.assertLess(time.perf_counter()-t0,3)
        self.assertEqual(result['status'],'failed')
        self.assertIn('deadline',result['failures'][0])
        self.assertLessEqual(result['allocated_gpu_hours'],3*4/3600)

    def test_freeze_imports_all_stage_helpers_without_cuda(self):
        names=['run_teacher_caption_probe.py','run_teacher_mechanism_probe.py','score_teacher_probe.py','gt_ref_metrics.py']
        manifest={'jobs':self.jobs,'own_sources':{n:c1.m1.file_record(ROOT/'scripts'/n) for n in names},
                  'p1_audit':{'p1_source':{n:c1.m1.file_record(ROOT/'scripts'/n)
                              for n in ['run_teacher_probe.py','gen_jobs.py']},'framework':{}}}
        frozen=c1.freeze(self.run,manifest,c1.budget('0,1,2,3',20,20,1),time.perf_counter()+20)
        for name,pin in frozen['snapshot_sha256'].items():
            self.assertEqual(c1.sha256(self.run/name),pin)
        code="import run_teacher_caption_probe; assert 'torch' not in __import__('sys').modules"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=15,
                              env=dict(os.environ,PYTHONPATH=str(self.run/'code'),CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(result.returncode,0,result.stderr)
        with self.assertRaises((ValueError,FileExistsError)):
            c1.freeze(self.run,manifest,{},time.perf_counter()+10)

    def test_worker_rejects_wrong_parent_before_importing_gpu(self):
        with mock.patch.dict(os.environ,{'C1_PARENT_PID':'not-my-parent'}):
            with self.assertRaises(ValueError):
                c1.worker(self.run,'gen',0,'x')


if __name__=='__main__':
    unittest.main()
