"""CPU-only real GNU timeout/process cleanup and adversarial gate tests."""
import copy
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_teacher_generation as generation
import prepare_teacher_recovery as recovery
from test_teacher_recovery import plan


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def record_json(self, name, value):
        path = self.root / name
        recovery.durable_write(path, value)
        return recovery.record(path, nonempty=True)

    def spawn(self, code):
        child = subprocess.Popen([sys.executable, '-c', code], start_new_session=True)
        row = generation.identity(child.pid)
        self.addCleanup(self.cleanup, child, row)
        return child, row

    def cleanup(self, child, row):
        if generation.session_members(row):
            generation.cleanup_owned(child, row, time.monotonic() + 3)
        child.wait(timeout=3)

    def test_bound_json_single_read_keeps_sha_duplicate_and_finite_gates(self):
        row = self.record_json('single.json', {'bound': 1})
        original = Path.read_bytes
        reads = []
        def observed(path):
            reads.append(str(path))
            return original(path)
        with mock.patch.object(Path, 'read_bytes', observed):
            self.assertEqual(generation.read_bound(row), {'bound': 1})
        self.assertEqual(reads, [row['path']])
        path = Path(row['path'])
        stat = path.stat()
        path.write_bytes(b'x' * row['bytes'])
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            generation.read_bound(row)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                generation.read_bound(recovery.record(path, nonempty=True))

    def worker_preflight_fixture(self):
        allocation = dict(shard=0, node_rank=0, gpu_index=0, allocated_gpus=1, cap_seconds=4500)
        plans = {m: self.record_json('worker-plan-' + m + '.json', dict(root=str(ROOT), method=m))
                 for m in generation.METHODS}
        value = dict(plans=plans, allocations={'base': [allocation]}, comparison=dict(path='approved-metadata', sha256='c'*64),
                     common_execution_sha256='d'*64, score_bundle={'sha256':'e'*64}, retirement={'alg':{},'dymos':{}},
                     source_files=[recovery.record(p, nonempty=True) for p in generation.sources(ROOT)])
        execution = self.record_json('worker-execution.json', value)
        args = [sys.executable, str(ROOT/'scripts/run_teacher_generation.py'), '--run', '--gpu-authorized',
                '--execution', execution['path'], '--approved-execution-sha256', execution['sha256'],
                '--method', 'base', '--shard', '0', '--rank', '0']
        owner = dict(pid=12345, ppid=1, pgid=12345, session=12345, start_ticks=777, state='S', argv=args)
        launch = dict(execution=execution, execution_sha256=execution['sha256'], method='base', shard=0,
                      allocation=allocation, plan=plans['base'], owner=owner)
        common = {k:value[k] for k in ('plans','common_execution_sha256','score_bundle')}
        return value, execution, launch, common

    def worker_preflight_patches(self, common, owner):
        import contextlib
        import prepare_teacher_comparison as comparison
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(generation, 'validate_sidecar', side_effect=lambda v:v))
        stack.enter_context(mock.patch.object(generation, 'validate_authorization'))
        stack.enter_context(mock.patch.object(generation, 'verify_scoring_owner'))
        stack.enter_context(mock.patch.object(generation, 'verify_retirement'))
        stack.enter_context(mock.patch.object(generation, 'same_process', return_value=True))
        stack.enter_context(mock.patch.object(generation, 'identity', return_value=owner))
        stack.enter_context(mock.patch.object(comparison, 'verify_metadata', return_value=common))
        stack.enter_context(mock.patch.object(comparison, 'verify', side_effect=AssertionError('Worker must not reload scoring assets')))
        return stack

    def test_generation_worker_never_rehashes_scoring_host_assets(self):
        value, row, launch, common = self.worker_preflight_fixture()
        import score_teacher_devset as scoring
        with self.worker_preflight_patches(common, launch['owner']), \
             mock.patch.object(scoring, 'verify_bundle', side_effect=AssertionError('No full metric environment verification inside generation')):
            self.assertEqual(generation.verify_worker_execution(row['path'],row['sha256'],launch),value)
        # Own generation model/runtime SHA checks are mandatory immediately before CUDA.
        import ast
        tree=ast.parse((ROOT/'scripts/run_teacher_generation.py').read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_worker')
        calls=[n.func.id for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('current_local_content',calls)
        self.assertIn('verify_worker_execution',calls)

    def test_generation_worker_reused_or_unapproved_parent_is_rejected(self):
        _value,row,launch,common=self.worker_preflight_fixture()
        for flag,bad in (('--method','alg'),('--shard','1'),('--approved-execution-sha256','f'*64)):
            owner=copy.deepcopy(launch['owner']);owner['argv'][owner['argv'].index(flag)+1]=bad
            with self.worker_preflight_patches(common,owner), self.assertRaises(ValueError):
                generation.verify_worker_execution(row['path'],row['sha256'],launch)
        with self.worker_preflight_patches(common,launch['owner']), \
             mock.patch.object(generation,'same_process',return_value=False), self.assertRaises(ValueError):
            generation.verify_worker_execution(row['path'],row['sha256'],launch)

    def test_generation_worker_source_same_size_drift_is_rejected(self):
        value,row,launch,common=self.worker_preflight_fixture()
        source=self.root/'worker-runtime.py';source.write_bytes(b'original')
        value['source_files'][0]=recovery.record(source,nonempty=True)
        # Keep the expected source set isolated, while actual file content hashing remains real.
        row=self.record_json('worker-execution-source.json',value)
        launch['execution']=row;launch['execution_sha256']=row['sha256']
        owner=copy.deepcopy(launch['owner']);owner['argv'][owner['argv'].index('--execution')+1]=row['path']
        owner['argv'][owner['argv'].index('--approved-execution-sha256')+1]=row['sha256'];launch['owner']=owner
        stat=source.stat();source.write_bytes(b'modified');os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.worker_preflight_patches(common,owner), \
             mock.patch.object(generation,'sources',return_value=[Path(r['path']) for r in value['source_files']]), \
             self.assertRaises(ValueError):
            generation.verify_worker_execution(row['path'],row['sha256'],launch)

    def test_no_cuda_import_and_explicit_execution_sha_required(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; sys.path.insert(0, 'scripts'); import prepare_teacher_recovery, run_teacher_generation; "
            "assert 'torch' not in sys.modules"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(ValueError):
            generation.supervise('absent', 'bad', 'alg', 0, 1)
        with self.assertRaises(ValueError):
            generation.main(['--run', '--execution', 'absent'])

    def test_real_gnu_timeout_term_and_fresh_owned_session_cleanup(self):
        child = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=1s', '.2s',
                                  sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        owner = generation.identity(child.pid)
        self.addCleanup(self.cleanup, child, owner)
        self.assertEqual(owner['session'], child.pid)
        self.assertEqual(owner['ppid'], os.getpid())
        self.assertEqual(child.wait(timeout=5), 124)
        self.assertEqual(generation.cleanup_owned(child, owner, time.monotonic() + 2), [])
        self.assertEqual(generation.session_members(owner), [])

    def test_cleanup_term_kill_stays_owned_and_preserves_external(self):
        external, external_identity = self.spawn('import time; time.sleep(30)')
        child, owner = self.spawn('import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)')
        time.sleep(.1)
        with mock.patch.object(generation, 'KILL_SECONDS', .1):
            signalled = generation.cleanup_owned(child, owner, time.monotonic() + 2)
        self.assertTrue(any(r['signal'] == signal.SIGTERM for r in signalled))
        self.assertTrue(any(r['signal'] == signal.SIGKILL for r in signalled))
        self.assertTrue(generation.same_process(external_identity))
        self.assertEqual(generation.session_members(owner), [])

    def test_observed_owned_child_in_new_session_is_cleaned_without_external(self):
        external, external_identity = self.spawn('import time; time.sleep(30)')
        child, owner = self.spawn("import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True); time.sleep(30)")
        tracked = []
        until = time.monotonic() + 2
        while time.monotonic() < until:
            members = generation.owned_members(owner, tracked)
            escaped = [r for r in members if r['session'] != owner['session']]
            if escaped:
                break
            time.sleep(.02)
        self.assertTrue(escaped)
        generation.cleanup_owned(child, owner, time.monotonic() + 2, tracked)
        self.assertFalse(any(generation.same_process(r) for r in tracked))
        self.assertTrue(generation.same_process(external_identity))

    def test_wrong_pid_start_cannot_signal_external(self):
        child, row = self.spawn('import time; time.sleep(30)')
        wrong = dict(row, start_ticks=row['start_ticks'] + 1)
        self.assertFalse(generation.same_process(wrong))
        with self.assertRaises(ValueError):
            generation.session_members(dict(row, session=os.getsid(0)))
        self.assertTrue(generation.same_process(row))

    def test_current_content_hash_rejects_same_size_restored_mtime(self):
        path = self.root / 'model'
        path.write_bytes(b'original')
        expected = recovery.record(path, nonempty=True)
        stat = path.stat()
        path.write_bytes(b'modified')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            generation.read_bound(expected)

    def test_all16_mapping_fixed_no_old_caps_or_foreign_owner(self):
        for method in ('alg', 'dymos'):
            value = plan(method)
            self.assertEqual(generation.fixed_allocations(value), recovery.allocations(method))
            self.assertEqual(sum(a['cap_seconds'] for a in generation.fixed_allocations(value))
                             + recovery.PRIOR[method][2] <= 86400, True)
        native = dict(method='base', allocations=[dict(a, cap_seconds=4500) for a in recovery.allocations('alg')])
        self.assertEqual(len(generation.fixed_allocations(native)), 16)
        for mutation in ('cap', 'foreign', 'missing'):
            bad = copy.deepcopy(native)
            if mutation == 'cap':
                bad['allocations'][0]['cap_seconds'] = 5000
            elif mutation == 'foreign':
                bad['allocations'][0]['node_rank'] = 3
            else:
                bad['allocations'].pop()
            with self.assertRaises(ValueError):
                generation.fixed_allocations(bad)

    def test_all16_and_both_immutable_cpu_owner_proofs_required_before_dispatch(self):
        import run_teacher_base as base
        objects = {}
        values = dict(schema=generation.SCHEMA, private=True, execution_id=recovery.EXECUTION_ID,
            GPU_authorized=True, GPU_used=False, quality_pass=False, sota_claim=False,
            cleanup_seconds=generation.CLEANUP_SECONDS, worker_margin_seconds=generation.WORKER_MARGIN_SECONDS,
            kill_seconds=generation.KILL_SECONDS,
            canary_shard=0, canary_job_count=12, retry_allowed=False, borrow_allowed=False,
            plans={}, allocations={}, owners={})
        models = [dict(asset=a, bytes=1, sha256='a' * 64) for a in generation.smoke.ASSETS]
        sources = [dict(path='cpu-protocol'), dict(path='cpu-test')]
        for method in generation.METHODS:
            p = plan(method if method != 'base' else 'alg')
            p.update(method=method, common_contract_frozen=True, source_files=sources,
                models=models, model_content_sha256='content', runtime=dict(files=[], python='CPU-TEST'))
            if method == 'base':
                p.pop('prior_generation', None)  # A fresh original base does not inherit the ALG prior fixture.
                p['allocations'] = [dict(a, cap_seconds=4500) for a in recovery.allocations('alg')]
                p['phase_seconds'] = dict(generation=72000, score=18000)
            values['plans'][method] = dict(path='plan-' + method, sha256='sha-' + method)
            objects['plan-' + method] = p
            rows = generation.fixed_allocations(p)
            values['allocations'][method] = rows
            values['owners'][method] = []
            for rank in sorted({r['node_rank'] for r in rows}):
                hostname = 'owner-' + str(rank)
                name = method + '-' + str(rank)
                values['owners'][method].append(dict(node_rank=rank, hostname=hostname, proof=dict(path='owner-' + name)))
                objects['owner-' + name] = dict(schema='teacher_generation_owner_cpu_proof_v1', GPU_used=False,
                    method=method, plan_sha256='sha-' + method, node_rank=rank, hostname=hostname,
                    allocations=[r for r in rows if r['node_rank'] == rank], source_files=sources,
                    model_content_sha256='content', models=models, runtime=p['runtime'],
                    probe=dict(path='probe-' + name), node_preflight=dict(path='pf-' + name),
                    local_root=str(recovery.LOCAL_ROOT), filesystem=dict(node_local=True, type='xfs', magic='58465342',
                        command=['stat', '-f', '-c', '%T %t', str(recovery.LOCAL_ROOT)],
                        stdout='xfs 58465342', stderr='', returncode=0, device=1))
                objects['probe-' + name] = dict(schema='teacher_real_local_cpu_probe_v1', GPU_used=False,
                    hostname=hostname, counter=40, unique_entries=40, errors=[], exit_codes=[0]*4,
                    read_checks=1, source=sources[0], test_source=sources[1])
                reservation = dict(schema='teacher_generation_owner_cpu_allocations_v1', plan_sha256='sha-' + method,
                    method=method, node_rank=rank, hostname=hostname, GPU_used=False,
                    allocations=[dict(a, status='reserved', GPU_used=False) for a in rows if a['node_rank'] == rank])
                local_path = recovery.LOCAL_ROOT / f'teacher_{method}_owner_cpu_{recovery.EXECUTION_ID}/reservations.json'
                objects['owner-' + name].update(reservation_snapshot=reservation, lock_identity=dict(device=1, inode=2),
                    local_reservations=dict(path=str(local_path), sha256=recovery.sha(recovery.encoded(reservation)),
                                            bytes=len(recovery.encoded(reservation))))
                objects['pf-' + name] = dict(schema='e3_subset_node_preflight_v1', GPU_used=False,
                    hostname=hostname, models=models, model_content_sha256='content')
                objects['owner-' + name].update(node_preflight_snapshot=objects['pf-' + name],
                    probe_snapshot=objects['probe-' + name])
        def peer_metadata(row):
            if row['path'].startswith(('pf-', 'probe-')):
                raise AssertionError('Peer-local preflight/probe path must not be accessed')
            return objects[row['path']]
        with mock.patch.object(generation, 'read_bound', side_effect=peer_metadata), \
             mock.patch.object(base, 'validate_contract'), mock.patch.object(recovery, 'validate_plan'):
            self.assertEqual(generation.validate_sidecar(values), values)
            for mutation in ('retry', 'missing-shard', 'foreign-owner', 'false-proof'):
                bad = copy.deepcopy(values)
                if mutation == 'retry':
                    bad['retry_allowed'] = True
                elif mutation == 'missing-shard':
                    bad['allocations']['alg'].pop()
                elif mutation == 'foreign-owner':
                    bad['owners']['alg'][0]['hostname'] = 'foreign'
                else:
                    bad['owners']['alg'][0]['proof'] = dict(path='owner-dymos-0')
                with self.assertRaises(ValueError):
                    generation.validate_sidecar(bad)

    def test_real_cpu_owner_own8_reservations_are_one_shot_and_inode_bound(self):
        p = plan('alg')
        digest = recovery.sha(recovery.encoded(p))
        path, snapshot, lock = generation.reserve_owner_cpu(p, digest, 1, self.root)
        self.assertEqual(generation.read_bound(recovery.record(path, nonempty=True)), snapshot)
        self.assertEqual([r['shard'] for r in snapshot['allocations']], list(range(8)))
        self.assertTrue(all(r['status'] == 'reserved' and r['GPU_used'] is False for r in snapshot['allocations']))
        stat = path.with_name(path.name + '.lock').stat()
        self.assertEqual(lock, dict(device=stat.st_dev, inode=stat.st_ino))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            generation.reserve_owner_cpu(p, digest, 1, self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_node_local_ledger_one_shot_only_own8(self):
        execution = dict(execution_id=recovery.EXECUTION_ID, allocations={'alg': recovery.allocations('alg')},
                         owners={'alg': [dict(node_rank=1, hostname=os.uname().nodename), dict(node_rank=2, hostname='peer')]})
        digest = recovery.sha(recovery.encoded(execution))
        path = generation.reserve_local(execution, digest, 'alg', 1, self.root)
        self.assertEqual(len(json.loads(path.read_bytes())['allocations']), 8)
        with self.assertRaises(ValueError):
            generation.reserve_local(execution, digest, 'alg', 1, self.root)
        with self.assertRaises(ValueError):
            generation.reserve_local(execution, digest, 'alg', 2, self.root)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            with generation.local_transaction(execution, digest, 'alg', 1, self.root) as value:
                value['allocations'][0]['shard'] = 8
        self.assertEqual(path.read_bytes(), before)
        lock = path.with_name(path.name + '.lock')
        inode = lock.stat().st_ino
        with generation.local_transaction(execution, digest, 'alg', 1, self.root) as value:
            value['allocations'][0]['status'] = 'running'
        self.assertEqual(lock.stat().st_ino, inode)

    def test_bad_ledger_preserved_before_owner_cleanup(self):
        execution = dict(execution_id=recovery.EXECUTION_ID, allocations={'alg': recovery.allocations('alg')},
                         owners={'alg': [dict(node_rank=1, hostname=os.uname().nodename)]})
        digest = recovery.sha(recovery.encoded(execution))
        path = generation.reserve_local(execution, digest, 'alg', 1, self.root)
        path.write_bytes(b'{bad')
        child, owner = self.spawn('import time; time.sleep(30)')
        try:
            with self.assertRaises(ValueError):
                with generation.local_transaction(execution, digest, 'alg', 1, self.root):
                    pass
        finally:
            generation.cleanup_owned(child, owner, time.monotonic() + 2)
        self.assertEqual(path.read_bytes(), b'{bad')
        self.assertFalse(generation.same_process(owner))

    def test_false_positive_old_and_zero_release_never_reduce_cap(self):
        p = plan('alg')
        digest = recovery.sha(recovery.encoded(p))
        for schema, seconds in [('teacher_recovery_parent_receipt_v1', 0),
                                ('teacher_recovery_parent_receipt_v1', 1),
                                (generation.RECEIPT_SCHEMA, 1)]:
            receipt = dict(p['allocations'][0], schema=schema, execution_id=recovery.EXECUTION_ID,
                method='alg', plan_sha256=digest, owner_pid=12345, hostname='fake', owned_tree_released=True,
                GPU_released=True, start=100, end=100 + seconds, elapsed_seconds=seconds)
            row = self.record_json(f'fake-{schema}-{seconds}.json', receipt)
            summary = recovery.collect_costs(p, digest, {0: row})
            self.assertEqual(summary['unknown_shards'], list(range(16)))
            self.assertAlmostEqual(summary['generation_charged_upper_seconds'], recovery.PRIOR['alg'][2] + 80000)
            with self.assertRaises((ValueError, KeyError)):
                generation.verify_parent_release(receipt, p)

    def test_reported_overcap_survives_bad_release(self):
        p = plan('dymos')
        digest = recovery.sha(recovery.encoded(p))
        receipt = dict(p['allocations'][0], schema=generation.RECEIPT_SCHEMA, execution_id=recovery.EXECUTION_ID,
            method='dymos', plan_sha256=digest, owner_pid=12345, hostname='fake', owned_tree_released=True,
            GPU_released=True, start=100, end=6000, elapsed_seconds=5900)
        row = self.record_json('overcap.json', receipt)
        summary = recovery.collect_costs(p, digest, {0: row})
        self.assertEqual(summary['over_cap_shards'], [0])
        self.assertTrue(summary['budget_exceeded'])
        self.assertAlmostEqual(summary['generation_charged_upper_seconds'], recovery.PRIOR['dymos'][2] + 15 * 4800 + 5900)

    def test_real_parent_supervision_records_release_after_cpu_child_exit(self):
        p = plan('alg')
        p['output_root'] = str(self.root)
        p['root'] = str(ROOT)
        execution = dict(execution_id=recovery.EXECUTION_ID, plans={'alg': self.record_json('plan.json', p)},
            allocations={'alg': recovery.allocations('alg')}, owners={'alg': [dict(node_rank=1, hostname=os.uname().nodename)]},
            common_execution_sha256='cpu-probe', score_bundle=dict(sha256='cpu-probe'))
        execution_row = self.record_json('execution.json', execution)
        generation.reserve_local(execution, execution_row['sha256'], 'alg', 1, self.root)
        manager = generation.identity(os.getpid())
        boundary = dict(gpu_uuid='CPU-TEST-NOT-A-GPU', manager=manager)
        folder = self.root / 'generation_shard0'
        code = "import json,time; from pathlib import Path; time.sleep(.15); " + repr('') + "; "
        code += "Path(" + repr(str(folder / 'worker.json')) + ").write_text(json.dumps({'success': True}))"
        command = ['timeout', '--signal=TERM', '--kill-after=1s', '2s', sys.executable, '-c', code]
        def query(_uuid):
            begin = time.monotonic()
            return dict(command=generation.QUERY_COMMAND, gpu_uuid=_uuid, start_monotonic=begin,
                start_epoch=time.time(), end_monotonic=time.monotonic(), returncode=0, stdout='', stderr='', pids=[])
        with mock.patch.object(generation, 'verify_execution', return_value=execution), \
             mock.patch.object(generation, 'current_local_content', return_value=(p, [])), \
             mock.patch.object(generation, 'manager_boundary', return_value=boundary), \
             mock.patch.object(generation, 'query_release', side_effect=query), \
             mock.patch.object(generation, 'command_for', return_value=command), \
             mock.patch.object(generation, 'verify_parent_release', return_value=True), \
             mock.patch.object(recovery, 'LOCAL_ROOT', self.root):
            try:
                path, _digest = generation.supervise(execution_row['path'], execution_row['sha256'], 'alg', 0, 1,
                    gpu_authorized=True, state_dir=self.root, local_root=self.root)
            except ValueError as exc:
                receipt = json.loads((folder / 'parent.json').read_bytes())
                self.fail(f"{exc}: error={receipt['error']}; exit={receipt['exit_code']}; worker={receipt['worker']}")
        receipt = json.loads(path.read_bytes())
        launch = generation.read_bound(receipt['launch'])
        supervised = generation.read_bound(receipt['supervised'])
        self.assertEqual(receipt['owner_pid'], os.getpid())
        self.assertEqual(receipt['session']['session'], receipt['session']['pid'])
        self.assertEqual(receipt['session']['ppid'], os.getpid())
        self.assertLessEqual(launch['start_monotonic'], supervised['observed_monotonic'])
        self.assertEqual(receipt['elapsed_seconds'], receipt['end_monotonic'] - launch['start_monotonic'])
        self.assertGreaterEqual(len(receipt['release_queries']), 2)
        self.assertTrue(receipt['GPU_released'])
        self.assertFalse(generation.same_process(receipt['session']))
        self.assertEqual(receipt['final_owned_processes'], [])
        # This tests the CPU process lifecycle only, never authenticates GPU evidence.
        self.assertEqual(receipt['release_queries'][0]['gpu_uuid'], 'CPU-TEST-NOT-A-GPU')

    def test_real_parent_user_term_cleans_owned_tree_even_with_corrupt_ledger(self):
        external, external_identity = self.spawn('import time; time.sleep(30)')
        p = plan('alg')
        p.update(output_root=str(self.root), root=str(ROOT))
        execution = dict(execution_id=recovery.EXECUTION_ID, plans={'alg': self.record_json('term-plan.json', p)},
            allocations={'alg': recovery.allocations('alg')}, owners={'alg': [dict(node_rank=1, hostname=os.uname().nodename)]},
            common_execution_sha256='CPU-TEST', score_bundle=dict(sha256='CPU-TEST'))
        row = self.record_json('term-execution.json', execution)
        ledger = generation.reserve_local(execution, row['sha256'], 'alg', 1, self.root)
        boundary = dict(gpu_uuid='CPU-TEST-NOT-A-GPU', manager=generation.identity(os.getpid()))
        code = "import os,signal,time; from pathlib import Path; time.sleep(.15); Path(" + repr(str(ledger))
        code += ").write_bytes(b'{bad'); os.kill(" + str(os.getpid()) + ", signal.SIGTERM); time.sleep(30)"
        command = ['timeout', '--signal=TERM', '--kill-after=1s', '2s', sys.executable, '-c', code]
        def query(uuid):
            begin = time.monotonic()
            return dict(command=generation.QUERY_COMMAND, gpu_uuid=uuid, start_monotonic=begin,
                start_epoch=time.time(), end_monotonic=time.monotonic(), returncode=0, stdout='', stderr='', pids=[])
        with mock.patch.object(generation, 'verify_execution', return_value=execution), \
             mock.patch.object(generation, 'current_local_content', return_value=(p, [])), \
             mock.patch.object(generation, 'manager_boundary', return_value=boundary), \
             mock.patch.object(generation, 'query_release', side_effect=query), \
             mock.patch.object(generation, 'command_for', return_value=command), \
             mock.patch.object(recovery, 'LOCAL_ROOT', self.root):
            with self.assertRaises(ValueError):
                generation.supervise(row['path'], row['sha256'], 'alg', 0, 1,
                    gpu_authorized=True, state_dir=self.root, local_root=self.root)
        receipt = json.loads((self.root / 'generation_shard0/parent.json').read_bytes())
        self.assertEqual(receipt['interrupted'], signal.SIGTERM)
        self.assertFalse(receipt['success'])
        self.assertEqual(receipt['final_owned_processes'], [])
        self.assertFalse(generation.same_process(receipt['session']))
        self.assertEqual(ledger.read_bytes(), b'{bad')
        self.assertTrue(generation.same_process(external_identity))

    def test_worker_cannot_self_claim_gpu_authorization_without_live_parent_chain(self):
        allocation = recovery.allocations('alg')[0]
        execution = dict(allocations={'alg': recovery.allocations('alg')})
        launch = dict(allocation=allocation, hostname=os.uname().nodename,
            owner=generation.identity(os.getpid()), gpu_uuid='NOT-A-GPU')
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='NOT-A-GPU', WORLD_SIZE='1', LOCAL_WORLD_SIZE='1'):
            with self.assertRaisesRegex(ValueError, 'authorized parent ancestry'):
                generation.verify_worker_scope(execution, launch, 'alg', 0, 1, self.root)

    def scoring_bundle_fixture(self):
        import score_teacher_devset as score
        roots = {name: '/missing-rank0-scoring/' + name for name in score.DEPENDENCIES}
        tools = {name: '/missing-rank0-scoring/bin/' + name for name in ('python', 'ffmpeg', 'ffprobe', 'timeout', 'nvidia_smi')}
        observation = dict(versions={name: 'CPU-TEST' for name in score.RUNTIME_PACKAGES},
            dependency_roots=roots, library_roots={'stdlib': '/missing-rank0-scoring/stdlib'},
            loaded_modules={}, python_invocation=tools['python'], python_prefix='/missing-rank0-scoring', base_prefix='/usr')
        row = lambda path: dict(path=path, bytes=1, sha256='a' * 64)
        return dict(schema=score.BUNDLE_SCHEMA, private=True, metric_protocol='score_e1_all8_gt_v1',
            dimensions=score.e1.DIMS, gt_keys=list(score.e1.GT_KEYS),
            sources=[recovery.record(p, nonempty=True) for p in score.required_sources()],
            dependency_roots=roots, dependencies=[row('/missing-rank0-scoring/dependency')],
            model_roots={'vbench': '/missing-rank0-scoring/models', 'dreamsim': '/missing-rank0-scoring/dreamsim'},
            models=[row('/missing-rank0-scoring/models/weights')],
            runtime=dict(versions=observation['versions'], files=[row(path) for path in tools.values()],
                tools={k: tools[k] for k in ('python', 'ffmpeg', 'ffprobe')}, observation=observation,
                library_roots=observation['library_roots'], library_files=[row('/missing-rank0-scoring/stdlib/os.py')]),
            execution=dict(python=tools['python'], vbench_repo='/missing-rank0-scoring/vbench',
                tools=tools, offline=True, gpu_dimensions=score.e1.DIMS, gt_K=12,
                vbench_local=True, cache_dir='/missing-rank0-scoring/models'))

    def scoring_owner_fixture(self, hostname='rank0-scoring-owner'):
        bundle = self.scoring_bundle_fixture()
        bundle_row = self.record_json('host-score-bundle.json', bundle)
        proof = dict(schema='teacher_generation_scoring_owner_proof_v1', private=True, GPU_used=False,
            node_rank=0, hostname=hostname, bundle=bundle_row, content=generation.scoring_owner_content(bundle),
            observed_epoch=time.time(), observed_monotonic=time.monotonic(),
            verifier_sources=[recovery.record(p, nonempty=True) for p in generation.scoring_owner_sources()])
        proof_row = self.record_json('host-score-owner.json', proof)
        return bundle, dict(score_bundle=bundle_row, scoring_owner=dict(node_rank=0, hostname=hostname, proof=proof_row))

    def test_peer_missing_scoring_environment_uses_approved_owner_metadata_only(self):
        import prepare_teacher_comparison as comparison
        import score_teacher_devset as score
        bundle, value = self.scoring_owner_fixture()
        self.assertFalse(Path(bundle['execution']['python']).exists())
        self.assertFalse(Path(bundle['model_roots']['vbench']).exists())
        value.update(plans={m: self.record_json('peer-' + m + '.json', dict(root=str(ROOT), method=m)) for m in generation.METHODS},
            comparison=dict(path='approved-comparison', sha256='c' * 64), common_execution_sha256='d' * 64,
            retirement={'alg': {}, 'dymos': {}},
            source_files=[recovery.record(p, nonempty=True) for p in generation.sources(ROOT)])
        execution = self.record_json('peer-execution.json', value)
        common = dict(plans=value['plans'], common_execution_sha256=value['common_execution_sha256'], score_bundle=value['score_bundle'])
        with mock.patch.object(generation, 'validate_sidecar', side_effect=lambda v: v), \
             mock.patch.object(generation, 'validate_authorization'), mock.patch.object(generation, 'verify_retirement'), \
             mock.patch.object(comparison, 'verify_metadata', return_value=common, create=True) as metadata, \
             mock.patch.object(comparison, 'verify', side_effect=AssertionError('peer read rank0 assets')), \
             mock.patch.object(score, 'verify_bundle', side_effect=AssertionError('peer read unavailable scoring environment')):
            self.assertEqual(generation.verify_execution(execution['path'], execution['sha256']), value)
        metadata.assert_called_once_with('approved-comparison', 'c' * 64)

    def test_scoring_owner_proof_is_mandatory_and_content_bound_not_a_boolean(self):
        _bundle, value = self.scoring_owner_fixture()
        generation.verify_scoring_owner(value)
        for bad in ({}, dict(value, scoring_owner=None), dict(value, scoring_owner=dict(node_rank=0, hostname='wrong-host', proof=value['scoring_owner']['proof']))):
            with self.assertRaises(ValueError):
                generation.verify_scoring_owner(bad)
        proof = generation.read_bound(value['scoring_owner']['proof'])
        proof['content']['content_sha256']['models'] = 'f' * 64
        value['scoring_owner']['proof'] = self.record_json('false-owner.json', proof)
        with self.assertRaises(ValueError):
            generation.verify_scoring_owner(value)

    def test_legacy_scoring_inventory_never_passes_real_generation_owner_gate(self):
        bundle = self.scoring_bundle_fixture()
        del bundle['execution']
        with self.assertRaisesRegex(ValueError, 'Actual scoring execution'):
            generation.scoring_owner_content(bundle)

    def test_rank0_owner_proof_constructor_and_revalidation_keep_actual_full_content_gate(self):
        import score_teacher_devset as score
        import run_teacher_scoring as scoring
        bundle = self.scoring_bundle_fixture()
        row = self.record_json('rank0-bundle.json', bundle)
        with mock.patch.object(score, 'verify_bundle', return_value=bundle) as full, \
             mock.patch.object(scoring, 'validate_execution_bundle', return_value=bundle['execution']) as actual:
            path, digest = generation.prepare_scoring_owner_proof(row['path'], row['sha256'], self.root / 'rank0-proof.json')
            proof = generation.read_bound(recovery.record(path, nonempty=True))
            value = dict(score_bundle=row, scoring_owner=dict(node_rank=0, hostname=os.uname().nodename,
                proof=dict(recovery.record(path, nonempty=True), sha256=digest)))
            self.assertEqual(generation.verify_scoring_owner(value, full_content=True), proof)
        self.assertEqual(full.call_count, 2)
        self.assertTrue(all(c.kwargs['verify_content'] is True for c in full.call_args_list))
        self.assertEqual(actual.call_count, 2)

    def test_own_nine_models_still_full_hash_with_peer_metadata_not_rank0_mtime(self):
        import verify_teacher_history as history
        smoke = generation.smoke
        local = self.root / 'own-node'
        models = []
        for asset in smoke.ASSETS:
            path = local / 'models' / asset
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original')
            models.append(dict(recovery.record(path, nonempty=True), asset=asset))
        content_sha = recovery.sha(recovery.encoded([{k: r[k] for k in ('asset', 'bytes', 'sha256')} for r in models]))
        pf = self.record_json('own-pf.json', dict(schema='e3_subset_node_preflight_v1', GPU_used=False,
            hostname=os.uname().nodename, models=models, model_content_sha256=content_sha))
        source = self.root / 'own-runtime'
        source.write_bytes(b'CPU-TEST')
        runtime = dict(files=[recovery.record(source, nonempty=True)])
        proof = self.record_json('own-cpu-proof.json', dict(node_preflight=pf, runtime=runtime))
        foreign = [dict(r, path='/absent-rank0/' + r['asset'], mtime_ns=1) for r in models]
        p = dict(root=str(ROOT), models=foreign, runtime=runtime, source_files=[], cases=[], inputs=[], cases_file={})
        p_row = self.record_json('own-plan.json', p)
        execution = dict(plans={'alg': p_row}, owners={'alg': [dict(node_rank=2, hostname=os.uname().nodename, proof=proof)]}, source_files=[])
        with mock.patch.object(history, 'inventory', return_value=dict(matches=[])), \
             mock.patch.object(smoke, 'LOCAL_ROOT', local), mock.patch.object(smoke, 'runtime_records', return_value=runtime), \
             mock.patch.object(generation.dev, 'frozen_cases', return_value=(b'', [], [], {})):
            checked_plan, actual_models = generation.current_local_content(execution, 'alg', 2)
            self.assertEqual(checked_plan, p)
            self.assertEqual(actual_models, models)
            runtime_stat = source.stat()
            source.write_bytes(b'DRIFTING')
            os.utime(source, ns=(runtime_stat.st_atime_ns, runtime_stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'Evidence content drift'):
                generation.current_local_content(execution, 'alg', 2)
            source.write_bytes(b'CPU-TEST')
            os.utime(source, ns=(runtime_stat.st_atime_ns, runtime_stat.st_mtime_ns))
            path = Path(models[0]['path'])
            stat = path.stat()
            path.write_bytes(b'modified')
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'hash or metadata drift'):
                generation.current_local_content(execution, 'alg', 2)

    def test_candidate_remaining_requires_worst_case_130_second_tail_projection(self):
        import run_teacher_base as base
        for method, times, load, elapsed, blocked in (
            ('alg', [400.] * 11 + [100.], 100., 4700., True),
            ('alg', [100.] * 12, 100., 1450., False),
            ('dymos', [(4800 - 130 - 100) / 12] * 11 + [100.], 100., 4500., True),
        ):
            p = plan(method)
            p.update(output_root=str(self.root / method), jobs=[dict(index=i) for i in range(188)])
            base_plan = dict(output_root=str(self.root / 'base'))
            objects = dict(candidate=p, native=base_plan)
            native_folder = generation.shard_folder(base_plan, 0)
            native_folder.mkdir(parents=True, exist_ok=True)
            recovery.durable_write(native_folder / 'parent.json', dict(success=True))
            candidate_folder = generation.shard_folder(p, 0)
            candidate_folder.mkdir(parents=True, exist_ok=True)
            rows = []
            for index, seconds in enumerate(times):
                name = method + '-' + str(index)
                objects[name] = dict(elapsed_seconds=seconds)
                rows.append(dict(job=p['jobs'][index * 16], parameters_before={'same': 'SHA'}, case_receipt=dict(path=name)))
            objects['worker-' + method] = dict(success=True, error=None, cases=rows, load_seconds=load)
            recovery.durable_write(candidate_folder / 'parent.json', dict(success=True, exit_code=0, error=None,
                interrupted=None, timed_out=False, elapsed_seconds=elapsed, worker=dict(path='worker-' + method)))
            execution = dict(plans={'base': dict(path='native'), method: dict(path='candidate', sha256='CPU-SHA')},
                allocations={method: recovery.allocations(method)})
            with mock.patch.object(generation, 'read_bound', side_effect=lambda row: objects[row['path']]), \
                 mock.patch.object(generation, 'verify_parent_release', return_value=True), \
                 mock.patch.object(generation, 'verify_case'), \
                 mock.patch.object(base, 'verify_native_canary_admission', return_value=True):
                if blocked:
                    with self.assertRaisesRegex(ValueError, 'worst-case|projection|tail'):
                        generation.remaining_admission(execution, method)
                else:
                    generation.remaining_admission(execution, method)

    def test_popen_identity_failure_reaps_real_owned_child_and_preserves_external(self):
        external, external_identity = self.spawn('import time; time.sleep(30)')
        p = plan('alg')
        p.update(output_root=str(self.root), root=str(ROOT))
        execution = dict(execution_id=recovery.EXECUTION_ID, plans={'alg': self.record_json('identity-plan.json', p)},
            allocations={'alg': recovery.allocations('alg')}, owners={'alg': [dict(node_rank=1, hostname=os.uname().nodename)]},
            common_execution_sha256='CPU-TEST', score_bundle=dict(sha256='CPU-TEST'))
        row = self.record_json('identity-execution.json', execution)
        generation.reserve_local(execution, row['sha256'], 'alg', 1, self.root)
        real_identity, real_popen, children = generation.identity, subprocess.Popen, []
        def popen(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        initialized = self.root / 'identity-child-ready'
        def identity(pid):
            if pid == os.getpid():
                return real_identity(pid)
            until = time.monotonic() + 2
            while not initialized.exists() and time.monotonic() < until:
                time.sleep(.01)
            raise OSError('injected child identity lookup failure')
        boundary = dict(gpu_uuid='CPU-TEST-NOT-A-GPU', manager=real_identity(os.getpid()))
        code = "import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        code += "Path(" + repr(str(initialized)) + ").touch(); time.sleep(30)"
        command = [sys.executable, '-c', code]
        try:
            with mock.patch.object(generation, 'verify_execution', return_value=execution), \
                 mock.patch.object(generation, 'current_local_content', return_value=(p, [])), \
                 mock.patch.object(generation, 'manager_boundary', return_value=boundary), \
                 mock.patch.object(generation, 'command_for', return_value=command), \
                 mock.patch.object(generation, 'identity', side_effect=identity), \
                 mock.patch.object(generation.subprocess, 'Popen', side_effect=popen), \
                 mock.patch.object(generation, 'KILL_SECONDS', .1), \
                 mock.patch.object(recovery, 'LOCAL_ROOT', self.root):
                with self.assertRaises(ValueError):
                    generation.supervise(row['path'], row['sha256'], 'alg', 0, 1,
                        gpu_authorized=True, state_dir=self.root, local_root=self.root)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll(), 'Spawned own child was left alive after failed identity lookup')
            self.assertEqual(children[0].returncode, -signal.SIGKILL)
            receipt = json.loads((self.root / 'generation_shard0/parent.json').read_bytes())
            self.assertFalse(receipt['success'])
            self.assertFalse(receipt['GPU_released'])
            self.assertIsNone(receipt['session'])
            self.assertTrue(generation.same_process(external_identity))
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=3)

    def test_retirement_delegates_to_independent_pinned_history_verifier(self):
        import verify_teacher_history as history
        p = plan('alg')
        row = self.record_json('history-adapter.json', dict(schema='teacher_generation_retirement_v1'))
        proof = generation.read_bound(row)
        with mock.patch.object(history, 'verify_retirement', return_value=proof) as verifier:
            self.assertEqual(generation.verify_retirement(row, 'alg', p), proof)
        verifier.assert_called_once_with(proof, p)

    def test_arbitrary_positive_retirement_records_cannot_pass_real_history_verifier(self):
        import verify_teacher_history as history
        p = plan('alg')
        p['root'] = str(ROOT)
        text = self.root / 'arbitrary.txt'
        text.write_text('not authenticated original tool logs')
        arbitrary = recovery.record(text, nonempty=True)
        p['prior_generation'].update(audit=arbitrary, old_plan=arbitrary, old_ledger=arbitrary)
        proof = dict(schema='teacher_generation_retirement_v1', method='alg', execution_id=recovery.EXECUTION_ID,
            old_actual_seconds=None, old_generation_upper_seconds=recovery.PRIOR['alg'][2],
            audit=arbitrary, old_plan=arbitrary, old_ledger=arbitrary, termination_verifier_source=arbitrary,
            source_files=[arbitrary], shards=[dict(shard=shard, retired=True, source_verified=True,
                complete_tool_logs=True, records=[arbitrary]) for shard in range(16)])
        row = self.record_json('arbitrary-retirement.json', proof)
        # No mock: this invokes the real pinned index/raw logs and all16 prior upper verifier.
        with self.assertRaises((ValueError, KeyError)):
            generation.verify_retirement(row, 'alg', p)

    def test_current_run_boundary_rejects_old_live_inventory_before_content_or_cuda(self):
        import verify_teacher_history as history
        with mock.patch.object(history, 'inventory', return_value=dict(matches=[dict(pid=123)])) as inventory, \
             mock.patch.object(generation, 'read_bound') as read:
            with self.assertRaisesRegex(ValueError, 'Old generation is still live'):
                generation.current_local_content({}, 'alg', 1)
        inventory.assert_called_once_with(1)
        read.assert_not_called()

    def test_reused_session_leader_never_reaches_any_cleanup_signal(self):
        child = mock.Mock(pid=9000001)
        owner = dict(pid=9000001, session=9000001, start_ticks=100)
        new_owner = dict(pid=9000001, session=9000001, start_ticks=200, state='S')
        with mock.patch.object(generation, 'identity', return_value=new_owner), \
             mock.patch.object(generation.os, 'kill') as kill:
            with self.assertRaisesRegex(ValueError, 'identity was reused'):
                generation.cleanup_owned(child, owner, time.monotonic() + 1)
        kill.assert_not_called()
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_unbound_auth_owner_retirement_completion_are_blocked(self):
        with self.assertRaises(ValueError):
            generation.validate_sidecar(dict(schema=generation.SCHEMA, private=True, execution_id=recovery.EXECUTION_ID,
                GPU_authorized=False, GPU_used=False, quality_pass=False, sota_claim=False))
        fake = self.record_json('retirement.json', dict(schema='teacher_generation_retirement_v1', method='alg',
            execution_id=recovery.EXECUTION_ID, old_actual_seconds=0, old_generation_upper_seconds=0))
        with self.assertRaises(ValueError):
            generation.verify_retirement(fake, 'alg', dict(plan(), root=str(ROOT)))
        fake = self.record_json('completion.json', dict(schema='teacher_generation_completion_v1', method='alg',
            generation_complete=True, quality_pass=False, sota_claim=False, common_execution_sha256='fake', score_bundle_sha256='fake'))
        with self.assertRaises(ValueError):
            generation.verify_generation_completion(fake['path'], fake['sha256'], method='alg',
                common_execution_sha256='approved', score_bundle_sha256='approved')


if __name__ == '__main__':
    unittest.main()
