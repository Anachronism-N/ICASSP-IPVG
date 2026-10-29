"""CPU-only tests of recovery budget and actual node-local lock/write primitives."""
import contextlib
import copy
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_teacher_recovery as recovery


def plan(method='alg'):
    return dict(schema=recovery.SCHEMA, execution_id=recovery.EXECUTION_ID, method=method,
                GPU_authorized=False, GPU_used=False, status='BLOCKED', old_reservations_retired=False,
                nshards=16, allocations=recovery.allocations(method),
                prior_generation=dict(actual_seconds=None, upper_seconds=recovery.PRIOR[method][2],
                                      audit=dict(sha256=recovery.PRIOR[method][1])),
                phase_seconds=dict(recovery.dev.PHASE_SECONDS), total_seconds=recovery.dev.TOTAL_SECONDS,
                video_count=188, seeds=[0, 1], cases_file=dict(sha256=recovery.dev.CASES_SHA256),
                call=dict(recovery.dev.SPEC), parameters=dict(recovery.dev.PARAMS[method]),
                historical_base_reuse_allowed=False, smoke_charged_seconds=1560.)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.local = Path(self.temp.name)
        self.plan = plan()
        self.digest = recovery.sha(recovery.encoded(self.plan))

    def test_cost_caps_include_old_generation(self):
        for method in ('alg', 'dymos'):
            value = plan(method)
            recovery.validate_plan(value)
            report = recovery.collect_costs(value, recovery.sha(recovery.encoded(value)), {})
            expected = recovery.PRIOR[method][2] + 16 * recovery.CAPS[method]
            self.assertAlmostEqual(report['generation_charged_upper_seconds'], expected)
            self.assertIsNone(report['new_actual_seconds'])
            self.assertEqual(report['unknown_shards'], list(range(16)))
            self.assertFalse(report['generation_complete'])
            self.assertFalse(report['budget_exceeded'])

    def test_cannot_reset_old_unknown_or_phase_caps(self):
        mutations = [('actual_seconds', 0), ('upper_seconds', 0)]
        for key, value in mutations:
            changed = copy.deepcopy(self.plan)
            changed['prior_generation'][key] = value
            with self.assertRaises(ValueError):
                recovery.validate_plan(changed)
        changed = copy.deepcopy(self.plan)
        changed['phase_seconds']['generation'] += 1
        with self.assertRaises(ValueError):
            recovery.validate_plan(changed)

    def test_old_5400_caps_and_authorization_rejected(self):
        for key, value in [('GPU_authorized', True), ('old_reservations_retired', True), ('execution_id', 'recovery_a2')]:
            changed = copy.deepcopy(self.plan)
            changed[key] = value
            with self.assertRaises(ValueError):
                recovery.validate_plan(changed)
        changed = copy.deepcopy(self.plan)
        for item in changed['allocations']:
            item['cap_seconds'] = 5400
        with self.assertRaises(ValueError):
            recovery.validate_plan(changed)

    def test_local_owns_only_eight_and_refuses_reset(self):
        path = recovery.reserve_local(self.plan, self.digest, 1, self.local)
        value = json.loads(path.read_bytes())
        self.assertEqual([a['shard'] for a in value['allocations']], list(range(8)))
        with recovery.local_transaction(path, self.plan, self.digest, 1, self.local) as value:
            value['cpu_revision'] += 1
        self.assertEqual(json.loads(path.read_bytes())['cpu_revision'], 1)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            recovery.reserve_local(self.plan, self.digest, 1, self.local)
        with self.assertRaises(ValueError):
            with recovery.local_transaction(path, self.plan, self.digest, 1, self.local) as value:
                value['allocations'][0]['node_rank'] = 2
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            with recovery.local_transaction(path, self.plan, self.digest, 2, self.local):
                pass

    def test_empty_namespace_source_is_not_an_empty_model_asset(self):
        source = self.local / '__init__.py'
        source.write_bytes(b'')
        expected = recovery.record(source)
        self.assertEqual(recovery.check_content(expected)['bytes'], 0)
        model = self.local / 'weights.safetensors'
        model.write_bytes(b'')
        with self.assertRaises(ValueError):
            recovery.check_content(recovery.record(model))
        source.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            recovery.check_content(expected)

    def test_same_size_mtime_content_drift_rejected(self):
        path = self.local / 'source'
        path.write_bytes(b'original')
        expected = recovery.record(path, nonempty=True)
        stat = path.stat()
        path.write_bytes(b'modified')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with self.assertRaises(ValueError):
            recovery.check_content(expected)

    def test_transaction_preserves_corrupt_bytes(self):
        path = recovery.reserve_local(self.plan, self.digest, 1, self.local)
        path.write_bytes(b'{corrupt')
        with self.assertRaises(ValueError):
            with recovery.local_transaction(path, self.plan, self.digest, 1, self.local):
                pass
        self.assertEqual(path.read_bytes(), b'{corrupt')

    def test_atomic_replace_failure_preserves_original_and_cleans_temp(self):
        path = self.local / 'state.json'
        path.write_bytes(b'old')
        with mock.patch.object(recovery.os, 'replace', side_effect=OSError('injected')):
            with self.assertRaises(OSError):
                recovery.durable_write(path, {'new': True})
        self.assertEqual(path.read_bytes(), b'old')
        self.assertEqual(list(self.local.glob('*.tmp')), [])

    def test_lock_inode_persistent_and_timeout(self):
        path = self.local / 'allocations.json'
        with recovery.local_locked(path, self.local):
            inode = path.with_name(path.name + '.lock').stat().st_ino
            with self.assertRaises(TimeoutError):
                with recovery.local_locked(path, self.local, timeout_seconds=0):
                    pass
        with recovery.local_locked(path, self.local):
            self.assertEqual(path.with_name(path.name + '.lock').stat().st_ino, inode)

    def test_symlink_and_wrong_local_root_rejected(self):
        target = self.local / 'target'
        target.write_bytes(b'x')
        link = self.local / 'link'
        link.symlink_to(target)
        for path in (link, Path('/not-the-approved-root/allocations.json')):
            with self.assertRaises(ValueError):
                with recovery.local_locked(path, self.local):
                    pass

    def test_real_four_worker_local_counter(self):
        path = self.local / 'probe.json'
        recovery.durable_write(path, dict(counter=0, entries=[], GPU_used=False))
        processes = [multiprocessing.get_context('fork').Process(target=recovery.cpu_probe,
                     args=(path, self.local, 10, worker)) for worker in range(4)]
        for p in processes:
            p.start()
        for p in processes:
            p.join(20)
            if p.is_alive():
                p.terminate()
                p.join(5)
            self.assertEqual(p.exitcode, 0)
        value = json.loads(path.read_bytes())
        self.assertEqual(value['counter'], 40)
        self.assertEqual(len({tuple(x) for x in value['entries']}), 40)

    def test_collector_bound_release_and_known_overcap_are_not_truncated(self):
        fixed = self.plan['allocations'][0]
        elapsed = fixed['cap_seconds'] + 100
        receipt = dict(fixed, schema='teacher_recovery_parent_receipt_v1', execution_id=recovery.EXECUTION_ID,
                       method='alg', plan_sha256=self.digest, owner_pid=12345, hostname='cpu-fixture',
                       owned_tree_released=True, GPU_released=True, start=100, end=100 + elapsed,
                       elapsed_seconds=elapsed)
        path = self.local / 'receipt.json'
        recovery.durable_write(path, receipt)
        report = recovery.collect_costs(self.plan, self.digest, {0: recovery.record(path, nonempty=True)})
        self.assertEqual(report['new_actual_known_seconds'], 0)
        self.assertEqual(report['unknown_shards'], list(range(16)))
        self.assertGreaterEqual(report['generation_charged_upper_seconds'],
                                self.plan['prior_generation']['upper_seconds'] + 16 * recovery.CAPS['alg'] + 100)
        self.assertEqual(report['over_cap_shards'], [0])
        self.assertTrue(report['budget_exceeded'])
        self.assertIsNone(report['new_actual_seconds'])
        self.assertFalse(report['generation_complete'])

    def test_bad_receipt_retains_cap_and_digest_drift_rejected(self):
        path = self.local / 'receipt.json'
        path.write_bytes(b'{bad')
        report = recovery.collect_costs(self.plan, self.digest, {0: recovery.record(path, nonempty=True)})
        self.assertEqual(len(report['unknown_shards']), 16)
        with self.assertRaises(ValueError):
            recovery.collect_costs(self.plan, 'wrong', {})
        with self.assertRaises(ValueError):
            recovery.reserve_local(self.plan, 'wrong', 1, self.local)

    def test_fabricated_zero_receipts_cannot_release_static_quotas(self):
        receipts = {}
        for fixed in self.plan['allocations']:
            path = self.local / f"fake{fixed['shard']}.json"
            value = dict(fixed, schema='teacher_recovery_parent_receipt_v1', execution_id=recovery.EXECUTION_ID,
                         method='alg', plan_sha256=self.digest, owner_pid=12345, hostname='not-attested',
                         owned_tree_released=True, GPU_released=True, start=0, end=0, elapsed_seconds=0)
            recovery.durable_write(path, value)
            receipts[fixed['shard']] = recovery.record(path, nonempty=True)
        result = recovery.collect_costs(self.plan, self.digest, receipts)
        self.assertEqual(result['unknown_shards'], list(range(16)))
        self.assertIsNone(result['new_actual_seconds'])
        self.assertFalse(result['allocation_evidence_complete'])
        self.assertEqual(result['generation_charged_upper_seconds'], self.plan['prior_generation']['upper_seconds'] + 80000)

    def test_positive_receipt_booleans_are_not_release_evidence(self):
        fixed = self.plan['allocations'][0]
        path = self.local / 'positive.json'
        recovery.durable_write(path, dict(fixed, schema='teacher_recovery_parent_receipt_v1',
            execution_id=recovery.EXECUTION_ID, method='alg', plan_sha256=self.digest, owner_pid=12345,
            hostname='not-attested', owned_tree_released=True, GPU_released=True, start=100, end=101, elapsed_seconds=1))
        result = recovery.collect_costs(self.plan, self.digest, {0: recovery.record(path, nonempty=True)})
        self.assertEqual(len(result['unknown_shards']), 16)
        self.assertEqual(result['generation_charged_upper_seconds'], self.plan['prior_generation']['upper_seconds'] + 80000)

    def test_explicit_execution_adapter_binds_immutable_cpu_plan_without_promoting_it(self):
        import run_teacher_generation as generation
        path = self.local / 'plan.blocked.json'
        recovery.dev.exclusive(path, self.plan)
        before = path.read_bytes()
        execution = dict(plans={'alg': recovery.record(path, nonempty=True)})
        arguments = ['--run', '--plan', str(path), '--approved-plan-sha256', self.digest,
            '--execution', str(self.local / 'execution.json'), '--approved-execution-sha256', 'a' * 64,
            '--rank', '1', '--shard', '0', '--gpu-authorized']
        with mock.patch.object(recovery, 'verify', return_value=self.plan), \
             mock.patch.object(generation, 'verify_execution', return_value=execution), \
             mock.patch.object(generation, 'supervise', return_value=(self.local / 'parent.json', 'b' * 64)) as supervise, \
             contextlib.redirect_stdout(io.StringIO()):
            recovery.main(arguments)
        supervise.assert_called_once_with(self.local / 'execution.json', 'a' * 64, 'alg', 0, 1,
            gpu_authorized=True, state_dir='/dockerdata/wm_logs/gpu_hold')
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(self.plan['GPU_authorized'])
        self.assertFalse(self.plan['old_reservations_retired'])
        self.assertEqual(self.plan['status'], 'BLOCKED')

    def test_cli_run_fails_before_preparing_anything(self):
        with self.assertRaisesRegex(ValueError, 'GPU recovery blocked'):
            recovery.main(['--run', '--method', 'alg'])


def cluster_probe(folder):
    folder = Path(folder).absolute()
    recovery.local_path(folder, recovery.LOCAL_ROOT)
    folder.mkdir(mode=0o700)
    path = folder / 'counter.json'
    recovery.durable_write(path, dict(counter=0, entries=[], GPU_used=False))
    command = [sys.executable, str(Path(__file__).resolve()), '--probe-worker', str(path)]
    children = [subprocess.Popen(command + [str(worker)], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True) for worker in range(4)]
    deadline = time.monotonic() + 45
    read_checks, errors, outputs = 0, [], []
    try:
        while any(p.poll() is None for p in children):
            value = json.loads(path.read_bytes())
            if value['counter'] != len(value['entries']):
                raise ValueError('Counter/entries drift')
            read_checks += 1
            if time.monotonic() >= deadline:
                raise TimeoutError('Bounded CPU probe deadline')
            time.sleep(.002)
        for p in children:
            out, err = p.communicate(timeout=2)
            outputs.append(json.loads(out) if p.returncode == 0 else {})
            if p.returncode != 0 or err:
                errors.append(dict(exit_code=p.returncode, stderr=err))
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=3)
    value = json.loads(path.read_bytes())
    result = dict(schema='teacher_real_local_cpu_probe_v1', hostname=os.uname().nodename,
                  path=str(path), counter=value['counter'], unique_entries=len({tuple(x) for x in value['entries']}),
                  read_checks=read_checks, errors=errors, exit_codes=[p.returncode for p in children],
                  record=recovery.record(path, nonempty=True), source=recovery.record(recovery.__file__, nonempty=True),
                  test_source=recovery.record(__file__, nonempty=True), workers=outputs, GPU_used=False)
    recovery.dev.exclusive(folder / 'proof.json', result)
    print(json.dumps(result))
    if errors or result['counter'] != 40 or result['unique_entries'] != 40:
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--cluster-probe':
        cluster_probe(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == '--probe-worker':
        print(json.dumps(recovery.cpu_probe(Path(sys.argv[2]), recovery.LOCAL_ROOT, 10, int(sys.argv[3]))))
    else:
        unittest.main()
