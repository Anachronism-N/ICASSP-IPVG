"""CPU mocks: monitor-only never spawns a CUDA holder, including idle/pause/error."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gpu_hold_monitor_test_module', ROOT / 'scripts/gpu_hold.py')
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


class StopAfter:
    def __init__(self, loops=3):
        self.loops = loops
        self.stopped = False

    def is_set(self):
        return self.stopped or self.loops <= 0

    def wait(self, seconds):
        self.loops -= 1

    def set(self):
        self.stopped = True


class MonitorTests(unittest.TestCase):
    def exercise(self, *, monitor=True, external=None, paused=False, query_error=False, capacity=40960):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            if paused:
                (folder / 'pause.0').touch()
            args = argparse.Namespace(memory_gib=30, interval=0, idle_seconds=0, monitor_only=monitor)
            mapping = {0: dict(uuid='GPU-CPU-fixture', total_mib=capacity, used_mib=0)}
            states = []
            child = mock.Mock(pid=12345, returncode=None)
            child.poll.return_value = None
            processes = {mapping[0]['uuid']: set(external or [])}
            with mock.patch.object(gpu, 'query_gpus', return_value=mapping),                  mock.patch.object(gpu, 'query_processes', side_effect=OSError('fixture query') if query_error else None,
                                   return_value=processes),                  mock.patch.object(gpu.threading, 'Event', return_value=StopAfter()),                  mock.patch.object(gpu.signal, 'signal'),                  mock.patch.object(gpu, 'write_state', side_effect=lambda folder, value: states.append(copy.deepcopy(value))),                  mock.patch.object(gpu.subprocess, 'Popen', return_value=child) as spawn,                  mock.patch.object(gpu, 'stop_child') as stop:
                gpu.supervisor(args, folder, [0])
            return states, spawn.call_count, stop.call_count

    def test_idle_monitor_has_no_holder_and_keeps_uuid_identity(self):
        states, spawn, stopped = self.exercise()
        self.assertEqual(spawn, 0)
        self.assertEqual(stopped, 0)
        running = [s for s in states if s['running']]
        self.assertEqual(len(running), 3)
        for value in running:
            self.assertTrue(value['monitor_only'])
            self.assertTrue(value['query_ok'])
            self.assertEqual(value['gpus']['0']['gpu_uuid'], 'GPU-CPU-fixture')
            self.assertIsNone(value['gpus']['0']['holder_pid'])
            self.assertEqual(value['gpus']['0']['memory_gib_target'], 0)
        self.assertFalse(states[-1]['running'])

    def test_external_processes_remain_visible_and_are_not_stopped(self):
        states, spawn, stopped = self.exercise(external=[98765])
        self.assertEqual((spawn, stopped), (0, 0))
        self.assertEqual(states[0]['gpus']['0']['external_pids'], [98765])

    def test_pause_acknowledged_without_spawning(self):
        states, spawn, stopped = self.exercise(paused=True)
        self.assertEqual((spawn, stopped), (0, 0))
        self.assertTrue(states[0]['gpus']['0']['paused'])
        self.assertIsNone(states[0]['gpus']['0']['holder_pid'])

    def test_query_failure_no_spawn_and_not_claimed_healthy(self):
        states, spawn, stopped = self.exercise(query_error=True)
        self.assertEqual((spawn, stopped), (0, 0))
        self.assertFalse(states[0]['query_ok'])
        self.assertIn('fixture query', states[0]['error'])

    def test_monitor_does_not_require_holder_memory_capacity(self):
        states, spawn, _ = self.exercise(capacity=8192)
        self.assertEqual(spawn, 0)
        self.assertTrue(states[0]['query_ok'])

    def test_existing_default_reservation_behavior_preserved(self):
        _, spawn, stopped = self.exercise(monitor=False)
        self.assertEqual(spawn, 1)
        self.assertEqual(stopped, 1)

    def test_monitor_option_cannot_be_used_for_cuda_worker(self):
        with mock.patch.object(sys, 'argv', ['gpu_hold.py', '--action', 'worker', '--gpu', '0', '--monitor-only']),              mock.patch.object(gpu, 'worker') as worker,              mock.patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                gpu.main()
        worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
