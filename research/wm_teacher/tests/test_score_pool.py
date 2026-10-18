import builtins
from collections import Counter
import importlib.util
import io
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "score_pool", Path(__file__).resolve().parents[1] / "scripts/score_pool.py")
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


class ScorePoolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.logs = Path(temporary.name)
        listdir = os.listdir
        for patcher in (mock.patch.object(pool, "L", str(self.logs)),
                        mock.patch.object(pool.os, "listdir",
                                          side_effect=lambda path: [] if path == "/proc" else listdir(path)),
                        mock.patch.object(pool, "print", create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_pool(self, run, *, gpus="0,1", conc=2, specs=("suite:model",), dims=None):
        argv = ["score_pool.py", *specs, "--gpus", gpus, "--conc", str(conc)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(pool.subprocess, "run", side_effect=run):
            if dims is None:
                pool.main()
            else:
                with mock.patch.object(pool, "DIMS", dims):
                    pool.main()

    def test_all_pairs_stage_once_before_full_dimension_coverage(self):
        pairs = [("suite", "model_a"), ("suite", "model_b"), ("other", "model_c")]
        stages, scored = [], []
        lock = threading.Lock()

        def run(command, **kwargs):
            suite, model, dim = command[3], command[5], command[7]
            self.assertEqual(command, [pool.PY, pool.EVAL, "--suite", suite, "--model", model, "--dims", dim])
            self.assertEqual(kwargs["cwd"], "/dockerdata/wm_eval/VBench")
            self.assertEqual(kwargs["env"]["VBENCH_CACHE_DIR"], "/dockerdata/vbench_cache")
            if dim == "none":
                self.assertEqual(set(kwargs), {"cwd", "env", "capture_output", "text"})
                self.assertTrue(kwargs["capture_output"])
                self.assertTrue(kwargs["text"])
                stages.append((suite, model))
            else:
                self.assertEqual(stages, pairs)
                self.assertIn(kwargs["env"]["CUDA_VISIBLE_DEVICES"], {"0", "1"})
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertEqual(Path(kwargs["stdout"].name).parent, self.logs)
                kwargs["stdout"].write("mock scoring or existing-score skip\n")
                with lock:
                    scored.append((suite, model, dim))
            return subprocess.CompletedProcess(command, 0, "staged", "")

        self.run_pool(run, specs=("suite:model_a,model_b", "other:model_c"))
        self.assertEqual(stages, pairs)
        expected = [(suite, model, dim) for suite, model in pairs for dim in pool.DIMS]
        self.assertEqual(Counter(scored), Counter(expected))
        self.assertEqual({p.name for p in self.logs.iterdir()},
                         {f"score_{suite}_{model}_{dim}.log" for suite, model, dim in expected})

    def test_slow_gpu_does_not_block_reuse_of_idle_gpu_or_overlap(self):
        slow_started, release_slow, fast_reused = threading.Event(), threading.Event(), threading.Event()
        active, assignments, errors = {}, {}, []
        lock = threading.Lock()

        def run(command, **kwargs):
            dim = command[-1]
            if dim == "none":
                return subprocess.CompletedProcess(command, 0, "staged", "")
            gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            with lock:
                self.assertNotIn(gpu, active, "two scoring processes overlap on one GPU")
                active[gpu] = dim
                assignments[dim] = gpu
                self.assertLessEqual(len(active), 2)
            try:
                if dim == "slow":
                    slow_started.set()
                    self.assertTrue(release_slow.wait(5), "slow scorer was not released")
                else:
                    self.assertTrue(slow_started.wait(5), "slow scorer did not start")
                    if dim in {"fast_2", "fast_3"}:
                        self.assertFalse(release_slow.is_set())
                        if dim == "fast_3":
                            fast_reused.set()
                return subprocess.CompletedProcess(command, 0)
            finally:
                with lock:
                    del active[gpu]

        def main():
            try:
                self.run_pool(run, dims=["slow", "fast_1", "fast_2", "fast_3"])
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=main, daemon=True)
        thread.start()
        try:
            self.assertTrue(fast_reused.wait(3), "idle GPU cannot accept new work while another GPU is slow")
            self.assertFalse(release_slow.is_set())
        finally:
            release_slow.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(set(assignments), {"slow", "fast_1", "fast_2", "fast_3"})
        self.assertNotEqual(assignments["slow"], assignments["fast_2"])
        self.assertEqual(assignments["fast_1"], assignments["fast_2"])
        self.assertEqual(assignments["fast_2"], assignments["fast_3"])
        self.assertEqual(active, {})

    def test_concurrency_is_bounded_by_argument_and_available_gpus(self):
        for gpus, conc, expected in (("0,1,2", 2, 2), ("0,0,1", 8, 2), ("0,1", 1, 1)):
            with self.subTest(gpus=gpus, conc=conc):
                barrier = threading.Barrier(expected)
                active, maximum, started = set(), 0, 0
                lock = threading.Lock()

                def run(command, **kwargs):
                    nonlocal maximum, started
                    if command[-1] == "none":
                        return subprocess.CompletedProcess(command, 0, "staged", "")
                    gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                    with lock:
                        self.assertNotIn(gpu, active)
                        active.add(gpu)
                        maximum = max(maximum, len(active))
                        started += 1
                        initial = started <= expected
                    try:
                        if initial:
                            barrier.wait(5)
                        return subprocess.CompletedProcess(command, 0)
                    finally:
                        with lock:
                            active.remove(gpu)

                self.run_pool(run, gpus=gpus, conc=conc, dims=["a", "b", "c", "d"])
                self.assertEqual(maximum, expected)
                self.assertEqual(started, 4)

    def test_wmb_reserved_gpus_are_excluded_and_duplicates_do_not_add_tokens(self):
        proc = {
            "/proc/11/cmdline": b"python\0/bench/evaluation.py\0--model_name\0model\0",
            "/proc/11/environ": b"OTHER=1\0CUDA_VISIBLE_DEVICES=0,2\0",
            "/proc/12/cmdline": b"python\0/vbench_eval.py\0--model_name\0model\0",
            "/proc/13/cmdline": b"python\0/bench/evaluation.py\0",
            "/proc/14/cmdline": b"python\0/bench/evaluation.py\0--model_name\0model\0",
        }
        tokens = queue.Queue()
        assignments = []

        def open_file(path, mode="r", **kwargs):
            if str(path).startswith("/proc/"):
                if path not in proc:
                    raise OSError("process exited")
                return io.BytesIO(proc[path])
            return builtins.open(path, mode, **kwargs)

        def run(command, **kwargs):
            if command[-1] != "none":
                assignments.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            return subprocess.CompletedProcess(command, 0, "staged", "")

        with mock.patch.object(pool.os, "listdir", return_value=["11", "12", "13", "14", "self"]), \
                mock.patch.object(pool, "open", side_effect=open_file, create=True), \
                mock.patch.object(pool.queue, "Queue", return_value=tokens):
            self.run_pool(run, gpus="0,1,1,2,3", conc=8, dims=["a", "b", "c", "d"])
        self.assertEqual(len(assignments), 4)
        self.assertTrue(set(assignments) <= {"1", "3"})
        self.assertEqual(tokens.qsize(), 2)
        self.assertEqual({tokens.get_nowait(), tokens.get_nowait()}, {"1", "3"})

    def test_all_gpus_reserved_prevents_staging_and_scoring(self):
        proc = mock.mock_open(read_data=b"python\0evaluation.py\0--model_name\0CUDA_VISIBLE_DEVICES=0,1\0")
        run = mock.Mock()
        with mock.patch.object(pool.os, "listdir", return_value=["11"]), \
                mock.patch.object(pool, "open", proc, create=True):
            with self.assertRaisesRegex(RuntimeError, "All requested GPUs are reserved"):
                self.run_pool(run)
        run.assert_not_called()
        self.assertEqual(list(self.logs.iterdir()), [])

    def test_nonzero_scoring_exit_releases_token_and_fails_main(self):
        tokens, scored = queue.Queue(), []

        def run(command, **kwargs):
            dim = command[-1]
            if dim != "none":
                scored.append((dim, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            return subprocess.CompletedProcess(command, 7 if dim == "bad" else 0, "staged", "")

        with mock.patch.object(pool.queue, "Queue", return_value=tokens):
            with self.assertRaisesRegex(SystemExit, "VBench scoring failed") as caught:
                self.run_pool(run, gpus="1", conc=2, dims=["bad", "good"])
        self.assertNotEqual(caught.exception.code, 0)
        self.assertEqual(scored, [("bad", "1"), ("good", "1")])
        self.assertEqual(tokens.qsize(), 1)
        self.assertEqual(tokens.get_nowait(), "1")

    def test_scoring_exception_propagates_and_releases_token(self):
        tokens = queue.Queue()
        failure = OSError("mock scorer cannot start")

        def run(command, **kwargs):
            if command[-1] != "none":
                raise failure
            return subprocess.CompletedProcess(command, 0, "staged", "")

        with mock.patch.object(pool.queue, "Queue", return_value=tokens):
            with self.assertRaises(OSError) as caught:
                self.run_pool(run, gpus="1", dims=["bad"])
        self.assertIs(caught.exception, failure)
        self.assertEqual(tokens.qsize(), 1)
        self.assertEqual(tokens.get_nowait(), "1")

    def test_log_open_exception_also_releases_gpu_token(self):
        tokens = queue.Queue()
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "staged", ""))
        with mock.patch.object(pool.queue, "Queue", return_value=tokens), \
                mock.patch.object(pool, "open", side_effect=OSError("mock log error"), create=True):
            with self.assertRaisesRegex(OSError, "mock log error"):
                self.run_pool(run, gpus="1", dims=["bad"])
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][-1], "none")
        self.assertEqual(tokens.qsize(), 1)
        self.assertEqual(tokens.get_nowait(), "1")

    def test_staging_failure_or_exception_never_starts_scoring(self):
        for failure in (subprocess.CompletedProcess([], 9, "", "mock staging failure"),
                        OSError("mock staging exception")):
            with self.subTest(failure=failure):
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    self.assertEqual(command[-1], "none", "scoring started before all staging succeeded")
                    if command[5] == "model_b":
                        if isinstance(failure, Exception):
                            raise failure
                        return failure
                    return subprocess.CompletedProcess(command, 0, "staged", "")

                error = OSError if isinstance(failure, Exception) else RuntimeError
                with self.assertRaisesRegex(error, "mock staging"):
                    self.run_pool(run, specs=("suite:model_a,model_b",))
                self.assertEqual([command[5] for command in calls], ["model_a", "model_b"])
                self.assertEqual(list(self.logs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
