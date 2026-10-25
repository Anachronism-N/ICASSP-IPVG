import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_e3_subset_node.py"
GPU_HOLD = SCRIPT.with_name("gpu_hold.py")
SPEC = importlib.util.spec_from_file_location("e3_subset_node", SCRIPT)
node = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(node)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def frozen(path):
    return {k: v for k, v in node.record(path).items() if k != "mtime_ns"}


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="E3 node with spaces ")
        self.root = Path(self.tmp.name)
        self.models = self.root / "local/models"
        self.state = self.root / "local/hold"
        self.patches = [patch.object(node, "MODELS", self.models),
                        patch.object(node, "LOCAL_ROOT", self.root / "local"),
                        patch.object(node, "STATE", self.state)]
        for p in self.patches:
            p.start()
        self.addCleanup(self.tmp.cleanup)
        for p in self.patches:
            self.addCleanup(p.stop)
        self.checkpoint = put(self.root / "checkpoint.safetensors", b"synthetic C checkpoint")
        self.cache = put(self.models / "wm_checkpoints" / (node.sha(str(self.checkpoint).encode()) + ".safetensors"),
                         self.checkpoint.read_bytes())
        for asset in node.ASSETS:
            put(self.models / asset, asset.encode())
        code = [put(self.root / "scripts" / name, name.encode()) for name in (
            "gen_jobs.py", "prepare_e3_subset.py", "run_e3_subset_node.py", "run_gen.sh", "gpu_hold.py")]
        code.append(put(self.root / "DiffSynth-Studio/diffsynth/__init__.py", b"# frozen fake imports"))
        runtime = put(self.root / "python", b"fake runtime")
        image = put(self.root / "image.jpg", b"mini input")
        self.folder = self.root / "eval/e3_subset_20261025"
        self.folder.mkdir(parents=True)
        jobs, cases = [], []
        for suite, count in node.COUNTS.items():
            for i in range(count):
                out = str(self.folder / "videos" / suite / f"{i}.mp4")
                j = {"id": f"{suite}_{i}", "model": node.TAG, "ckpt": str(self.checkpoint),
                     "out": out, "image": str(image), "prompt": "Neutral sample", "seed": 0}
                jobs.append(j)
                cases.append(dict(j, suite=suite, video=out))
        jp = put(self.folder / "jobs.jsonl", b"".join(node.encoded(j) for j in jobs))
        cp = put(self.folder / "cases.jsonl", b"".join(node.encoded(c) for c in cases))
        self.plan = {"schema": "e3_subset_v1", "private": True, "tag": node.TAG,
                     "selection": {"counts": node.COUNTS}, "checkpoint": frozen(self.checkpoint),
                     "runtime": {"python": str(runtime), "packages": {}},
                     "budget": {"per_node_wall_seconds": 3000, "generation_nodes": 4, "gpus_per_node": 8,
                                "total_generation_gpus": 32, "generation_gpu_hours": 28, "score_gpu_hours": 12,
                                "total_gpu_hours": 40, "training_authorized": False, "full_1366_authorized": False},
                     "jobs_path": str(jp), "cases_path": str(cp),
                     "manifest_files": [frozen(jp), frozen(cp)],
                     "frozen_files": [frozen(p) for p in code + [runtime, image, self.checkpoint]]}
        # Deliberately inaccessible rank0-only base/scorer assets must not be read.
        self.plan["frozen_files"].append({"path": str(self.root / "absent_base_video.mp4"),
                                          "sha256": "0" * 64, "bytes": 999})
        self.plan_path = put(self.folder / "plan.json", node.encoded(self.plan))
        self.plan_sha = node.sha(self.plan_path.read_bytes())

    def preflight(self):
        path, digest = node.preflight(self.plan_path, 0, self.plan_sha)
        value = json.loads(path.read_bytes())
        for rank in range(1, 4):
            node.exclusive(node.node_path(self.plan_path, rank, "_preflight.json"), dict(value, rank=rank))
        return path, digest

    def fake_command(self, body):
        worker = put(self.root / "fake executable.py", body.encode())
        return patch.object(node, "command_for", return_value=(
            [sys.executable, str(GPU_HOLD), "--state-dir", str(self.state), "--action", "exec",
             "--gpus", node.GPUS, "--", sys.executable, str(worker)], dict(os.environ)))

    def test_preflight_assets_and_duplicate(self):
        path, _ = self.preflight()
        pf = json.loads(path.read_bytes())
        self.assertEqual(len(pf["models"]), 9)
        self.assertEqual(pf["checkpoint_cache"]["sha256"], node.sha(self.checkpoint.read_bytes()))
        self.assertFalse(pf["GPU_used"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            node.preflight(self.plan_path, 0, self.plan_sha)
        self.assertFalse(node.node_path(self.plan_path, 0, ".launch.json").exists())

    def test_frozen_empty_package_init_is_hashed_and_allowed(self):
        source = self.root / "DiffSynth-Studio/diffsynth/__init__.py"
        source.write_bytes(b"")
        expected = frozen(source)
        self.assertEqual(expected["bytes"], 0)
        self.assertEqual(expected["sha256"], node.sha(b""))
        self.plan["frozen_files"] = [expected if r["path"] == str(source) else r
                                     for r in self.plan["frozen_files"]]
        self.plan_path.write_bytes(node.encoded(self.plan))
        self.plan_sha = node.sha(self.plan_path.read_bytes())
        path, _ = self.preflight()
        self.assertFalse(json.loads(path.read_bytes())["GPU_used"])
        source.write_bytes(b"# drift from frozen empty source")
        with self.assertRaisesRegex(ValueError, "Frozen content SHA256/size mismatch"):
            node.verify_inputs(self.plan_path, self.plan_sha)

    def test_empty_model_assets_are_rejected(self):
        for asset in node.ASSETS:
            with self.subTest(asset=asset):
                path = self.models / asset
                original = path.read_bytes()
                path.write_bytes(b"")
                with self.assertRaisesRegex(ValueError, "Empty required model asset"):
                    node.preflight(self.plan_path, 0, self.plan_sha)
                self.assertFalse(node.node_path(self.plan_path, 0, "_preflight.json").exists())
                path.write_bytes(original)

    def test_empty_checkpoint_cache_is_rejected(self):
        self.cache.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty required checkpoint cache"):
            node.preflight(self.plan_path, 0, self.plan_sha)
        self.assertFalse(node.node_path(self.plan_path, 0, "_preflight.json").exists())

    def test_frozen_empty_checkpoint_is_rejected(self):
        self.checkpoint.write_bytes(b"")
        expected = frozen(self.checkpoint)
        self.plan["checkpoint"] = expected
        self.plan["frozen_files"] = [expected if r["path"] == str(self.checkpoint) else r
                                     for r in self.plan["frozen_files"]]
        self.plan_path.write_bytes(node.encoded(self.plan))
        with self.assertRaisesRegex(ValueError, "Empty required checkpoint"):
            node.preflight(self.plan_path, 0, node.sha(self.plan_path.read_bytes()))

    def test_frozen_empty_manifest_is_rejected(self):
        manifest = self.folder / "jobs.jsonl"
        manifest.write_bytes(b"")
        self.plan["manifest_files"] = [frozen(manifest) if r["path"] == str(manifest) else r
                                       for r in self.plan["manifest_files"]]
        self.plan_path.write_bytes(node.encoded(self.plan))
        with self.assertRaisesRegex(ValueError, "Empty required manifest"):
            node.preflight(self.plan_path, 0, node.sha(self.plan_path.read_bytes()))

    def test_wrong_approved_plan_and_budget(self):
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            node.preflight(self.plan_path, 0, "0" * 64)
        self.plan["budget"]["per_node_wall_seconds"] = 3090
        self.plan_path.write_bytes(node.encoded(self.plan))
        with self.assertRaisesRegex(ValueError, "GPU budget"):
            node.preflight(self.plan_path, 0, node.sha(self.plan_path.read_bytes()))

    def test_checkpoint_cache_drift(self):
        self.cache.write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError, "checkpoint cache SHA256"):
            node.preflight(self.plan_path, 0, self.plan_sha)

    def test_missing_asset_or_shared_symlink(self):
        p = self.models / node.ASSETS[-1]
        p.unlink()
        with self.assertRaises(FileNotFoundError):
            node.preflight(self.plan_path, 0, self.plan_sha)
        shared = put(self.root / "shared-model", b"shared")
        p.symlink_to(shared)
        with self.assertRaisesRegex(ValueError, "node-local"):
            node.preflight(self.plan_path, 0, self.plan_sha)

    def test_new_import_is_not_frozen(self):
        put(self.root / "DiffSynth-Studio/diffsynth/new.py", b"# new source")
        with self.assertRaisesRegex(ValueError, "imports not frozen"):
            node.preflight(self.plan_path, 0, self.plan_sha)

    def test_model_stat_drift_rejected_before_launch(self):
        _, digest = self.preflight()
        p = self.models / node.ASSETS[0]
        stat = p.stat()
        os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, "model drift"):
            node.run_node(self.plan_path, 0, self.plan_sha, digest)
        self.assertFalse(node.node_path(self.plan_path, 0, ".launch.json").exists())

    def test_manifest_source_and_cache_drift_rejected(self):
        for target in (self.folder / "jobs.jsonl", self.root / "scripts/gen_jobs.py", self.cache):
            with self.subTest(target=target):
                _, digest = self.preflight()
                original = target.read_bytes()
                target.write_bytes(original + b"drift")
                with self.assertRaises(ValueError):
                    node.run_node(self.plan_path, 0, self.plan_sha, digest)
                self.assertFalse(node.node_path(self.plan_path, 0, ".launch.json").exists())
                target.write_bytes(original)
                for rank in range(4):
                    node.node_path(self.plan_path, rank, "_preflight.json").unlink()

    def test_changed_plan_and_wrong_preflight_approval(self):
        _, digest = self.preflight()
        with self.assertRaisesRegex(ValueError, "preflight SHA256"):
            node.run_node(self.plan_path, 0, self.plan_sha, "0" * 64)
        self.plan_path.write_bytes(self.plan_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            node.run_node(self.plan_path, 0, self.plan_sha, digest)
        self.assertFalse(node.node_path(self.plan_path, 0, ".launch.json").exists())

    def test_extra_shard_drift_and_existing_receipt_rejected(self):
        _, digest = self.preflight()
        extra = put(self.models / "Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model_extra.safetensors", b"unexpected")
        with self.assertRaisesRegex(ValueError, "shard set changed"):
            node.run_node(self.plan_path, 0, self.plan_sha, digest)
        extra.unlink()
        node.exclusive(node.node_path(self.plan_path, 0, ".finished.json"), {"previous": True})
        with self.assertRaisesRegex(ValueError, "receipt already exists"):
            node.run_node(self.plan_path, 0, self.plan_sha, digest)
        self.assertFalse(node.node_path(self.plan_path, 0, ".launch.json").exists())

    def test_four_node_content_match_required(self):
        _, digest = self.preflight()
        other = node.node_path(self.plan_path, 3, "_preflight.json")
        value = json.loads(other.read_bytes())
        value["model_content_sha256"] = "0" * 64
        other.write_bytes(node.encoded(value))
        with self.assertRaisesRegex(ValueError, "four node"):
            node.run_node(self.plan_path, 0, self.plan_sha, digest)

    def test_success_duplicate_launch_and_actual_cost(self):
        _, digest = self.preflight()
        self.state.mkdir(parents=True)
        (self.state / "pause.7").touch()
        with self.fake_command("import time; time.sleep(0.05)"):
            receipt = node.run_node(self.plan_path, 0, self.plan_sha, digest)
        self.assertEqual(receipt["exit_code"], 0)
        self.assertTrue(receipt["owned_tree_released"])
        self.assertTrue(receipt["complete_not_proven"])
        self.assertFalse(receipt["timed_out"])
        self.assertEqual(receipt["gpu_hours"], 8 * receipt["elapsed_seconds"] / 3600)
        self.assertAlmostEqual(receipt["end"] - receipt["start"], receipt["elapsed_seconds"], places=3)
        self.assertTrue((self.state / "pause.7").exists())
        self.assertFalse((self.state / "pause.0").exists())
        self.assertEqual(json.loads(node.node_path(self.plan_path, 0, ".finished.json").read_bytes()), receipt)
        with self.assertRaisesRegex(ValueError, "already launched"):
            node.run_node(self.plan_path, 0, self.plan_sha, digest)

    def test_failed_spawn_still_has_finished_receipt(self):
        _, digest = self.preflight()
        with patch.object(node, "command_for", return_value=([str(self.root / "missing-executable")], dict(os.environ))):
            receipt = node.run_node(self.plan_path, 0, self.plan_sha, digest)
        self.assertIn("FileNotFoundError", receipt["error"])
        self.assertTrue(receipt["owned_tree_released"])
        self.assertTrue(node.node_path(self.plan_path, 0, ".finished.json").exists())
        self.assertFalse((self.state / "pause.0").exists())

    def test_existing_experiment_is_not_stopped(self):
        _, digest = self.preflight()
        self.state.mkdir(parents=True)
        with (self.state / "experiment.0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.fake_command("raise RuntimeError('must not execute')"):
                receipt = node.run_node(self.plan_path, 0, self.plan_sha, digest)
        self.assertIn("BlockingIOError", receipt["error"])
        self.assertIsNone(receipt["exit_code"])

    def stubborn_worker(self):
        pidfile = self.root / "grandchild.pid"
        body = ("import os, signal, subprocess, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
                f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n"
                "time.sleep(30)\n")
        return self.fake_command(body), pidfile

    def assert_descendant_dead(self, pidfile):
        pid = int(pidfile.read_text())
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return
        self.assertIn(fields[0], ("Z", "X"))

    def test_timeout_kills_owned_group_not_unrelated_holder(self):
        _, digest = self.preflight()
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            command, pidfile = self.stubborn_worker()
            with command, patch.object(node, "TERM_SECONDS", 0.35), patch.object(node, "WALL_SECONDS", 1.6):
                receipt = node.run_node(self.plan_path, 0, self.plan_sha, digest)
            self.assertTrue(receipt["timed_out"])
            self.assertTrue(receipt["owned_tree_released"])
            self.assertLess(receipt["elapsed_seconds"], 1.6)
            self.assertIsNone(unrelated.poll())
            self.assert_descendant_dead(pidfile)
            self.assertFalse(any(self.state.glob("pause.*")))
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_user_term_cleanup_and_receipt(self):
        _, digest = self.preflight()
        command, pidfile = self.stubborn_worker()
        timer = threading.Timer(0.35, lambda: os.kill(os.getpid(), signal.SIGTERM))
        with command, patch.object(node, "WALL_SECONDS", 1.6):
            timer.start()
            try:
                receipt = node.run_node(self.plan_path, 0, self.plan_sha, digest)
            finally:
                timer.cancel()
                timer.join()
        self.assertEqual(receipt["interrupted_signal"], signal.SIGTERM)
        self.assertFalse(receipt["timed_out"])
        self.assertTrue(receipt["owned_tree_released"])
        self.assert_descendant_dead(pidfile)
        self.assertFalse(any(self.state.glob("pause.*")))

    def test_command_uses_argument_array_and_one_model_tag(self):
        command, env = node.command_for(self.plan, 2)
        self.assertIsInstance(command, list)
        self.assertEqual(command[command.index("--gpus") + 1], "0,1,2,3,4,5,6,7")
        self.assertEqual(command[-2:], ["--models", node.TAG])
        self.assertEqual(env["NODE_RANK"], "2")
        self.assertEqual(env["NNODES"], "4")
        self.assertEqual(node.TERM_SECONDS, 2940)
        self.assertEqual(node.WALL_SECONDS, 3000)
        self.assertEqual(32 * 3000 / 3600, 26.666666666666668)
        self.assertAlmostEqual(28 - 32 * 3000 / 3600, 1.3333333333333321)


if __name__ == "__main__":
    unittest.main()
