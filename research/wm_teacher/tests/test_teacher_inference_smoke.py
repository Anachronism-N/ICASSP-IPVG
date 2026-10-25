"""CPU-only smoke-driver regression tests; fake inference is not CUDA evidence."""
import builtins
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("smoke_driver", ROOT / "scripts/run_teacher_inference_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
REAL_RUNTIME_RECORDS = smoke.runtime_records


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "repo"
        self.local = self.base / "local"
        self.state = self.local / "wm_logs/gpu_hold"
        self.state.mkdir(parents=True)
        (self.root / "scripts").mkdir(parents=True)
        for name in ("gen_jobs.py", "run_teacher_inference_smoke.py", "teacher_alg.py", "teacher_dymos.py", "gpu_hold.py"):
            (self.root / "scripts" / name).write_text('NEG = "original negative"\n')
        (self.root / "DiffSynth-Studio/diffsynth").mkdir(parents=True)
        (self.root / "DiffSynth-Studio/diffsynth/module.py").write_text("value = 1\n")
        (self.root / "docs").mkdir()
        (self.root / "docs/51_免训练Teacher多维策略有限实验预注册.md").write_text("Frozen test protocol\n")
        cases_dir = self.root / "eval/e1_devset_20261021"
        cases_dir.mkdir(parents=True)
        image = cases_dir / "image.png"
        image.write_bytes(b"CPU fixture image; loader is explicitly mocked")
        first = dict(id="fixture_safe_first_case", image=str(image), image_sha256=smoke.record(image)["sha256"],
                     prompt="A person walks in a room.", prompt_sha256=smoke.sha(b"A person walks in a room."))
        self.cases_path = cases_dir / "cases.jsonl"
        self.cases_path.write_bytes(b"".join(smoke.encoded(dict(first, id=first["id"] if i == 0 else f"case_{i}")) for i in range(94)))
        self.models = []
        for i, asset in enumerate(smoke.ASSETS):
            path = self.local / "models" / asset
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"mock asset {i}".encode())
            self.models.append(dict(smoke.record(path), asset=asset))
        self.pf = self.base / "node_preflight.json"
        self.pf.write_bytes(smoke.encoded(dict(schema="e3_subset_node_preflight_v1", hostname=os.uname().nodename,
                           GPU_used=False, models=self.models,
                           model_content_sha256=smoke.sha(smoke.encoded([{k: m[k] for k in ("asset", "bytes", "sha256")} for m in self.models])))))
        self.tools = {n: shutil.which(n) for n in ("ffmpeg", "ffprobe", "timeout")}
        self.runtime = dict(python=str(Path(sys.executable).resolve()), tools=self.tools, packages={"mock": "CPU"},
                            files=[smoke.record(Path(sys.executable).resolve())])
        patches = [mock.patch.object(smoke, "FIRST_CASE_SHA256", smoke.sha(b"fixture_safe_first_case")),
                   mock.patch.object(smoke, "LOCAL_ROOT", self.local),
                   mock.patch.object(smoke, "runtime_records", return_value=self.runtime)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def prepare(self, method="alg", name="request"):
        return smoke.prepare(method, self.base / name, self.pf, smoke.sha(self.pf.read_bytes()), root=self.root)

    def change_request(self, path, change):
        value = json.loads(path.read_bytes())
        change(value)
        path.write_bytes(smoke.encoded(value))
        return smoke.sha(path.read_bytes())

    def state_value(self, **selected):
        scope = dict(paused=False, holder_pid=None, external_pids=[], gpu_uuid="GPU-fixture")
        scope.update(selected)
        return dict(pid=12345, running=True, query_ok=True, updated_at=time.time(), gpus={"2": scope})

    def state_write(self, value):
        (self.state / "status.json").write_bytes(smoke.encoded(value))

    def scope_patches(self):
        self.state_write(self.state_value())
        return contextlib.ExitStack()


class PrepareTests(Fixture):
    def test_empty_python_source_is_legally_frozen_and_verified(self):
        source = self.root / "DiffSynth-Studio/diffsynth/utils/state_dict_converters/__init__.py"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"")
        path, digest = self.prepare()
        request = smoke.verify_request(path, digest)
        frozen = next(r for r in request["source_files"] if r["path"] == str(source))
        self.assertEqual(frozen["bytes"], 0)
        self.assertEqual(frozen["sha256"], smoke.sha(b""))
        source.write_bytes(b"changed = True\n")
        with self.assertRaisesRegex(ValueError, "drift"):
            smoke.verify_request(path, digest)

    def test_generic_record_empty_source_has_sha256(self):
        source = self.base / "empty.py"
        source.write_bytes(b"")
        self.assertEqual(smoke.record(source)["sha256"], smoke.sha(b""))
        with self.assertRaisesRegex(ValueError, "Empty required"):
            smoke.record(source, nonempty=True)

    def test_empty_model_rejected_even_if_preflight_hash_matches(self):
        value = json.loads(self.pf.read_bytes())
        model = value["models"][0]
        Path(model["path"]).write_bytes(b"")
        model.update(smoke.record(model["path"]))
        value["model_content_sha256"] = smoke.sha(smoke.encoded([{k: m[k] for k in ("asset", "bytes", "sha256")} for m in value["models"]]))
        self.pf.write_bytes(smoke.encoded(value))
        with self.assertRaisesRegex(ValueError, "Empty required model"):
            self.prepare()
        self.assertFalse((self.base / "request").exists())

    def test_empty_image_rejected_even_if_case_hash_matches(self):
        cases = [json.loads(line) for line in self.cases_path.read_text().splitlines()]
        Path(cases[0]["image"]).write_bytes(b"")
        cases[0]["image_sha256"] = smoke.sha(b"")
        self.cases_path.write_bytes(b"".join(smoke.encoded(c) for c in cases))
        with self.assertRaisesRegex(ValueError, "Empty required input"):
            self.prepare()

    def test_empty_case_manifest_rejected(self):
        self.cases_path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty required input"):
            self.prepare()

    def test_empty_node_preflight_rejected(self):
        self.pf.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty required input"):
            self.prepare()

    def test_prepare_imports_no_torch_generator_or_cuda(self):
        original_import = builtins.__import__
        def reject(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "gen_jobs", "diffsynth", "teacher_alg", "teacher_dymos"):
                raise AssertionError("CPU prepare imported GPU runtime: " + name)
            return original_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=reject), mock.patch.object(smoke.subprocess, "run", side_effect=AssertionError("prepare ran subprocess")):
            path, digest = self.prepare()
        request = json.loads(path.read_bytes())
        self.assertEqual(digest, smoke.sha(path.read_bytes()))
        self.assertFalse(request["GPU_used"])
        self.assertFalse(request["quality_pass"])
        self.assertEqual(request["checkpoint"], "NONE")
        self.assertEqual(request["call"], smoke.SPEC)
        self.assertEqual(request["video_count"], 1)
        self.assertEqual(path.parent.stat().st_mode & 0o077, 0)
        self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_default_cli_action_is_cpu_prepare(self):
        with mock.patch.object(smoke, "prepare", return_value=(Path("private/request.json"), "a" * 64)) as prepare, mock.patch("builtins.print"):
            smoke.main(["--method", "alg", "--request-dir", "private", "--node-preflight", str(self.pf),
                        "--approved-node-preflight-sha256", "b" * 64])
        prepare.assert_called_once()

    def test_prepare_rejects_gpu_arguments(self):
        with mock.patch("sys.stderr", new_callable=__import__("io").StringIO), self.assertRaises(SystemExit):
            smoke.main(["--method", "alg", "--request-dir", "private", "--node-preflight", str(self.pf),
                        "--approved-node-preflight-sha256", "b" * 64, "--gpu", "2"])

    def test_methods_get_independent_new_requests(self):
        alg, alg_sha = self.prepare("alg", "alg")
        dymos, dymos_sha = self.prepare("dymos", "dymos")
        self.assertNotEqual(alg_sha, dymos_sha)
        self.assertEqual(json.loads(alg.read_bytes())["parameters"], {"factor": 0.4})
        self.assertEqual(json.loads(dymos.read_bytes())["parameters"], {"gamma": 0.6, "phi": 1})
        self.assertNotEqual(alg.parent, dymos.parent)

    def test_prepare_never_overwrites(self):
        path, _ = self.prepare()
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare()
        self.assertEqual(original, path.read_bytes())

    def test_unapproved_preflight_rejected(self):
        with self.assertRaisesRegex(ValueError, "preflight SHA"):
            smoke.prepare("alg", self.base / "bad", self.pf, "0" * 64, self.root)
        self.assertFalse((self.base / "bad").exists())

    def test_wrong_first_case_rejected(self):
        cases = self.cases_path.read_text().splitlines()
        case = json.loads(cases[0])
        case["id"] = "not_first"
        cases[0] = json.dumps(case)
        self.cases_path.write_text("\n".join(cases))
        with self.assertRaisesRegex(ValueError, "fixed first"):
            self.prepare()

    def test_missing_ninth_asset_rejected(self):
        value = json.loads(self.pf.read_bytes())
        value["models"].pop()
        self.pf.write_bytes(smoke.encoded(value))
        with self.assertRaisesRegex(ValueError, "nine"):
            self.prepare()

    def test_shared_asset_symlink_rejected(self):
        model = self.models[0]
        shared = self.base / "shared_model"
        shared.write_bytes(Path(model["path"]).read_bytes())
        Path(model["path"]).unlink()
        Path(model["path"]).symlink_to(shared)
        with self.assertRaisesRegex(ValueError, "local"):
            self.prepare()


class FrozenRequestTests(Fixture):
    def test_runtime_package_version_drift_rejected(self):
        path, digest = self.prepare()
        self.runtime["packages"]["mock"] = "changed"
        with self.assertRaisesRegex(ValueError, "Runtime version"):
            smoke.verify_request(path, digest)

    def test_correct_request_verified(self):
        path, digest = self.prepare()
        self.assertEqual(smoke.verify_request(path, digest)["case"]["id"], "fixture_safe_first_case")

    def test_wrong_request_sha_before_any_launch(self):
        path, _ = self.prepare()
        with mock.patch.object(smoke, "supervisor_scope", side_effect=AssertionError("looked at GPU")), self.assertRaisesRegex(ValueError, "request SHA"):
            smoke.run(path, "0" * 64, 2, 12345, self.state, True)
        self.assertFalse((path.parent / "launch.json").exists())

    def test_input_content_drift_rejected(self):
        path, digest = self.prepare()
        Path(json.loads(path.read_bytes())["image"]["path"]).write_bytes(b"bad image")
        with self.assertRaisesRegex(ValueError, "image hash"):
            smoke.verify_request(path, digest)

    def test_model_content_drift_rejected(self):
        path, digest = self.prepare()
        Path(self.models[0]["path"]).write_bytes(b"bad weight")
        with self.assertRaisesRegex(ValueError, "drift"):
            smoke.verify_request(path, digest)

    def test_extra_model_shard_rejected(self):
        path, digest = self.prepare()
        (Path(self.models[0]["path"]).parent / "diffusion_pytorch_model-extra.safetensors").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "shard"):
            smoke.verify_request(path, digest)

    def test_source_content_drift_rejected(self):
        path, digest = self.prepare()
        (self.root / "scripts/teacher_alg.py").write_text('NEG = "changed"\n')
        with self.assertRaisesRegex(ValueError, "drift"):
            smoke.verify_request(path, digest)

    def test_source_import_addition_rejected(self):
        path, digest = self.prepare()
        (self.root / "DiffSynth-Studio/diffsynth/new_module.py").write_text("new = True\n")
        with self.assertRaisesRegex(ValueError, "set drift"):
            smoke.verify_request(path, digest)

    def test_frozen_metadata_drift_rejected(self):
        path, digest = self.prepare()
        image = Path(json.loads(path.read_bytes())["image"]["path"])
        st = image.stat()
        os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        with self.assertRaisesRegex(ValueError, "image drift"):
            smoke.verify_request(path, digest)

    def test_over_budget_request_rejected_even_with_new_sha(self):
        path, _ = self.prepare()
        digest = self.change_request(path, lambda r: r["budget"].update(gpu_hours=2.0))
        with self.assertRaisesRegex(ValueError, "budget"):
            smoke.verify_request(path, digest)

    def test_bool_seed_is_not_integer_zero(self):
        path, _ = self.prepare()
        digest = self.change_request(path, lambda r: r["call"].update(seed=False))
        with self.assertRaisesRegex(ValueError, "configuration"):
            smoke.verify_request(path, digest)

    def test_lora_or_reduced_spec_rejected(self):
        for key, value in (("checkpoint", "LORA"), ("video_count", 2), ("active_steps", 9), ("fps", 25)):
            with self.subTest(key=key):
                path, _ = self.prepare(name=key)
                digest = self.change_request(path, lambda r: r.update({key: value}))
                with self.assertRaisesRegex(ValueError, "configuration"):
                    smoke.verify_request(path, digest)

    def test_public_request_permissions_rejected(self):
        path, digest = self.prepare()
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private"):
            smoke.verify_request(path, digest)


class RuntimeRecordTests(Fixture):
    def test_runtime_selects_entrypoints_metadata_and_critical_binaries_only(self):
        package_root = self.base / "site-packages"
        package_root.mkdir()
        names = {"torch": "torch", "transformers": "transformers", "safetensors": "safetensors",
                 "pillow": "PIL", "imageio": "imageio", "numpy": "numpy"}
        distributions, expected = [], set()
        for name, module in names.items():
            entry = package_root / module / "__init__.py"
            entry.parent.mkdir()
            entry.write_bytes(b"")
            metadata = Path(name + "-1.0.dist-info/METADATA")
            metadata_path = package_root / metadata
            metadata_path.parent.mkdir()
            metadata_path.write_text("Name: " + name + "\nVersion: 1.0\n")
            unrelated = package_root / module / "unrelated.py"
            unrelated.write_text("not exhaustively frozen\n")
            distributions.append(types.SimpleNamespace(metadata={"Name": name}, version="1.0",
                files=[metadata, unrelated.relative_to(package_root)],
                locate_file=lambda item, root=package_root: root / item))
            expected.update((str(entry), str(metadata_path)))
        binary = package_root / "torch/_C.fixture.so"
        binary.write_bytes(b"selected critical binary")
        native_not_selected = package_root / "torch/transitive_native.so"
        native_not_selected.write_bytes(b"not an exhaustive native attestation")
        with mock.patch.object(smoke.importlib.metadata, "distributions", return_value=distributions):
            value = REAL_RUNTIME_RECORDS()
        files = {r["path"] for r in value["files"]}
        self.assertTrue(expected <= files)
        self.assertIn(str(binary), files)
        self.assertNotIn(str(native_not_selected), files)
        self.assertFalse(any(p.endswith("unrelated.py") for p in files))
        self.assertIn("not exhaustively hashed", value["caveat"])
        self.assertEqual(len(value["packages"]), 6)
        self.assertLessEqual(len(files), 17)


class GPUAuthorizationTests(Fixture):
    def test_gpu_authorization_required_before_verification(self):
        with mock.patch.object(smoke, "verify_request", side_effect=AssertionError("verification should not run")), self.assertRaisesRegex(ValueError, "authorized"):
            smoke.run(self.base / "missing", "x", 2, 12345, self.state)

    def test_single_gpu_scope_rejected(self):
        for gpu in (None, "2,3", -1, 8, True):
            with self.subTest(gpu=gpu), self.assertRaisesRegex(ValueError, "one physical GPU"):
                smoke.supervisor_scope(self.state, 12345, gpu)

    def test_live_supervisor_pid_required(self):
        with mock.patch.object(smoke, "proc_args", return_value=[]), self.assertRaisesRegex(ValueError, "live gpu_hold"):
            smoke.supervisor_scope(self.state, 12345, 2)

    def test_external_current_workload_rejected_without_pause(self):
        self.state_write(self.state_value(external_pids=[999]))
        with mock.patch.object(smoke, "proc_args", return_value=[b"python", b"gpu_hold.py", b"--action", b"run"]), self.assertRaisesRegex(ValueError, "existing workload"):
            smoke.supervisor_scope(self.state, 12345, 2)
        self.assertFalse((self.state / "pause.2").exists())

    def test_already_paused_card_rejected(self):
        self.state_write(self.state_value(paused=True))
        with mock.patch.object(smoke, "proc_args", return_value=[b"python", b"gpu_hold.py", b"--action", b"run"]), self.assertRaisesRegex(ValueError, "already paused"):
            smoke.supervisor_scope(self.state, 12345, 2)

    def test_live_query_detects_new_workload(self):
        self.state_write(self.state_value())
        with mock.patch.object(smoke, "proc_args", return_value=[b"python", b"gpu_hold.py", b"--action", b"run"]), mock.patch.object(smoke.subprocess, "run", return_value=types.SimpleNamespace(stdout="GPU-fixture, 999\n")), self.assertRaisesRegex(ValueError, "unrelated"):
            smoke.supervisor_scope(self.state, 12345, 2)

    def test_stale_supervisor_rejected(self):
        state = self.state_value()
        state["updated_at"] -= 20
        self.state_write(state)
        with mock.patch.object(smoke, "proc_args", return_value=[b"python", b"gpu_hold.py", b"--action", b"run"]), self.assertRaisesRegex(ValueError, "Stale"):
            smoke.supervisor_scope(self.state, 12345, 2)

    def test_timeout_holderexec_scope_is_single_gpu(self):
        path, digest = self.prepare()
        request = smoke.verify_request(path, digest)
        command = smoke.run_command(request, digest, 2, 12345, self.state, 100.0)
        self.assertIn("3450s", command)
        self.assertIn("--kill-after=30s", command)
        self.assertEqual(command[command.index("--gpus") + 1], "2")
        self.assertEqual(command[command.index("--action") + 1], "exec")
        self.assertIn(digest, command)
        self.assertLessEqual(smoke.BUDGET["outer_cap_seconds"], 3600)
        self.assertLess(smoke.BUDGET["internal_stop_seconds"], smoke.BUDGET["timeout_seconds"])

    def test_direct_worker_has_no_managed_ancestry(self):
        path, digest = self.prepare()
        request = smoke.verify_request(path, digest)
        smoke.exclusive(path.parent / "launch.json", dict(gpu=2, gpu_uuid="GPU-fixture", supervisor_pid=12345, parent_pid=99999999, budget=smoke.BUDGET))
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="GPU-fixture"), self.assertRaisesRegex(ValueError, "outer supervisor"):
            smoke.worker_scope(request, 2, 12345, self.state, time.monotonic())


class OuterReceiptTests(Fixture):
    def test_launch_failure_is_billed_and_default_failure(self):
        path, digest = self.prepare()
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), mock.patch.object(smoke.subprocess, "Popen", side_effect=OSError("fixture launch failed")):
            result = smoke.run(path, digest, 2, 12345, self.state, True)
        self.assertFalse(result["success"])
        self.assertTrue(result["failure"])
        self.assertFalse(result["mechanism_PASS"])
        self.assertGreater(result["elapsed_GPUh"], 0)
        self.assertTrue(result["owned_tree_released"])
        self.assertIn("fixture launch failed", result["error"])
        self.assertFalse((self.state / "pause.2").exists())
        self.assertTrue((path.parent / "finished.json").exists())

    def test_preexisting_run_never_overwritten(self):
        path, digest = self.prepare()
        smoke.exclusive(path.parent / "finished.json", {"old": True})
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), self.assertRaisesRegex(ValueError, "Old run"):
            smoke.run(path, digest, 2, 12345, self.state, True)
        self.assertEqual(json.loads((path.parent / "finished.json").read_bytes()), {"old": True})

    def test_zero_exit_without_completed_evidence_is_failure(self):
        path, digest = self.prepare()
        child = types.SimpleNamespace(pid=99999999, returncode=0, poll=lambda: 0, wait=lambda **kw: 0)
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), mock.patch.object(smoke.subprocess, "Popen", return_value=child), mock.patch.object(smoke, "stop_group"):
            result = smoke.run(path, digest, 2, 12345, self.state, True)
        self.assertFalse(result["success"])
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_timeout_carries_last_stats_and_is_billed(self):
        path, digest = self.prepare()
        def launch(*args, **kwargs):
            smoke.checkpoint(path.parent / "progress.json", {"success": False, "stats": {"observed": {"scheduler_steps": 3}}, "peak_memory_bytes": 1000})
            return types.SimpleNamespace(pid=99999999, returncode=124, poll=lambda: 124, wait=lambda **kw: 124)
        with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), mock.patch.object(smoke.subprocess, "Popen", side_effect=launch), mock.patch.object(smoke, "stop_group"):
            result = smoke.run(path, digest, 2, 12345, self.state, True)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["success"])
        self.assertEqual(result["stats"]["observed"]["scheduler_steps"], 3)
        self.assertEqual(result["peak_memory_bytes"], 1000)
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_experiment_lock_contention_does_not_pause(self):
        path, digest = self.prepare()
        with (self.state / "experiment.2.lock").open("a") as held:
            smoke.fcntl.flock(held, smoke.fcntl.LOCK_EX | smoke.fcntl.LOCK_NB)
            with mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), self.assertRaises(BlockingIOError):
                smoke.run(path, digest, 2, 12345, self.state, True)
        self.assertFalse((self.state / "pause.2").exists())
        self.assertFalse((path.parent / "launch.json").exists())


class MechanismTests(Fixture):
    def setUp(self):
        super().setUp()
        import torch
        torch.set_num_threads(1)
        self.torch = torch

    def fake_runtime(self, request, fail_at=None, mutate=False, nonfinite=False):
        torch = self.torch
        method_name = request["method"]
        stats = dict(completed_steps=0, active_steps=0, model_forwards=0, bypassed=False)
        if method_name == "alg":
            stats["first_frame_sha256"] = "a" * 64
        else:
            stats.update(biased_attention_calls=0, original_attention_calls=0, reference_tokens=880, sequence_tokens=27280)
        dit = torch.nn.Linear(2, 2, bias=False)
        scheduler = types.SimpleNamespace(timesteps=torch.arange(50, dtype=torch.float32))
        scheduler.step = lambda prediction, timestep, sample: sample - prediction * 0.001
        pipe = types.SimpleNamespace(dit=dit, scheduler=scheduler)
        def predict(**kwargs):
            stats["model_forwards"] += 1
            if nonfinite:
                return torch.full_like(kwargs["latents"], float("nan"))
            return kwargs["latents"] * 0.01
        pipe.model_fn = predict
        class FakePipeline:
            def __init__(self):
                self.__dict__.update(vars(pipe))
            def __call__(self, **kwargs):
                self.call = kwargs
                latent = torch.ones(1, 2, 3, 2, 2)
                reference = latent[:, :, :1].clone()
                for i, timestep in enumerate(kwargs["progress_bar_cmd"](self.scheduler.timesteps)):
                    if fail_at == i:
                        raise RuntimeError("fixture inference failure")
                    inputs = dict(latents=latent, first_frame_latents=reference, context=torch.ones(1, 2, 2))
                    forwards = 3 if method_name == "alg" and i < 10 else 2
                    for _ in range(forwards):
                        prediction = self.model_fn(**inputs)
                    latent = self.scheduler.step(prediction, timestep, latent)
                    latent[:, :, :1] = reference
                    stats["completed_steps"] += 1
                    stats["active_steps"] += int(i < 10)
                    if method_name == "dymos":
                        stats["biased_attention_calls"] += 30 * int(i < 10)
                        stats["original_attention_calls"] += 30 * (1 if i < 10 else 2)
                if mutate:
                    with torch.no_grad():
                        self.dit.weight.add_(1)
                return [None] * 121
        fake_pipe = FakePipeline()
        generator = types.ModuleType("gen_jobs")
        generator.NEG = request["negative_prompt"]
        generator.load_image = mock.Mock(return_value="CPU mock image")
        def save_video(video, path, fps, quality):
            if not all(self.tools.values()):
                self.skipTest("ffmpeg/ffprobe not installed")
            subprocess.run([self.tools["ffmpeg"], "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=1280x704:r=24",
                            "-frames:v", "121", "-c:v", "mpeg4", "-threads", "1", "-pix_fmt", "yuv420p", path],
                           check=True, capture_output=True, timeout=60)
        generator.save_video = save_video
        @contextlib.contextmanager
        def method(pipe, **kwargs):
            self.assertEqual(kwargs["checkpoint"], "NONE")
            self.assertEqual({k: kwargs[k] for k in request["parameters"]}, request["parameters"])
            yield stats
        module = types.ModuleType("teacher_" + method_name)
        setattr(module, "teacher_" + method_name, method)
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(sys.modules, {"gen_jobs": generator, "teacher_" + method_name: module}))
        stack.enter_context(mock.patch.object(smoke, "build_pipe", return_value=fake_pipe))
        for name, value in (("device_count", 1), ("set_device", None), ("reset_peak_memory_stats", None), ("max_memory_allocated", 123456)):
            stack.enter_context(mock.patch.object(torch.cuda, name, return_value=value))
        return stack, fake_pipe

    def execute(self, method="alg", **changes):
        path, digest = self.prepare(method)
        request = smoke.verify_request(path, digest)
        stack, pipe = self.fake_runtime(request, **changes)
        original_model, original_step = pipe.model_fn, pipe.scheduler.step
        with stack:
            result = smoke.execute_worker(request, time.monotonic(), digest)
        self.assertIs(pipe.model_fn, original_model)
        self.assertIs(pipe.scheduler.step, original_step)
        self.assertTrue((path.parent / "worker.json").exists())
        return result, path, pipe

    def test_alg_complete_fakepass_with_real_full_decode(self):
        result, path, pipe = self.execute("alg")
        self.assertTrue(result["success"], result["error"])
        self.assertTrue(result["mechanism_PASS"])
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["sota_claim"])
        self.assertTrue(result["parameters_unchanged"])
        self.assertEqual(result["parameters_before"], result["parameters_after"])
        self.assertEqual(result["stats"]["method"]["model_forwards"], 110)
        self.assertEqual(result["stats"]["observed"]["scheduler_steps"], 50)
        self.assertEqual(result["video"]["decoded_frames"], 121)
        self.assertEqual(result["video"]["file"]["sha256"], smoke.record(path.parent / "video.mp4")["sha256"])
        self.assertEqual(pipe.call["seed"], 0)
        self.assertTrue(pipe.call["tiled"])
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_outer_full_fakepass_propagates_verified_worker_and_billing(self):
        path, digest = self.prepare()
        request = smoke.verify_request(path, digest)
        stack, _ = self.fake_runtime(request)
        def launch(command, **kwargs):
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-fixture")
            self.assertTrue(kwargs["start_new_session"])
            start = float(command[command.index("--allocation-start") + 1])
            result = smoke.execute_worker(request, start, digest)
            self.assertTrue(result["success"], result["error"])
            return types.SimpleNamespace(pid=99999999, returncode=0, poll=lambda: 0, wait=lambda **kw: 0)
        original_popen = smoke.subprocess.Popen
        def mock_launch(command, **kwargs):
            if "--worker" in command:
                return launch(command, **kwargs)
            return original_popen(command, **kwargs)
        with stack, mock.patch.object(smoke, "supervisor_scope", return_value="GPU-fixture"), mock.patch.object(smoke.subprocess, "Popen", side_effect=mock_launch), mock.patch.object(smoke, "stop_group"):
            receipt = smoke.run(path, digest, 2, 12345, self.state, True)
        self.assertTrue(receipt["success"], receipt["error"])
        self.assertTrue(receipt["mechanism_PASS"])
        self.assertFalse(receipt["quality_pass"])
        self.assertGreaterEqual(receipt["elapsed_GPUh"], receipt["evidence"]["elapsed_GPUh"])
        self.assertEqual(receipt["stats"]["observed"]["scheduler_steps"], 50)
        self.assertEqual(receipt["peak_memory_bytes"], 123456)
        self.assertFalse((self.state / "pause.2").exists())

    def test_dymos_complete_fakepass_not_cuda_evidence(self):
        result, _, _ = self.execute("dymos")
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["stats"]["method"]["biased_attention_calls"], 300)
        self.assertEqual(result["stats"]["method"]["original_attention_calls"], 2700)
        self.assertEqual(result["stats"]["method"]["reference_tokens"], 880)
        self.assertEqual(result["stats"]["method"]["sequence_tokens"], 27280)
        self.assertFalse(result["quality_pass"])

    def test_mid_inference_failure_has_real_stats_billing_and_restoration(self):
        result, _, _ = self.execute(fail_at=3)
        self.assertFalse(result["success"])
        self.assertTrue(result["failure"])
        self.assertIn("fixture inference failure", result["error"])
        self.assertEqual(result["stats"]["observed"]["scheduler_steps"], 3)
        self.assertTrue(result["parameters_unchanged"])
        self.assertEqual(result["peak_memory_bytes"], 123456)
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_nonfinite_model_prediction_rejected_and_billed(self):
        result, _, _ = self.execute(nonfinite=True)
        self.assertFalse(result["success"])
        self.assertIn("model prediction", result["error"])
        self.assertEqual(result["stats"]["observed"]["scheduler_steps"], 0)
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_each_dit_parameter_independently_hash_checked(self):
        result, _, _ = self.execute(mutate=True)
        self.assertFalse(result["success"])
        self.assertFalse(result["parameters_unchanged"])
        self.assertNotEqual(result["parameters_before"]["weight"], result["parameters_after"]["weight"])
        self.assertIn("parameters changed", result["error"])

    def test_internal_budget_failure_before_cuda_import_is_billed(self):
        path, digest = self.prepare()
        request = smoke.verify_request(path, digest)
        with mock.patch.object(self.torch.cuda, "device_count", side_effect=AssertionError("touched CUDA")):
            result = smoke.execute_worker(request, time.monotonic() - 3351, digest)
        self.assertFalse(result["success"])
        self.assertIn("3350s", result["error"])
        self.assertGreater(result["elapsed_GPUh"], 3350 / 3600)
        self.assertIsNone(result["peak_memory_bytes"])

    def test_missing_steps_stats_never_pass(self):
        observed = dict(scheduler_steps=49, progress_steps=49, finite_checks=10, model_forwards=110)
        stats = dict(completed_steps=49, active_steps=10, bypassed=False, model_forwards=110, first_frame_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            smoke.validate_stats("alg", stats, observed)

    def test_active_steps_and_biased_calls_are_strict(self):
        observed = dict(scheduler_steps=50, progress_steps=50, finite_checks=10, model_forwards=100)
        stats = dict(completed_steps=50, active_steps=10, bypassed=False, model_forwards=100,
                     biased_attention_calls=299, original_attention_calls=2700, reference_tokens=880, sequence_tokens=27280)
        with self.assertRaisesRegex(ValueError, "attention"):
            smoke.validate_stats("dymos", stats, observed)
        stats.update(biased_attention_calls=300, active_steps=11)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            smoke.validate_stats("dymos", stats, observed)

    def test_local_builder_calls_none_and_never_downloads_or_reuses(self):
        generator = types.SimpleNamespace()
        original_config = mock.Mock(side_effect=lambda **kwargs: kwargs)
        generator.ModelConfig = original_config
        configs = []
        def builder(checkpoint):
            self.assertEqual(checkpoint, "NONE")
            for model_id, pattern in (("Wan-AI/Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth"),
                    ("Wan-AI/Wan2.2-TI2V-5B", "diffusion_pytorch_model*.safetensors"),
                    ("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"), ("Wan-AI/Wan2.1-T2V-1.3B", "google/umt5-xxl/")):
                configs.append(generator.ModelConfig(model_id=model_id, origin_file_pattern=pattern))
            return object()
        generator.build_pipe = builder
        first = smoke.build_pipe("NONE", self.models, generator)
        second = smoke.build_pipe("NONE", self.models, generator)
        self.assertIsNot(first, second)
        self.assertIs(generator.ModelConfig, original_config)
        self.assertEqual(len(configs), 8)
        self.assertTrue(all(c["skip_download"] is True and "model_id" not in c for c in configs))
        self.assertEqual(configs[1]["path"], [m["path"] for m in self.models[:3]])
        with self.assertRaisesRegex(ValueError, "NONE"):
            smoke.build_pipe("LORA", self.models, generator)

    def test_observer_composes_with_real_alg_context_cpu(self):
        spec = importlib.util.spec_from_file_location("alg_test_fixture", ROOT / "tests/test_teacher_alg.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        pipe = fixture.FakePipeline()
        observed = dict(model_forwards=0, scheduler_steps=0, progress_steps=50, finite_checks=0)
        original_model, original_step = pipe.model_fn, pipe.scheduler.step
        with smoke.observe(pipe, self.torch, observed, lambda: None):
            with fixture.alg.teacher_alg(pipe, call_kwargs=fixture.CALL, checkpoint="NONE", factor=0.4) as stats:
                pipe.run()
        smoke.validate_stats("alg", stats, observed)
        self.assertIs(pipe.model_fn, original_model)
        self.assertEqual(pipe.scheduler.step, original_step)
        self.assertEqual(observed["model_forwards"], 110)

    def test_observer_composes_with_real_dymos_context_cpu_mock_kernel(self):
        spec = importlib.util.spec_from_file_location("dymos_test_fixture", ROOT / "tests/test_teacher_dymos.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        pipe = fixture.FixturePipe(tiny=True)
        pipe.scheduler.step = lambda prediction, timestep, sample: sample
        observed = dict(model_forwards=0, scheduler_steps=0, progress_steps=50, finite_checks=0)
        with mock.patch.dict(fixture.dymos._FIXED_CALL, fixture.TINY_FIXED), mock.patch.object(fixture.dymos._FusedAttention, "__call__", side_effect=lambda q, *args: q):
            with smoke.observe(pipe, self.torch, observed, lambda: None):
                with fixture.dymos.teacher_dymos(pipe, call_kwargs=pipe.call, checkpoint="NONE", gamma=0.6, phi=1) as stats:
                    for i in range(50):
                        pipe.model_fn(**pipe.kwargs(i, True))
                        prediction = pipe.model_fn(**pipe.kwargs(i, False))
                        pipe.scheduler.step(prediction, pipe.scheduler.timesteps[i], pipe.x)
        self.assertEqual(observed["scheduler_steps"], 50)
        self.assertEqual(observed["model_forwards"], 100)
        self.assertEqual(stats["active_steps"], 10)
        self.assertEqual(stats["biased_attention_calls"], 300)
        self.assertEqual(stats["original_attention_calls"], 2700)

    def test_bad_mp4_never_mechanism_passes(self):
        if not all(self.tools.values()):
            self.skipTest("ffmpeg/ffprobe not installed")
        path = self.base / "bad.mp4"
        path.write_bytes(b"not an mp4")
        with self.assertRaises(subprocess.CalledProcessError):
            smoke.validate_video(path, self.tools)

    def test_actual_frame_count_geometry_and_fps_are_checked(self):
        stream = dict(codec_type="video", width=1280, height=704, nb_read_frames="120", avg_frame_rate="24/1", r_frame_rate="24/1")
        def fake(command, **kwargs):
            return types.SimpleNamespace(stdout=json.dumps({"streams": [stream]}))
        with mock.patch.object(smoke.subprocess, "run", side_effect=fake), self.assertRaisesRegex(ValueError, "121f"):
            smoke.validate_video(self.base / "ignored", self.tools)
        stream.update(nb_read_frames="121", avg_frame_rate="25/1")
        with mock.patch.object(smoke.subprocess, "run", side_effect=fake), self.assertRaisesRegex(ValueError, "fps"):
            smoke.validate_video(self.base / "ignored", self.tools)


if __name__ == "__main__":
    unittest.main()
