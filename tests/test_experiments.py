"""Regression tests for resource admission, split isolation and resume integrity."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ipvg.cli import main
from ipvg.common import load_json, write_json
from ipvg.data import write_manifest
from ipvg.experiments import plan_experiment, run_campaign, run_experiment, run_process


def record(number=1, split="train", identity=None, source=None):
    return {"sample_id": "sample-{}".format(number), "identity_id": identity or "id-{}".format(number),
            "source_video_id": source or "video-{}".format(number), "reference_images": ["face.png"],
            "prompt": "A person walks.", "track": 1, "split": split}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "data.jsonl"
        write_manifest(self.manifest, [record()])

    def config(self, run_id="trial", **changes):
        value = {"schema_version": 1, "run_id": run_id, "task": "training", "purpose": "development",
                 "hypothesis": "Verify a small test checkpoint is persisted.", "manifest": "data.jsonl",
                 "split": "train", "resources": {"gpus": 0, "timeout_s": 3, "max_gpu_hours": 0},
                 "expected_artifacts": ["*.safetensors"],
                 "backend": {"kind": "command", "command": [sys.executable, "-c",
                     "from pathlib import Path; import sys; (Path(sys.argv[1])/'model.safetensors').write_bytes(b'test fixture')",
                     "{output_dir}"]}}
        value.update(changes)
        path = self.root / (run_id + ".json")
        write_json(path, value)
        return path

    def test_training_must_declare_artifacts_and_reserve_full_timeout(self):
        with self.assertRaisesRegex(ValueError, "expected_artifacts"):
            plan_experiment(self.config(expected_artifacts=[]), root=self.root)
        with self.assertRaisesRegex(ValueError, "exceeds declared"):
            plan_experiment(self.config(resources={"gpus": 8, "timeout_s": 3600, "max_gpu_hours": 7}), root=self.root)
        plan = plan_experiment(self.config(resources={"gpus": 8, "timeout_s": 3600, "max_gpu_hours": 8}), root=self.root)
        self.assertEqual(plan["max_backend_gpu_hours"], 8)
        self.assertFalse((self.root / "runs").exists())

    def test_portable_run_and_sample_names_fail_before_running(self):
        for name in ("CON", "NUL.txt", "trailing."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                plan_experiment(self.config(run_id=name), root=self.root)
        rows = [record(1), record(2)]
        rows[0]["sample_id"], rows[1]["sample_id"] = "Alice", "alice"
        write_manifest(self.manifest, rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plan_experiment(self.config(), root=self.root)

    def test_leakage_is_rejected_before_requested_split_filtering(self):
        write_manifest(self.manifest, [record(1, "train", source="shared"), record(2, "dev", source="shared")])
        with self.assertRaisesRegex(ValueError, "leaks"):
            plan_experiment(self.config(split="train"), root=self.root)

    def test_training_cannot_use_dev_or_final_generation_to_access_holdout(self):
        write_manifest(self.manifest, [record(1, "dev")])
        with self.assertRaisesRegex(ValueError, "training cannot consume"):
            plan_experiment(self.config(split="dev"), root=self.root)
        write_manifest(self.manifest, [record(1, "holdout")])
        path = self.config(purpose="final_generation")
        value = load_json(path)
        value.pop("split")
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "inference only"):
            plan_experiment(path, root=self.root)

    def test_development_holdout_denied_but_explicit_final_inference_can_plan(self):
        write_manifest(self.manifest, [record(1, "holdout")])
        path = self.config(task="inference")
        value = load_json(path)
        value.pop("split")
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "holdout"):
            plan_experiment(path, root=self.root)
        value["purpose"] = "final_generation"
        write_json(path, value)
        self.assertEqual(plan_experiment(path, root=self.root)["sample_count"], 1)

    def test_exit_zero_without_checkpoint_is_failed(self):
        result = run_experiment(self.config(backend={"kind": "command", "command": [sys.executable, "-c", "pass"]}), root=self.root)
        self.assertEqual(result["status"], "failed")
        self.assertIn("artifact missing", result["error"])
        self.assertEqual(load_json(self.root / "runs/trial/result.json")["status"], "failed")

    def test_nonzero_exit_is_failed_and_failed_run_is_not_resumed(self):
        path = self.config(backend={"kind": "command", "command": [sys.executable, "-c", "raise SystemExit(7)"]})
        result = run_experiment(path, root=self.root)
        self.assertEqual(result["status"], "failed")
        self.assertIn("code 7", result["error"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_experiment(path, root=self.root, resume=True)

    def test_success_resume_checks_artifacts_frozen_manifest_and_resources_request(self):
        path = self.config()
        first = run_experiment(path, root=self.root)
        self.assertEqual(first["status"], "completed")
        request = load_json(self.root / "runs/trial/request.json")
        self.assertEqual(request["resources"], load_json(path)["resources"])
        self.assertEqual(run_experiment(path, root=self.root, resume=True), first)
        frozen = self.root / "runs/trial/manifest.jsonl"
        frozen.write_text(frozen.read_text(encoding="utf-8").replace("walks", "sits"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            run_experiment(path, root=self.root, resume=True)

    def test_resume_rejects_modified_checkpoint_or_deleted_hash_inventory(self):
        path = self.config()
        original = run_experiment(path, root=self.root)
        artifact = self.root / "runs/trial/outputs/model.safetensors"
        artifact.write_bytes(b"modified fixture")
        with self.assertRaisesRegex(ValueError, "hashes changed"):
            run_experiment(path, root=self.root, resume=True)
        artifact.write_bytes(b"test fixture")
        original["artifact_sha256"] = {}
        write_json(self.root / "runs/trial/result.json", original)
        with self.assertRaisesRegex(ValueError, "hashes changed"):
            run_experiment(path, root=self.root, resume=True)

    def test_timeout_records_failure(self):
        path = self.config(resources={"gpus": 0, "timeout_s": .15, "max_gpu_hours": 0},
                           backend={"kind": "command", "command": [sys.executable, "-c", "import time; time.sleep(5)"]})
        start = time.monotonic()
        result = run_experiment(path, root=self.root)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timed out", result["error"])
        self.assertLess(time.monotonic() - start, 4)

    def test_backend_stdin_is_closed(self):
        path = self.config(backend={"kind": "command", "command": [sys.executable, "-c",
            "from pathlib import Path; import sys; assert sys.stdin.read()==''; (Path(sys.argv[1])/'model.safetensors').write_bytes(b'ok')", "{output_dir}"]})
        self.assertEqual(run_experiment(path, root=self.root)["status"], "completed")

    def test_artifact_path_patterns_are_portable_and_confined(self):
        for pattern in ("../escape", "..\\escape", "C:/escape", "/escape"):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, "relative output patterns"):
                plan_experiment(self.config(expected_artifacts=[pattern]), root=self.root)

    def campaign(self, configs, **changes):
        campaign = {"schema_version": 1, "campaign_id": "finite", "experiments": [str(p) for p in configs],
                    "max_trials": 2, "max_gpu_hours": 0, "report": "campaign-report.json"}
        campaign.update(changes)
        path = self.root / "campaign.json"
        write_json(path, campaign)
        return path

    def test_campaign_reserves_timeouts_before_any_backend_launch(self):
        configs = [self.config(run_id=name, resources={"gpus": 8, "timeout_s": 3600, "max_gpu_hours": 8})
                   for name in ("a", "b")]
        path = self.campaign(configs, max_gpu_hours=15)
        with patch("ipvg.experiments.run_experiment") as execute:
            with self.assertRaisesRegex(ValueError, "total GPU-hour budget"):
                run_campaign(path, root=self.root)
            execute.assert_not_called()

    def test_campaign_enforces_trial_limit_and_rejects_final_generation_search(self):
        configs = [self.config(run_id=name) for name in ("a", "b")]
        with self.assertRaisesRegex(ValueError, "max_trials"):
            run_campaign(self.campaign(configs, max_trials=1), root=self.root)
        value = load_json(configs[0])
        value.update(task="inference", purpose="final_generation")
        write_json(configs[0], value)
        with self.assertRaisesRegex(ValueError, "final_generation"):
            run_campaign(self.campaign(configs[:1]), root=self.root)

    def test_campaign_backend_error_is_recorded_and_stops_search(self):
        configs = [self.config(run_id=name) for name in ("a", "b")]
        with patch("ipvg.experiments.run_experiment", side_effect=OSError("run directory inaccessible")) as execute:
            result = run_campaign(self.campaign(configs), root=self.root)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("inaccessible", result["results"][0]["error"])
        self.assertEqual(load_json(self.root / "campaign-report.json")["status"], "failed")

    def test_campaign_report_cannot_overwrite_inputs_or_run_artifacts(self):
        config = self.config()
        for target in (str(config), str(self.manifest), "runs/trial/result.json", "campaign.json"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "must not overwrite"):
                run_campaign(self.campaign([config], report=target), root=self.root)

    def test_cli_split_report_cannot_overwrite_source_or_new_manifest(self):
        original = self.manifest.read_bytes()
        output = self.root / "split.jsonl"
        for report in (self.manifest, output):
            with self.subTest(report=report), contextlib.redirect_stderr(io.StringIO()):
                code = main(["data", "split", str(self.manifest), "--output", str(output), "--report", str(report)])
                self.assertEqual(code, 2)
                self.assertEqual(self.manifest.read_bytes(), original)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
