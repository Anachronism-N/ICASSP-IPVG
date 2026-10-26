import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_e3_subset_scoring as d


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / "project"
        experiment = root / "eval/e3_subset_20261025"
        experiment.mkdir(parents=True)
        (experiment / "plan.json").write_text('{}')
        self.args = argparse.Namespace(root=root, experiment=experiment, work_dir=Path(self.temp.name) / "stage",
            bench=root / "bench", vbench_repo=root / "vbench", gpu_ledger=experiment / "score_ledger.json",
            plan_sha256=d.s.sha256(experiment / "plan.json"), score_source_sha256=d.s.sha256(d.s.__file__),
            run=False, gpu_authorized=False, supervisor_pid=None, occupied_gpus=1, gpu="0", max_gpu_seconds=1800,
            vbench_python="vbench-python", piq_python="piq-python", wmb_python="wmb-python", judge="original-judge",
            bootstrap=2000, seed=20261025, approved_bundle_sha256=None)
        for path in (self.args.root / "scripts/gt_ref_metrics.py", root / "scripts/summarize_eval.py", root / "scripts/analyze_merge_metrics.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Original CPU fixture source\n")
        bundle = d.score_bundle(self.args)
        self.args.approved_bundle_sha256 = bundle["bundle_sha256"]
        d.s.write_new(experiment / "score_source_bundle.json", bundle)
        self.plan = {"tag": "C", "checkpoint": {"path": "/private/checkpoint"}}
        self.groups, self.info = {}, []
        self.old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.addCleanup(self.restore_env)
        for i in range(6):
            rank = i if i < 4 else i - 3
            hours = .201 if i in (1, 2) else 1.
            receipt = dict(rank=rank, plan_sha256=self.args.plan_sha256, owned_tree_released=True,
                cap_exceeded=False, gpu_hours=hours, elapsed_seconds=hours * 3600 / 8,
                exit_code=1 if i in (1, 2) else 0, error=None)
            if i >= 4:
                receipt["run_id"] = "recovery_a1"
            name = f"node_{rank}" + ("_recovery_a1" if i >= 4 else "") + ".finished.json"
            d.s.write_new(experiment / name, receipt)

    def restore_env(self):
        if self.old_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.old_env

    def update_receipt(self, name, **changes):
        path = self.args.experiment / name
        data = d.s.read_json(path)
        path.write_text(json.dumps(dict(data, **changes)))

    def fixture(self):
        def decode(path, generated=True):
            return dict(path=str(path), sha256=d.s.sha256(path), all_frames_decoded=True,
                        frames=121, fps=24, width=1280, height=704)
        self.decoder = Mock(side_effect=decode)
        def stage(*args, **kwargs):
            return d.s.stage_jobs(*args, **kwargs, decoder=self.decoder)
        self.stage = stage
        source_score = self.args.root / "historical.json"
        source_score.write_text('{}')
        for suite, count in d.s.COUNTS.items():
            jobs = []
            model = "C_po" if suite == "vbench_po" else "C"
            base = "base_po" if suite == "vbench_po" else "base"
            for i in range(count):
                name = f"case{i}"
                out = self.args.experiment / "videos" / suite / (name + "-0.mp4")
                original = self.args.root / "historical" / suite / out.name
                for path in (out, original):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"same video {suite} {i}".encode())
                jobs.append(dict(id=name, out=str(out), base_out=str(original), base_out_sha256=d.s.sha256(original)))
            self.groups[suite] = jobs
            if suite == "vbench":
                self.info = [dict(prompt_en=j["id"], dimension=d.s.DIMS) for j in jobs]
            stage(jobs, suite, base, self.args.work_dir, source="base_out")
            directory = d.s.score_path(self.args.experiment, suite, base, "unused").parent
            d.s.write_new(directory / "provenance.json", dict(plan_sha256=self.args.plan_sha256,
                source_sha256={str(source_score): d.s.sha256(source_score)}))
            if suite in ("vbench", "vbench_po", "wm"):
                for dim in d.s.DIMS:
                    rows = [dict(video_path=j["id"] + "-0.mp4", video_results=50 if dim == "imaging_quality" else .5,
                                 cnt_per_video=120, video_sim=60) for j in jobs]
                    d.s.write_new(directory / (base + "__" + dim + "_eval_results.json"), {dim: [.5, rows]})
                if suite == "wm":
                    for j in jobs:
                        j["gt"] = j["base_out"]
                    d.s.write_new(directory / "gtref.json", {j["id"]: {k: .5 for k in d.s.GT_KEYS} for j in jobs})
            elif suite == "wmb":
                preds = {Path(j["out"]).stem: dict(instruction=["Score: 2"], physical_laws=["No"] * 5,
                                                   common_sense=["Yes", "No"]) for j in jobs}
                d.s.write_new(directory / "result.json", dict(model_name="base", preds=preds,
                    accs=dict(instruction=[2.] * count, physical_laws=[True] * (5 * count), common_sense=[False, True] * count)))
            else:
                (directory / "physics.csv").write_text("fixture")
                d.s.write_new(directory / "physics_metrics.json", {"final_score_orig": .5})
        self.decoder.reset_mock()

    def drive(self, cpu=None):
        with patch.object(d, "PLAN_SHA", self.args.plan_sha256), patch.object(d, "ORIGINAL_STAGE", self.stage),              patch.object(d.s, "load_context", return_value=(self.plan, self.groups, self.info)),              patch.object(d, "cpu_call", cpu or Mock()), patch.object(d, "ORIGINAL_GPU") as gpu:
            result = d.drive(self.args)
            gpu.assert_not_called()
            return result

    def test_missing_receipt_and_nonfinite_cost_fail(self):
        path = self.args.experiment / "node_3.finished.json"
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaises(RuntimeError):
            d.generation(self.args)
        path.write_bytes(saved)
        self.update_receipt(path.name, gpu_hours=float("nan"))
        with self.assertRaises(RuntimeError):
            d.generation(self.args)

    def test_failed_original_cost_included_and_latest_return_required(self):
        hours, _ = d.generation(self.args)
        self.assertAlmostEqual(hours, 4.402)
        self.update_receipt("node_2_recovery_a1.finished.json", exit_code=1)
        with self.assertRaises(RuntimeError):
            d.generation(self.args)

    def test_generation_budget_and_release_fail_closed(self):
        for name in ("node_0.finished.json", "node_3.finished.json", "node_1_recovery_a1.finished.json", "node_2_recovery_a1.finished.json"):
            self.update_receipt(name, gpu_hours=6.6, elapsed_seconds=2970)
        for rank in (1, 2):
            self.update_receipt(f"node_{rank}.finished.json", gpu_hours=1.1, elapsed_seconds=495)
        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            d.generation(self.args)
        self.update_receipt("node_3.finished.json", owned_tree_released=False)
        with self.assertRaisesRegex(RuntimeError, "release/cap"):
            d.generation(self.args)

    def test_ledger_preserves_usage_and_rejects_changed_phase_hash(self):
        hours, receipts = d.generation(self.args)
        d.ledger(self.args, hours, receipts)
        data = d.s.read_json(self.args.gpu_ledger)
        data.update(score_gpu_hours=2, runs=[dict(phase="score", gpus=1, used_gpu_hours=2)])
        self.args.gpu_ledger.write_text(json.dumps(data))
        before = self.args.gpu_ledger.read_bytes()
        self.assertEqual(d.ledger(self.args, hours, receipts)["score_gpu_hours"], 2)
        self.assertEqual(self.args.gpu_ledger.read_bytes(), before)
        for key, bad in (("phase", "generation"), ("plan_sha256", "wrong"), ("score_source_sha256", "wrong"), ("approved_bundle_sha256", "wrong")):
            self.args.gpu_ledger.write_text(json.dumps(dict(data, **{key: bad})))
            with self.assertRaises(RuntimeError):
                d.ledger(self.args, hours, receipts)

    def amendment(self):
        row = dict(schema="e3_subset_budget_amendment_v1", parent_plan_sha256=self.args.plan_sha256,
                   original=d.BUDGET, approved=d.completion.AMENDED_BUDGET, user_authorized=True,
                   approved_additional_generation_gpu_hours=2)
        path = self.args.experiment / d.completion.AMENDMENT_FILE
        path.write_text(json.dumps(row))
        self.args.budget_amendment_sha256 = d.s.sha256(path)
        return path

    def test_amended_ledger_42_exact_sidecar_hash_preserves_cost_and_score_cap(self):
        amendment = self.amendment()
        with patch.object(d.completion, "PLAN_SHA", self.args.plan_sha256):
            state = d.ledger(self.args, 29.5, {"prior-failed-and-completion": "hash"})
            self.assertEqual(state["budget_gpu_hours"], dict(generation=30, score=12, total=42))
            self.assertEqual(state["budget_amendment_sha256"], d.s.sha256(amendment))
            state.update(score_gpu_hours=11.99, runs=[dict(phase="score", gpus=1, used_gpu_hours=11.99)])
            self.args.gpu_ledger.write_text(json.dumps(state))
            before = self.args.gpu_ledger.read_bytes()
            self.assertEqual(d.ledger(self.args, 29.5, {"prior-failed-and-completion": "hash"})["score_gpu_hours"], 11.99)
            self.assertEqual(self.args.gpu_ledger.read_bytes(), before)
            for changes in (dict(budget_amendment_sha256="0" * 64), dict(budget_gpu_hours=d.BUDGET),
                            dict(score_gpu_hours=12.1), dict(generation_gpu_hours=29.4)):
                self.args.gpu_ledger.write_text(json.dumps(dict(state, **changes)))
                with self.assertRaises((RuntimeError, ValueError)):
                    d.ledger(self.args, 29.5, {"prior-failed-and-completion": "hash"})
            self.args.gpu_ledger.write_bytes(before)
            amendment.write_text(amendment.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "exact approved sidecar SHA"):
                d.ledger(self.args, 29.5, {"prior-failed-and-completion": "hash"})
            self.assertEqual(self.args.gpu_ledger.read_bytes(), before)

    def test_amended_summary_only_with_exact_verified_sidecar_not_arbitrary_extra_score(self):
        self.amendment()
        report = self.summary_fixture()
        report.update(budget=d.completion.AMENDED_BUDGET, budget_amendment_sha256=self.args.budget_amendment_sha256)
        with patch.object(d.completion, "PLAN_SHA", self.args.plan_sha256):
            public = d.public_report(report, self.args)
            self.assertEqual(public["budget"], dict(generation=30, score=12, total=42))
            self.assertEqual(public["guards"]["generation_cap"], 30)
            self.assertEqual(public["guards"]["score_cap"], 12)
            with self.assertRaisesRegex(RuntimeError, "verified sidecar context"):
                d.public_report(report)
            report["budget_amendment_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "amendment SHA"):
                d.public_report(report, self.args)
            report["budget"] = dict(generation=30, score=13, total=43)
            with self.assertRaisesRegex(RuntimeError, "Unapproved summary budget"):
                d.public_report(report, self.args)

    def test_default_prepare_zero_gpu_and_cache_same_bytes(self):
        self.fixture()
        result = self.drive()
        self.assertEqual(result["status"], "prepared")
        self.assertFalse(result["GPU_used"])
        self.assertEqual(self.decoder.call_count, 276)
        self.decoder.reset_mock()
        self.drive()
        self.decoder.assert_not_called()

    def test_cache_source_and_stage_mutations_rejected(self):
        self.fixture()
        cache = {}
        jobs = self.groups["vbench"]
        stage = d.cached_stage(cache)
        videos, rows = stage(jobs, "vbench", "base", self.args.work_dir, source="base_out")
        self.assertIs(rows, stage(jobs, "vbench", "base", self.args.work_dir, source="base_out")[1])
        for path in (Path(jobs[0]["base_out"]), videos / "case0-0.mp4"):
            original = path.read_bytes()
            path.write_bytes(b"x" * len(original))
            with self.assertRaises(RuntimeError):
                stage(jobs, "vbench", "base", self.args.work_dir, source="base_out")
            path.write_bytes(original)

    def test_duplicate_and_nonfinite_existing_metric_preserved(self):
        self.fixture()
        path = d.result_paths(self.args, "vbench", "base", d.s.DIMS[0])[0]
        original = d.s.read_json(path)
        for change in ("duplicate", "nonfinite"):
            data = copy.deepcopy(original)
            if change == "duplicate":
                data[d.s.DIMS[0]][1][1] = data[d.s.DIMS[0]][1][0]
            else:
                data[d.s.DIMS[0]][1][0]["video_results"] = float("nan")
            path.write_text(json.dumps(data))
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                self.drive()
            self.assertEqual(path.read_bytes(), before)

    def test_existing_candidate_without_hash_evidence_not_overwritten(self):
        self.fixture()
        self.args.run, self.args.gpu_authorized = True, True
        self.args.supervisor_pid = os.getppid()
        target = d.result_paths(self.args, "vbench", "C", d.s.DIMS[0])[0]
        d.s.write_new(target, {"partial": True})
        before = target.read_bytes()
        with self.assertRaises(RuntimeError):
            self.drive()
        self.assertEqual(target.read_bytes(), before)

    def test_pause_boundary_wrapped_inside_original_accounting(self):
        self.args.gpu_authorized = True
        self.args.supervisor_pid = os.getppid()
        original = Mock()
        run = d.boundary(self.args, original)
        def inspect(command, args, **kwargs):
            self.assertEqual(command[2:8], ["--action", "exec", "--gpus", "0", "--", "original-python"])
            self.assertEqual(command[-1], "metric")
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(args.max_gpu_seconds, 1800)
        original.side_effect = inspect
        run(["original-python", "metric"], self.args)
        original.assert_called_once()
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        data = d.s.read_json(self.args.gpu_ledger)
        data.update(score_gpu_hours=12, runs=[dict(phase="score", gpus=1, used_gpu_hours=12)])
        self.args.gpu_ledger.write_text(json.dumps(data))
        with self.assertRaises(RuntimeError):
            run(["original-python", "metric"], self.args)
        self.assertEqual(original.call_count, 1)

    def test_full_mocked_scoring_and_complete_resume_no_gpu(self):
        self.fixture()
        self.args.run, self.args.gpu_authorized = True, True
        self.args.supervisor_pid = os.getppid()
        def engine(args, context, action, suite="", dim=""):
            if action == "validate-physics":
                return
            if action == "summary":
                path = args.experiment / "summary.json"
                if not path.exists():
                    d.s.write_new(path, self.summary_fixture())
                return
            base = "base_po" if suite == "vbench_po" else "base"
            model = "C_po" if suite == "vbench_po" else "C"
            metric = dim if action == "vbench" else action
            source = d.result_paths(args, suite, base, metric)
            destination = d.result_paths(args, suite, model, metric)
            for src, dst in zip(source, destination):
                d.s.verified_copy(src, dst)
            if action in ("vbench", "wmb"):
                state = d.s.read_json(args.gpu_ledger)
                state["score_gpu_hours"] += .01
                state["runs"].append(dict(phase="score", gpus=1, used_gpu_hours=.01))
                args.gpu_ledger.write_text(json.dumps(state))
        mocked = Mock(side_effect=engine)
        result = self.drive(mocked)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["GPU_used"])
        self.assertAlmostEqual(result["score_gpu_hours"], .25)
        mocked.reset_mock()
        result = self.drive(mocked)
        self.assertFalse(result["GPU_used"])
        self.assertFalse(any(call.args[2] in ("vbench", "wmb", "gtref", "physics") for call in mocked.call_args_list))

    def test_run_auth_and_sidecar_gates_before_gpu(self):
        self.fixture()
        self.args.run = True
        (self.args.experiment / "score_source_bundle.json").unlink()
        for changes in (dict(gpu_authorized=False), dict(gpu_authorized=True, supervisor_pid=os.getpid()),
                        dict(gpu_authorized=True, supervisor_pid=os.getppid(), occupied_gpus=2),
                        dict(gpu_authorized=True, supervisor_pid=os.getppid(), occupied_gpus=1)):
            for key, value in changes.items():
                setattr(self.args, key, value)
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                self.drive()

    def test_partial_late_suite_prevents_all_new_metric_actions(self):
        self.fixture()
        self.args.run, self.args.gpu_authorized = True, True
        self.args.supervisor_pid = os.getppid()
        target = d.result_paths(self.args, "wmb", "C", "wmb")[0]
        d.s.write_new(target, {"partial": True})
        cpu = Mock()
        with self.assertRaisesRegex(RuntimeError, "Partial result"):
            self.drive(cpu)
        self.assertFalse(any(call.args[2] in ("vbench", "wmb", "gtref", "physics") for call in cpu.call_args_list))

    def test_dynamic_remaining_budget_bounds_action_and_failed_parent(self):
        self.args.gpu_authorized, self.args.supervisor_pid = True, os.getppid()
        hours, receipts = d.generation(self.args)
        state = d.ledger(self.args, hours, receipts)
        state.update(score_gpu_hours=11.99, runs=[dict(phase="score", gpus=1, used_gpu_hours=11.99)])
        self.args.gpu_ledger.write_text(json.dumps(state))
        original = Mock()
        d.boundary(self.args, original)(["fake"], self.args)
        self.assertGreater(self.args.max_gpu_seconds, 0)
        self.assertLessEqual(self.args.max_gpu_seconds, 21)
        with patch.object(d.os, "kill", side_effect=ProcessLookupError):
            with self.assertRaises(ProcessLookupError):
                d.boundary(self.args, original)(["fake"], self.args)
        self.assertEqual(original.call_count, 1)

    def test_extra_staged_video_and_claims_block_before_gpu(self):
        self.fixture()
        extra = self.args.work_dir / "vbench/base/videos/extra.mp4"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(RuntimeError, "Local staged video coverage"):
            self.drive()
        extra.unlink()
        claim = Path(self.groups["wm"][0]["out"] + ".claim")
        claim.write_text("pending")
        with self.assertRaisesRegex(RuntimeError, "claims/temporary"):
            self.drive()

    def test_source_bundle_no_sidecar_cycle_and_rejects_mutation(self):
        bundle = d.verify_bundle(self.args)
        self.assertEqual(len(bundle["files"]), 6)
        self.assertIn("run_e3_subset_completion.py", bundle["files"])
        self.assertNotIn("score_source_bundle.json", bundle["files"])
        self.assertEqual(bundle["files"]["gt_ref_metrics.py"]["path"], str(self.args.root / "scripts/gt_ref_metrics.py"))
        self.assertFalse((self.args.vbench_repo / "gt_ref_metrics.py").exists())
        self.assertEqual(d.score_bundle(self.args), bundle)
        for path in (self.args.root / "scripts/gt_ref_metrics.py", self.args.root / "scripts/summarize_eval.py"):
            before = path.read_bytes()
            path.write_bytes(before + b"# changed")
            with self.assertRaisesRegex(RuntimeError, "bundle/sidecar"):
                d.verify_bundle(self.args)
            path.write_bytes(before)
        self.args.approved_bundle_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "bundle/sidecar"):
            d.verify_bundle(self.args)

    def test_original_gpu_runner_charges_failed_pause_wall(self):
        # CPU fake Popen only: pause/wrapped command fails after a measured wall interval.
        self.args.gpu_authorized, self.args.supervisor_pid = True, os.getppid()
        hours, receipts = d.generation(self.args)
        d.ledger(self.args, hours, receipts)
        process = Mock(returncode=1)
        process.poll.return_value = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with patch.object(d.s.subprocess, "Popen", return_value=process), patch.object(d.s.time, "monotonic", side_effect=[0, 90]):
            with self.assertRaises(RuntimeError):
                d.ORIGINAL_GPU(["fake-holder-pause"], self.args)
        row = d.s.read_json(self.args.gpu_ledger)
        self.assertAlmostEqual(row["score_gpu_hours"], 90 / 3600)
        self.assertFalse(row["runs"][0]["success"])
        self.assertIsNone(row["active"])

    def test_cpu_call_uses_separate_original_environments_without_cuda(self):
        context = self.args.experiment / "ctx.json"
        context.write_text('{}')
        with patch.object(d.subprocess, "run") as run:
            for action, python in (("gtref", "vbench-python"), ("physics", "piq-python"), ("summary", "piq-python")):
                d.cpu_call(self.args, context, action)
                self.assertEqual(run.call_args.args[0][0], python)
                self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def summary_fixture(self):
        pair = dict(delta=0., ci95=[0., 0.])
        point = dict(pair, left=.5, right=.5)
        row = dict(left={dim: .5 for dim in d.s.DIMS}, right={dim: .5 for dim in d.s.DIMS},
                   paired={dim: copy.deepcopy(pair) for dim in d.s.DIMS}, n_by_dim={dim: 22 for dim in d.s.DIMS},
                   aggregate_only_total=copy.deepcopy(point))
        wmb = dict(point, n_cases=70, eight_question_paired=[copy.deepcopy(pair) for _ in range(8)])
        return dict(subset_counts=d.s.COUNTS, bootstrap=2000, seed=20261025,
            not_full_acceptance=True, not_sota=True, vbench={k: copy.deepcopy(row) for k in ("vbench", "vbench_po", "wm")},
            gtref=dict(n_cases=22, K=12, metrics={k: copy.deepcopy(point) for k in d.s.GT_KEYS[:3]}, case_ids=["private-case"]),
            physics=dict(point, n_scenes=14, views_per_scene=3, scene_ids=["private-scene"]),
            wmb=dict(n_cases=70, questions_per_case=8, aggregate=wmb, domains={str(i): dict(copy.deepcopy(wmb), n_cases=10) for i in range(7)}),
            budget=d.BUDGET, source_bundle_sha256=self.args.approved_bundle_sha256,
            checkpoint={"path": "/private/ckpt"}, source_sha256={"/private/result": "abc"})

    def test_summary_missing_dimension_metric_and_private_identity_scrub(self):
        report = self.summary_fixture()
        report["future_secret"] = "private-secret"
        report["vbench"]["wm"]["aggregate_only_total"]["future_path"] = "/private/future"
        report["physics"]["future_secret"] = ["private-case"]
        report["wmb"]["aggregate"]["extra_per_case"] = {"private-case": .5}
        public = d.public_report(report)
        self.assertNotIn("private", json.dumps(public))
        self.assertTrue(public["quality_false"])
        report["vbench"]["wm"]["paired"].pop(d.s.DIMS[0])
        with self.assertRaises(RuntimeError):
            d.public_report(report)
        report["vbench"]["wm"]["paired"][d.s.DIMS[0]] = dict(delta=0., ci95=[0., 0.])
        report["gtref"]["metrics"].pop("dreamsim")
        with self.assertRaises(RuntimeError):
            d.public_report(report)


if __name__ == "__main__":
    unittest.main()
