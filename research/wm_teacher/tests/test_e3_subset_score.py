import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import score_e3_subset as score


class FakeIQ:
    views = ["left", "center", "right"]

    @classmethod
    def get_list_columns(cls):
        return ["iou_" + view for view in cls.views]

    @classmethod
    def get_scalar_keys(cls):
        return ["variance"]


class ScoreTests(unittest.TestCase):
    def result(self, dim, names, values=None):
        values = values if values is not None else [0.5] * len(names)
        rows = [{"video_path": "/local/" + name, "video_results": value} for name, value in zip(names, values)]
        return {dim: [0.5, rows]}

    def test_official_applicability_is_not_all_71(self):
        info = [{"prompt_en": "one", "dimension": score.DIMS},
                {"prompt_en": "two", "dimension": ["i2v_background", "background_consistency"]},
                {"prompt_en": "three", "dimension": score.DIMS}]
        selected = score.filter_full_info(info, ["one", "two"])
        expected = score.applicable(selected)
        self.assertEqual(expected["i2v_subject"], {"one-0.mp4"})
        self.assertEqual(expected["i2v_background"], {"one-0.mp4", "two-0.mp4"})
        rows = score.dimension_rows(self.result("i2v_subject", ["one-0.mp4", "three-0.mp4"]),
                                    "i2v_subject", expected["i2v_subject"], {"one-0.mp4", "three-0.mp4"})
        self.assertEqual(set(rows), {"one-0.mp4"})

    def test_missing_duplicate_and_nonfinite_score_rejected(self):
        expected = {"a-0.mp4", "b-0.mp4"}
        for names, values in [(["a-0.mp4"], [0.5]),
                              (["a-0.mp4", "a-0.mp4"], [0.5, 0.5]),
                              (["a-0.mp4", "b-0.mp4"], [0.5, float("nan")]),
                              (["a-0.mp4", "b-0.mp4"], [0.5, None])]:
            with self.assertRaises(RuntimeError):
                score.dimension_rows(self.result("aesthetic_quality", names, values), "aesthetic_quality", expected)

    def test_boolean_dynamic_degree_is_official_output(self):
        rows = score.dimension_rows(self.result("dynamic_degree", ["a-0.mp4"], [True]),
                                    "dynamic_degree", {"a-0.mp4"})
        self.assertEqual(score.pool_dimension("dynamic_degree", rows), 1)

    def test_musiq_100_scale_and_weighted_original_consistency(self):
        rows = {"a.mp4": {"video_path": "a.mp4", "video_results": 20},
                "b.mp4": {"video_path": "b.mp4", "video_results": 80}}
        self.assertEqual(score.pool_dimension("imaging_quality", rows), 0.5)
        rows["a.mp4"]["video_results"] = 0.2
        rows["b.mp4"]["video_results"] = 0.8
        self.assertAlmostEqual(score.pool_dimension("subject_consistency", rows, {"a.mp4": 2, "b.mp4": 4}), 0.65)
        with self.assertRaises(RuntimeError):
            score.pool_dimension("subject_consistency", rows)
        for name, count in [("a.mp4", 1), ("b.mp4", 3)]:
            rows[name].update(cnt_per_video=count, video_sim=count * rows[name]["video_results"])
        self.assertAlmostEqual(score.pool_dimension("background_consistency", rows), 0.65)
        rows["b.mp4"]["video_sim"] = 100
        with self.assertRaises(RuntimeError):
            score.pool_dimension("background_consistency", rows)

    def test_total_same_case_bootstrap_with_dimension_specific_ids(self):
        left, right = {}, {}
        names = ["a-0.mp4", "b-0.mp4", "c-0.mp4"]
        for dim in score.DIMS:
            use = names[:2] if dim == "i2v_subject" else names[1:] if dim == "i2v_background" else names
            left[dim] = score.dimension_rows(self.result(dim, use, [60 if dim == "imaging_quality" else 0.6] * len(use)), dim, set(use))
            right[dim] = score.dimension_rows(self.result(dim, use, [50 if dim == "imaging_quality" else 0.5] * len(use)), dim, set(use))
            if dim == "background_consistency":
                for collection in [left, right]:
                    for row in collection[dim].values():
                        row.update(cnt_per_video=120, video_sim=120 * row["video_results"])
        counts = {name: 121 for name in names}
        report = score.paired_vbench(left, right, counts, counts, names, bootstrap=25, seed=3)
        self.assertEqual(report["n_by_dim"]["i2v_subject"], 2)
        self.assertEqual(report["paired"]["imaging_quality"]["ci95"], [0.09999999999999998] * 2)
        self.assertNotIn("per_case_total", report)
        right["i2v_subject"].pop("a-0.mp4")
        with self.assertRaises(RuntimeError):
            score.paired_vbench(left, right, counts, counts, names, bootstrap=3)

    def test_verified_copy_rejects_source_and_stage_wrong_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "input", Path(tmp) / "stage/video"
            src.write_bytes(b"test video bytes")
            with self.assertRaises(RuntimeError):
                score.verified_copy(src, dst, "0" * 64)
            digest = score.verified_copy(src, dst)
            self.assertEqual(digest, score.sha256(src))
            self.assertEqual(dst.read_bytes(), src.read_bytes())
            dst.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                score.verified_copy(src, dst, digest)

    def test_physics_complete_three_views_only(self):
        names = [f"000{i}_{view}_scene.mp4" for i, view in enumerate(sorted(score.VIEWS))]
        self.assertEqual(score.physics_scenes(names), ["scene.mp4"])
        with self.assertRaises(RuntimeError):
            score.physics_scenes(names[:-1])
        with self.assertRaises(RuntimeError):
            score.physics_scenes(names + [names[0]])

    def test_physics_table_missing_view_nonfinite_duplicate(self):
        row = {"scenario": "scene.mp4", **{f"iou_{v}": "[0.5, 0.6]" for v in FakeIQ.views},
               **{f"variance_{v}": "0.7" for v in FakeIQ.views}}
        score.validate_physics_rows([row], {"scene.mp4"}, FakeIQ)
        for broken in [{k: v for k, v in row.items() if k != "iou_right"},
                       {**row, "variance_right": "nan"}, {**row, "iou_right": "[]"}]:
            with self.assertRaises(RuntimeError):
                score.validate_physics_rows([broken], {"scene.mp4"}, FakeIQ)
        with self.assertRaises(RuntimeError):
            score.validate_physics_rows([row, row], {"scene.mp4"}, FakeIQ)

    def wmb_data(self):
        return {"model_name": "base", "preds": {
            "one": {"instruction": ["Score: 2"], "physical_laws": ["No"] * 5, "common_sense": ["Yes", "No"]},
            "two": {"instruction": ["Score: 3"], "physical_laws": ["Yes"] * 5, "common_sense": ["No", "No"]}},
            "accs": {"instruction": [2., 3.], "physical_laws": [True] * 5 + [False] * 5,
                     "common_sense": [False, True, True, True]}}

    def test_wmb_exact_eight_questions_and_original_reward(self):
        data = self.wmb_data()
        result = score.filter_wmb(data, {"two"}, {"one", "two"})
        self.assertEqual(result["accs"]["instruction"], [3.])
        self.assertEqual(score.wmb_vectors(result, ["two"]), [[3., False, False, False, False, False, True, True]])
        data["preds"]["one"]["physical_laws"].pop()
        with self.assertRaises(RuntimeError):
            score.filter_wmb(data, {"two"}, {"one", "two"})
        data = self.wmb_data()
        data["accs"]["instruction"][0] = 1
        with self.assertRaises(RuntimeError):
            score.filter_wmb(data, {"two"}, {"one", "two"})
        with self.assertRaises(RuntimeError):
            score.filter_wmb(self.wmb_data(), {"missing"}, {"one", "two"})

    def test_json_duplicate_keys_and_publish_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.json"
            path.write_text('{"id":1,"id":2}')
            with self.assertRaises(RuntimeError):
                score.read_json(path)
            output = Path(tmp) / "new.json"
            score.write_new(output, {"ok": True})
            with self.assertRaises(RuntimeError):
                score.write_new(output, {"ok": False})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})

    def test_gtref_missing_nonfinite_and_exact_base_filter(self):
        data = {"one": {k: 0.5 for k in score.GT_KEYS}, "two": {k: 0.7 for k in score.GT_KEYS}}
        self.assertEqual(set(score.validate_gtref(data, {"one"}, {"one", "two"})), {"one"})
        with self.assertRaises(RuntimeError):
            score.validate_gtref(data, {"missing"}, {"one", "two"})
        for bad in [None, float("inf")]:
            data["two"]["dreamsim"] = bad
            with self.assertRaises(RuntimeError):
                score.validate_gtref(data, {"one"}, {"one", "two"})

    def test_original_physics_bootstrap_recomputes_ratios_not_average_scene_scores(self):
        import numpy as np
        values = np.asarray([[0.2, 1, 0.2, 1, 0.2, 1, 0, 0], [0.8, 2, 0.8, 2, 0.8, 2, 0, 0]])
        self.assertAlmostEqual(float(score.original_score(values.mean(axis=0))), 1 / 3)
        self.assertNotAlmostEqual(float(score.original_score(values).mean()), 1 / 3)
        with self.assertRaises(RuntimeError):
            score.original_score(np.zeros(8))

    def test_stage_full_decode_wrong_format_hash_and_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.mp4"
            source.write_bytes(b"synthetic bytes")
            work = Path(tmp) / "work"
            jobs = [{"id": "one", "out": str(source)}]
            def decode(path, generated=True):
                return {"path": str(path), "sha256": score.sha256(path), "all_frames_decoded": True,
                        "frames": 121, "fps": 24, "width": 1280, "height": 704}
            _, outputs = score.stage_jobs(jobs, "vbench", "candidate", work, decoder=decode)
            self.assertEqual(outputs["one-0.mp4"]["frames"], 121)
            for key, value in [("frames", 120), ("fps", 16), ("sha256", "0" * 64), ("all_frames_decoded", False)]:
                def broken(path, generated=True):
                    result = decode(path, generated)
                    result[key] = value
                    return result
                with self.assertRaises(RuntimeError):
                    score.stage_jobs(jobs, "vbench", "broken_" + key, work, decoder=broken)
            source.unlink()
            with self.assertRaises(RuntimeError):
                score.stage_jobs(jobs, "vbench", "missing", work, decoder=decode)

    def test_gpu_action_requires_real_explicit_supervision_without_starting_process(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        args = SimpleNamespace(gpu_authorized=False, supervisor_pid=None, gpu_ledger=None)
        with patch.object(score.subprocess, "Popen") as spawn:
            with self.assertRaises(RuntimeError):
                score.gpu_run(["never-start"], args)
            spawn.assert_not_called()

    def test_context_exact_276_no_missing_jobs_or_wrong_official_hash(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import prepare_e3_subset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            experiment = root / "eval/e3_subset_20261025"
            experiment.mkdir(parents=True)
            vbrepo = root / "local-vbench"
            fi = vbrepo / "vbench2_beta_i2v/vbench2_i2v_full_info.json"
            fi.parent.mkdir(parents=True)
            official = [{"prompt_en": f"v{i}", "dimension": list(score.DIMS)} for i in range(71)]
            fi.write_text(json.dumps(official))
            jobs, cases, records = [], [], [{"path": str(fi), "sha256": score.sha256(fi)}]
            tag, checkpoint = "e3C_subset_20261025", str(root / "fixed.safetensors")
            for suite, size in score.COUNTS.items():
                for i in range(size):
                    model = tag + ("_po" if suite == "vbench_po" else "")
                    ident = f"v{i}" if suite in ("vbench", "vbench_po") else f"{suite}{i}"
                    name = ident + ".mp4"
                    if suite == "physiq":
                        view = sorted(score.VIEWS)[i % 3]
                        name = f"{i:04d}_{view}_scene{i // 3}.mp4"
                    output = str(experiment / "videos" / suite / model / name)
                    base = str(root / "base" / suite / name)
                    image = str(root / "input" / suite / (ident + ".png"))
                    job = {"id": ident, "model": tag, "ckpt": checkpoint, "out": output,
                           "image": image, "prompt": "synthetic", "seed": 0}
                    case = {**job, "model": model, "suite": suite, "video": output, "base_video": base}
                    if suite in ("vbench", "vbench_po"):
                        case["dims"] = list(score.DIMS)
                    if suite == "wm":
                        case["gt"] = image
                    if suite == "wmb":
                        case["domain"] = f"domain{i // 10}"
                    jobs.append(job); cases.append(case)
                    records.extend([{"path": base, "sha256": "1" * 64}, {"path": image, "sha256": "2" * 64}])
            jp, cp = experiment / "jobs.jsonl", experiment / "cases.jsonl"
            def write_rows(path, rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_rows(jp, jobs); write_rows(cp, cases)
            plan = {"tag": tag, "checkpoint": {"path": checkpoint}, "selection": {"counts": score.COUNTS},
                    "budget": {"total_gpu_hours": 40, "generation_gpu_hours": 28, "score_gpu_hours": 12},
                    "jobs_path": str(jp), "cases_path": str(cp), "frozen_files": records}
            args = SimpleNamespace(root=root, experiment=experiment, plan_sha256=None, vbench_repo=vbrepo)
            with patch.object(prepare_e3_subset, "verify_plan", side_effect=lambda *a, **kw: (plan, score.load_jsonl(jp), score.load_jsonl(cp))):
                _, groups, _ = score.load_context(args)
                self.assertEqual({key: len(value) for key, value in groups.items()}, score.COUNTS)
                write_rows(jp, jobs[:-1])
                with self.assertRaises(RuntimeError):
                    score.load_context(args)
                write_rows(jp, jobs)
                records[0]["sha256"] = "0" * 64
                with self.assertRaises(RuntimeError):
                    score.load_context(args)

    def test_frozen_default_bootstrap_is_2000_and_default_cpu_hides_gpu(self):
        import inspect
        import os
        from unittest.mock import patch
        self.assertEqual(inspect.signature(score.paired_vbench).parameters["bootstrap"].default, 2000)
        with patch.object(sys, "argv", ["score_e3_subset.py", "validate"]), \
             patch.object(score, "load_context", return_value=({}, {}, [])), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch("builtins.print"):
            score.main()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")

    def test_gpu_ledger_serial_reservation_and_failure_accounting_without_gpu(self):
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            initial = {"budget_gpu_hours": {"generation": 28, "score": 12, "total": 40},
                       "generation_gpu_hours": 20, "score_gpu_hours": 1, "active": None,
                       "runs": [{"phase": "score", "gpus": 1, "used_gpu_hours": 1}]}
            ledger.write_text(json.dumps(initial))
            args = SimpleNamespace(gpu_authorized=True, supervisor_pid=12345, gpu_ledger=ledger,
                                   max_gpu_seconds=60, occupied_gpus=1)
            process = MagicMock()
            process.poll.return_value = 1
            process.returncode = 1
            with patch.dict(score.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                 patch.object(score.os, "kill"), patch.object(score.subprocess, "Popen", return_value=process), \
                 patch.object(score.time, "monotonic", side_effect=[0, 10]):
                with self.assertRaises(RuntimeError):
                    score.gpu_run(["synthetic-original-scorer"], args)
            actual = json.loads(ledger.read_text())
            self.assertIsNone(actual["active"])
            self.assertFalse(actual["runs"][-1]["success"])
            self.assertAlmostEqual(actual["score_gpu_hours"], 1 + 10 / 3600)
            ledger.write_text(json.dumps(initial))
            args.occupied_gpus = 8
            with patch.dict(score.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}),                  patch.object(score.os, "kill"), patch.object(score.subprocess, "Popen", return_value=process),                  patch.object(score.time, "monotonic", side_effect=[0, 10]):
                with self.assertRaises(RuntimeError):
                    score.gpu_run(["synthetic-original-scorer"], args)
            actual = json.loads(ledger.read_text())
            self.assertAlmostEqual(actual["score_gpu_hours"], 1 + 80 / 3600)
            self.assertEqual(actual["runs"][-1]["gpus"], 8)
            actual["active"] = {"phase": "other"}
            ledger.write_text(json.dumps(actual))
            with patch.dict(score.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
                 patch.object(score.os, "kill"), patch.object(score.subprocess, "Popen") as spawn:
                with self.assertRaises(RuntimeError):
                    score.gpu_run(["synthetic-original-scorer"], args)
                spawn.assert_not_called()

    def test_amended_gpu_ledger_exact_authorization_42_failure_billing_and_cap_without_gpu(self):
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        import run_e3_subset_completion as completion
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp)
            plan = experiment / "plan.json"
            plan.write_text('{}')
            plan_sha = score.sha256(plan)
            amendment = experiment / completion.AMENDMENT_FILE
            auth = dict(schema="e3_subset_budget_amendment_v1", parent_plan_sha256=plan_sha,
                        original=completion.ORIGINAL_BUDGET, approved=completion.AMENDED_BUDGET,
                        user_authorized=True, approved_additional_generation_gpu_hours=2)
            amendment.write_text(json.dumps(auth))
            args = SimpleNamespace(experiment=experiment, plan_sha256=plan_sha,
                budget_amendment_sha256=score.sha256(amendment), score_source_sha256=score.sha256(score.__file__),
                gpu_authorized=True, supervisor_pid=12345, gpu_ledger=experiment / "ledger.json",
                max_gpu_seconds=60, occupied_gpus=1)
            initial = dict(phase="score", plan_sha256=plan_sha, score_source_sha256=args.score_source_sha256,
                budget_gpu_hours=completion.AMENDED_BUDGET, budget_amendment_sha256=args.budget_amendment_sha256,
                generation_gpu_hours=29.5, score_gpu_hours=11.9, active=None,
                runs=[dict(phase="score", gpus=1, used_gpu_hours=11.9)])
            args.gpu_ledger.write_text(json.dumps(initial))
            process = MagicMock(returncode=1)
            process.poll.return_value = 1
            with patch.object(completion, "PLAN_SHA", plan_sha), patch.dict(score.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                 patch.object(score.os, "kill"), patch.object(score.subprocess, "Popen", return_value=process) as spawn, \
                 patch.object(score.time, "monotonic", side_effect=[0, 10]):
                with self.assertRaisesRegex(RuntimeError, "exited 1"):
                    score.gpu_run(["synthetic-original-scorer"], args)
                spawn.assert_called_once()
            actual = score.read_json(args.gpu_ledger)
            self.assertAlmostEqual(actual["score_gpu_hours"], 11.9 + 10 / 3600)
            self.assertFalse(actual["runs"][-1]["success"])
            self.assertEqual(actual["generation_gpu_hours"], 29.5)
            self.assertIsNone(actual["active"])
            for changes in (dict(budget_amendment_sha256="0" * 64),
                            dict(budget_gpu_hours=dict(generation=31, score=12, total=43)),
                            dict(budget_gpu_hours=dict(generation=30, score=13, total=43)),
                            dict(score_gpu_hours=11.99, runs=[dict(phase="score", gpus=1, used_gpu_hours=11.99)]),
                            dict(generation_gpu_hours=30.1), dict(runs=[])):
                args.gpu_ledger.write_text(json.dumps(dict(initial, **changes)))
                before = args.gpu_ledger.read_bytes()
                with patch.object(completion, "PLAN_SHA", plan_sha), patch.dict(score.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                     patch.object(score.os, "kill"), patch.object(score.subprocess, "Popen") as spawn:
                    with self.assertRaises(RuntimeError):
                        score.gpu_run(["never-start"], args)
                    spawn.assert_not_called()
                self.assertEqual(args.gpu_ledger.read_bytes(), before)
            args.gpu_ledger.write_text(json.dumps(initial))
            amendment.write_text(json.dumps(dict(auth, user_authorized=False)))
            args.budget_amendment_sha256 = score.sha256(amendment)
            with patch.object(completion, "PLAN_SHA", plan_sha), patch.dict(score.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
                 patch.object(score.os, "kill"), patch.object(score.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "authorization"):
                    score.gpu_run(["never-start"], args)
                spawn.assert_not_called()

    def test_complete_synthetic_summary_all_dimensions_gtref_physics_wmb_domains(self):
        import numpy as np
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import patch
        class BootIQ:
            def __init__(self, frame):
                self.frame = frame
            def compute_final_score_orig(self):
                return float(score.original_score(self.frame.to_numpy().mean(axis=0)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment, work = root / "eval/e3_subset_20261025", root / "work"
            experiment.mkdir(parents=True)
            (experiment / "plan.json").write_text('{"synthetic":true}')
            digest = score.sha256(experiment / "plan.json")
            groups = {suite: [] for suite in score.COUNTS}
            tag = "e3C_subset_20261025"
            official = [{"prompt_en": "a", "dimension": score.DIMS},
                        {"prompt_en": "b", "dimension": [d for d in score.DIMS if d != "i2v_subject"]}]
            for suite in groups:
                size = 14 if suite == "wmb" else 6 if suite == "physiq" else 2
                model = tag + ("_po" if suite == "vbench_po" else "")
                base = "base_po" if suite == "vbench_po" else "base"
                for i in range(size):
                    ident = "ab"[i] if suite in ("vbench", "vbench_po", "wm") else f"item{i}"
                    name = ident + "-0.mp4" if suite in ("vbench", "vbench_po", "wm") else ident + ".mp4"
                    if suite == "physiq":
                        name = f"{i:04d}_{sorted(score.VIEWS)[i % 3]}_scene{i // 3}.mp4"
                    row = {"id": ident, "model": model, "out": str(experiment / "videos" / suite / name)}
                    if suite == "wm":
                        row["gt"] = "synthetic-gt"
                    if suite == "wmb":
                        row["domain"] = f"domain{i // 2}"
                    groups[suite].append(row)
                for arm in (model, base):
                    outputs = {}
                    for row in groups[suite]:
                        name = row["id"] + "-0.mp4" if suite in ("vbench", "vbench_po", "wm") else Path(row["out"]).name
                        local = work / suite / arm / "videos" / name
                        local.parent.mkdir(parents=True, exist_ok=True)
                        local.write_bytes(b"synthetic full decoded video")
                        outputs[name] = {"path": str(local), "source": str(local), "sha256": score.sha256(local),
                                         "all_frames_decoded": True, "frames": 121, "fps": 24, "width": 1280, "height": 704}
                    score.write_new(work / suite / arm / "decode.json", {"suite": suite, "model": arm,
                                    "all_frames_decoded": True, "outputs": outputs})
                score.write_new(score.score_path(experiment, suite, base, "provenance.json"),
                                {"source_sha256": {}, "plan_sha256": digest})
                if suite in ("vbench", "vbench_po", "wm"):
                    info = official if suite != "wm" else [{"prompt_en": x, "dimension": score.DIMS} for x in "ab"]
                    expected = score.applicable(info)
                    for arm, value in [(model, 0.6), (base, 0.5)]:
                        for dim in score.DIMS:
                            names = sorted(expected[dim])
                            result = self.result(dim, names, [100*value if dim == "imaging_quality" else value]*len(names))
                            for row in result[dim][1]:
                                if dim == "background_consistency":
                                    row.update(cnt_per_video=120, video_sim=120*value)
                            result[dim][0] = value
                            score.write_new(score.score_path(experiment, suite, arm, arm + "__" + dim + "_eval_results.json"), result)
                if suite == "wm":
                    for arm, value in [(model, 0.4), (base, 0.5)]:
                        score.write_new(score.score_path(experiment, suite, arm, "gtref.json"),
                                        {x: {k: value for k in score.GT_KEYS} for x in "ab"})
                if suite == "physiq":
                    for arm in (model, base):
                        path = score.score_path(experiment, suite, arm, "physics.csv")
                        path.parent.mkdir(parents=True, exist_ok=True); path.write_text("synthetic")
                        score.write_new(path.with_name("physics_metrics.json"), {"final_score_orig": 0.5})
                if suite == "wmb":
                    for arm, instruction in [(model, 2), (base, 1)]:
                        data = {"model_name": arm, "preds": {}, "accs": {"instruction": [], "physical_laws": [], "common_sense": []}}
                        for row in groups[suite]:
                            data["preds"][row["id"]] = {"instruction": [f"Score: {instruction}"],
                                                       "physical_laws": ["No"]*5, "common_sense": ["No"]*2}
                            data["accs"]["instruction"].append(float(instruction))
                            data["accs"]["physical_laws"].extend([True]*5)
                            data["accs"]["common_sense"].extend([True]*2)
                        score.write_new(score.score_path(experiment, suite, arm, "result.json"), data)
            args = SimpleNamespace(experiment=experiment, work_dir=work, bootstrap=8, seed=3, bench=root)
            plan = {"tag": tag, "checkpoint": {"path": "synthetic"}}
            values = np.asarray([[0.5, 1, 0.5, 1, 0.5, 1, 0, 0]]*2)
            frame = pd.DataFrame(values)
            with patch.object(score, "load_iqtable", return_value=BootIQ), \
                 patch.object(score, "physics_values", return_value=(values, frame, 0.5)):
                result = score.summary(args, plan, groups, official)
                self.assertEqual(result["bootstrap"], 8)
                self.assertEqual(result["vbench"]["vbench"]["n_by_dim"]["i2v_subject"], 1)
                self.assertAlmostEqual(result["gtref"]["metrics"]["dreamsim"]["delta"], -0.1)
                self.assertEqual(result["physics"]["delta"], 0)
                self.assertEqual(len(result["wmb"]["domains"]), 7)
                self.assertEqual(result["wmb"]["aggregate"]["delta"], 1)
                self.assertTrue(result["not_full_acceptance"] and result["not_sota"])
                missing = score.score_path(experiment, "vbench", tag, tag + "__i2v_subject_eval_results.json")
                missing.unlink()
                with self.assertRaises(OSError):
                    score.summary(args, plan, groups, official)

    def test_missing_piq_decoders_use_original_cpu_vbench_bridge(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        row = {"path": "/tmp/staged.mp4", "sha256": "1"*64, "all_frames_decoded": True,
               "frames": 121, "fps": 24, "width": 1280, "height": 704}
        with patch.object(score.importlib.util, "find_spec", return_value=None), \
             patch.object(score.sys, "executable", "/tmp/piq-python"), \
             patch.object(score.subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(row))) as execute:
            self.assertEqual(score.decode_local(Path(row["path"])), row)
            command = execute.call_args.args[0]
            self.assertEqual(command[0], "/dockerdata/venv_vbench/bin/python")
            self.assertEqual(command[-1], "true")
            self.assertEqual(execute.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(execute.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
