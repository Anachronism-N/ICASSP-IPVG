import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_e3_subset_completion as c


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.no_writers_patch = patch.object(c, "no_writers")
        self.no_writers_patch.start()
        self.addCleanup(self.no_writers_patch.stop)
        self.root = Path(self.temp.name) / "project"
        self.exp = self.root / "eval/e3_subset_20261025"
        self.exp.mkdir(parents=True)
        scripts = self.root / "scripts"
        scripts.mkdir()
        frozen = []
        for name in ("gen_jobs.py", "run_gen.sh", "gpu_hold.py", "run_e3_subset_node.py", "run_e3_subset_recovery.py"):
            path = scripts / name
            shutil.copyfile(ROOT / "scripts" / name, path)
            frozen.append(c.n.record(path))
        image = self.root / "static.png"
        image.write_bytes(b"frozen static image")
        frozen.append(c.n.record(image))
        cp = self.root / "C.safetensors"
        cp.write_bytes(b"frozen C")
        frozen.append(c.n.record(cp))
        self.jobs, cases = [], []
        for suite, count in c.n.COUNTS.items():
            for i in range(count):
                out = self.exp / "videos" / suite / f"case{i}.mp4"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(f"video fixture {suite} {i}".encode())
                job = dict(id=f"case{i}", model=c.n.TAG, ckpt=str(cp), image=str(image), prompt=f"object motion — {i}",
                           out=str(out), seed=0, cfg=5.0, shift=5.0, steps=50)
                self.jobs.append(job)
                contract = {k: job[k] for k in ("image", "prompt", "seed", "cfg", "shift", "steps")}
                cases.append(dict(job, suite=suite, video=str(out), input_contract_sha256=c.n.sha(c.input_encoded(contract))))
        job_path, case_path = self.exp / "jobs.jsonl", self.exp / "cases.jsonl"
        job_path.write_bytes(b"".join(c.n.encoded(j) for j in self.jobs))
        case_path.write_bytes(b"".join(c.n.encoded(j) for j in cases))
        self.plan = dict(schema="e3_subset_v1", private=True, tag=c.n.TAG, selection=dict(counts=c.n.COUNTS),
                         budget=dict(generation_gpu_hours=28, score_gpu_hours=12, total_gpu_hours=40),
                         jobs_path=str(job_path), cases_path=str(case_path), manifest_files=[c.n.record(job_path), c.n.record(case_path)],
                         frozen_files=frozen, checkpoint=c.n.record(cp), runtime=dict(python="/opt/conda/envs/torch-base/bin/python"))
        c.n.exclusive(self.exp / "plan.json", self.plan)
        digest = c.sha(self.exp / "plan.json")
        self.pf = dict(schema="e3_subset_node_preflight_v1", rank=0, plan_sha256=digest, hostname=os.uname().nodename,
                       model_content_sha256="a" * 64)
        for rank in range(4):
            c.n.exclusive(self.exp / f"node_{rank}_preflight.json", dict(self.pf, rank=rank))
        self.args = argparse.Namespace(root=self.root, experiment=self.exp, plan_sha256=digest, run=True, node_rank=0,
            gpu_authorized=True, hostname=os.uname().nodename, preflight_sha256=c.sha(self.exp / "node_0_preflight.json"),
            source_sha256=c.sha(Path(c.__file__)), max_repair_gpu_hours=2, work_dir=Path(self.temp.name) / "stage")
        for index, path in enumerate(c.original_paths(self.exp)):
            rank = index if index < 4 else index - 3
            elapsed = 90.6 if index in (1, 2) else 2941 if index == 3 else 900
            row = dict(rank=rank, plan_sha256=digest, start=10000, end=10000 + elapsed, elapsed_seconds=elapsed,
                       gpu_hours=8 * elapsed / 3600, exit_code=1 if index in (1, 2) else 130 if index == 3 else 0,
                       error=None, interrupted_signal=None, owned_tree_released=True, cap_exceeded=False)
            command, _ = c.n.command_for(self.plan, rank)
            launch = dict(rank=rank, plan_sha256=digest, command=command)
            if index < 4:
                row.update(timed_out=index == 3, cap_seconds=3000, model_content_sha256=self.pf["model_content_sha256"])
            else:
                row["run_id"] = "recovery_a1"
                launch.update(run_id="recovery_a1", max_seconds=3000, script_sha256=c.RECOVERY_SHA)
            c.n.exclusive(path, row)
            c.n.exclusive(path.with_name(path.name.replace(".finished.json", ".launch.json")), launch)
        self.before = {p: p.read_bytes() for p in c.original_paths(self.exp)}
        self.decoder = Mock(side_effect=self.decoded)

    def decoded(self, path):
        return dict(sha256=c.sha(path), frames=121, actual_count_frames=121, fps=24, width=1280, height=704,
                    all_frames_decoded=True, decoder="ffprobe-count_frames+ffmpeg-xerror")

    def update(self, path, **changes):
        row = c.read(path)
        path.write_bytes(c.n.encoded(dict(row, **changes)))

    def run_complete(self, repair=None):
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256), patch.object(c, "preflight", return_value=(self.pf, self.plan)), \
                patch.object(c, "run_repair", repair or Mock(side_effect=AssertionError("GPU forbidden"))):
            return c.complete(self.args, decoder=self.decoder)

    def verify(self):
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256):
            return c.validate_global(self.exp, self.args.plan_sha256, getattr(self.args, "budget_amendment_sha256", None))

    def test_real_no_writers_rejects_generation_process(self):
        self.no_writers_patch.stop()
        for executable in (b"gen_jobs.py", b"run_gen.sh"):
            entry = Mock(name="process")
            entry.name = "987654"
            cmdline = Mock()
            cmdline.read_bytes.return_value = b"python" + bytes([0]) + executable + bytes([0])
            entry.__truediv__ = Mock(return_value=cmdline)
            with patch.object(Path, "iterdir", return_value=iter([entry])):
                with self.assertRaisesRegex(ValueError, "Active generation"):
                    c.no_writers()
        self.no_writers_patch.start()

    def test_all_276_already_valid_zero_gpu_preserves_original_failures(self):
        gpu = Mock(side_effect=AssertionError("GPU forbidden"))
        result = self.run_complete(gpu)
        gpu.assert_not_called()
        self.assertEqual((result["status"], result["exit_code"], result["decode_ok"], result["total_frames"]),
                         ("COMPLETE", 0, 276, 33396))
        self.assertEqual(result["actual_repair_gpu_hours"], 0)
        self.assertEqual(self.decoder.call_count, 552)
        used, receipts = self.verify()
        self.assertAlmostEqual(used, sum(c.read(p)["gpu_hours"] for p in self.before))
        self.assertEqual(len(receipts), 7)
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertEqual(c.read(self.exp / "node_3.finished.json")["exit_code"], 130)
        self.assertEqual(c.read(self.exp / "node_1.finished.json")["exit_code"], 1)
        self.assertFalse(any(Path(r["path"]).is_symlink() for r in c.read(Path(result["decode_manifest"]))["outputs"].values()))
        with self.assertRaisesRegex(ValueError, "Single global"):
            self.run_complete(gpu)

    def test_node0_preflight_rejects_node3_rank_sha_or_wrong_host_before_fullhash(self):
        with patch.object(c.n, "verify_inputs") as verify:
            self.args.node_rank = 3
            with self.assertRaisesRegex(ValueError, "node-rank 0"):
                c.preflight(self.args)
            self.args.node_rank = 0
            self.args.preflight_sha256 = c.sha(self.exp / "node_3_preflight.json")
            with self.assertRaisesRegex(ValueError, "node0 preflight SHA"):
                c.preflight(self.args)
            self.args.preflight_sha256 = c.sha(self.exp / "node_0_preflight.json")
            self.args.hostname = "other-node3-host"
            with self.assertRaisesRegex(ValueError, "Wrong node0 hostname"):
                c.preflight(self.args)
            verify.assert_not_called()
        self.assertFalse((self.exp / (c.RUN_ID + ".launch.json")).exists())

    def test_global_score_requires_same_node0_host_and_exact_launch_rank(self):
        result = self.run_complete()
        receipt = self.exp / (c.RUN_ID + ".finished.json")
        launch = self.exp / (c.RUN_ID + ".launch.json")
        original = receipt.read_bytes()
        self.assertEqual((result["node_rank"], result["preflight_path"], result["hostname"]),
                         (0, str(self.exp / "node_0_preflight.json"), os.uname().nodename))
        self.assertEqual(c.read(Path(result["decode_manifest"]))["hostname"], os.uname().nodename)
        for changes in (dict(node_rank=3), dict(hostname="another-node"),
                        dict(preflight_path=str(self.exp / "node_3_preflight.json"))):
            self.update(receipt, **changes)
            with self.assertRaisesRegex(ValueError, "same node0 hostname"):
                self.verify()
            receipt.write_bytes(original)
        self.update(launch, node_rank=3)
        self.update(receipt, launch_sha256=c.sha(launch))
        with self.assertRaisesRegex(ValueError, "launch node0/hostname identity"):
            self.verify()

    def test_default_inspect_never_gpu_or_launch(self):
        self.args.run, self.args.gpu_authorized = False, False
        result = self.run_complete()
        self.assertEqual(result["status"], "INSPECTED")
        self.assertEqual(result["decode_ok"], 276)
        self.assertFalse((self.exp / (c.RUN_ID + ".launch.json")).exists())

    def test_wait_six_and_released_before_any_quarantine(self):
        path = c.original_paths(self.exp)[-1]
        path.rename(path.with_suffix(".saved"))
        claim = Path(self.jobs[0]["out"] + ".claim")
        claim.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "all six"):
            self.run_complete()
        self.assertTrue(claim.exists())
        self.assertFalse((self.exp / (c.RUN_ID + ".quarantine")).exists())

    def test_unknown_claim_preserved(self):
        unknown = self.exp / "videos/wmb/unapproved.mp4.claim"
        unknown.write_bytes(b"unknown")
        with self.assertRaisesRegex(ValueError, "Unknown output/claim/tmp"):
            self.run_complete()
        self.assertEqual(unknown.read_bytes(), b"unknown")
        self.assertFalse((self.exp / (c.RUN_ID + ".launch.json")).exists())

    def test_owned_remnants_only_moved_with_stat_and_hash(self):
        out = Path(self.jobs[0]["out"])
        before = out.read_bytes()
        claim, tmp = Path(str(out) + ".claim"), Path(str(out) + ".tmp.mp4")
        claim.write_bytes(b"")
        tmp.write_bytes(b"partial")
        result = self.run_complete()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(out.read_bytes(), before)
        self.assertFalse(claim.exists())
        self.assertFalse(tmp.exists())
        self.assertEqual(len(result["quarantine"]), 2)
        for record in result["quarantine"]:
            self.assertEqual(c.sha(record["destination"]), record["sha256"])
            self.assertIn("st_ino", record["stat"])

    def test_user_interrupt_unknown_errors_and_missing_cost_refused(self):
        path = self.exp / "node_3.finished.json"
        for changes in (dict(interrupted_signal=2), dict(error="unknown"), dict(gpu_hours=None),
                        dict(exit_code=1), dict(owned_tree_released=False), dict(cap_exceeded=True)):
            path.write_bytes(self.before[path])
            self.update(path, **changes)
            with self.assertRaises(ValueError):
                c.originals(self.exp, self.args.plan_sha256)
        path.write_bytes(self.before[path])

    def test_recovery_expected_timeout_requires_fixed_source_cap_and_no_interrupt(self):
        path = self.exp / "node_1_recovery_a1.finished.json"
        self.update(path, exit_code=130, elapsed_seconds=2941, gpu_hours=8 * 2941 / 3600, end=12941)
        used, _ = c.originals(self.exp, self.args.plan_sha256)
        self.assertGreater(used, 0)
        self.update(path, interrupted_signal=15)
        with self.assertRaisesRegex(ValueError, "User interrupt"):
            c.originals(self.exp, self.args.plan_sha256)
        self.update(path, interrupted_signal=None)
        launch = self.exp / "node_1_recovery_a1.launch.json"
        self.update(launch, script_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Recovery cap/source"):
            c.originals(self.exp, self.args.plan_sha256)

    def amendment(self):
        row = dict(schema="e3_subset_budget_amendment_v1", parent_plan_sha256=self.args.plan_sha256,
                   original=c.ORIGINAL_BUDGET, approved=c.AMENDED_BUDGET, user_authorized=True,
                   approved_additional_generation_gpu_hours=2)
        path = self.exp / c.AMENDMENT_FILE
        path.write_bytes(c.n.encoded(row))
        self.args.budget_amendment_sha256 = c.sha(path)
        self.args.max_repair_gpu_hours = 3.4
        return path, row

    def test_exact_amendment_all_original_timeouts_twenty_repairs_budget_sha_and_cost(self):
        path, _ = self.amendment()
        amendment_before = path.read_bytes()
        for i, receipt in enumerate(c.original_paths(self.exp)):
            if i in (1, 2):
                continue
            changes = dict(exit_code=130, elapsed_seconds=2941, end=12941, gpu_hours=8 * 2941 / 3600)
            if i < 4:
                changes["timed_out"] = True
            self.update(receipt, **changes)
        before = {p: p.read_bytes() for p in c.original_paths(self.exp)}
        for job in self.jobs[:20]:
            Path(job["out"]).unlink()
        def repair(args, plan, manifest, prior, used):
            jobs = [json.loads(line) for line in manifest.read_bytes().splitlines()]
            self.assertEqual(jobs, self.jobs[:20])
            for job in jobs:
                Path(job["out"]).write_bytes(b"amended same-input repaired fixture")
            allowed, wall, term = c.allowance(used, args.max_repair_gpu_hours, c.AMENDED_BUDGET)
            return dict(start=20000, end=24080, elapsed_seconds=4080, gpu_hours=3.4, gpus=list(c.GPUS),
                        allowed_gpu_hours=allowed, wall_cap_seconds=wall, term_seconds=term, cap_exceeded=False,
                        owned_tree_released=True, interrupted_signal=None, timed_out=False, error=None,
                        worker_exit_codes=[0, 0, 0], commands=[r[0] for r in c.repair_commands(plan, manifest, args.root)])
        result = self.run_complete(Mock(side_effect=repair))
        self.assertEqual((result["exit_code"], result["repair_job_count"], result["decode_ok"]), (0, 20, 276))
        self.assertGreater(result["total_generation_gpu_hours"], 28)
        self.assertLessEqual(result["total_generation_gpu_hours"], 30)
        self.assertEqual(result["budget_amendment_sha256"], c.sha(path))
        self.assertEqual(c.read(self.exp / (c.RUN_ID + ".launch.json"))["budget_amendment_sha256"], c.sha(path))
        hours, hashes = self.verify()
        self.assertEqual(len(hashes), 7)
        self.assertAlmostEqual(hours, result["actual_prior_gpu_hours"] + 3.4)
        import run_e3_subset_scoring as driver
        self.args.score_source_sha256 = driver.ENGINE_SHA
        self.args.approved_bundle_sha256 = "b" * 64
        self.args.gpu_ledger = self.exp / "score_ledger.json"
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256):
            score_hours, score_receipts = driver.generation(self.args)
            self.assertEqual((score_hours, score_receipts), (hours, hashes))
            ledger = driver.ledger(self.args, score_hours, score_receipts)
            self.assertEqual(ledger["budget_gpu_hours"], c.AMENDED_BUDGET)
            self.assertEqual(ledger["global_completion"]["actual_repair_gpu_hours"], 3.4)
            self.assertEqual(ledger["generation_receipts_sha256"], hashes)
        self.assertEqual(path.read_bytes(), amendment_before)
        for receipt, data in before.items():
            self.assertEqual(receipt.read_bytes(), data)
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256):
            with self.assertRaisesRegex(ValueError, "exact approved sidecar SHA"):
                c.validate_global(self.exp, self.args.plan_sha256)
        self.update(self.exp / (c.RUN_ID + ".finished.json"), budget_amendment_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "budget amendment identity"):
            self.verify()

    def test_amendment_wrong_schema_authorization_parent_caps_or_sha_refused_before_launch(self):
        path, original = self.amendment()
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256):
            for changes in (dict(schema="wrong"), dict(user_authorized=False), dict(parent_plan_sha256="0" * 64),
                            dict(approved=dict(generation=31, score=12, total=43)),
                            dict(approved=dict(generation=30, score=13, total=43)),
                            dict(original=dict(generation=28.0, score=12, total=40))):
                with self.subTest(changes=changes):
                    path.write_bytes(c.n.encoded(dict(original, **changes)))
                    self.args.budget_amendment_sha256 = c.sha(path)
                    with self.assertRaises(ValueError):
                        c.complete(self.args, self.decoder)
                    self.assertFalse((self.exp / (c.RUN_ID + ".launch.json")).exists())
            path.write_bytes(c.n.encoded(original))
            for digest in (None, "0" * 64):
                self.args.budget_amendment_sha256 = digest
                with self.assertRaises(ValueError):
                    c.complete(self.args, self.decoder)
            path.unlink()
            self.args.budget_amendment_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "Missing approved"):
                c.complete(self.args, self.decoder)
        self.decoder.assert_not_called()
        self.assertEqual(c.allowance(29.5, 3.4, c.AMENDED_BUDGET), (.5, 600, 570))
        with self.assertRaises(ValueError):
            c.allowance(26.5, 3.41, c.AMENDED_BUDGET)

    def test_negative_exhausted_budget_and_no_future_reservations(self):
        for used, approved in ((-1, 2), (28, 2), (28.1, 2), (float("nan"), 2), (20, 3), (20, -1)):
            with self.assertRaises(ValueError):
                c.allowance(used, approved)
        remaining, wall, term = c.allowance(27.5, 2)
        self.assertEqual((remaining, wall, term), (.5, 600, 570))
        self.assertEqual(c.allowance(20, 1), (1, 1200, 1170))

    def test_no_budget_blocks_missing_output_without_cleanup_or_launch(self):
        Path(self.jobs[0]["out"]).unlink()
        claim = Path(self.jobs[0]["out"] + ".claim")
        claim.touch()
        with patch.object(c, "allowance", side_effect=ValueError("BLOCKED: insufficient actual remaining generation budget")):
            with self.assertRaisesRegex(ValueError, "insufficient"):
                self.run_complete()
        self.assertTrue(claim.exists())
        self.assertFalse((self.exp / (c.RUN_ID + ".launch.json")).exists())

    def repair_success(self, args, plan, manifest, prior, used):
        jobs = [json.loads(line) for line in manifest.read_bytes().splitlines()]
        self.assertEqual(jobs, [self.jobs[0]])
        for job in jobs:
            Path(job["out"]).write_bytes(b"repaired video fixture")
        allowed, wall, term = c.allowance(used, args.max_repair_gpu_hours)
        return dict(start=20000, end=20120, elapsed_seconds=120, gpu_hours=.1, gpus=list(c.GPUS),
                    allowed_gpu_hours=allowed, wall_cap_seconds=wall, term_seconds=term, cap_exceeded=False,
                    owned_tree_released=True, interrupted_signal=None, timed_out=False, error=None, worker_exit_codes=[0, 0, 0],
                    commands=[r[0] for r in c.repair_commands(plan, manifest, args.root)])

    def test_missing_only_manifest_identical_protocol_and_three_gpu_billing(self):
        Path(self.jobs[0]["out"]).unlink()
        unaffected = {Path(j["out"]): Path(j["out"]).read_bytes() for j in self.jobs[1:]}
        result = self.run_complete(Mock(side_effect=self.repair_success))
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["repair_job_count"], 1)
        self.assertAlmostEqual(result["actual_repair_gpu_hours"], .1)
        used, receipts = self.verify()
        self.assertAlmostEqual(used, result["actual_prior_gpu_hours"] + .1)
        self.assertIn(str(self.exp / (c.RUN_ID + ".finished.json")), receipts)
        for path, original in unaffected.items():
            self.assertEqual(path.read_bytes(), original)
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_bad_format_quarantined_not_success_overwritten(self):
        bad = Path(self.jobs[0]["out"])
        original = bad.read_bytes()
        def decoder(path):
            if Path(path).read_bytes() == original:
                return dict(self.decoded(path), frames=120, actual_count_frames=120)
            return self.decoded(path)
        self.decoder.side_effect = decoder
        result = self.run_complete(Mock(side_effect=self.repair_success))
        self.assertEqual(result["exit_code"], 0)
        preserved = result["quarantine"][0]
        self.assertEqual(Path(preserved["destination"]).read_bytes(), original)
        self.assertNotEqual(bad.read_bytes(), original)

    def test_failed_repair_billed_nonzero_receipt_no_original_rewrite(self):
        Path(self.jobs[0]["out"]).unlink()
        def failed(*args):
            return dict(self.repair_success(*args), worker_exit_codes=[1, 0, 0], error="worker failure")
        result = self.run_complete(Mock(side_effect=failed))
        self.assertEqual((result["status"], result["exit_code"]), ("INCOMPLETE", 1))
        self.assertEqual(result["actual_repair_gpu_hours"], .1)
        self.assertAlmostEqual(result["total_generation_gpu_hours"], result["actual_prior_gpu_hours"] + .1)
        with self.assertRaisesRegex(ValueError, "failed/interrupted"):
            self.verify()
        for path, original in self.before.items():
            self.assertEqual(path.read_bytes(), original)

    def test_hash_drift_and_fabricated_completion_refused(self):
        result = self.run_complete()
        path = self.exp / (c.RUN_ID + ".finished.json")
        row = c.read(path)
        self.update(path, decode_ok=275)
        with self.assertRaisesRegex(ValueError, "aggregate decode"):
            self.verify()
        path.write_bytes(c.n.encoded(row))
        decoded = Path(result["decode_manifest"])
        record = c.read(decoded)
        record["outputs"].pop(self.jobs[0]["out"])
        decoded.write_bytes(c.n.encoded(record))
        self.update(path, decode_manifest_sha256=c.sha(decoded))
        with self.assertRaisesRegex(ValueError, "Fake/partial"):
            self.verify()

    def test_original_hash_drift_and_source_stage_hash_drift_refused(self):
        self.run_complete()
        original = c.original_paths(self.exp)[0]
        self.update(original, extra="drift")
        with self.assertRaisesRegex(ValueError, "six SHA"):
            self.verify()
        original.write_bytes(self.before[original])
        Path(self.jobs[0]["out"]).write_bytes(b"source drift")
        with self.assertRaisesRegex(ValueError, "source/local stage SHA"):
            self.verify()

    def test_frame_format_faked_manifest_refused(self):
        result = self.run_complete()
        manifest = Path(result["decode_manifest"])
        decoded = c.read(manifest)
        decoded["outputs"][self.jobs[0]["out"]]["width"] = 640
        manifest.write_bytes(c.n.encoded(decoded))
        self.update(self.exp / (c.RUN_ID + ".finished.json"), decode_manifest_sha256=c.sha(manifest))
        with self.assertRaisesRegex(ValueError, "frame/format"):
            self.verify()

    def test_generator_argv_and_env_are_original_three_shards(self):
        commands = c.repair_commands(self.plan, self.exp / "repair.jsonl", self.root)
        self.assertEqual(len(commands), 3)
        for shard, (argv, env) in enumerate(commands):
            self.assertEqual(argv[:3], [self.plan["runtime"]["python"], str(self.root / "scripts/gen_jobs.py"), str(self.exp / "repair.jsonl")])
            self.assertEqual(argv[3:], [str(shard), "3", "--models", c.n.TAG])
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], str((2, 3, 4)[shard]))
            self.assertEqual(env["PYTHONPATH"], str(self.root / "DiffSynth-Studio"))
            self.assertEqual(env["DIFFSYNTH_MODEL_BASE_PATH"], "/dockerdata/models")
            self.assertEqual(env["WM_CHECKPOINT_CACHE_DIR"], "/dockerdata/models/wm_checkpoints")
            self.assertEqual(env["DIFFSYNTH_SKIP_DOWNLOAD"], "true")

    def test_static_image_or_case_contract_drift_refused(self):
        Path(self.jobs[0]["image"]).write_bytes(b"changed static image")
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256):
            with self.assertRaisesRegex(ValueError, "SHA256/size mismatch"):
                c.protocol(self.exp / "plan.json", self.args.plan_sha256)

    def test_external_process_existing_pause_and_unowned_files_never_touched(self):
        state = Path(self.temp.name) / "hold"
        state.mkdir()
        mapping = {g: dict(uuid=f"gpu-{g}", used_mib=0) for g in c.GPUS}
        with patch.object(c.n, "STATE", state), patch.object(c.hold, "query_gpus", return_value=mapping), \
                patch.object(c.hold, "query_processes", return_value={"gpu-2": {777}}):
            with self.assertRaisesRegex(ValueError, "External/current-scope"):
                c.free_gpus()
            (state / "pause.2").write_bytes(b"other scope")
            with self.assertRaisesRegex(ValueError, "Existing GPU pause"):
                c.free_gpus()
            self.assertEqual((state / "pause.2").read_bytes(), b"other scope")

    def test_cpu_supervisor_actual_elapsed_billing_shell_false_and_flags_release(self):
        state = Path(self.temp.name) / "hold"
        state.mkdir()
        used, hashes = c.originals(self.exp, self.args.plan_sha256)
        mapping = {g: dict(uuid=f"gpu-{g}", used_mib=0) for g in c.GPUS}
        (self.exp / "repair.jsonl").write_bytes(c.n.encoded(self.jobs[0]))
        class Clock:
            now = 0
            def monotonic(self):
                self.now += 1
                return self.now
            def wall(self):
                return 10000 + self.now
        clock = Clock()
        children = []
        def spawn(*args, **kwargs):
            self.assertIs(kwargs["shell"], False)
            self.assertIs(kwargs["start_new_session"], True)
            child = Mock(pid=90000 + len(children), returncode=0)
            child.poll.return_value = 0
            children.append(child)
            return child
        with patch.object(c, "PLAN_SHA", self.args.plan_sha256), patch.object(c.n, "STATE", state), \
                patch.object(c, "preflight", return_value=(self.pf, self.plan)), patch.object(c, "free_gpus", return_value=mapping), \
                patch.object(c.hold, "query_gpus", return_value=mapping), patch.object(c.hold, "query_processes", return_value={}), \
                patch.object(c, "time") as tm, patch.object(c.subprocess, "Popen", side_effect=spawn), \
                patch.object(c.n, "group_alive", return_value=False):
            tm.monotonic.side_effect = clock.monotonic
            tm.time.side_effect = clock.wall
            tm.sleep.side_effect = lambda _: None
            stage = c.run_repair(self.args, self.plan, self.exp / "repair.jsonl", hashes, used)
        self.assertIsNone(stage["error"])
        self.assertEqual(stage["worker_exit_codes"], [0, 0, 0])
        self.assertEqual(stage["gpu_hours"], 3 * stage["elapsed_seconds"] / 3600)
        self.assertGreater(stage["gpu_hours"], 0)
        self.assertTrue(stage["owned_tree_released"])
        self.assertEqual(list(state.glob("pause.*")), [])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class RealCPUDecodeTests(unittest.TestCase):
    def test_real_full_frame_decode_and_wrong_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for frames in (121, 120):
                path = Path(tmp) / f"{frames}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1", "-f", "lavfi", "-i",
                    "color=c=black:s=1280x704:r=24", "-frames:v", str(frames), "-c:v", "mpeg4", "-q:v", "3",
                    "-threads", "1", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
                if frames == 121:
                    row = c.decode(path)
                    self.assertEqual((row["actual_count_frames"], row["frames"], row["width"], row["height"], row["fps"]),
                                     (121, 121, 1280, 704, 24))
                    self.assertEqual(row["sha256"], c.sha(path))
                else:
                    with self.assertRaises(c.BadVideo):
                        c.decode(path)
            broken = Path(tmp) / "broken.mp4"
            broken.write_bytes(b"not a video")
            with self.assertRaises(c.BadVideo):
                c.decode(broken)


if __name__ == "__main__":
    unittest.main()
