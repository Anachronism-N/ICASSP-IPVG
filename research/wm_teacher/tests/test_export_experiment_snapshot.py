import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "export_experiment_snapshot", Path(__file__).resolve().parents[1] /
    "scripts/export_experiment_snapshot.py")
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)


class PublicationAllowlistTests(unittest.TestCase):
    def test_p1_code_and_plan_are_selected_without_private_cases(self):
        selected = {"docs/33_Teacher优化开发消融计划.md",
                    "docs/34_Teacher_CFG开发消融结论.md",
                    "eval/diag/teacher_p1_results_20261019.json",
                    "docs/35_Teacher机制诊断与随机标签审计计划.md",
                    "docs/36_Teacher优化相关工作与可证伪方案.md",
                    "docs/38_标签语义校准与C1推理条件实验.md",
                    "docs/39_C1原与校订Caption完整结论.md", "eval/diag/C1_caption_public_results_20261020.json",
    "eval/diag/C1_FT1_membership_public_summary_20261020.json",
                    "docs/40_四节点受控训练资源规划.md",
                    "docs/41_窗口标注链与命名歧义验收.md",
                    "docs/42_LoRA有限训练技术Smoke.md",
                    "docs/43_四节点追平SOTA实验计划.md",
                    "docs/44_委托代审17例caption结论.md",
                    "eval/diag/delegated_label_review_17_public_summary_20261021.json",
                    "eval/diag/LoRA_native_CPU_microcheck_20261020.json",
                    "eval/diag/teacher_training_smoke_public_results_20261020.json",
                    "eval/e1_devset_20261021/e1b_summary.json",
                    "eval/diag/teacher_e1_devset_public_summary_20261021.json",
                    "eval/diag/teacher_e1b_cfgzero_public_summary_20261021.json",
                    "docs/46_E2完整基准结论.md",
                    "eval/e2_s12c5_20261021/plan.json",
                    "scripts/run_teacher_training_smoke.py", "tests/test_teacher_training_smoke.py",
                    "scripts/serve_label_review.py", "tests/test_serve_label_review.py",
                    "scripts/prepare_e1_devset.py", "scripts/build_e1_jobs.py", "scripts/score_e1.py",
                    "tests/test_e1_devset.py", "tests/test_e1_jobs.py", "tests/test_e1_score.py",
                    "DiffSynth-Studio/diffsynth/diffusion/training_module.py",
                    "DiffSynth-Studio/diffsynth/diffusion/loss.py",
                    "training_metadata/teacher_namespace_candidates_20261020_v1/eligibility_summary.json",
                    "scripts/prepare_teacher_eligibility.py", "tests/test_prepare_teacher_eligibility.py",
                    "eval/diag/namespace_inventory_public_summary_20261020.json",
                    "eval/diag/window_caption_chain_public_summary_20261020.json",
                    "tests/test_reslice_namespace.py",
                    "eval/diag/teacher_L1_semantic_pilot_summary_20261020.json",
                    "docs/37_Teacher机制诊断M1实测结论.md", "eval/diag/teacher_M1_public_results_20261020.json",
                    "docs/37_M1同状态预测分解实验结论.md",
                    "eval/diag/teacher_m1_public_summary_20261020.json",
                    "eval/diag/teacher_label_audit_L1_public_summary_20261020.json",
                    "scripts/audit_teacher_labels.py", "scripts/run_teacher_mechanism_probe.py", "scripts/run_teacher_caption_probe.py",
                    "tests/test_audit_teacher_labels.py", "tests/test_teacher_mechanism_probe.py", "tests/test_teacher_caption_probe.py",
                    "scripts/run_teacher_probe.py", "scripts/score_teacher_probe.py",
                    "tests/test_run_teacher_probe.py", "tests/test_score_teacher_probe.py"}
        self.assertTrue(selected <= set(snapshot.FILES))
        self.assertEqual(len(snapshot.FILES), len(set(snapshot.FILES)))
        self.assertFalse(any("teacher_opt_state" in name or "p1_jobs" in name
                             or "p1_provenance" in name or "idle_reservation" in name
                             or name.endswith("namespace_candidates.csv") or name.endswith("namespace_quarantine.csv")
                             for name in snapshot.FILES))


class ExportSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.destination = self.base / "snapshot"
        self.files = ["README.md", "scripts/required.py", "eval/diag/result.json"]
        self.required = {"scripts/required.py"}
        self.write("README.md", "# Measured experiment\n")
        self.write("scripts/required.py", "print('source snapshot only')\n")
        self.write("eval/diag/result.json", '{"score": 0.5}\n')
        old_umask = os.umask(0o077)
        os.umask(old_umask)
        self.addCleanup(os.umask, old_umask)
        for name, value in (("FILES", self.files), ("REQUIRED", self.required)):
            patch = mock.patch.object(snapshot, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def assert_no_snapshot(self):
        self.assertFalse(self.destination.exists())
        self.assertEqual({path.name for path in self.base.iterdir()}, {"source"})

    def test_exact_allowlist_excludes_git_cookies_data_and_logs(self):
        excluded = [".git/config", "cookies_netscape.txt", "dataset/private.txt",
                    "models/private.safetensors", "dataset/download_logs/private.log",
                    "eval/diag/merge_monitor_events.jsonl", "unselected.py"]
        for relative in excluded:
            self.write(relative, b"private artifact, never selected")
        status = {"at": "2026-10-18 17:00 CST", "complete_nonempty": 1676,
                  "total": 3033, "by_model": {"m_full1_a05": 1011},
                  "nodes": [{"ip": "11." + "239.36.34"}], "task": "private-session"}
        self.write("eval/diag/merge_monitor_latest.json", json.dumps(status))
        scope = snapshot.export(self.root, self.destination)
        actual = {str(path.relative_to(self.destination)) for path in
                  self.destination.rglob("*") if path.is_file()}
        self.assertEqual(actual, set(self.files) | {
            "PUBLICATION_MANIFEST.json", "PUBLICATION.md", ".gitignore"})
        self.assertFalse((self.destination / ".git").exists())
        self.assertEqual({record["path"] for record in scope["files"]}, set(self.files))
        for relative in excluded:
            self.assertFalse((self.destination / relative).exists())
        self.assertEqual(scope["experiment_progress"], {
            key: status[key] for key in ("at", "complete_nonempty", "total", "by_model")})
        self.assertNotIn("private-session", (self.destination / "PUBLICATION_MANIFEST.json").read_text())
        self.assertIn("not source Git history", scope["caveat"])
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o700)

    def test_manifest_sha256_matches_original_and_exported_bytes(self):
        originals = {relative: (self.root / relative).read_bytes() for relative in self.files}
        scope = snapshot.export(self.root, self.destination)
        for record in scope["files"]:
            relative = record["path"]
            exported = (self.destination / relative).read_bytes()
            self.assertEqual(exported, originals[relative])
            self.assertEqual(record["source_sha256"], hashlib.sha256(originals[relative]).hexdigest())
            self.assertEqual(record["snapshot_sha256"], hashlib.sha256(exported).hexdigest())
        self.assertEqual(json.loads((self.destination / "PUBLICATION_MANIFEST.json").read_text()), scope)

    def test_internal_node_address_is_placeholder_and_hashes_record_transformation(self):
        text = "deployment node=" + "11." + "239.36.34" + "\n"
        path = self.write("README.md", text)
        raw = path.read_bytes()
        scope = snapshot.export(self.root, self.destination)
        output = (self.destination / "README.md").read_bytes()
        self.assertEqual(output, b"deployment node=PRIVATE_NODE_ADDRESS\n")
        record = next(row for row in scope["files"] if row["path"] == "README.md")
        self.assertEqual(record["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["snapshot_sha256"], hashlib.sha256(output).hexdigest())
        self.assertNotEqual(record["source_sha256"], record["snapshot_sha256"])
        self.assertEqual(path.read_bytes(), raw)

    def test_credential_families_rejected_without_values_or_partial_output(self):
        secrets = ["github_" + "pat_" + "A" * 30,
                   "gh" + "p_" + "B" * 30,
                   "h" + "f_" + "C" * 30,
                   "-----BEGIN " + "OPENSSH " + "PRIVATE KEY-----",
                   "https://" + "user:" + "secret-value" + "@example.invalid/project",
                   "password = '" + "D" * 20 + "'"]
        for secret in secrets:
            with self.subTest(family=secrets.index(secret)):
                self.write("eval/diag/result.json", "secret fixture: " + secret)
                with self.assertRaisesRegex(RuntimeError, "Credential-like content") as caught:
                    snapshot.export(self.root, self.destination)
                self.assertNotIn(secret, str(caught.exception))
                self.assert_no_snapshot()

    def test_selected_file_symlink_escape_is_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("not repository-local")
        selected = self.root / "README.md"
        selected.unlink()
        selected.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "repository-local regular file"):
            snapshot.export(self.root, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(outside.read_text(), "not repository-local")

    def test_selected_parent_symlink_escape_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "file.py").write_text("private outside file")
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        self.files.append("linked/file.py")
        with self.assertRaisesRegex(RuntimeError, "repository-local regular file"):
            snapshot.export(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_absolute_source_escape_is_rejected(self):
        outside = self.base / "outside.py"
        outside.write_text("unselected external file")
        self.files.append(str(outside))
        with self.assertRaisesRegex(RuntimeError, "repository-local regular file"):
            snapshot.export(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_destination_inside_source_is_rejected(self):
        inside = self.root / "public-snapshot"
        with self.assertRaisesRegex(RuntimeError, "outside the working repository"):
            snapshot.export(self.root, inside)
        self.assertFalse(inside.exists())
        self.assert_no_snapshot()

    def test_destination_symlink_parent_pointing_into_source_is_rejected(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        inside = self.root / "escaped-snapshot"
        with self.assertRaisesRegex(RuntimeError, "outside the working repository"):
            snapshot.export(self.root, alias / "escaped-snapshot")
        self.assertFalse(inside.exists())

    def test_existing_destination_is_preserved(self):
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("existing user snapshot")
        with self.assertRaisesRegex(RuntimeError, "Destination already exists"):
            snapshot.export(self.root, self.destination)
        self.assertEqual(marker.read_text(), "existing user snapshot")
        self.assertEqual(list(self.destination.iterdir()), [marker])

    def test_missing_required_code_is_rejected_without_output(self):
        (self.root / "scripts/required.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "Required experiment implementation not ready"):
            snapshot.export(self.root, self.destination)
        self.assert_no_snapshot()

    def test_optional_missing_file_is_not_exported(self):
        self.files.append("docs/optional_missing.md")
        scope = snapshot.export(self.root, self.destination)
        self.assertNotIn("docs/optional_missing.md", {record["path"] for record in scope["files"]})

    def test_nul_bytes_are_rejected_without_partial_output(self):
        self.write("eval/diag/result.json", b"text" + bytes([0]) + b"binary")
        with self.assertRaisesRegex(RuntimeError, "Not a small text publication asset"):
            snapshot.export(self.root, self.destination)
        self.assert_no_snapshot()

    def test_oversized_text_is_rejected_without_partial_output(self):
        self.write("eval/diag/result.json", b"x" * (10 * 1024 * 1024 + 1))
        with self.assertRaisesRegex(RuntimeError, "Not a small text publication asset"):
            snapshot.export(self.root, self.destination)
        self.assert_no_snapshot()

    def test_invalid_utf8_is_rejected_without_partial_output(self):
        self.write("eval/diag/result.json", bytes([255]))
        with self.assertRaises((UnicodeDecodeError, RuntimeError)):
            snapshot.export(self.root, self.destination)
        self.assert_no_snapshot()

    def test_credential_in_selected_monitor_metadata_is_rejected_without_output(self):
        secret = "github_" + "pat_" + "Z" * 30
        self.write("eval/diag/merge_monitor_latest.json", json.dumps({
            "at": secret, "complete_nonempty": 1, "total": 3033, "by_model": {}}))
        with self.assertRaisesRegex(RuntimeError, "Credential-like content") as caught:
            snapshot.export(self.root, self.destination)
        self.assertNotIn(secret, str(caught.exception))
        self.assert_no_snapshot()

    def test_broken_existing_destination_symlink_is_preserved(self):
        target = self.base / "unused-target"
        self.destination.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "Destination already exists"):
            snapshot.export(self.root, self.destination)
        self.assertTrue(self.destination.is_symlink())
        self.assertEqual(self.destination.readlink(), target)
        self.assertFalse(target.exists())

    def test_published_monitor_metadata_replaces_internal_address(self):
        address = "11." + "239.36.34"
        self.write("eval/diag/merge_monitor_latest.json", json.dumps({
            "at": "2026-10-18", "complete_nonempty": 1, "total": 3033,
            "by_model": {address: 1}}))
        scope = snapshot.export(self.root, self.destination)
        self.assertEqual(scope["experiment_progress"]["by_model"], {"PRIVATE_NODE_ADDRESS": 1})
        self.assertNotIn(address, (self.destination / "PUBLICATION_MANIFEST.json").read_text())

    def test_monitor_nonobject_and_nonfinite_metadata_rejected_without_output(self):
        for value in ([], {"complete_nonempty": float("nan")}):
            with self.subTest(kind=type(value).__name__):
                self.write("eval/diag/merge_monitor_latest.json", json.dumps(value))
                with self.assertRaises((ValueError, RuntimeError)):
                    snapshot.export(self.root, self.destination)
                self.assert_no_snapshot()

    def test_partial_monitor_json_is_rejected_without_partial_output(self):
        self.write("eval/diag/merge_monitor_latest.json", '{"complete_nonempty":')
        with self.assertRaises((ValueError, RuntimeError)):
            snapshot.export(self.root, self.destination)
        self.assert_no_snapshot()


if __name__ == "__main__":
    unittest.main()
