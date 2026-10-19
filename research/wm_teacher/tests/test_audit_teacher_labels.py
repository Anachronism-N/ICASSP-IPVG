import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "label_audit", Path(__file__).resolve().parents[1] / "scripts/audit_teacher_labels.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "eval/diag").mkdir(parents=True)
        (self.root / "training_metadata/ft2").mkdir(parents=True)
        self.csv = self.root / audit.CSV_REL
        self.manifest_path = self.root / audit.MANIFEST_REL
        self.rows = [{"video": f"dataset/wan22_5s/{dataset}_5s/v{i}_w001.mp4",
                      "prompt": "untouched prompt\nsecond physical line" if i == 0 else f"original prompt {i}"}
                     for i, dataset in enumerate(["beta", "alpha", "beta", "alpha", "beta", "alpha", "beta", "beta"])]
        with self.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
            writer.writeheader()
            writer.writerows(self.rows)
        self.csv_sha = audit.sha256(self.csv)
        groups = {dataset: [i for i, row in enumerate(self.rows) if audit.dataset_name(row["video"]) == dataset]
                  for dataset in ("alpha", "beta")}
        rng = random.Random(audit.SEED)
        cases = []
        for dataset in sorted(groups):
            for index in sorted(rng.sample(groups[dataset], 2)):
                row = self.rows[index]
                cases.append({"audit_id": f"{dataset}_{index}", "dataset": dataset, "csv_record_index0": index,
                              "csv_physical_line": index + 2 + (index > 0), "video": row["video"],
                              "prompt": row["prompt"], "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                              "inclusion_probability": 2 / len(groups[dataset]), "canonical_parent": "UNRESOLVED",
                              "semantic_review": "pending", "decode_review": "pending"})
        self.manifest = {"design": audit.DESIGN, "rng_seed": audit.SEED, "csv_sha256": self.csv_sha,
                         "population_windows": 8, "sources": {
                             dataset: {"population_windows": len(rows), "sample_windows": 2,
                                       "inclusion_probability": 2 / len(rows)} for dataset, rows in groups.items()},
                         "sample_windows": 4, "parent_source_independence": "NOT_CLAIMED; possible duplicates/UNKNOWN",
                         "semantic_review": "pending", "media_uploaded": False, "cases": cases}
        self.manifest_path.write_text(json.dumps(self.manifest))
        for name, value in (("POPULATION", 8), ("DATASETS", 2), ("SAMPLE_PER_DATASET", 2),
                            ("CSV_SHA", self.csv_sha), ("MANIFEST_SHA", audit.sha256(self.manifest_path))):
            patcher = mock.patch.object(audit, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.out = self.root / "eval/diag/new_private_run"

    def changed_manifest(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest))
        return mock.patch.object(audit, "MANIFEST_SHA", audit.sha256(self.manifest_path))

    def test_exact_rng_csv_record_physical_line_and_unmodified_inputs(self):
        before = (self.csv.read_bytes(), self.manifest_path.read_bytes())
        actual = audit.validate_inputs(self.root)
        self.assertEqual(actual, self.manifest)
        self.assertEqual((self.csv.read_bytes(), self.manifest_path.read_bytes()), before)
        self.assertFalse(self.out.exists())

    def test_hashes_fail_closed(self):
        self.csv.write_bytes(self.csv.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            audit.validate_inputs(self.root)
        self.assertFalse(self.out.exists())

    def test_manifest_hash_fail_closed(self):
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            audit.validate_inputs(self.root)

    def test_provenance_sampler_probability_and_semantic_tampering(self):
        bad = []
        for key, value in (("video", "dataset/wan22_5s/alpha_5s/replaced.mp4"),
                           ("prompt", "replacement label"), ("prompt_sha256", "0" * 64),
                           ("csv_physical_line", 12345), ("csv_record_index0", 0),
                           ("inclusion_probability", 1.0), ("canonical_parent", "guessed"),
                           ("semantic_review", "complete"), ("audit_id", "../escape")):
            manifest = copy.deepcopy(self.manifest)
            manifest["cases"][0][key] = value
            bad.append(manifest)
        reverse = copy.deepcopy(self.manifest)
        reverse["cases"].reverse()
        bad.append(reverse)
        missing = copy.deepcopy(self.manifest)
        missing["cases"].pop()
        bad.append(missing)
        for manifest in bad:
            with self.subTest(case=manifest["cases"][0]), self.changed_manifest(manifest), self.assertRaises(ValueError):
                audit.validate_inputs(self.root)
        self.assertFalse(self.out.exists())

    def test_safe_paths_absolute_traversal_backslash_and_symlink(self):
        for path in ("../outside.mp4", "/tmp/outside.mp4", "dataset/../x", "dataset\\x", "./dataset/x", "dataset//x", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                audit.safe_path(self.root, path)
        (self.root / "outside_link").symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            audit.safe_path(self.root, "outside_link/media.mp4")
        self.assertTrue(audit.safe_path(self.root, self.manifest["cases"][0]["video"]).is_relative_to(self.root))

    def test_budget_and_existing_or_nonprivate_output_rejected(self):
        for workers, seconds in ((0, 3600), (9, 3600), (8, 3601), (8, 14), (8, float("nan")), (8, float("inf"))):
            with self.subTest(workers=workers, seconds=seconds), self.assertRaises(ValueError):
                audit.budget(workers, seconds)
        self.out.mkdir()
        sentinel = self.out / "keep.json"
        sentinel.write_text("user work")
        with self.assertRaises(ValueError):
            audit.run_audit(self.root, self.out, 1, 15)
        self.assertEqual(sentinel.read_text(), "user work")
        for output in (self.root / "public_run", self.root / "eval/diag", self.root.parent / "escape"):
            with self.assertRaises(ValueError):
                audit.run_audit(self.root, output, 1, 15)

    def test_atomic_json_hash_and_permissions(self):
        path = self.root / "private_result.json"
        digest = audit.atomic_json(path, {"semantic_review": "pending"})
        self.assertEqual(digest, audit.sha256(path))
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertFalse(list(self.root.glob("*.tmp.*")))

    def test_missing_and_corrupt_preserved_without_replacement_or_semantic_label(self):
        case = self.manifest["cases"][0]
        output = self.root / "contact.jpg"
        missing = audit.decode_case(self.root, case, output)
        self.assertEqual(missing["decode_status"], "missing")
        self.assertEqual(missing["video"], case["video"])
        self.assertEqual(missing["prompt_sha256"], case["prompt_sha256"])
        self.assertFalse(missing["replacement"])
        self.assertEqual(missing["canonical_parent"], "UNKNOWN")
        self.assertEqual(missing["semantic_label"], "NOT_REVIEWED")
        video = audit.safe_path(self.root, case["video"])
        video.parent.mkdir(parents=True)
        video.write_bytes(b"not a video")
        corrupt = audit.decode_case(self.root, case, output)
        self.assertEqual(corrupt["decode_status"], "decode_failed")
        self.assertEqual(corrupt["video_sha256"], audit.sha256(video))
        self.assertFalse(output.exists())

    def make_video(self, case, frames, fps):
        import av
        from PIL import Image
        path = audit.safe_path(self.root, case["video"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = 64, 48
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 1
            for index in range(frames):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (index % 256, 50, 70)))
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    def test_real_full_decode_non121_protocol_and_sparse_pts_contacts(self):
        case = self.manifest["cases"][0]
        path = self.make_video(case, 41, 10)
        result = audit.decode_case(self.root, case, self.root / "contact.jpg")
        self.assertEqual(result["decode_status"], "decoded_all_frames")
        self.assertEqual(result["decoded_frames"], 41)
        self.assertEqual(result["metadata"]["declared_frames"], 41)
        self.assertIn("frame_count_not_121_observation_not_rejection", result["metadata_observations"])
        self.assertEqual(result["metadata"]["decoder_threads"], 1)
        self.assertEqual(result["video_sha256"], audit.sha256(path))
        self.assertTrue(result["file_unchanged"])
        self.assertEqual(result["contact_status"], "created")
        self.assertEqual([f["decoded_index0"] for f in result["contact"]["frames"]], [0, 5, 15, 30, 40])
        self.assertEqual(result["semantic_review"], "pending")
        self.assertEqual(result["canonical_parent"], "UNKNOWN")
        self.assertTrue(result["contact"]["not_evidence_of_absence"])

    def test_short_video_does_not_substitute_missing_contact_times(self):
        case = self.manifest["cases"][0]
        self.make_video(case, 3, 10)
        result = audit.decode_case(self.root, case, self.root / "short_contact.jpg")
        self.assertEqual(result["decode_status"], "decoded_all_frames")
        self.assertEqual(result["navigation_times_unavailable"], [0.5, 1.5, 3.0])
        self.assertEqual([f["requested"] for f in result["contact"]["frames"]], ["0s", "last"])

    def test_file_change_is_failure_not_hidden(self):
        case = self.manifest["cases"][0]
        path = self.make_video(case, 3, 10)
        original = audit.sha256
        count = 0
        def modified_hash(target):
            nonlocal count
            digest = original(target)
            if Path(target) == path:
                count += 1
                if count == 2:
                    return "f" * 64
            return digest
        with mock.patch.object(audit, "sha256", side_effect=modified_hash):
            result = audit.decode_case(self.root, case, self.root / "changed_contact.jpg")
        self.assertEqual(result["decode_status"], "file_changed")
        self.assertFalse(result["file_unchanged"])

    def test_declared_frame_mismatch_is_not_silently_accepted(self):
        import av
        case = self.manifest["cases"][0]
        path = self.make_video(case, 41, 10)
        actual_open = av.open
        class TruncatedContainer:
            def __init__(self):
                self.actual = actual_open(str(path))
                self.streams = self.actual.streams
                self.duration = self.actual.duration
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.actual.close()
            def decode(self, stream):
                for index, frame in enumerate(self.actual.decode(stream)):
                    if index == 40:
                        break
                    yield frame
        with mock.patch.object(av, "open", side_effect=lambda *args, **kwargs: TruncatedContainer()):
            result = audit.decode_case(self.root, case, self.root / "count_mismatch.jpg")
        self.assertEqual(result["decode_status"], "decoded_count_mismatch")
        self.assertEqual(result["decoded_frames"], 40)
        self.assertFalse(result["full_decode_matches_declared_count"])

    def test_real_wall_alarm_finishes_with_all_cases_accounted_for(self):
        import time
        self.terminated = False
        owner = self
        class BlockingPool:
            def imap_unordered(self, function, tasks, chunksize):
                time.sleep(60)
                return iter(())
            def terminate(self):
                owner.terminated = True
            def join(self):
                pass
        context = types.SimpleNamespace(Pool=lambda workers: BlockingPool())
        started = time.monotonic()
        with mock.patch.object(audit.multiprocessing, "get_context", return_value=context):
            self.assertEqual(audit.run_audit(self.root, self.out, 2, 15), 2)
        self.assertLess(time.monotonic() - started, 15)
        self.assertTrue(self.terminated)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["decode_status"], {"not_completed_wall_deadline": 4})
        self.assertEqual(report["termination"], "wall_deadline")
        self.assertEqual(report["result_count"], 4)

    def fake_pool(self, fail=False):
        owner = self
        class Pool:
            def imap_unordered(self, function, tasks, chunksize):
                for index, task in enumerate(tasks):
                    result = function(task)
                    if fail and index == 0:
                        raise audit.DeadlineExceeded()
                    yield result
            def close(self):
                pass
            def join(self):
                pass
            def terminate(self):
                owner.terminated = True
        return types.SimpleNamespace(Pool=lambda workers: Pool())

    def test_runner_full_failure_coverage_and_priority_without_replacement(self):
        before = self.csv.read_bytes(), self.manifest_path.read_bytes()
        with mock.patch.object(audit.multiprocessing, "get_context", return_value=self.fake_pool()):
            self.assertEqual(audit.run_audit(self.root, self.out, 2, 15), 2)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["decode_status"], {"missing": 4})
        self.assertEqual(report["result_count"], 4)
        self.assertEqual(report["semantic_labels_assigned"], 0)
        self.assertIsNone(report["semantic_accuracy"])
        self.assertEqual(report["replacements"], 0)
        self.assertTrue(report["all_frozen_cases_accounted_for"])
        self.assertEqual((self.csv.read_bytes(), self.manifest_path.read_bytes()), before)
        for row in report["per_dataset_coverage"].values():
            self.assertEqual(row["results"], 2)
            self.assertEqual(row["canonical_parent_unknown"], 2)
        selected = json.loads((self.out / "first_17_contacts.json").read_text())["cases"]
        for row in selected:
            self.assertEqual(row["csv_record_index0"], min(c["csv_record_index0"] for c in self.manifest["cases"] if c["dataset"] == row["dataset"]))
        hashes = json.loads((self.out / "result_hashes.json").read_text())
        self.assertEqual(hashes["report_sha256"], audit.sha256(self.out / "report.json"))
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)

    def test_deadline_salvages_atomic_results_and_covers_unfinished(self):
        self.terminated = False
        with mock.patch.object(audit.multiprocessing, "get_context", return_value=self.fake_pool(fail=True)):
            self.assertEqual(audit.run_audit(self.root, self.out, 2, 15), 2)
        self.assertTrue(self.terminated)
        report = json.loads((self.out / "report.json").read_text())
        self.assertEqual(report["termination"], "wall_deadline")
        self.assertEqual(report["result_count"], 4)
        self.assertEqual(report["decode_status"], {"missing": 1, "not_completed_wall_deadline": 3})
        self.assertEqual(len(list((self.out / "cases").glob("*.json"))), 4)
        self.assertEqual(report["semantic_labels_assigned"], 0)


if __name__ == "__main__":
    unittest.main()
