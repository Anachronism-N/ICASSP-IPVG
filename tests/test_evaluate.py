import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ipvg.evaluate import evaluate_outputs, load_manifest_records, probe_video


def successful_probe(width=720, height=480, fps="24/1", frames=121, duration="5.041667"):
    return subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264",
                                        "width": width, "height": height, "avg_frame_rate": fps,
                                        "nb_read_frames": str(frames), "duration": duration}],
                           "format": {"duration": duration}}), stderr="",
    )


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.manifest = self.root / "manifest.jsonl"

    def manifest_rows(self, *ids):
        self.manifest.write_text("\n".join(json.dumps({"sample_id": name, "identity_id": "person1",
             "source_video_id": "source1", "reference_images": ["private.png"],
             "prompt": "A person walks", "track": 1, "timeline": []}) for name in ids) + "\n", encoding="utf-8")

    def video(self, sample_id):
        (self.outputs / (sample_id + ".mp4")).write_bytes(b"generated-video")

    @patch("ipvg.evaluate.subprocess.run")
    def test_missing_outputs_report_coverage_without_fabricated_scores(self, run):
        self.manifest_rows("one", "two")
        self.video("one")
        run.return_value = successful_probe()
        report = evaluate_outputs(self.manifest, self.outputs)
        self.assertFalse(report["ok"])
        self.assertEqual(report["summary"]["valid_output_coverage"], 0.5)
        self.assertEqual(report["summary"]["failure_classes"], {"missing_output": 1})
        self.assertEqual(report["summary"]["metrics"], {})
        self.assertIsNone(report["official_score"])
        self.assertEqual(report["score_kind"], "local_proxy")

    @patch("ipvg.evaluate.subprocess.run")
    def test_explicit_spec_tolerances_and_fractional_fps(self, run):
        self.manifest_rows("one")
        self.video("one")
        run.return_value = successful_probe(fps="24000/1001", duration="5.2")
        report = evaluate_outputs(self.manifest, self.outputs, expected_width=720,
             expected_height=480, expected_fps=23.976, expected_duration=5,
             duration_tolerance=0.25, fps_tolerance=0.001)
        self.assertTrue(report["ok"])
        command = run.call_args.args[0]
        self.assertIn("-count_frames", command)
        self.assertNotIn("shell", run.call_args.kwargs)

    @patch("ipvg.evaluate.subprocess.run")
    def test_wrong_spec_lists_each_failure(self, run):
        self.manifest_rows("one")
        self.video("one")
        run.return_value = successful_probe()
        report = evaluate_outputs(self.manifest, self.outputs, expected_width=1280,
            expected_height=720, expected_fps=30, expected_duration=10)
        self.assertEqual(set(report["samples"][0]["failure_classes"]),
                         {"width_mismatch", "height_mismatch", "fps_mismatch", "duration_mismatch"})

    @patch("ipvg.evaluate.subprocess.run")
    def test_decoder_errors_fail_even_with_zero_exit(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="{}", stderr="decoder: corrupt frame")
        self.assertEqual(probe_video("bad.mp4")["failure_class"], "decode_error")

    @patch("ipvg.evaluate.subprocess.run")
    def test_missing_probe_and_timeout_are_explicit(self, run):
        run.side_effect = FileNotFoundError()
        self.assertEqual(probe_video("bad.mp4")["failure_class"], "probe_unavailable")
        run.side_effect = subprocess.TimeoutExpired("ffprobe", 1)
        self.assertEqual(probe_video("bad.mp4")["failure_class"], "probe_timeout")

    @patch("ipvg.evaluate.subprocess.run")
    def test_metadata_without_decoded_frames_is_not_accepted(self, run):
        run.return_value = successful_probe(frames=0)
        self.assertFalse(probe_video("bad.mp4")["ok"])

    @patch("ipvg.evaluate.subprocess.run")
    def test_proxy_metrics_only_average_reported_values(self, run):
        self.manifest_rows("one", "two")
        self.video("one")
        self.video("two")
        run.return_value = successful_probe()
        scores = self.root / "scores.jsonl"
        scores.write_text(json.dumps({"sample_id": "one", "scores": {"identity": 0.8},
            "failure_classes": ["identity_drift", "identity_drift"]}) + "\n", encoding="utf-8")
        report = evaluate_outputs(self.manifest, self.outputs, proxy_scores_path=scores)
        metric = report["summary"]["metrics"]["identity"]
        self.assertEqual(metric["mean"], 0.8)
        self.assertEqual(metric["coverage"], 0.5)
        self.assertEqual(report["summary"]["proxy_failure_classes"], {"identity_drift": 1})
        self.assertTrue(report["ok"], "local quality proxies do not change media-format validity")

    def test_invalid_proxy_records_rejected(self):
        self.manifest_rows("one")
        path = self.root / "scores.jsonl"
        cases = [
            [{"sample_id": "unknown", "scores": {}}],
            [{"sample_id": "one", "scores": {"identity": float("nan")}}],
            [{"sample_id": "one", "scores": {"identity": True}}],
            [{"sample_id": "one", "score_kind": "official", "scores": {}}],
            [{"sample_id": "one", "scores": {}}] * 2,
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    evaluate_outputs(self.manifest, self.outputs, proxy_scores_path=path)

    def test_unsafe_or_duplicate_manifest_ids_rejected(self):
        for ids in [("../escape",), ("a/b",), ("a\\b",), ("CON",), ("trailing.",), ("one", "ONE")]:
            with self.subTest(ids=ids):
                self.manifest_rows(*ids)
                with self.assertRaises(ValueError):
                    load_manifest_records(self.manifest)

    def test_invalid_spec_is_rejected_before_probe(self):
        self.manifest_rows("one")
        for kwargs in [{"expected_width": 0}, {"expected_height": 1.5}, {"expected_fps": float("inf")},
                       {"duration_tolerance": -1}, {"fps_tolerance": True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate_outputs(self.manifest, self.outputs, **kwargs)

    @patch("ipvg.evaluate.subprocess.run")
    def test_unexpected_video_detection_and_json_report(self, run):
        self.manifest_rows("one")
        self.video("one")
        self.video("old-run")
        run.return_value = successful_probe()
        report_path = self.root / "reports" / "evaluation.json"
        report = evaluate_outputs(self.manifest, self.outputs, report_path, fail_on_extra=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["summary"]["unexpected_videos"], ["old-run.mp4"])
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["summary"], report["summary"])

    @patch("ipvg.evaluate.subprocess.run")
    def test_symlink_video_is_rejected(self, run):
        self.manifest_rows("one")
        private = self.root / "private.mp4"
        private.write_bytes(b"not-generated")
        try:
            (self.outputs / "one.mp4").symlink_to(private)
        except OSError:
            self.skipTest("symlinks unavailable in this environment")
        report = evaluate_outputs(self.manifest, self.outputs)
        self.assertEqual(report["summary"]["failure_classes"], {"unsafe_output_path": 1})
        run.assert_not_called()
