import hashlib
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ipvg.submission import create_draft_submission


def probe_ok():
    return subprocess.CompletedProcess([], 0, stdout=json.dumps({"streams": [{
        "codec_type": "video", "codec_name": "h264", "width": 720, "height": 480,
        "avg_frame_rate": "24/1", "nb_read_frames": "120", "duration": "5.0",
    }], "format": {"duration": "5.0"}}), stderr="")


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps({"sample_id": "one", "identity_id": "private-identity",
            "source_video_id": "private-source", "reference_images": ["restricted-reference.png"],
            "prompt": "private prompt", "track": 1, "timeline": []}) + "\n", encoding="utf-8")
        self.video = self.outputs / "one.mp4"
        self.video.write_bytes(b"generated-video")
        self.archive = self.root / "draft.zip"

    @patch("ipvg.evaluate.subprocess.run")
    def test_archive_contains_only_allowlisted_generated_video_and_hash_manifest(self, run):
        run.return_value = probe_ok()
        (self.outputs / "restricted-reference.png").write_bytes(b"private image")
        (self.outputs / "raw-data.zip").write_bytes(b"private dataset")
        result = create_draft_submission(self.manifest, self.outputs, self.archive,
            expected_width=720, expected_height=480, expected_fps=24, expected_duration=5)
        self.assertEqual(result["status"], "local_draft_not_submitted")
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(set(archive.namelist()), {"videos/one.mp4", "draft_manifest.json"})
            draft = json.loads(archive.read("draft_manifest.json"))
            self.assertEqual(draft["samples"][0]["sha256"], hashlib.sha256(b"generated-video").hexdigest())
            self.assertEqual(draft["source_manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
            self.assertFalse(draft["official_submission_format_confirmed"])
            content = archive.read("draft_manifest.json").decode("utf-8")
            self.assertNotIn("private-identity", content)
            self.assertNotIn("private prompt", content)
            self.assertNotIn("restricted-reference.png", content)

    @patch("ipvg.evaluate.subprocess.run")
    def test_missing_or_wrong_spec_video_refuses_packaging(self, run):
        run.return_value = probe_ok()
        with self.assertRaises(ValueError):
            create_draft_submission(self.manifest, self.outputs, self.archive, expected_fps=30)
        self.assertFalse(self.archive.exists())
        self.video.unlink()
        with self.assertRaises(ValueError):
            create_draft_submission(self.manifest, self.outputs, self.archive)
        self.assertFalse(self.archive.exists())

    @patch("ipvg.evaluate.subprocess.run")
    def test_extra_video_refused(self, run):
        run.return_value = probe_ok()
        (self.outputs / "old.mp4").write_bytes(b"old output")
        with self.assertRaisesRegex(ValueError, "validation failed"):
            create_draft_submission(self.manifest, self.outputs, self.archive)

    def test_existing_archive_not_overwritten(self):
        self.archive.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            create_draft_submission(self.manifest, self.outputs, self.archive)
        self.assertEqual(self.archive.read_bytes(), b"keep")

    def test_path_traversal_manifest_rejected(self):
        self.manifest.write_text('{"sample_id":"../escape"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            create_draft_submission(self.manifest, self.outputs, self.archive)
        self.assertFalse(self.archive.exists())

    @patch("ipvg.evaluate.subprocess.run")
    def test_reference_input_cannot_be_packaged_as_generated_output(self, run):
        run.return_value = probe_ok()
        row = json.loads(self.manifest.read_text(encoding="utf-8"))
        row["reference_images"] = [str(self.video)]
        self.manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference input"):
            create_draft_submission(self.manifest, self.outputs, self.archive)

    @patch("ipvg.submission.evaluate_outputs")
    def test_changed_video_is_detected_and_partial_zip_removed(self, evaluate):
        evaluate.return_value = {"ok": True, "manifest_sha256": "manifest-hash", "expected_spec": {},
            "samples": [{"sample_id": "one", "filename": "one.mp4", "sha256": "old-hash",
                         "bytes": 1, "media": {}}]}
        with self.assertRaisesRegex(ValueError, "changed after validation"):
            create_draft_submission(self.manifest, self.outputs, self.archive)
        self.assertFalse(self.archive.exists())
