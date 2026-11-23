import csv
import io
import json
import tarfile
import tempfile
import unittest
import urllib.error
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

from dataset_tools import build_manifests, download_queue


class DownloadQueueTests(unittest.TestCase):
    def test_load_export_and_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "Example_download_urls.json"
            export.write_text(
                json.dumps(
                    {
                        "sequences": {
                            "seq/1": {
                                "video_main_rgb": {
                                    "filename": "preview.mp4",
                                    "download_url": "https://example.invalid/video",
                                    "file_size_bytes": 123,
                                    "sha1sum": "abc",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = download_queue.load_export(export, "video_main_rgb")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "Example")
            self.assertEqual(
                download_queue.destination(root, rows[0]),
                root / "Example" / "seq_1" / "preview.mp4",
            )

    def test_zero_retries_still_attempts_once_and_records_error(self):
        row = {
            "dataset": "Example",
            "sequence": "seq1",
            "filename": "preview.mp4",
            "url": "https://example.invalid/video",
            "expected_bytes": 123,
            "sha1": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            error = urllib.error.URLError("proxy DNS unavailable")
            with mock.patch("urllib.request.urlopen", side_effect=error) as urlopen:
                result = download_queue.download_one(row, Path(directory), timeout=1, retries=0)

            self.assertEqual(urlopen.call_count, 1)
            self.assertEqual(result["status"], "failed")
            self.assertIn("proxy DNS unavailable", result["error"])


class ManifestTests(unittest.TestCase):
    def test_hoigen_aggregates_caption_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "videos_part_1.zip"
            filename = "abcdefghijk-0:00:01.000-0:00:04.000.mp4"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(filename, b"video")
            caption_csv = root / "captions.csv"
            with caption_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "text", "fps", "width", "height", "duration", "num_frames"])
                writer.writerow([filename, "first caption", 24, 1280, 720, 3, 72])
                writer.writerow([filename, "second caption", 24, 1280, 720, 3, 72])
            output = root / "manifest.jsonl"
            result = build_manifests.build_hoigen(
                Namespace(
                    root=root,
                    archive_glob="videos_part_*.zip",
                    caption_csv=caption_csv,
                    output=output,
                )
            )
            self.assertEqual(result, 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_sequence_id"], "abcdefghijk")
            self.assertEqual(rows[0]["caption_variants"], ["first caption", "second caption"])

    def test_seamless_requires_complete_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "0000.tar"
            metadata = {
                "id": "sample",
                "metadata:transcript": [{"transcript": "hello", "start": 0, "end": 1}],
                "metadata:vad": [{"start": 0, "end": 1}],
            }
            with tarfile.open(archive, "w") as handle:
                for suffix, content in {
                    ".mp4": b"video",
                    ".wav": b"audio",
                    ".npz": b"motion",
                    ".json": json.dumps(metadata).encode(),
                }.items():
                    info = tarfile.TarInfo(f"sample{suffix}")
                    info.size = len(content)
                    handle.addfile(info, io.BytesIO(content))
            output = root / "manifest.jsonl"
            result = build_manifests.build_seamless(
                Namespace(root=root, archive_glob="*.tar", output=output)
            )
            self.assertEqual(result, 0)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["transcript"], "hello")
            self.assertEqual(row["video_member"], "sample.mp4")


if __name__ == "__main__":
    unittest.main()
