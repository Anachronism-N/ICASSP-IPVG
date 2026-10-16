import copy
import hashlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ipvg.resources import ResourceError, build_download_plan, download_plan


REVISION = "a" * 40


def metadata(gated=False):
    return {"sha": REVISION, "gated": gated, "cardData": {"license": "apache-2.0"},
            "siblings": [{"rfilename": "README.md", "size": 5},
                         {"rfilename": "weights/model.safetensors", "lfs": {"size": 100}},
                         {"rfilename": "Videos/huge.tar", "size": 8_000_000_000_000}]}


def plan(meta=None, **changes):
    args = {"repo_id": "owner/repo", "repo_type": "model", "revision": REVISION,
            "allow_patterns": ["README.md", "weights/*"], "max_bytes": 105}
    args.update(changes)
    return build_download_plan(meta or metadata(), **args)


class ResourceTests(unittest.TestCase):
    def test_allowlist_and_exact_byte_cap(self):
        result = plan()
        self.assertEqual(result["total_bytes"], 105)
        self.assertEqual(result["file_count"], 2)
        self.assertEqual([item["path"] for item in result["files"]], ["README.md", "weights/model.safetensors"])
        with self.assertRaisesRegex(ResourceError, "exceeding cap"):
            plan(max_bytes=104)

    def test_default_run_makes_no_network_calls_or_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "not-created"
            with patch("ipvg.resources.fetch_hf_metadata", side_effect=AssertionError("network")):
                result = download_plan(plan(), target)
            self.assertEqual(result["status"], "planned")
            self.assertFalse(target.exists())

    def test_full_large_dataset_wildcard_is_rejected_before_download(self):
        with self.assertRaisesRegex(ResourceError, "exceeding cap"):
            plan(allow_patterns=["*"], max_bytes=1_000_000_000)

    def test_unpinned_revision_unknown_size_and_no_matches_fail_closed(self):
        with self.assertRaisesRegex(ResourceError, "immutable"):
            plan(revision="main")
        changed = metadata()
        changed["sha"] = "b" * 40
        with self.assertRaisesRegex(ResourceError, "SHA"):
            plan(changed)
        changed = metadata()
        changed["siblings"][1] = {"rfilename": "weights/model.safetensors"}
        with self.assertRaisesRegex(ResourceError, "unknown"):
            plan(changed)
        with self.assertRaisesRegex(ResourceError, "match no"):
            plan(allow_patterns=["absent/*"])

    def test_unsafe_names_and_duplicates_rejected(self):
        for filename in ("../outside.bin", "/outside.bin", "C:/outside.bin", "a\\b.bin"):
            changed = metadata()
            changed["siblings"][0]["rfilename"] = filename
            with self.subTest(filename=filename), self.assertRaises(ResourceError):
                plan(changed)
        changed = metadata()
        changed["siblings"].append(dict(changed["siblings"][0]))
        with self.assertRaisesRegex(ResourceError, "duplicate"):
            plan(changed)

    def test_manual_gate_cannot_be_silently_bypassed(self):
        result = plan(metadata(gated="manual"))
        self.assertTrue(result["requires_access_approval"])
        with tempfile.TemporaryDirectory() as folder:
            with patch("ipvg.resources.fetch_hf_metadata", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ResourceError, "approval"):
                    download_plan(result, folder, execute=True)

    def test_saved_plan_cannot_smuggle_a_non_allowlisted_file(self):
        changed = plan()
        changed["files"].append({"path": "Videos/huge.tar", "size_bytes": 1})
        with self.assertRaises(ResourceError):
            download_plan(changed, "unused")

    def test_execution_rechecks_actual_metadata_size_before_download(self):
        changed = plan()
        changed["files"][1]["size_bytes"] = 1
        changed["total_bytes"] = 6
        changed["max_bytes"] = 6
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "not-created"
            with patch("ipvg.resources.fetch_hf_metadata", return_value=metadata()):
                with self.assertRaisesRegex(ResourceError, "exceeding cap"):
                    download_plan(changed, target, execute=True)
            self.assertFalse(target.exists())

    def test_execution_downloads_only_planned_files_at_pinned_revision(self):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x" * (5 if kwargs["filename"] == "README.md" else 100))
            return str(target)

        module = types.SimpleNamespace(hf_hub_download=fake_download)
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("sys.modules", {"huggingface_hub": module}), \
                    patch("ipvg.resources.fetch_hf_metadata", return_value=metadata()):
                result = download_plan(plan(), folder, execute=True)
        self.assertEqual(result["file_count"], 2)
        self.assertTrue(all(call["revision"] == REVISION for call in calls))
        self.assertNotIn("Videos/huge.tar", [call["filename"] for call in calls])

    def test_checksum_mismatch_detected(self):
        meta = {"sha": REVISION, "siblings": [{"rfilename": "a.bin", "size": 3,
                 "lfs": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}]}
        selected = plan(meta, allow_patterns=["a.bin"], max_bytes=3)

        def fake_download(**kwargs):
            (Path(kwargs["local_dir"]) / kwargs["filename"]).write_bytes(b"bad")

        with tempfile.TemporaryDirectory() as folder:
            with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(hf_hub_download=fake_download)}), \
                    patch("ipvg.resources.fetch_hf_metadata", return_value=meta):
                with self.assertRaisesRegex(ResourceError, "checksum mismatch"):
                    download_plan(selected, folder, execute=True)


if __name__ == "__main__":
    unittest.main()
