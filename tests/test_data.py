import copy
import json
import tempfile
import unittest
from pathlib import Path

from ipvg.data import ManifestError, read_manifest, split_manifest, split_report, validate_manifest, write_manifest


def sample(number, identity=None, source=None, **extra):
    row = {"sample_id": "sample-{}".format(number), "identity_id": identity or "person-{}".format(number),
           "source_video_id": source or "video-{}".format(number),
           "reference_images": ["faces/{}.png".format(number)], "prompt": "The person walks.", "track": 1}
    row.update(extra)
    return row


class ManifestTests(unittest.TestCase):
    def test_jsonl_round_trip_preserves_extra_fields_and_unicode(self):
        rows = [sample(1, prompt="A person smiles.", note={"quality": 0.9})]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "manifest.jsonl"
            write_manifest(path, rows)
            self.assertEqual(read_manifest(path), rows)

    def test_duplicate_ids_rejected_across_lines(self):
        with self.assertRaisesRegex(ManifestError, "duplicate sample_id"):
            validate_manifest([sample(1), sample(1, identity="other")])

    def test_invalid_json_reports_physical_line_number(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.jsonl"
            path.write_text(json.dumps(sample(1)) + "\n\n{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, r":3: invalid JSON"):
                read_manifest(path)

    def test_paths_are_optional_but_use_caller_root_when_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = sample(1, reference_images=["face.png"], video_path="video.mp4")
            validate_manifest([row])
            (root / "face.png").write_bytes(b"face")
            with self.assertRaisesRegex(ManifestError, "media file not found"):
                validate_manifest([row], root=root, check_paths=True)
            (root / "video.mp4").write_bytes(b"video")
            self.assertEqual(validate_manifest([row], root=root, check_paths=True)["total_samples"], 1)

    def test_timeline_accepts_touching_intervals_in_any_order(self):
        row = sample(1, track=2, timeline=[{"start_s": 2, "end_s": 4, "action": "sit"},
                                          {"start_s": 0, "end_s": 2, "action": "walk"}])
        validate_manifest([row])

    def test_bad_times_and_overlap_are_rejected(self):
        bad = [(-1, 2), (2, 2), (3, 2), (0, float("inf")), (float("nan"), 2), (False, 2)]
        for start, end in bad:
            with self.subTest(start=start, end=end), self.assertRaises(ManifestError):
                validate_manifest([sample(1, track=2, timeline=[{"start_s": start, "end_s": end, "action": "walk"}])])
        with self.assertRaisesRegex(ManifestError, "overlapping"):
            validate_manifest([sample(1, timeline=[{"start_s": 0, "end_s": 3, "action": "walk"},
                                                  {"start_s": 2, "end_s": 4, "action": "sit"}])])

    def test_invalid_schema_values(self):
        for change in ({"track": True}, {"track": 3}, {"reference_images": []},
                       {"split": "test"}, {"prompt": " "}, {"sample_id": "../escape"},
                       {"track": 2, "timeline": []}):
            with self.subTest(change=change), self.assertRaises(ManifestError):
                validate_manifest([sample(1, **change)])

    def test_transitive_identity_source_chain_stays_together(self):
        rows = [sample(1, "alice", "shared-video"), sample(2, "bob", "shared-video"),
                sample(3, "bob", "second-video")]
        rows += [sample(number) for number in range(4, 31)]
        result, report = split_manifest(rows, ratios=(.6, .2, .2), seed=17)
        self.assertEqual(len({row["split"] for row in result[:3]}), 1)
        self.assertTrue(report["leakage"]["safe"])
        self.assertEqual(report["connected_group_count"], 28)
        self.assertNotIn("split", rows[0])

    def test_split_is_independent_of_input_order(self):
        rows = [sample(number) for number in range(100)]
        first, report = split_manifest(rows, seed=23)
        second, _ = split_manifest(list(reversed(rows)), seed=23)
        self.assertEqual({row["sample_id"]: row["split"] for row in first},
                         {row["sample_id"]: row["split"] for row in second})
        self.assertEqual(report["counts"], {"train": 80, "dev": 10, "holdout": 10, "unassigned": 0})
        other, _ = split_manifest(rows, seed=24)
        self.assertNotEqual([row["split"] for row in first], [row["split"] for row in other])

    def test_existing_leak_can_be_repaired_or_explicitly_rejected(self):
        rows = [sample(1, "alice", "video-a", split="dev"), sample(2, "alice", "video-b", split="train")]
        self.assertFalse(split_report(rows)["leakage"]["safe"])
        with self.assertRaisesRegex(ManifestError, "conflict"):
            split_manifest(rows, respect_existing=True)
        repaired, report = split_manifest(rows, respect_existing=False)
        self.assertTrue(report["leakage"]["safe"])
        self.assertEqual(repaired[0]["split"], repaired[1]["split"])

    def test_respect_existing_propagates_to_connected_unassigned_rows(self):
        rows = [sample(1, "alice", "a", split="holdout"), sample(2, "alice", "b"), sample(3)]
        result, _ = split_manifest(rows, respect_existing=True)
        self.assertEqual([row["split"] for row in result[:2]], ["holdout", "holdout"])

    def test_bad_ratios_and_zero_weight_splits(self):
        for ratios in ((0, 0, 0), (-1, 1, 1), (1, 1), (float("nan"), 1, 1)):
            with self.subTest(ratios=ratios), self.assertRaises(ManifestError):
                split_manifest([sample(1)], ratios=ratios)
        rows, _ = split_manifest([sample(i) for i in range(20)], ratios=(1, 0, 0))
        self.assertEqual({row["split"] for row in rows}, {"train"})


if __name__ == "__main__":
    unittest.main()
