import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reslice_namespace", ROOT / "scripts/reslice_5s.py")
reslice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reslice)


class NamespaceTests(unittest.TestCase):
    def run_record(self, source="shared_source", start=0, directory="dataset/source"):
        return {"dataset": "egoexo4d", "source": source, "segs": [
            directory + "/" + source + "_" + str(i).zfill(3) + ".mp4"
            for i in range(start, start + 3)], "windows": 2,
            "width": 1280, "height": 704, "orient": "landscape"}

    def test_distinct_contiguous_runs_same_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous window namespace"):
            reslice.validate_run_names([(self.run_record(start=0), [0]),
                                        (self.run_record(start=112), [0])])

    def test_chunks_of_exact_same_run_are_allowed(self):
        run = self.run_record()
        reslice.validate_run_names([(run, [0]), (copy.deepcopy(run), [1])])

    def test_source_sanitization_collision_rejected(self):
        with self.assertRaises(ValueError):
            reslice.validate_run_names([(self.run_record("source/a"), [0]),
                                        (self.run_record("source:a"), [1])])

    def test_truncation_collision_rejected(self):
        prefix = "x" * 150
        with self.assertRaises(ValueError):
            reslice.validate_run_names([(self.run_record(prefix + "a"), [0]),
                                        (self.run_record(prefix + "b"), [1])])

    def test_distinct_source_directories_with_same_basename_rejected(self):
        with self.assertRaises(ValueError):
            reslice.validate_run_names([(self.run_record(directory="dataset/take_a"), [0]),
                                        (self.run_record(directory="dataset/take_b"), [0])])

    def test_same_name_in_different_datasets_is_allowed(self):
        first = self.run_record()
        second = self.run_record(start=112)
        second["dataset"] = "direct"
        reslice.validate_run_names([(first, [0]), (second, [0])])

    def test_unchanged_valid_window_paths(self):
        run = self.run_record("unique_source")
        row = reslice._row(run, 0, "dataset/wan22_5s", "unique_source")
        self.assertTrue(row["video_path"].endswith("/unique_source_w000.mp4"))
        self.assertEqual(row["prompt"], "")

    def test_main_rejects_before_any_writes_or_workers(self):
        runs = {"egoexo4d": [self.run_record(), self.run_record(start=112)]}
        args = ["reslice", "--all-windows", "--datasets", "egoexo4d"]
        with mock.patch.object(sys, "argv", args), mock.patch.object(reslice, "load_runs", return_value=runs), mock.patch.object(reslice.os, "makedirs") as mkdir, mock.patch.object(reslice, "ProcessPoolExecutor") as pool, mock.patch("builtins.open") as output:
            with self.assertRaisesRegex(ValueError, "Ambiguous window namespace"):
                reslice.main()
            mkdir.assert_not_called()
            pool.assert_not_called()
            output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
