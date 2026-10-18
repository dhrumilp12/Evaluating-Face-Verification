"""External-artifact and filename-association guards for the author benchmark."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.project2_benchmark import load_official, load_pairs


class Project2BenchmarkTests(unittest.TestCase):
    def write_fixture(self, directory: Path, filenames=None):
        if filenames is None:
            filenames = ["./lfw/Person/Person_0001.jpg", "./lfw/Person/Person_0002.jpg"]
        np.save(directory / "embeddings_lfw.npy", np.full((2, 512), 0.25))
        np.save(directory / "filenames_lfw.npy", np.asarray(filenames))
        (directory / "lfw_pairs.txt").write_text("Person\t1\t2\n" * 5999 + "Missing\t1\t2\n")

    def test_missing_features_remain_visible_at_original_pair_index(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_fixture(directory)
            _, pairs, labels, records = load_pairs(directory)
        self.assertEqual(pairs.shape, (5999, 2))
        self.assertTrue(labels.all())
        self.assertEqual(len(records), 6000)
        self.assertEqual(records[-1]["pair_index"], 5999)
        self.assertEqual(records[-1]["original_fold"], 9)
        self.assertEqual(records[-1]["status"], "excluded")
        self.assertIn("Missing_0001.jpg", records[-1]["exclusion_reason"])

    def test_duplicate_filenames_cannot_silently_change_feature_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_fixture(directory, ["./lfw/Person/Person_0001.jpg"] * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_pairs(directory)

    def test_pickle_backed_filename_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_fixture(directory)
            np.save(directory / "filenames_lfw.npy", np.asarray(["a", "b"], dtype=object))
            with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
                load_pairs(directory)

    def test_changed_upstream_code_is_rejected_before_execution(self):
        prior = sys.modules.get("similarity")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "similarity").mkdir()
            (directory / "similarity/base.py").write_text("raise RuntimeError('must never execute')\n")
            with self.assertRaisesRegex(ValueError, "Unexpected artifact contents"):
                load_official(directory)
        self.assertIs(sys.modules.get("similarity"), prior)


if __name__ == "__main__":
    unittest.main()
