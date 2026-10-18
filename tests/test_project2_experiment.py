import tempfile
from pathlib import Path
import unittest

import numpy as np

from src.project2_experiment import pair_rows, read_feature, save_feature


class ExperimentTests(unittest.TestCase):
    def test_feature_cache_retains_raw_magnitudes_and_checks_source(self):
        raw = np.random.default_rng(7).normal(size=512).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npz"
            self.assertIsNone(read_feature(path, "input-one"))
            save_feature(path, raw, "input-one")
            np.testing.assert_array_equal(read_feature(path, "input-one"), raw)
            self.assertIsNone(read_feature(path, "different-input"))
            with np.load(path, allow_pickle=False) as data:
                self.assertAlmostEqual(float(np.linalg.norm(data["unit"])), 1.0, places=12)
                self.assertAlmostEqual(float(data["quality"]), np.linalg.norm(raw.astype(float)), places=12)

    def test_corrupt_quality_fails_instead_of_reusing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feature.npz"
            np.savez(path, raw=np.ones(512), unit=np.ones(512), quality=1.0, input_sha256="hash")
            with self.assertRaisesRegex(ValueError, "quality"):
                read_feature(path, "hash")

    def test_pair_rows_retain_failures_and_fixed_reference(self):
        first = np.zeros(512, dtype=np.float32)
        first[0] = 20
        second = first * 0.5
        pairs = [{"pair_index": 0, "fold": 0, "label": 1, "reference": "ref", "probe": "probe"},
                 {"pair_index": 1, "fold": 0, "label": 0, "reference": "ref", "probe": "bad"}]
        records = {"ref": {"status": "ok"}, "probe": {"status": "ok"}, "bad": {"status": "no_face_detected"}}
        rows = pair_rows(pairs, records, {"ref": first}, {"probe": second}, "brightness_0.5")
        self.assertEqual(rows[0]["reference_quality"], 20)
        self.assertEqual(rows[0]["probe_quality"], 10)
        self.assertEqual(rows[0]["cosine_similarity"], 1)
        self.assertEqual(rows[1]["status"], "failed")
        self.assertIsNone(rows[1]["qmagface_score"])
        self.assertIn("no_face_detected", rows[1]["error"])

    def test_missing_eligible_feature_is_fatal(self):
        pair = {"pair_index": 0, "fold": 0, "label": 1, "reference": "ref", "probe": "probe"}
        with self.assertRaisesRegex(ValueError, "eligible embedding"):
            pair_rows([pair], {"ref": {"status": "ok"}, "probe": {"status": "ok"}}, {}, {}, "original")


if __name__ == "__main__":
    unittest.main()
