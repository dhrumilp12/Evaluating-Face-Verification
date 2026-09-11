"""Resolution transforms, fixed evaluation, and synthetic cache integration."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from degradation import reduce_resolution
import resolution_experiment as experiment
from baseline_verification import write_json, write_csv
from lfw_dataset import sha256


def small_rows():
    rows = []
    for fold in range(10):
        for label, probe, score in ((1, "A2", 1.0), (0, "B1", 0.0)):
            rows.append({"pair_index": len(rows), "fold": fold, "label": label,
                         "reference": "A1", "probe": probe, "status": "scored", "cosine_similarity": score})
    rows.append({"pair_index": 20, "fold": 0, "label": 1, "reference": "missing",
                 "probe": "A2", "status": "preprocessing_failed", "cosine_similarity": None})
    return rows


class ResolutionTests(unittest.TestCase):
    def test_control_exact_copy(self):
        raw = np.random.default_rng(42).uniform(0, 255, (3, 160, 160)).astype(np.float32)
        original = raw.copy()
        result = reduce_resolution(raw, 160)
        np.testing.assert_array_equal(result, original)
        result[:] = 0
        np.testing.assert_array_equal(raw, original)

    def test_constant_preserved_and_checkerboard_loses_detail(self):
        constant = np.full((3, 160, 160), 117.5, dtype=np.float32)
        checker = np.tile((np.indices((160, 160)).sum(axis=0) % 2 * 255)[None], (3, 1, 1)).astype(np.float32)
        for size in (80, 40, 20):
            np.testing.assert_allclose(reduce_resolution(constant, size), constant, atol=1e-5)
            output = reduce_resolution(checker, size)
            self.assertEqual(output.shape, checker.shape)
            self.assertLess(output.std(), checker.std())
            self.assertTrue(np.isfinite(output).all())

    def test_invalid_input(self):
        for raw, size in ((np.zeros((3, 160, 160)), 30), (np.zeros((3, 160, 160)), 80.0),
                          (np.zeros((160, 160, 3)), 80), (np.full((3,160,160),np.nan), 80),
                          (np.full((3,160,160),256),80)):
            with self.assertRaises(ValueError):
                reduce_resolution(raw, size)

    def test_fixed_threshold_and_pair_membership(self):
        rows = small_rows()
        thresholds = {str(i): .5 for i in range(10)}
        scores = {r["pair_index"]: r["cosine_similarity"] for r in rows if r["status"] == "scored"}
        a, folds, predictions = experiment.evaluate_fixed(rows, scores, thresholds, 160)
        self.assertEqual(a["scored_pairs"], 20)
        self.assertEqual(a["excluded_pairs"], 1)
        scores[0] = .1
        changed, new_folds, _ = experiment.evaluate_fixed(rows, scores, thresholds, 20)
        self.assertEqual([r["threshold"] for r in new_folds], [r["threshold"] for r in folds])
        self.assertEqual(changed["false_nonmatches"], 1)
        self.assertIsNone(predictions[-1]["prediction"])
        del scores[0]
        with self.assertRaises(ValueError):
            experiment.evaluate_fixed(rows, scores, thresholds, 20)

    def test_runner_resume_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            for name in ("degradation.py", "resolution_experiment.py", "face_embedding.py", "evaluation.py"):
                (root / "src" / name).write_text("# Synthetic source fingerprint\n")
            metrics = root / "results/metrics"
            cache = root / "data/processed/baseline/fixture"
            cache.mkdir(parents=True)
            records, vectors = [], {}
            for name, intensity in (("A1",80),("A2",80),("B1",160)):
                vector = np.zeros(512,dtype=np.float32)
                vector[0 if intensity < 100 else 1] = 1
                vectors[name] = vector
                payload = cache / (name + ".npz")
                np.savez_compressed(payload, crop=np.full((3,160,160),intensity,dtype=np.float32), embedding=vector)
                records.append({"path":name,"status":"ok","payload":payload.name,"payload_sha256":sha256(payload)})
            fingerprint = {"fixture": True}
            write_json(cache / "manifest.json", {"fingerprint":fingerprint,"images":{r["path"]:r for r in records}})
            rows = small_rows()
            thresholds = {str(i):.5 for i in range(10)}
            scores = {r["pair_index"]:r["cosine_similarity"] for r in rows if r["status"]=="scored"}
            aggregate, _, _ = experiment.evaluate_fixed(rows,scores,thresholds,160)
            write_csv(metrics / "baseline_scores.csv", rows)
            write_json(metrics / "baseline_images.json", records)
            saved = {"model_fingerprint":fingerprint,"cache_directory":"data/processed/baseline/fixture",
                     "eligible_pair_indices":list(scores),"thresholds":thresholds}
            write_json(metrics / "baseline_thresholds.json", saved)
            baseline = {"status":"completed","requested_pairs":len(rows),"aggregate":aggregate,
                        "model_fingerprint":fingerprint,"cache_directory":saved["cache_directory"],
                        "output_sha256":{n:sha256(metrics/n) for n in ("baseline_scores.csv","baseline_images.json","baseline_thresholds.json")}}
            baseline["aggregate"]["confusion_counts"] = {"false_matches":0,"false_nonmatches":0}
            write_json(metrics / "baseline_summary.json", baseline)
            before = {p:sha256(p) for p in cache.iterdir()}
            class FakeModel:
                calls = 0
                def __init__(self, root): pass
                def crop(self, image): raise AssertionError("Detection/cropping must never run")
                def embed(self, raw):
                    FakeModel.calls += 1
                    return vectors["A1" if raw.mean()<100 else "B1"].copy()
            with patch.object(experiment,"FaceEmbedder",FakeModel), patch.object(experiment,"verify_model"):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = experiment.run_resolution(root)
                self.assertEqual(result["status"],"completed")
                self.assertEqual(FakeModel.calls,7)  # six degraded probes plus one control check
                for row in result["conditions"]:
                    self.assertEqual(row["scored_pairs"],20)
                    self.assertEqual(row["excluded_pairs"],1)
                FakeModel.calls = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    repeated = experiment.run_resolution(root)
                self.assertEqual(FakeModel.calls,1)
                self.assertEqual(repeated["conditions"][1]["reused_probe_embeddings"],2)
                self.assertEqual(before,{p:sha256(p) for p in cache.iterdir()})
                (cache / "A1.npz").write_bytes(b"broken")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    experiment.run_resolution(root)
                failed = json.loads((metrics/"resolution_summary.json").read_text())
                self.assertEqual(failed["status"],"failed")


if __name__ == "__main__":
    unittest.main()
