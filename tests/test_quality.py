"""Numerical transform properties and synthetic fixed-threshold integration."""
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
from quality_degradation import gaussian_blur, reduce_brightness, apply_condition, CONDITIONS
import quality_experiment as experiment
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


class QualityTests(unittest.TestCase):
    def test_identity_and_input_preservation(self):
        raw = np.random.default_rng(42).uniform(0,255,(3,160,160)).astype(np.float32)
        original = raw.copy()
        for result in (gaussian_blur(raw,0),reduce_brightness(raw,1),apply_condition(raw,CONDITIONS[0])):
            np.testing.assert_array_equal(result,original)
            result[:] = 0
        for condition in CONDITIONS:
            result = apply_condition(raw,condition)
            self.assertEqual(result.shape,raw.shape)
            self.assertEqual(result.dtype,np.float32)
            self.assertTrue(np.isfinite(result).all())
            self.assertGreaterEqual(result.min(),0)
            self.assertLessEqual(result.max(),255)
        np.testing.assert_array_equal(raw,original)

    def test_gaussian_impulse_spread_and_channels(self):
        raw = np.zeros((3,160,160),dtype=np.float32)
        raw[0,80,80] = 255
        last_peak = 256
        for sigma in (1,2,3):
            out = gaussian_blur(raw,sigma)
            self.assertAlmostEqual(float(out[0].sum()),255,places=3)
            self.assertEqual(float(out[1:].sum()),0)  # no channel mixing
            self.assertLess(out[0,80,80],last_peak)
            last_peak = out[0,80,80]
            np.testing.assert_allclose(out[0,80,71:90],out[0,71:90,80])
            offsets = np.arange(160)-80
            variance = float((out[0].sum(axis=0) * offsets**2).sum()/255)
            self.assertAlmostEqual(variance,sigma**2,delta=.04*sigma**2)
        constant = np.full((3,160,160),117.5,dtype=np.float32)
        np.testing.assert_allclose(gaussian_blur(constant,3),constant,atol=1e-5)

    def test_brightness_scales_raw_pixels_without_blurring(self):
        raw = np.tile((np.indices((160,160)).sum(axis=0)%2*255)[None],(3,1,1)).astype(np.float32)
        for factor in (.75,.5,.25):
            out=reduce_brightness(raw,factor)
            np.testing.assert_array_equal(out,raw*factor)
            self.assertEqual(float(out[0,0,0]),0)
            self.assertEqual(float(out[0,0,1]),255*factor)

    def test_invalid_inputs(self):
        for raw in (np.zeros((160,160,3)),np.full((3,160,160),np.nan),np.full((3,160,160),-1)):
            for transform,level in ((gaussian_blur,1),(reduce_brightness,.5)):
                with self.assertRaises(ValueError): transform(raw,level)
        raw=np.zeros((3,160,160))
        for transform,levels in ((gaussian_blur,[-1,4,np.nan,True]),(reduce_brightness,[0,1.5,np.inf,True])):
            for level in levels:
                with self.assertRaises(ValueError): transform(raw,level)

    def test_labels_thresholds_and_pair_membership(self):
        rows=small_rows()
        scores={r["pair_index"]:r["cosine_similarity"] for r in rows if r["status"]=="scored"}
        thresholds={str(i):.5 for i in range(10)}
        scores[0]=.1
        a,folds,predictions=experiment.evaluate_condition(rows,scores,thresholds,CONDITIONS[1])
        self.assertEqual(a["false_nonmatches"],1)
        self.assertEqual(a["excluded_pairs"],1)
        self.assertTrue(all(f["threshold"]==.5 for f in folds))
        self.assertTrue(all(p["condition"]=="blur_sigma_1" for p in predictions))
        self.assertNotIn("resolution",a)
        self.assertIsNone(predictions[-1]["prediction"])
        del scores[0]
        with self.assertRaises(ValueError):
            experiment.evaluate_condition(rows,scores,thresholds,CONDITIONS[1])

    def test_runner_resume_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            for name in ("quality_degradation.py", "quality_experiment.py", "degradation.py", "resolution_experiment.py",
                         "baseline_verification.py", "lfw_dataset.py", "face_embedding.py", "evaluation.py"):
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
                    result = experiment.run_quality(root)
                self.assertEqual(result["status"],"completed")
                self.assertEqual(FakeModel.calls,13)
                self.assertEqual(len(result["conditions"]),7)
                # B1 darkens below the synthetic identity cutoff at 50% and 25%.
                self.assertEqual(result["conditions"][-1]["false_matches"],10)
                self.assertEqual(result["conditions"][-2]["false_matches"],10)
                self.assertEqual(result["conditions"][4]["false_matches"],0)  # twelve degraded probes plus one control check
                for row in result["conditions"]:
                    self.assertEqual(row["scored_pairs"],20)
                    self.assertEqual(row["excluded_pairs"],1)
                FakeModel.calls = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    repeated = experiment.run_quality(root)
                self.assertEqual(FakeModel.calls,1)
                self.assertEqual(repeated["conditions"][1]["reused_probe_embeddings"],2)
                self.assertEqual(before,{p:sha256(p) for p in cache.iterdir()})
                # New invalid embeddings must abort instead of silently dropping pairs.
                payload = next((root / repeated["cache_directory"]).rglob("*.npy"))
                payload.write_bytes(b"broken")
                original_embed = FakeModel.embed
                def bad_embed(self, raw):
                    if FakeModel.calls >= 1:
                        return np.full(512, np.nan, dtype=np.float32)
                    return original_embed(self, raw)
                FakeModel.calls = 0
                with patch.object(FakeModel, "embed", bad_embed):
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                        experiment.run_quality(root)
                self.assertEqual(json.loads((metrics/"quality_summary.json").read_text())["status"],"failed")
                FakeModel.calls = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    repaired = experiment.run_quality(root)
                self.assertEqual(repaired["status"],"completed")
                self.assertEqual(FakeModel.calls,2)  # control plus one repaired payload
                self.assertEqual(before,{p:sha256(p) for p in cache.iterdir()})
                (cache / "A1.npz").write_bytes(b"broken")
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    experiment.run_quality(root)
                failed = json.loads((metrics/"quality_summary.json").read_text())
                self.assertEqual(failed["status"],"failed")


if __name__ == "__main__":
    unittest.main()
