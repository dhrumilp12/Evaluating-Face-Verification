"""Checks for pair grouping, withheld conditions and saved calibration models."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from src.project3_calibration import (
    METHODS, evaluate_regime, feature_matrix, fit_model, score_rows,
    validate_records,
)


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("original", "resolution_14", "blur_2.1", "brightness_0.25")


def example_rows(n_folds=3):
    """Several noisy scores in each class, reused only within the same pair."""
    rows = []
    examples = ((1, .85, 32.), (1, .58, 24.), (1, .31, 22.),
                (0, .44, 29.), (0, .18, 26.), (0, -.05, 20.))
    for condition_index, condition in enumerate(CONDITIONS):
        for fold in range(n_folds):
            for offset, (label, score, quality) in enumerate(examples):
                index = fold * len(examples) + offset
                cosine = score + .01 * fold - .04 * condition_index * label
                reference_quality = 28. + offset
                probe_quality = quality - condition_index
                qmin = min(reference_quality, probe_quality)
                adjusted = cosine + min(0., .125926 * cosine - .077428) * qmin
                rows.append({
                    "pair_index": index, "fold": fold, "label": label,
                    "reference_path": f"person_{index}/ref.jpg",
                    "probe_path": f"person_{index}/probe.jpg",
                    "condition": condition, "status": "scored", "error": "",
                    "cosine_similarity": cosine, "qmagface_score": adjusted,
                    "reference_quality": reference_quality, "probe_quality": probe_quality,
                })
    return rows


def fail_pair(rows, index):
    for row in rows:
        if row["pair_index"] == index:
            row.update(status="preprocessing_failed", cosine_similarity=None,
                       qmagface_score=None, reference_quality=None,
                       error="reference:low_detection_probability")


def evaluate(rows, train=CONDITIONS, test=CONDITIONS):
    return evaluate_regime(validate_records(rows, 3), regime="test",
                           train_conditions=list(train), test_conditions=list(test),
                           n_folds=3)


class Project3CalibrationTests(unittest.TestCase):
    def test_features_use_pair_minimum_and_explicit_interaction(self):
        row = example_rows()[1]
        s = row["cosine_similarity"]
        q = min(row["reference_quality"], row["probe_quality"])
        np.testing.assert_allclose(feature_matrix([row], "logistic_score"), [[s]])
        np.testing.assert_allclose(feature_matrix([row], "logistic_quality"), [[s, q]])
        np.testing.assert_allclose(feature_matrix([row], "logistic_interaction"), [[s, q, s*q]])
        swapped = dict(row, reference_quality=row["probe_quality"],
                       probe_quality=row["reference_quality"], label=1-row["label"])
        np.testing.assert_array_equal(feature_matrix([row], "logistic_interaction"),
                                      feature_matrix([swapped], "logistic_interaction"))

    def test_rejects_duplicate_missing_or_mismatched_pair_metadata(self):
        mutations = (
            lambda rows: rows.append(dict(rows[-1])),
            lambda rows: rows.pop(),
            lambda rows: rows[-1].update(label=1),
            lambda rows: rows[-1].update(fold=0),
            lambda rows: rows[-1].update(probe_path="other.jpg"),
        )
        for change in mutations:
            rows = example_rows()
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_records(rows, 3)

    def test_rejects_invalid_scores_qualities_or_status(self):
        for changes in ({"reference_quality": 0}, {"probe_quality": None},
                        {"probe_quality": float("inf")}, {"cosine_similarity": float("nan")},
                        {"cosine_similarity": 1.1}, {"qmagface_score": None},
                        {"status": "failed"}, {"label": True}, {"fold": 1.5}):
            rows = example_rows()
            rows[-1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_records(rows, 3)

    def test_rejects_condition_specific_eligibility_and_changed_reference_quality(self):
        rows = example_rows()
        rows[-1].update(status="failed", cosine_similarity=None, qmagface_score=None,
                        probe_quality=None)
        with self.assertRaises(ValueError):
            validate_records(rows, 3)
        rows = example_rows()
        rows[-1]["reference_quality"] += 1.
        with self.assertRaises(ValueError):
            validate_records(rows, 3)

    def test_heldout_labels_scores_and_quality_cannot_change_the_fitted_models(self):
        rows = example_rows()
        before = evaluate(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["fold"] == 0:
                row["label"] = 1-row["label"]
                row["cosine_similarity"] *= -.5
                row["probe_quality"] += 100.
                s = row["cosine_similarity"]
                q = min(row["reference_quality"], row["probe_quality"])
                row["qmagface_score"] = s + min(0., .125926*s - .077428)*q
        after = evaluate(changed)
        old_models = [m for m in before["models"] if m["fold"] == 0]
        new_models = [m for m in after["models"] if m["fold"] == 0]
        self.assertEqual(len(old_models), len(METHODS))
        self.assertEqual(old_models, new_models)

    def test_withheld_family_never_fits_scalers_coefficients_or_thresholds(self):
        train = tuple(c for c in CONDITIONS if not c.startswith("resolution_"))
        rows = example_rows()
        before = evaluate(rows, train=train, test=("original", "resolution_14"))
        for row in rows:
            if row["condition"].startswith("resolution_"):
                row["cosine_similarity"] = .999 if row["label"] == 0 else -.999
                row["probe_quality"] = 2.
                s = row["cosine_similarity"]
                row["qmagface_score"] = s + min(0., .125926*s - .077428)*2.
        after = evaluate(rows, train=train, test=("original", "resolution_14"))
        self.assertEqual(before["models"], after["models"])
        for model in before["models"]:
            self.assertNotIn("resolution_14", model["train_conditions"])

    def test_every_augmented_pair_stays_on_one_side_of_the_outer_split(self):
        result = evaluate(example_rows())
        for model in result["models"]:
            self.assertEqual(set(model["training_folds"]), set(range(3)) - {model["fold"]})
            heldout = list(range(model["fold"]*6, (model["fold"]+1)*6))
            training = sorted(set(range(18))-set(heldout))
            digest = lambda ids: hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            self.assertEqual(model["training_pair_indices_sha256"], digest(training))
            self.assertEqual(model["heldout_pair_indices_sha256"], digest(heldout))
            self.assertAlmostEqual(model["training_weight_sum"], 12.)

    def test_condition_replication_preserves_effective_regularization(self):
        original = [r for r in example_rows() if r["condition"] == "original"]
        duplicated = [dict(r, condition=f"copy_{n}") for n in range(4) for r in original]
        for method in ("logistic_score", "logistic_quality", "logistic_interaction"):
            first = fit_model(original, method, n_conditions=1)
            second = fit_model(duplicated, method, n_conditions=4)
            np.testing.assert_allclose(score_rows(first, original), score_rows(second, original),
                                       rtol=1e-6, atol=1e-6)

    def test_serialized_models_reconstruct_scores_and_fixed_logit_decision(self):
        rows = [r for r in example_rows() if r["condition"] == "original"]
        for method in ("logistic_score", "logistic_quality", "logistic_interaction"):
            model = fit_model(rows, method)
            restored = json.loads(json.dumps(model, allow_nan=False))
            x = feature_matrix(rows, method)
            expected = ((x-np.asarray(model["scaler_mean"]))/np.asarray(model["scaler_scale"])) @ np.asarray(model["coefficients"]) + model["intercept"]
            np.testing.assert_allclose(score_rows(restored, rows), expected, rtol=0, atol=1e-12)
            self.assertEqual(model["threshold"], 0.)

    def test_failures_remain_explicit_and_metrics_use_correct_denominators(self):
        rows = example_rows()
        fail_pair(rows, 0)
        result = evaluate(rows)
        predictions = list(result["predictions"])
        failed = [r for r in predictions if r["pair_index"] == 0]
        self.assertEqual(len(failed), len(CONDITIONS)*len(METHODS))
        for row in failed:
            self.assertIsNone(row["score"])
            self.assertIsNone(row["prediction"])
            self.assertIsNone(row["correct"])
        for row in result["aggregates"]:
            self.assertEqual(row["requested_pairs"], 18)
            self.assertEqual(row["scored_pairs"], 17)
            self.assertEqual(row["excluded_genuine_pairs"], 1)
            self.assertAlmostEqual(row["coverage"], 17/18)

    def test_validation_is_order_independent_and_does_not_mutate_input(self):
        rows = example_rows()
        before = copy.deepcopy(rows)
        result = validate_records(rows, 3)
        reversed_result = validate_records(list(reversed(rows)), 3)
        self.assertEqual(result, reversed_result)
        self.assertEqual(rows, before)

    def test_original_threshold_controls_recover_the_saved_project2_thresholds(self):
        source = ROOT / "results/project2/robustness"
        with (source / "pair_scores.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        original = [r for r in rows if r["condition"] == "original" and r["status"] == "scored"]
        thresholds = json.loads((source / "evaluation.json").read_text())["thresholds"]
        for fold in range(10):
            training = [r for r in original if r["fold"] != fold]
            for method, saved in (("cosine_threshold", "magface_cosine"),
                                  ("qmagface_threshold", "qmagface")):
                with self.subTest(fold=fold, method=method):
                    model = fit_model(training, method)
                    self.assertEqual(model["threshold"], thresholds[saved][str(fold)])

    def test_training_without_both_classes_fails_instead_of_using_test_rows(self):
        rows = [r for r in example_rows() if r["label"] == 1]
        for method in METHODS:
            with self.subTest(method=method), self.assertRaises(ValueError):
                fit_model(rows, method)


if __name__ == "__main__":
    unittest.main()
