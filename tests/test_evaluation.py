"""Small independent examples for metrics and calibration leakage protection."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import select_threshold, verification_metrics, cross_validate


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_counts(self):
        result = verification_metrics([.9, .4, .6, .2], [1, 1, 0, 0], .5)
        self.assertEqual(result["true_accepts"], 1)
        self.assertEqual(result["true_rejects"], 1)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["fmr"], .5)
        self.assertEqual(result["fnmr"], .5)

    def test_threshold_between_separated_classes(self):
        threshold = select_threshold([.8, .9, .1, .2], [1, 1, 0, 0])
        self.assertAlmostEqual(threshold, .5)

    def test_ties_choose_reject_all(self):
        threshold = select_threshold([.5, .5], [1, 0])
        self.assertGreater(threshold, .5)
        result = verification_metrics([.5, .5], [1, 0], threshold)
        self.assertEqual(result["fmr"], 0)
        self.assertEqual(result["fnmr"], 1)

    def test_boundary_is_accepted(self):
        result = verification_metrics([.5, .5], [1, 0], .5)
        self.assertEqual(result["false_matches"], 1)
        self.assertEqual(result["false_nonmatches"], 0)

    def test_test_fold_cannot_set_its_threshold(self):
        rows = []
        for fold in range(3):
            for label, score in [(1, .8), (0, .2)]:
                rows.append(dict(pair_index=len(rows), fold=fold, label=label, status="scored", cosine_similarity=score))
        original, predictions, aggregate = cross_validate(rows, n_folds=3)
        rows[0]["cosine_similarity"] = -.9
        rows[1]["cosine_similarity"] = .95
        changed, _, _ = cross_validate(rows, n_folds=3)
        self.assertEqual(original[0]["threshold"], changed[0]["threshold"])
        self.assertEqual(original[0]["accuracy"], 1)
        self.assertEqual(changed[0]["accuracy"], 0)
        self.assertEqual(len(predictions), 6)
        self.assertEqual(aggregate["mean_accuracy"], 1)

    def test_failures_count_in_coverage_not_conditional_errors(self):
        rows = []
        for fold in range(2):
            for label, score, status in [(1, .9, "scored"), (0, .1, "scored"), (1, None, "preprocessing_failed")]:
                rows.append(dict(pair_index=len(rows), fold=fold, label=label, status=status, cosine_similarity=score))
        folds, predictions, aggregate = cross_validate(rows, n_folds=2)
        self.assertEqual(aggregate["mean_fnmr"], 0)
        self.assertAlmostEqual(aggregate["coverage"], 2 / 3)
        self.assertEqual(aggregate["excluded_pairs"], 2)
        self.assertEqual(folds[0]["reject_on_failure_genuine_rejection_rate"], .5)
        self.assertIsNone(predictions[2]["prediction"])

    def test_invalid_inputs_fail(self):
        for scores, labels in [([], []), ([.5], [1]), ([float("nan"), .2], [1, 0]), ([.1, .2], [1, 2])]:
            with self.assertRaises(ValueError):
                select_threshold(scores, labels)

    def test_accuracy_optimum_agrees_with_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            scores = np.round(rng.uniform(-1, 1, 30), 1)
            labels = np.array([0, 1] * 15)
            threshold = select_threshold(scores, labels)
            attained = np.mean((scores >= threshold) == labels)
            candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
            expected = max(np.mean((scores >= t) == labels) for t in candidates)
            self.assertEqual(attained, expected)


if __name__ == "__main__":
    unittest.main()
