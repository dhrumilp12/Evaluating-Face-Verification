"""Checks for independently calibrated biometric operating thresholds."""
import copy
import math
import unittest

import numpy as np

from src.project3_calibration import validate_records
from src.project3_operating_points import (
    METHODS, evaluate_operating_regime, select_fmr_threshold, split_folds,
)


CONDITIONS = ("original", "resolution_14", "blur_2.1", "brightness_0.25")


def example_rows(n_folds=4):
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
                q = min(reference_quality, probe_quality)
                rows.append({
                    "pair_index": index, "fold": fold, "label": label,
                    "reference_path": f"person_{index}/ref.jpg",
                    "probe_path": f"person_{index}/probe.jpg",
                    "condition": condition, "status": "scored", "error": "",
                    "cosine_similarity": cosine,
                    "qmagface_score": cosine + min(0., .125926*cosine - .077428)*q,
                    "reference_quality": reference_quality, "probe_quality": probe_quality,
                })
    return rows


def update_score(row, score, quality=None):
    row["cosine_similarity"] = score
    if quality is not None:
        row["probe_quality"] = quality
    q = min(row["reference_quality"], row["probe_quality"])
    row["qmagface_score"] = score + min(0., .125926*score - .077428)*q


def evaluate(rows, train=CONDITIONS, test=CONDITIONS):
    return evaluate_operating_regime(validate_records(rows, 4), regime="test",
                                     train_conditions=list(train), test_conditions=list(test),
                                     n_folds=4, target_fmr=.01)


class Project3OperatingPointTests(unittest.TestCase):
    def test_exact_integer_budget_and_acceptance_at_equality(self):
        scores = np.linspace(-3., 2., 600)
        result = select_fmr_threshold(scores)
        boundary = np.sort(scores)[-7]
        self.assertEqual(result["allowed_false_matches"], 6)
        self.assertEqual(result["observed_false_matches"], 6)
        self.assertEqual(result["n_impostors"], 600)
        self.assertEqual(result["threshold"], np.nextafter(boundary, np.inf))
        self.assertFalse(boundary >= result["threshold"])
        self.assertEqual(int(np.sum(scores >= result["threshold"])), 6)

    def test_boundary_ties_are_rejected_together(self):
        scores = np.r_[np.full(7, .9), np.linspace(-.7, .8, 593)]
        result = select_fmr_threshold(scores)
        self.assertEqual(result["allowed_false_matches"], 6)
        self.assertEqual(result["observed_false_matches"], 0)
        self.assertEqual(result["threshold"], np.nextafter(.9, np.inf))
        tied = select_fmr_threshold(np.full(600, -.25))
        self.assertEqual(tied["observed_false_matches"], 0)
        self.assertEqual(tied["threshold"], np.nextafter(-.25, np.inf))

    def test_small_samples_and_zero_target_reject_all_calibration_impostors(self):
        for target in (0., .01):
            result = select_fmr_threshold([-.8, -.4, -.1], target)
            self.assertEqual(result["allowed_false_matches"], 0)
            self.assertEqual(result["observed_false_matches"], 0)
            self.assertGreater(result["threshold"], -.1)

    def test_decimal_budget_does_not_lose_a_match_to_binary_rounding(self):
        # .29 * 100 is 28.999999999999996 in binary floating-point arithmetic.
        result = select_fmr_threshold(np.arange(100, dtype=float), .29)
        self.assertEqual(result["allowed_false_matches"], 29)
        self.assertEqual(result["observed_false_matches"], 29)

    def test_threshold_is_monotone_in_target_and_independent_of_input_order(self):
        scores = np.linspace(-20, 20, 600)
        thresholds = [select_fmr_threshold(scores, x)["threshold"] for x in (0., .01, .1, .5)]
        self.assertEqual(thresholds, sorted(thresholds, reverse=True))
        self.assertEqual(select_fmr_threshold(scores), select_fmr_threshold(scores[::-1]))

    def test_rejects_invalid_threshold_inputs(self):
        for scores in ([], [[.1, .2]], [.1, math.nan], [.1, math.inf], [-math.inf]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_fmr_threshold(scores)
        for target in (-.1, 1., 1.1, math.nan, math.inf, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                select_fmr_threshold([.1, .2], target)

    def test_rotation_has_disjoint_seven_two_one_fold_roles(self):
        for fold in range(10):
            split = split_folds(fold)
            self.assertEqual(split["test_folds"], [fold])
            self.assertEqual(set(split["calibration_folds"]), {(fold+1)%10, (fold+2)%10})
            self.assertEqual(len(split["fit_folds"]), 7)
            roles = [set(split[k]) for k in ("fit_folds", "calibration_folds", "test_folds")]
            self.assertEqual(set.union(*roles), set(range(10)))
            self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
        for args in ((0, 3), (10, 10), (-1, 10), (True, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                split_folds(*args)

    def test_calibration_and_test_data_cannot_fit_the_model(self):
        rows = example_rows()
        before = evaluate(rows)
        changed = copy.deepcopy(rows)
        # For outer fold zero, folds 1 and 2 calibrate and fold 3 alone fits.
        for row in changed:
            if row["fold"] in (0, 1, 2):
                row["label"] = 1-row["label"]
                update_score(row, -.8 if row["label"] else .8, quality=2.)
        after = evaluate(changed)
        a = [m for m in before["models"] if m["fold"] == 0]
        b = [m for m in after["models"] if m["fold"] == 0]
        self.assertEqual(len(a), len(METHODS))
        self.assertEqual(a, b)

    def test_outer_test_labels_scores_and_quality_cannot_select_thresholds(self):
        rows = example_rows()
        before = evaluate(rows)
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["fold"] == 0:
                row["label"] = 1-row["label"]
                update_score(row, -.8 if row["label"] else .8, quality=2.)
        after = evaluate(changed)
        a = [r for r in before["thresholds"] if r["fold"] == 0]
        b = [r for r in after["thresholds"] if r["fold"] == 0]
        self.assertEqual(a, b)

    def test_genuine_calibration_scores_do_not_select_operating_threshold(self):
        rows = example_rows()
        before = evaluate(rows)
        for row in rows:
            if row["fold"] in (1, 2) and row["label"] == 1:
                update_score(row, -.99, quality=2.)
        after = evaluate(rows)
        a = {(r["method"], r["policy"]): r["threshold"]
             for r in before["thresholds"] if r["fold"] == 0}
        b = {(r["method"], r["policy"]): r["threshold"]
             for r in after["thresholds"] if r["fold"] == 0}
        self.assertEqual(a, b)

    def test_original_only_policies_are_identical(self):
        result = evaluate(example_rows(), train=("original",))
        thresholds = {(r["method"], r["policy"], r["fold"]): r["threshold"] for r in result["thresholds"]}
        for method in METHODS:
            for fold in range(4):
                self.assertEqual(thresholds[(method, "pooled", fold)],
                                 thresholds[(method, "condition_cap", fold)])
        predictions = list(result["predictions"])
        p = {(r["method"], r["condition"], r["pair_index"]): r["prediction"]
             for r in predictions if r["policy"] == "pooled"}
        for row in predictions:
            if row["policy"] == "condition_cap":
                self.assertEqual(row["prediction"], p[(row["method"], row["condition"], row["pair_index"])])

    def test_condition_cap_threshold_and_acceptances_are_conservative(self):
        result = evaluate(example_rows())
        thresholds = {(r["method"], r["policy"], r["fold"]): r["threshold"] for r in result["thresholds"]}
        for method in METHODS:
            for fold in range(4):
                self.assertGreaterEqual(thresholds[(method, "condition_cap", fold)],
                                        thresholds[(method, "pooled", fold)])
        predictions = list(result["predictions"])
        p = {(r["method"], r["condition"], r["pair_index"]): r["prediction"]
             for r in predictions if r["policy"] == "pooled"}
        for row in predictions:
            if row["policy"] == "condition_cap" and row["prediction"] is not None:
                self.assertLessEqual(row["prediction"], p[(row["method"], row["condition"], row["pair_index"])])

    def test_cap_exposes_a_tail_hidden_by_the_pooled_calibration_mixture(self):
        rows = []
        for repetition in range(40):
            for source in example_rows():
                row = dict(source)
                row["pair_index"] += 24*repetition
                row["reference_path"] = f"person_{row['pair_index']}/ref.jpg"
                row["probe_path"] = f"person_{row['pair_index']}/probe.jpg"
                if row["label"] == 0:
                    offset = .3 if row["condition"] == "original" else -.3
                    update_score(row, row["cosine_similarity"]+offset+repetition*.0001)
                rows.append(row)
        result = evaluate_operating_regime(validate_records(rows, 4), regime="test",
                                          train_conditions=CONDITIONS, test_conditions=CONDITIONS,
                                          n_folds=4, methods=("cosine_threshold",))
        thresholds = {(r["policy"], r["fold"]): r["threshold"] for r in result["thresholds"]}
        for fold in range(4):
            self.assertGreater(thresholds[("condition_cap", fold)], thresholds[("pooled", fold)])
        cap_rows = [r for r in result["calibration"] if r["policy"] == "condition_cap"]
        self.assertTrue(all(r["observed_false_matches"] <= r["allowed_false_matches"] for r in cap_rows))
        pooled_high_tail = [r for r in result["calibration"]
                            if r["policy"] == "pooled" and r["condition"] == "original"]
        self.assertTrue(all(r["observed_false_matches"] > r["allowed_false_matches"] for r in pooled_high_tail))

    def test_failed_pair_remains_explicit_with_common_coverage(self):
        rows = example_rows()
        for row in rows:
            if row["pair_index"] == 0:
                row.update(status="preprocessing_failed", cosine_similarity=None,
                           qmagface_score=None, reference_quality=None,
                           error="reference:low_detection_probability")
        result = evaluate(rows)
        failed = [r for r in result["predictions"] if r["pair_index"] == 0]
        self.assertEqual(len(failed), len(CONDITIONS)*len(METHODS)*2)
        self.assertTrue(all(r["score"] is None and r["prediction"] is None and r["correct"] is None for r in failed))
        for row in result["aggregates"]:
            self.assertEqual(row["requested_pairs"], 24)
            self.assertEqual(row["scored_pairs"], 23)
            self.assertEqual(row["excluded_genuine_pairs"], 1)
            self.assertAlmostEqual(row["coverage"], 23/24)


if __name__ == "__main__":
    unittest.main()
