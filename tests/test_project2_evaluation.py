"""Leakage, paired denominators, and independent numerical evaluation checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.project2_evaluation import evaluate_conditions, make_plots


def example_rows(n_folds=3):
    rows = []
    for condition in ("original", "brightness_0.25"):
        for fold in range(n_folds):
            for offset, (label, cosine, adjusted) in enumerate(((1, .9, .8), (1, .7, .5),
                                                               (0, .3, -1.0), (0, .1, -2.0))):
                index = fold * 4 + offset
                rows.append({"pair_index": index, "fold": fold, "label": label,
                             "reference_path": f"person_{index}/1.jpg", "probe_path": f"person_{index}/2.jpg",
                             "condition": condition, "status": "scored", "error": "",
                             "cosine_similarity": cosine, "qmagface_score": adjusted,
                             "reference_quality": 30., "probe_quality": 25.})
    return rows


def get_record(result, table, **selectors):
    return next(r for r in result[table] if all(r[key] == value for key, value in selectors.items()))


def fail(row, reason="probe alignment failed"):
    row.update(status="preprocessing_failed", cosine_similarity=None, qmagface_score=None,
               probe_quality=None, error=reason)


class Project2EvaluationTests(unittest.TestCase):
    def test_separate_method_thresholds_and_original_reuse(self):
        result = evaluate_conditions(example_rows(), n_folds=3)
        for fold in range(3):
            self.assertAlmostEqual(result["thresholds"]["magface_cosine"][str(fold)], .5)
            self.assertAlmostEqual(result["thresholds"]["qmagface"][str(fold)], -.25)
            for method in ("magface_cosine", "qmagface"):
                original = get_record(result, "folds", condition="original", method=method, fold=fold)
                degraded = get_record(result, "folds", condition="brightness_0.25", method=method, fold=fold)
                self.assertEqual(original["threshold"], degraded["threshold"])
                self.assertEqual(original["calibration_pairs"], 8)
        self.assertEqual(len(result["predictions"]), 48)
        self.assertTrue(all(r["correct"] == 1 for r in result["predictions"]))

    def test_degraded_scores_and_labels_never_calibrate(self):
        rows = example_rows()
        expected = evaluate_conditions(rows, 3)["thresholds"]
        for row in rows:
            if row["condition"] != "original":
                row["cosine_similarity"] = -1 if row["label"] else 1
                row["qmagface_score"] = -20 if row["label"] else 20
        actual = evaluate_conditions(rows, 3)
        self.assertEqual(actual["thresholds"], expected)
        self.assertEqual(get_record(actual, "aggregates", condition="brightness_0.25",
                                    method="qmagface")["pooled_accuracy"], 0.)

    def test_held_out_original_fold_cannot_set_its_own_threshold(self):
        rows = example_rows()
        before = evaluate_conditions(rows, 3)
        for row in rows:
            if row["condition"] == "original" and row["fold"] == 0:
                row["cosine_similarity"] = -1 if row["label"] else 1
                row["qmagface_score"] = -20 if row["label"] else 20
        after = evaluate_conditions(rows, 3)
        for method in ("magface_cosine", "qmagface"):
            self.assertEqual(before["thresholds"][method]["0"], after["thresholds"][method]["0"])
            self.assertEqual(get_record(after, "folds", condition="original", method=method, fold=0)["accuracy"], 0)

    def test_failure_coverage_and_reject_policy_use_requested_denominators(self):
        rows = example_rows()
        for row in rows:
            if row["condition"] == "brightness_0.25" and row["pair_index"] in (0, 2):
                fail(row)
        result = evaluate_conditions(rows, 3)
        record = get_record(result, "folds", condition="brightness_0.25", method="qmagface", fold=0)
        self.assertEqual(record["requested_pairs"], 4)
        self.assertEqual(record["scored_pairs"], 2)
        self.assertEqual(record["coverage"], .5)
        self.assertEqual(record["genuine_coverage"], .5)
        self.assertEqual(record["impostor_coverage"], .5)
        self.assertEqual(record["accuracy"], 1.)
        self.assertEqual(record["fmr"], 0.)
        self.assertEqual(record["fnmr"], 0.)
        self.assertEqual(record["reject_on_failure_accuracy"], .75)
        self.assertEqual(record["reject_on_failure_fnmr"], .5)
        self.assertEqual(record["failure_reasons"], {"probe alignment failed": 2})
        for method in ("magface_cosine", "qmagface"):
            prediction = get_record(result, "predictions", condition="brightness_0.25", method=method, pair_index=0)
            self.assertIsNone(prediction["prediction"])
            self.assertIsNone(prediction["correct"])
            self.assertEqual(prediction["reference_quality"], 30.)

    def test_original_failures_are_excluded_only_from_calibration_with_counts(self):
        rows = example_rows()
        fail(rows[0])
        result = evaluate_conditions(rows, 3)
        self.assertEqual(get_record(result, "folds", condition="original", method="qmagface", fold=0)["calibration_pairs"], 8)
        self.assertEqual(get_record(result, "folds", condition="original", method="qmagface", fold=1)["calibration_pairs"], 7)
        paired = get_record(result, "condition_changes", condition="brightness_0.25", method="qmagface", fold=None)
        self.assertEqual(paired["common_scored_pairs"], 11)
        self.assertEqual(paired["excluded_from_paired_comparison"], 1)

    def test_paired_decisions_count_improvements_and_regressions(self):
        rows = example_rows()
        changed = [r for r in rows if r["condition"] == "brightness_0.25" and r["fold"] == 0]
        changed[0]["cosine_similarity"], changed[0]["qmagface_score"] = .4, .2  # C corrects rejection
        changed[1]["cosine_similarity"], changed[1]["qmagface_score"] = .6, -.5  # C introduces rejection
        changed[2]["cosine_similarity"], changed[2]["qmagface_score"] = .6, -.5  # C corrects false match
        fail(changed[3])
        result = evaluate_conditions(rows, 3)
        record = get_record(result, "paired_changes", condition="brightness_0.25", fold=0)
        self.assertEqual(record["qmagface_only_correct"], 2)
        self.assertEqual(record["cosine_only_correct"], 1)
        self.assertEqual(record["net_correct_gain"], 1)
        self.assertEqual(record["changed_decisions"], 3)
        self.assertEqual(record["changed_to_accept"], 1)
        self.assertEqual(record["changed_to_reject"], 2)
        self.assertEqual(record["excluded_pairs"], 1)
        self.assertEqual(record["changed_fraction_of_scored"], 1.)

    def test_paired_condition_delta_uses_same_successful_pairs(self):
        rows = example_rows()
        for row in rows:
            if row["condition"] == "brightness_0.25" and row["fold"] == 0:
                if row["pair_index"] == 0:
                    row["cosine_similarity"] = 0.
                if row["pair_index"] == 2:
                    fail(row)
        result = evaluate_conditions(rows, 3)
        record = get_record(result, "condition_changes", condition="brightness_0.25", method="magface_cosine", fold=0)
        self.assertEqual(record["common_scored_pairs"], 3)
        self.assertEqual(record["common_genuine_pairs"], 2)
        self.assertEqual(record["common_impostor_pairs"], 1)
        self.assertEqual(record["original_accuracy"], 1)
        self.assertAlmostEqual(record["delta_accuracy"], -1/3)
        self.assertEqual(record["delta_fnmr"], .5)
        self.assertEqual(record["correct_to_wrong"], 1)

    def test_all_failed_test_fold_has_null_rates_and_explicit_valid_fold_counts(self):
        rows = example_rows()
        for row in rows:
            if row["condition"] == "brightness_0.25" and row["fold"] == 0:
                fail(row)
        result = evaluate_conditions(rows, 3)
        record = get_record(result, "folds", condition="brightness_0.25", method="qmagface", fold=0)
        self.assertEqual(record["coverage"], 0)
        self.assertIsNone(record["accuracy"])
        self.assertIsNone(record["auc"])
        self.assertEqual(record["reject_on_failure_accuracy"], .5)
        self.assertEqual(record["reject_on_failure_fnmr"], 1.)
        aggregate = get_record(result, "aggregates", condition="brightness_0.25", method="qmagface")
        self.assertIsNone(aggregate["mean_accuracy"])
        self.assertEqual(aggregate["valid_folds_accuracy"], 2)
        self.assertEqual(aggregate["pooled_accuracy"], 1.)
        json.dumps(result, allow_nan=False)

    def test_missing_test_class_does_not_claim_zero_error_rate(self):
        rows = example_rows()
        for row in rows:
            if row["condition"] == "brightness_0.25" and row["fold"] == 0 and row["label"] == 0:
                fail(row)
        result = evaluate_conditions(rows, 3)
        record = get_record(result, "folds", condition="brightness_0.25", method="magface_cosine", fold=0)
        self.assertIsNone(record["fmr"])
        self.assertEqual(record["fnmr"], 0.)
        self.assertIsNone(record["eer_interpolated"])

    def test_roc_auc_eer_with_ties_and_scores_outside_cosine_range(self):
        rows = example_rows()
        for row in rows:
            if row["condition"] != "original":
                row["qmagface_score"] = -3.
        result = evaluate_conditions(rows, 3)
        original = get_record(result, "aggregates", condition="original", method="qmagface")
        tied = get_record(result, "aggregates", condition="brightness_0.25", method="qmagface")
        self.assertEqual(original["pooled_auc"], 1.)
        self.assertEqual(original["pooled_eer_interpolated"], 0.)
        self.assertEqual(tied["pooled_auc"], .5)
        self.assertEqual(tied["pooled_eer_interpolated"], .5)
        curve = [r for r in result["curves"] if r["condition"] == "brightness_0.25" and r["method"] == "qmagface"]
        self.assertEqual(len(curve), 2)
        self.assertIsNone(curve[0]["threshold"])
        self.assertEqual(curve[0]["decision_rule"], "reject_all")
        self.assertEqual(curve[1]["threshold"], -3.)
        json.dumps(result, allow_nan=False)

    def test_rejects_missing_duplicate_or_changed_pair_metadata(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[-1])),
                       lambda rows: rows[-1].update(fold=0), lambda rows: rows[-1].update(label=1),
                       lambda rows: rows[-1].update(probe_path="other.jpg"),
                       lambda rows: rows[-1].update(pair_index=9000)):
            rows = example_rows()
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                evaluate_conditions(rows, 3)

    def test_rejects_invalid_or_method_specific_missing_scores(self):
        for changes in ({"qmagface_score": None}, {"cosine_similarity": float("nan")},
                        {"cosine_similarity": 1.1}, {"qmagface_score": float("inf")},
                        {"reference_quality": 0}, {"reference_quality": None},
                        {"status": "failed"}, {"label": True}, {"fold": 1.5}):
            rows = example_rows()
            rows[-1].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate_conditions(rows, 3)

    def test_rejects_incomplete_calibration_without_silently_using_test_scores(self):
        rows = example_rows()
        for row in rows:
            if row["condition"] == "original" and row["label"] == 0:
                fail(row)
        with self.assertRaisesRegex(ValueError, "Original calibration"):
            evaluate_conditions(rows, 3)

    def test_input_order_does_not_change_results_or_mutate_rows(self):
        rows = example_rows()
        untouched = copy.deepcopy(rows)
        result = evaluate_conditions(rows, 3)
        self.assertEqual(rows, untouched)
        permuted = evaluate_conditions(list(reversed(rows)), 3)
        self.assertEqual(result["thresholds"], permuted["thresholds"])
        sort = lambda values: sorted(values, key=lambda r: (r["condition"], r["method"], r["pair_index"]))
        self.assertEqual(sort(result["predictions"]), sort(permuted["predictions"]))

    def test_plot_artifacts_are_created_from_evaluation_without_inference(self):
        result = evaluate_conditions(example_rows(), 3)
        with tempfile.TemporaryDirectory() as directory:
            paths = make_plots(result, directory)
            self.assertEqual(len(paths), 5)
            for path in paths:
                self.assertTrue(Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertGreater(Path(path).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
