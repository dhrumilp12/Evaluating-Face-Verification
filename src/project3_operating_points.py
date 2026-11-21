"""Independent fitting/calibration/test splits for target-FMR verification.

The two calibration folds set thresholds from impostor scores only. Learned
scores fit on seven other folds and are never refitted after calibration.
Calibration FMR constraints are empirical, not guarantees on held-out data.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from numbers import Integral

import numpy as np

from .project3_calibration import (fit_model, score_rows, indices_sha256,
                                   _metrics, _aggregate)

METHODS = ("cosine_threshold", "qmagface_threshold", "logistic_interaction")
POLICIES = ("pooled", "condition_cap")
PREDICTION_FIELDS = ("regime", "method", "policy", "condition", "pair_index", "fold",
                     "label", "status", "score", "threshold", "prediction", "correct")


def split_folds(test_fold, n_folds=10):
    """Reserve the next two cyclic folds for calibration, all others for fitting."""
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 4:
        raise ValueError("At least four integer folds are required")
    if isinstance(test_fold, bool) or not isinstance(test_fold, Integral) or test_fold not in range(n_folds):
        raise ValueError("test_fold must identify one of the predefined folds")
    calibration = [(int(test_fold) + offset) % int(n_folds) for offset in (1, 2)]
    return {"fit_folds": [fold for fold in range(n_folds) if fold != test_fold and fold not in calibration],
            "calibration_folds": calibration, "test_folds": [int(test_fold)]}


def select_fmr_threshold(scores, target_fmr=.01):
    """Choose the lowest finite float threshold satisfying an impostor budget.

    Acceptance is score >= threshold. The disallowed sorted score is excluded
    by nextafter(..., +inf), including its whole tie block. Decimal-string exact
    arithmetic computes floor(target*N), avoiding a binary floating-point budget
    error. Repeated impostor variants are counts, not independent observations.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Impostor scores must be a nonempty finite one-dimensional vector")
    if isinstance(target_fmr, (bool, np.bool_)):
        raise ValueError("target_fmr must be between zero inclusive and one exclusive")
    try:
        target = Fraction(str(target_fmr))
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError("target_fmr must be finite and numeric") from error
    if not 0 <= target < 1:
        raise ValueError("target_fmr must be between zero inclusive and one exclusive")
    budget = target.numerator * len(values) // target.denominator
    descending = np.sort(values)[::-1]
    with np.errstate(over="ignore"):
        threshold = float(np.nextafter(descending[budget], np.inf))
    if not np.isfinite(threshold):
        raise ValueError("No finite threshold can exclude the disallowed maximum score")
    observed = int(np.sum(values >= threshold))
    if observed > budget:
        raise ValueError("Computed threshold violated the calibration impostor budget")
    return {"threshold": threshold, "allowed_false_matches": int(budget),
            "observed_false_matches": observed, "n_impostors": len(values),
            "target_fmr": float(target), "target_fmr_exact": str(target),
            "observed_fmr": observed / len(values)}


def _score_model(rows, method, n_conditions):
    if method == "logistic_interaction":
        model = fit_model(rows, method, c=1.0, n_conditions=n_conditions)
        model.pop("threshold")
        model.update(model_type="learned_logistic", fit_rows_used=len(rows))
        return model
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    # Fixed score definitions do not estimate parameters or inspect fitting labels.
    return {"method": method, "model_type": "fixed_score", "fit_rows_used": 0,
            "score_key": "cosine_similarity" if method == "cosine_threshold" else "qmagface_score",
            "feature_names": [], "coefficients": [], "intercept": None,
            "scaler_mean": [], "scaler_scale": [], "c": None, "n_iter": 0,
            "training_rows": 0, "n_conditions": int(n_conditions), "training_weight_sum": 0.0,
            "quality_range_train": None, "cosine_slope_min_train": None,
            "cosine_slope_max_train": None}


def evaluate_operating_regime(groups, *, regime, train_conditions, test_conditions,
                              n_folds=10, methods=METHODS, target_fmr=.01, progress=None):
    """Return decision evidence from disjoint fitting/calibration/test pair folds.

    groups must come from project3_calibration.validate_records. Both calibration
    policies use the same score object. The condition-cap policy takes the maximum
    of per-condition target-FMR thresholds and never uses a test condition label
    to pick a threshold. All variants of each pair stay in its original fold.
    """
    split_folds(0, n_folds)  # Validate the fold count before allocating arrays.
    methods, train_conditions, test_conditions = tuple(methods), tuple(train_conditions), tuple(test_conditions)
    if (not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS) or
            not train_conditions or len(set(train_conditions)) != len(train_conditions) or
            not test_conditions or len(set(test_conditions)) != len(test_conditions)):
        raise ValueError("Distinct, nonempty supported methods and condition lists are required")
    if not set(train_conditions + test_conditions) <= set(groups):
        raise ValueError("Unknown fitting/calibration/evaluation condition")
    original = groups["original"]
    fold_ids = np.asarray([row["fold"] for row in original])
    if set(fold_ids.tolist()) != set(range(n_folds)):
        raise ValueError("Expected every predefined pair fold")
    n_pairs = len(original)
    scores = {method: {condition: np.full(n_pairs, np.nan) for condition in test_conditions} for method in methods}
    decision_thresholds = {method: {policy: np.full(n_pairs, np.nan) for policy in POLICIES} for method in methods}
    models, threshold_records, calibration_records, folds, aggregates = [], [], [], [], []
    for fold in range(n_folds):
        if progress is not None:
            progress(f"{regime}: operating-point fold {fold + 1}/{n_folds}")
        split = split_folds(fold, n_folds)
        fitting = [row for condition in train_conditions for row in groups[condition]
                   if row["fold"] in split["fit_folds"] and row["status"] == "scored"]
        fit_ids = {row["pair_index"] for row in fitting}
        calibration_ids = {row["pair_index"] for row in original if row["fold"] in split["calibration_folds"]}
        heldout_ids = {row["pair_index"] for row in original if row["fold"] == fold}
        if fit_ids & calibration_ids or fit_ids & heldout_ids or calibration_ids & heldout_ids:
            raise ValueError("Fitting, calibration, and held-out pair IDs must be disjoint")
        if set(Counter(row["pair_index"] for row in fitting).values()) != {len(train_conditions)}:
            raise ValueError("Every fitting pair needs one successful row per fitting condition")
        calibration_rows = {condition: [row for row in groups[condition]
                                       if row["fold"] in split["calibration_folds"]]
                            for condition in train_conditions}
        test_indices = np.flatnonzero(fold_ids == fold)
        for method in methods:
            model = _score_model(fitting, method, len(train_conditions))
            model_id = f"{regime}/{method}/fold_{fold}"
            model.update(model_id=model_id, regime=regime, fold=fold, **split,
                         train_conditions=list(train_conditions), test_conditions=list(test_conditions),
                         fit_rows_available=len(fitting), fit_unique_pairs_available=len(fit_ids),
                         fit_pair_indices_sha256=indices_sha256(fit_ids),
                         calibration_requested_pair_indices_sha256=indices_sha256(calibration_ids),
                         heldout_requested_pair_indices_sha256=indices_sha256(heldout_ids))
            models.append(model)
            calibration_scores = {condition: score_rows(model, rows) for condition, rows in calibration_rows.items()}
            negatives, condition_choices, calibration_impostor_ids = {}, {}, set()
            for condition, rows in calibration_rows.items():
                mask = np.asarray([row["label"] == 0 and row["status"] == "scored" for row in rows])
                negatives[condition] = calibration_scores[condition][mask]
                condition_choices[condition] = select_fmr_threshold(negatives[condition], target_fmr)
                calibration_impostor_ids.update(row["pair_index"] for row, include in zip(rows, mask) if include)
            pooled_negatives = np.concatenate(list(negatives.values()))
            pooled_choice = select_fmr_threshold(pooled_negatives, target_fmr)
            thresholds = {"pooled": pooled_choice["threshold"],
                          "condition_cap": max(choice["threshold"] for choice in condition_choices.values())}
            for policy, threshold in thresholds.items():
                decision_thresholds[method][policy][test_indices] = threshold
                violations = []
                for condition, rows in calibration_rows.items():
                    labels = np.asarray([row["label"] for row in rows])
                    eligible = np.asarray([row["status"] == "scored" for row in rows])
                    accept = calibration_scores[condition] >= threshold
                    n_impostors, n_genuine = int(np.sum(eligible & (labels == 0))), int(np.sum(eligible & (labels == 1)))
                    fp = int(np.sum(eligible & (labels == 0) & accept))
                    fn = int(np.sum(eligible & (labels == 1) & ~accept))
                    choice = condition_choices[condition]
                    satisfies = fp <= choice["allowed_false_matches"]
                    violations.append(not satisfies)
                    calibration_records.append({
                        "regime": regime, "method": method, "fold": fold, "policy": policy,
                        "condition": condition, "n_impostors": n_impostors,
                        "allowed_false_matches": choice["allowed_false_matches"],
                        "observed_false_matches": fp, "observed_fmr": fp / n_impostors,
                        "target_fmr": pooled_choice["target_fmr"], "threshold": threshold,
                        "condition_minimum_threshold": choice["threshold"],
                        "condition_constraint_met": int(satisfies), "n_genuine": n_genuine,
                        "observed_false_nonmatches": fn, "observed_fnmr": fn / n_genuine if n_genuine else None})
                observed = int(np.sum(pooled_negatives >= threshold))
                constraint_met = observed <= pooled_choice["allowed_false_matches"] if policy == "pooled" else not any(violations)
                if not constraint_met:
                    raise ValueError("Selected calibration threshold violated its declared policy")
                threshold_records.append({
                    "model_id": model_id, "threshold_id": f"{model_id}/{policy}",
                    "regime": regime, "method": method, "fold": fold, "policy": policy,
                    "threshold": threshold, "target_fmr": pooled_choice["target_fmr"],
                    "target_fmr_exact": pooled_choice["target_fmr_exact"],
                    "n_calibration_impostor_variants": len(pooled_negatives),
                    "n_calibration_unique_impostor_pairs": len(calibration_impostor_ids),
                    "allowed_false_matches": pooled_choice["allowed_false_matches"],
                    "sum_condition_budgets": sum(choice["allowed_false_matches"] for choice in condition_choices.values()),
                    "observed_false_matches": observed, "observed_fmr": observed / len(pooled_negatives),
                    "calibration_constraint_met": bool(constraint_met),
                    "calibration_conditions": list(train_conditions),
                    "calibration_folds": split["calibration_folds"],
                    "calibration_impostor_pair_indices_sha256": indices_sha256(calibration_impostor_ids),
                    "condition_thresholds": {condition: choice["threshold"] for condition, choice in condition_choices.items()},
                })
            for condition in test_conditions:
                rows = [groups[condition][i] for i in test_indices]
                score = score_rows(model, rows)
                scores[method][condition][test_indices] = score
                for policy, threshold in thresholds.items():
                    metrics = _metrics(rows, score, np.full(len(rows), threshold))
                    folds.append({"regime": regime, "method": method, "policy": policy,
                                  "condition": condition, "fold": fold, "threshold": threshold, **metrics})
    for method in methods:
        for policy in POLICIES:
            for condition in test_conditions:
                subset = [row for row in folds if row["method"] == method and row["policy"] == policy and row["condition"] == condition]
                aggregates.append({"policy": policy, **_aggregate(subset, regime, method, condition)})

    def predictions():
        for method in methods:
            for policy in POLICIES:
                for condition in test_conditions:
                    for i, row in enumerate(groups[condition]):
                        scored = row["status"] == "scored"
                        value = float(scores[method][condition][i]) if scored else None
                        threshold = float(decision_thresholds[method][policy][i])
                        prediction = int(value >= threshold) if scored else None
                        yield {"regime": regime, "method": method, "policy": policy, "condition": condition,
                               "pair_index": row["pair_index"], "fold": row["fold"], "label": row["label"],
                               "status": row["status"], "score": value, "threshold": threshold,
                               "prediction": prediction, "correct": int(prediction == row["label"]) if scored else None}
    return {"models": models, "thresholds": threshold_records, "calibration": calibration_records,
            "folds": folds, "aggregates": aggregates, "predictions": predictions()}
