"""Train-only comparison calibration on the frozen Project 2 face scores.

Every transformed version of a pair stays in its original LFW fold. The learned
scores are decision functions, not validated probabilities. AUC/EER are computed
within each held-out fold because learned score scales can differ between folds.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .evaluation import select_threshold
from .project2_evaluation import _validate as validate_project2_records
from .score_analysis import empirical_curve

METHODS = ("cosine_threshold", "qmagface_threshold", "logistic_score",
           "logistic_quality", "logistic_interaction")
FEATURES = {
    "logistic_score": ("cosine_similarity",),
    "logistic_quality": ("cosine_similarity", "minimum_quality"),
    "logistic_interaction": ("cosine_similarity", "minimum_quality", "cosine_x_quality"),
}
CONDITIONS = ("original", "resolution_56", "resolution_28", "resolution_14",
              "blur_0.7", "blur_1.4", "blur_2.1",
              "brightness_0.75", "brightness_0.5", "brightness_0.25")
PREDICTION_FIELDS = ("regime", "method", "condition", "pair_index", "fold", "label",
                     "status", "score", "threshold", "prediction", "correct")
COUNT_KEYS = ("requested_pairs", "scored_pairs", "genuine_pairs", "impostor_pairs",
              "excluded_pairs", "excluded_genuine_pairs", "excluded_impostor_pairs",
              "true_accepts", "true_rejects", "false_matches", "false_nonmatches")
RATE_KEYS = ("accuracy", "fmr", "fnmr", "auc", "eer_interpolated")


def indices_sha256(indices):
    """Fingerprint sorted unique pair IDs, independently of record order."""
    value = json.dumps(sorted({int(x) for x in indices}), separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_records(rows, n_folds=10):
    """Validate Project 2 schema and identical eligibility across conditions.

    Input order is irrelevant. Metadata and successful reference qualities must
    be unchanged across transformed probes. Failed pairs stay in the dataset.
    """
    groups = validate_project2_records(rows, n_folds)
    original = groups["original"]
    for condition, records in groups.items():
        for reference, row in zip(original, records):
            if (reference["status"] == "scored") != (row["status"] == "scored"):
                raise ValueError(f"Eligibility changed in {condition} at pair {row['pair_index']}")
            if row["status"] == "scored" and not np.isclose(
                    row["reference_quality"], reference["reference_quality"], rtol=0, atol=1e-10):
                raise ValueError("Reference quality changed across probe conditions")
    return {name: groups[name] for name in sorted(groups, key=lambda x: (x != "original", x))}


def _method_kind(method):
    if method in METHODS:
        return method
    if method in ("logistic_interaction_c0.1", "logistic_interaction_c10"):
        return "logistic_interaction"
    raise ValueError(f"Unknown method: {method}")


def feature_matrix(rows, method):
    """Build features from successful records without consulting labels/folds."""
    kind = _method_kind(method)
    if kind not in FEATURES:
        raise ValueError("feature_matrix is defined only for learned methods")
    if not rows or any(r["status"] != "scored" for r in rows):
        raise ValueError("Feature rows must be nonempty and successfully scored")
    s = np.asarray([r["cosine_similarity"] for r in rows], dtype=float)
    q = np.asarray([min(r["reference_quality"], r["probe_quality"]) for r in rows], dtype=float)
    if not np.isfinite(s).all() or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("Features must be finite with positive raw qualities")
    columns = {"cosine_similarity": s, "minimum_quality": q, "cosine_x_quality": s * q}
    values = np.column_stack([columns[name] for name in FEATURES[kind]])
    if not np.isfinite(values).all():
        raise ValueError("Feature product overflowed")
    return values


def fit_model(training_rows, method, *, c=1.0, n_conditions=1):
    """Fit one decision model using only explicitly supplied training records.

    Each variant has weight 1/n_conditions, so each successful pair contributes
    total weight one. C is fixed in advance. No threshold is tuned on learned
    outputs: acceptance uses decision_function >= 0 throughout.
    """
    kind = _method_kind(method)
    if isinstance(n_conditions, bool) or not isinstance(n_conditions, Integral) or n_conditions < 1:
        raise ValueError("n_conditions must be a positive integer")
    if isinstance(c, bool) or not isinstance(c, Real) or not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")
    rows = [r for r in training_rows if r["status"] == "scored"]
    if not rows:
        raise ValueError("Training requires successful records")
    labels = np.asarray([r["label"] for r in rows])
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Training requires both genuine and impostor classes")
    weights = np.full(len(rows), 1.0 / n_conditions, dtype=float)
    model = {"method": method, "training_rows": len(rows), "n_conditions": int(n_conditions),
             "training_weight_sum": float(weights.sum()), "threshold": 0.0,
             "feature_names": [], "coefficients": [], "intercept": None,
             "scaler_mean": [], "scaler_scale": [], "c": None, "n_iter": 0,
             "score_key": None, "quality_range_train": None,
             "cosine_slope_min_train": None, "cosine_slope_max_train": None}
    if kind in ("cosine_threshold", "qmagface_threshold"):
        score_key = "cosine_similarity" if kind == "cosine_threshold" else "qmagface_score"
        model.update(score_key=score_key, threshold=select_threshold([r[score_key] for r in rows], labels))
        return model
    values = feature_matrix(rows, kind)
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    classifier = LogisticRegression(C=float(c), solver="lbfgs", max_iter=2000, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaled, labels, sample_weight=weights)
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Unexpected learned class ordering")
    coefficients = classifier.coef_[0]
    if not np.isfinite(coefficients).all() or not np.isfinite(classifier.intercept_).all():
        raise ValueError("Nonfinite fitted parameters")
    q = np.asarray([min(r["reference_quality"], r["probe_quality"]) for r in rows])
    slopes = np.full(2, coefficients[0] / scaler.scale_[0])
    if kind == "logistic_interaction":
        slopes += coefficients[2] / scaler.scale_[2] * np.array([q.min(), q.max()])
    model.update(feature_names=list(FEATURES[kind]), coefficients=coefficients.tolist(),
                 intercept=float(classifier.intercept_[0]), scaler_mean=scaler.mean_.tolist(),
                 scaler_scale=scaler.scale_.tolist(), c=float(c), n_iter=int(classifier.n_iter_[0]),
                 quality_range_train=[float(q.min()), float(q.max())],
                 cosine_slope_min_train=float(slopes.min()), cosine_slope_max_train=float(slopes.max()))
    return model


def score_rows(model, rows):
    """Apply a serialized model without reading any evaluation labels."""
    scored_indices = [i for i, row in enumerate(rows) if row["status"] == "scored"]
    output = np.full(len(rows), np.nan, dtype=float)
    if not scored_indices:
        return output
    successful = [rows[i] for i in scored_indices]
    if model["score_key"] is not None:
        values = np.asarray([row[model["score_key"]] for row in successful], dtype=float)
    else:
        features = feature_matrix(successful, model["method"])
        means, scales = np.asarray(model["scaler_mean"]), np.asarray(model["scaler_scale"])
        coefficients = np.asarray(model["coefficients"])
        if features.shape[1] != len(means) or means.shape != scales.shape or scales.shape != coefficients.shape:
            raise ValueError("Serialized model dimensions do not match features")
        if np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError("Serialized scales must be finite and positive")
        values = ((features - means) / scales) @ coefficients + model["intercept"]
    if not np.isfinite(values).all():
        raise ValueError("Model produced nonfinite scores")
    output[scored_indices] = values
    return output


def experiment_regimes(conditions):
    """Create original-only, mixed, and three unseen-family experiments."""
    names = tuple(conditions)
    if "original" not in names:
        raise ValueError("An original condition is required")
    regimes = {"original_only": {"train_conditions": ("original",), "test_conditions": names},
               "mixed": {"train_conditions": names, "test_conditions": names}}
    for family in ("resolution", "blur", "brightness"):
        heldout = tuple(name for name in names if name.startswith(family + "_"))
        if heldout:
            regimes[f"leave_{family}_out"] = {
                "train_conditions": tuple(name for name in names if name not in heldout),
                "test_conditions": ("original", *heldout),
            }
    return regimes


def _metrics(rows, scores, thresholds):
    labels = np.asarray([r["label"] for r in rows])
    eligible = np.asarray([r["status"] == "scored" for r in rows])
    if np.any(~np.isfinite(scores[eligible])) or np.any(np.isfinite(scores[~eligible])):
        raise ValueError("Scores and successful input records disagree")
    y, s, t = labels[eligible], scores[eligible], thresholds[eligible]
    accept = s >= t
    genuine, impostor = int(np.sum(y == 1)), int(np.sum(y == 0))
    if not genuine or not impostor:
        raise ValueError("Every evaluation fold requires both successful classes")
    tp, tn = int(np.sum(accept & (y == 1))), int(np.sum(~accept & (y == 0)))
    fp, fn = int(np.sum(accept & (y == 0))), int(np.sum(~accept & (y == 1)))
    excluded_genuine = int(np.sum(~eligible & (labels == 1)))
    excluded_impostor = int(np.sum(~eligible & (labels == 0)))
    curve = empirical_curve(s, y)
    return {"requested_pairs": len(rows), "scored_pairs": int(eligible.sum()),
            "genuine_pairs": genuine, "impostor_pairs": impostor,
            "excluded_pairs": int((~eligible).sum()), "excluded_genuine_pairs": excluded_genuine,
            "excluded_impostor_pairs": excluded_impostor, "true_accepts": tp, "true_rejects": tn,
            "false_matches": fp, "false_nonmatches": fn, "coverage": float(eligible.mean()),
            "accuracy": (tp + tn) / len(y), "fmr": fp / impostor, "fnmr": fn / genuine,
            "reject_on_failure_accuracy": (tp + tn + excluded_impostor) / len(rows),
            "reject_on_failure_fmr": fp / (impostor + excluded_impostor),
            "reject_on_failure_fnmr": (fn + excluded_genuine) / (genuine + excluded_genuine),
            "auc": curve["auc"], "eer_interpolated": curve["eer_interpolated"]}


def _aggregate(folds, regime, method, condition):
    totals = {key: sum(row[key] for row in folds) for key in COUNT_KEYS}
    out = {"regime": regime, "method": method, "condition": condition, **totals,
           "coverage": totals["scored_pairs"] / totals["requested_pairs"],
           "pooled_accuracy": (totals["true_accepts"] + totals["true_rejects"]) / totals["scored_pairs"],
           "pooled_fmr": totals["false_matches"] / totals["impostor_pairs"],
           "pooled_fnmr": totals["false_nonmatches"] / totals["genuine_pairs"],
           "reject_on_failure_accuracy": (totals["true_accepts"] + totals["true_rejects"] +
                                          totals["excluded_impostor_pairs"]) / totals["requested_pairs"],
           "reject_on_failure_fmr": totals["false_matches"] / (totals["impostor_pairs"] + totals["excluded_impostor_pairs"]),
           "reject_on_failure_fnmr": (totals["false_nonmatches"] + totals["excluded_genuine_pairs"]) /
                                    (totals["genuine_pairs"] + totals["excluded_genuine_pairs"])}
    for key in RATE_KEYS:
        values = np.asarray([row[key] for row in folds])
        out[f"mean_{key}"] = float(values.mean())
        out[f"sd_{key}"] = float(values.std(ddof=1))
    return out


def evaluate_regime(groups, *, regime, train_conditions, test_conditions,
                    n_folds=10, c=1.0, methods=METHODS, progress=None):
    """Fit on other folds, score held-out pairs, and return exportable evidence.

    groups must come from validate_records. Returned predictions is an iterator
    over compact rows; call list(...) only for small fixtures. Every fitted model
    records its pair-membership fingerprints and the exact training conditions.
    """
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    train_conditions, test_conditions, methods = tuple(train_conditions), tuple(test_conditions), tuple(methods)
    if (not train_conditions or not test_conditions or not methods or
            len(set(train_conditions)) != len(train_conditions) or
            len(set(test_conditions)) != len(test_conditions) or len(set(methods)) != len(methods)):
        raise ValueError("Training/test conditions and methods must be nonempty and distinct")
    if not set(train_conditions + test_conditions) <= set(groups):
        raise ValueError("Unknown calibration/evaluation condition")
    for method in methods:
        _method_kind(method)
    original = groups["original"]
    fold_ids = np.asarray([r["fold"] for r in original])
    if set(fold_ids.tolist()) != set(range(n_folds)):
        raise ValueError("Expected all predefined folds")
    n_pairs = len(original)
    scores = {method: {condition: np.full(n_pairs, np.nan) for condition in test_conditions} for method in methods}
    thresholds = {method: np.full(n_pairs, np.nan) for method in methods}
    models, folds, aggregates, paired = [], [], [], []
    for fold in range(n_folds):
        if progress is not None:
            progress(f"{regime}: fold {fold + 1}/{n_folds}")
        train_rows = [row for condition in train_conditions for row in groups[condition]
                      if row["fold"] != fold and row["status"] == "scored"]
        train_ids = {row["pair_index"] for row in train_rows}
        test_ids = {row["pair_index"] for row in original if row["fold"] == fold}
        if train_ids & test_ids:
            raise ValueError("Training and held-out pair IDs overlap")
        if set(Counter(row["pair_index"] for row in train_rows).values()) != {len(train_conditions)}:
            raise ValueError("Every training pair must have one successful row per training condition")
        test_indices = np.flatnonzero(fold_ids == fold)
        for method in methods:
            effective_c = .1 if method.endswith("_c0.1") else 10.0 if method.endswith("_c10") else c
            model = fit_model(train_rows, method, c=effective_c, n_conditions=len(train_conditions))
            model.update(regime=regime, fold=fold, train_conditions=list(train_conditions),
                         test_conditions=list(test_conditions), training_folds=[f for f in range(n_folds) if f != fold],
                         training_pair_indices_sha256=indices_sha256(train_ids),
                         heldout_pair_indices_sha256=indices_sha256(test_ids),
                         train_rows=len(train_rows), train_unique_pairs=len(train_ids),
                         heldout_requested_pairs=len(test_ids))
            models.append(model)
            thresholds[method][test_indices] = model["threshold"]
            for condition in test_conditions:
                rows = [groups[condition][i] for i in test_indices]
                score = score_rows(model, rows)
                scores[method][condition][test_indices] = score
                metric = _metrics(rows, score, np.full(len(rows), model["threshold"]))
                folds.append({"regime": regime, "method": method, "condition": condition,
                              "fold": fold, "threshold": model["threshold"], **metric})
    for method in methods:
        for condition in test_conditions:
            subset = [row for row in folds if row["method"] == method and row["condition"] == condition]
            aggregates.append(_aggregate(subset, regime, method, condition))
    for condition in test_conditions:
        labels = np.asarray([r["label"] for r in groups[condition]])
        eligible = np.asarray([r["status"] == "scored" for r in groups[condition]])
        for reference in ("cosine_threshold", "qmagface_threshold"):
            if reference not in methods:
                continue
            ref_correct = (scores[reference][condition] >= thresholds[reference]) == labels
            for method in methods:
                if method == reference:
                    continue
                correct = (scores[method][condition] >= thresholds[method]) == labels
                corrected = int(np.sum(eligible & ~ref_correct & correct))
                introduced = int(np.sum(eligible & ref_correct & ~correct))
                paired.append({"regime": regime, "condition": condition, "reference_method": reference,
                               "method": method, "scored_pairs": int(eligible.sum()),
                               "corrected_errors": corrected, "introduced_errors": introduced,
                               "net_correct_gain": corrected - introduced,
                               "both_correct": int(np.sum(eligible & ref_correct & correct)),
                               "both_wrong": int(np.sum(eligible & ~ref_correct & ~correct))})

    def predictions():
        for method in methods:
            for condition in test_conditions:
                for i, row in enumerate(groups[condition]):
                    scored = row["status"] == "scored"
                    value = float(scores[method][condition][i]) if scored else None
                    threshold = float(thresholds[method][i])
                    prediction = int(value >= threshold) if scored else None
                    yield {"regime": regime, "method": method, "condition": condition,
                           "pair_index": row["pair_index"], "fold": row["fold"], "label": row["label"],
                           "status": row["status"], "score": value, "threshold": threshold,
                           "prediction": prediction, "correct": int(prediction == row["label"]) if scored else None}
    return {"models": models, "folds": folds, "aggregates": aggregates,
            "paired_changes": paired, "predictions": predictions()}
