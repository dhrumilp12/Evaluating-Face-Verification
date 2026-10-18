"""Paired Project 2 evaluation, calibrated on original-quality training folds.

Every input row contains both scores from the same pair of MagFace embeddings.
Failures remain in all exports and coverage denominators. ROC/EER sweeps are
descriptive only; they never select the thresholds used for predictions.
"""
from collections import Counter
from numbers import Integral, Real
from pathlib import Path
from statistics import NormalDist

import numpy as np

try:
    from .evaluation import select_threshold
    from .score_analysis import empirical_curve
except ImportError:  # Direct script imports with src on sys.path.
    from evaluation import select_threshold
    from score_analysis import empirical_curve


METHODS = {"magface_cosine": "cosine_similarity", "qmagface": "qmagface_score"}
PAIR_KEYS = ("fold", "label", "reference_path", "probe_path")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def _validate(rows, n_folds):
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    groups = {}
    for source in rows:
        row = dict(source)
        required = ("pair_index", *PAIR_KEYS, "condition", "status", *METHODS.values(),
                    "reference_quality", "probe_quality")
        if any(key not in row for key in required):
            raise ValueError("Pair records are missing required fields")
        for key in ("pair_index", "fold", "label"):
            if isinstance(row[key], bool) or not isinstance(row[key], Integral):
                raise ValueError(f"{key} must be an integer")
            row[key] = int(row[key])
        if row["pair_index"] < 0 or row["fold"] not in range(n_folds) or row["label"] not in (0, 1):
            raise ValueError("Invalid pair index, fold, or label")
        for key in ("condition", "status", "reference_path", "probe_path"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"{key} must be a nonempty string")
        row.setdefault("error", "")
        if not isinstance(row["error"], str):
            raise ValueError("error must be a string")
        for key in ("reference_quality", "probe_quality"):
            if row[key] is not None and (not _finite(row[key]) or row[key] <= 0):
                raise ValueError("Raw qualities must be finite and positive, or absent on failures")
            if row[key] is not None:
                row[key] = float(row[key])
        if row["status"] == "scored":
            if any(not _finite(row[key]) for key in METHODS.values()):
                raise ValueError("Both methods require finite scores on every scored pair")
            if not -1 <= row["cosine_similarity"] <= 1:
                raise ValueError("Cosine similarity must be in [-1, 1]")
            if row["reference_quality"] is None or row["probe_quality"] is None:
                raise ValueError("Scored pairs require both raw qualities")
            for key in METHODS.values():
                row[key] = float(row[key])
        elif any(row[key] is not None for key in METHODS.values()):
            raise ValueError("Failure rows must have no score for either method")
        indexed = groups.setdefault(row["condition"], {})
        if row["pair_index"] in indexed:
            raise ValueError("Duplicate pair index within a condition")
        indexed[row["pair_index"]] = row
    if "original" not in groups:
        raise ValueError("An original-quality condition is required for calibration")
    original = groups["original"]
    if {r["fold"] for r in original.values()} != set(range(n_folds)):
        raise ValueError("Expected all predefined folds in the original condition")
    for fold in range(n_folds):
        if {r["label"] for r in original.values() if r["fold"] == fold} != {0, 1}:
            raise ValueError("Each requested fold must contain genuine and impostor pairs")
    for condition, indexed in groups.items():
        if set(indexed) != set(original):
            raise ValueError(f"Condition {condition} has missing or extra pair indices")
        for index, row in indexed.items():
            if any(row[key] != original[index][key] for key in PAIR_KEYS):
                raise ValueError(f"Condition {condition} changed pair metadata at index {index}")
    return {name: [indexed[i] for i in sorted(indexed)] for name, indexed in groups.items()}


def _rates(counts):
    tp, tn = counts["true_accepts"], counts["true_rejects"]
    fp, fn = counts["false_matches"], counts["false_nonmatches"]
    return {
        "coverage": _ratio(counts["scored_pairs"], counts["requested_pairs"]),
        "genuine_coverage": _ratio(counts["genuine_pairs"], counts["requested_genuine_pairs"]),
        "impostor_coverage": _ratio(counts["impostor_pairs"], counts["requested_impostor_pairs"]),
        "accuracy": _ratio(tp + tn, counts["scored_pairs"]),
        "fmr": _ratio(fp, counts["impostor_pairs"]),
        "fnmr": _ratio(fn, counts["genuine_pairs"]),
        "reject_on_failure_accuracy": _ratio(tp + tn + counts["excluded_impostor_pairs"], counts["requested_pairs"]),
        "reject_on_failure_fmr": _ratio(fp, counts["requested_impostor_pairs"]),
        "reject_on_failure_fnmr": _ratio(fn + counts["excluded_genuine_pairs"], counts["requested_genuine_pairs"]),
    }


def _metrics(predictions):
    scored = [r for r in predictions if r["prediction"] is not None]
    genuine = sum(r["label"] == 1 for r in scored)
    impostor = len(scored) - genuine
    requested_genuine = sum(r["label"] == 1 for r in predictions)
    counts = {
        "requested_pairs": len(predictions), "requested_genuine_pairs": requested_genuine,
        "requested_impostor_pairs": len(predictions) - requested_genuine,
        "scored_pairs": len(scored), "genuine_pairs": genuine, "impostor_pairs": impostor,
        "excluded_pairs": len(predictions) - len(scored),
        "excluded_genuine_pairs": requested_genuine - genuine,
        "excluded_impostor_pairs": len(predictions) - requested_genuine - impostor,
        "true_accepts": sum(r["label"] == 1 and r["prediction"] == 1 for r in scored),
        "true_rejects": sum(r["label"] == 0 and r["prediction"] == 0 for r in scored),
        "false_matches": sum(r["label"] == 0 and r["prediction"] == 1 for r in scored),
        "false_nonmatches": sum(r["label"] == 1 and r["prediction"] == 0 for r in scored),
    }
    failure_reasons = Counter(r["error"] or r["status"] for r in predictions if r["prediction"] is None)
    return {**counts, **_rates(counts), "failure_reasons": dict(sorted(failure_reasons.items()))}


def _curve(rows, score_key):
    selected = [r for r in rows if r["status"] == "scored"]
    if {r["label"] for r in selected} != {0, 1}:
        return None
    return empirical_curve([r[score_key] for r in selected], [r["label"] for r in selected])


def _describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "sd": None, "p05": None, "median": None, "p95": None}
    return {"count": len(values), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "p05": float(np.quantile(values, .05)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95))}


def _paired_changes(rows, thresholds, condition, fold):
    selected = [r for r in rows if r["status"] == "scored"]
    counters = Counter()
    for row in selected:
        a, b = (int(row[key] >= thresholds[method][str(row["fold"])]) for method, key in METHODS.items())
        correct_a, correct_b = a == row["label"], b == row["label"]
        name = ("both_correct" if correct_a and correct_b else "cosine_only_correct" if correct_a
                else "qmagface_only_correct" if correct_b else "both_wrong")
        counters[name] += 1
        if a != b:
            counters["changed_to_accept" if b else "changed_to_reject"] += 1
    keys = ("both_correct", "cosine_only_correct", "qmagface_only_correct", "both_wrong",
            "changed_to_accept", "changed_to_reject")
    changed = counters["cosine_only_correct"] + counters["qmagface_only_correct"]
    return {"condition": condition, "fold": fold, "requested_pairs": len(rows),
            "scored_pairs": len(selected), "excluded_pairs": len(rows) - len(selected),
            "coverage": _ratio(len(selected), len(rows)), **{key: counters[key] for key in keys},
            "changed_decisions": changed, "changed_fraction_of_scored": _ratio(changed, len(selected)),
            "net_correct_gain": counters["qmagface_only_correct"] - counters["cosine_only_correct"]}


def _original_changes(original, rows, method, thresholds, condition, fold):
    key = METHODS[method]
    common = [(original[r["pair_index"]], r) for r in rows
              if r["status"] == "scored" and original[r["pair_index"]]["status"] == "scored"]
    original_predictions, degraded_predictions = [], []
    for source, row in common:
        threshold = thresholds[method][str(row["fold"])]
        original_predictions.append({**source, "prediction": int(source[key] >= threshold)})
        degraded_predictions.append({**row, "prediction": int(row[key] >= threshold)})
    before, after = _metrics(original_predictions), _metrics(degraded_predictions)
    result = {"condition": condition, "method": method, "fold": fold,
              "requested_pairs": len(rows), "common_scored_pairs": len(common),
              "excluded_from_paired_comparison": len(rows) - len(common),
              "common_genuine_pairs": before["genuine_pairs"], "common_impostor_pairs": before["impostor_pairs"],
              "common_coverage": _ratio(len(common), len(rows))}
    for metric in ("accuracy", "fmr", "fnmr"):
        result[f"original_{metric}"] = before[metric]
        result[f"condition_{metric}"] = after[metric]
        result[f"delta_{metric}"] = after[metric] - before[metric] if before[metric] is not None else None
    result["mean_score_change"] = _ratio(sum(b[key] - a[key] for a, b in common), len(common))
    result["correct_to_wrong"] = sum(a["prediction"] == a["label"] and b["prediction"] != b["label"]
                                     for a, b in zip(original_predictions, degraded_predictions))
    result["wrong_to_correct"] = sum(a["prediction"] != a["label"] and b["prediction"] == b["label"]
                                     for a, b in zip(original_predictions, degraded_predictions))
    return result


def evaluate_conditions(rows, n_folds=10):
    """Return JSON-safe tables with one shared eligibility set for both methods.

    Required fields: pair_index, fold, label, reference_path, probe_path,
    condition, status, cosine_similarity, qmagface_score, reference_quality,
    probe_quality. An optional error string identifies failures. Failed records
    must have null scores; qualities may remain if one embedding was available.

    Only successful original-quality pairs in the other folds calibrate each
    method. A calibration set without both classes raises ValueError. Missing
    test classes produce null rates, rather than fabricated zeros. Fold means
    are null unless every fold has that metric; valid-fold counts are explicit.
    The returned tables are ready to export as JSON/CSV without NaN/Infinity.
    """
    groups = _validate(rows, n_folds)
    thresholds, calibration_counts = {}, {}
    for method, key in METHODS.items():
        thresholds[method], calibration_counts[method] = {}, {}
        for fold in range(n_folds):
            training = [r for r in groups["original"] if r["fold"] != fold and r["status"] == "scored"]
            try:
                threshold = select_threshold([r[key] for r in training], [r["label"] for r in training])
                if not np.isfinite(threshold):
                    raise ValueError("Calibration score range produced a nonfinite threshold")
            except ValueError as exc:
                raise ValueError(f"Original calibration for {method}, fold {fold}: {exc}") from exc
            thresholds[method][str(fold)] = threshold
            calibration_counts[method][str(fold)] = len(training)
    result = {"thresholds": thresholds, "folds": [], "predictions": [], "aggregates": [],
              "paired_changes": [], "curves": [], "distributions": [], "condition_changes": []}
    original_index = {r["pair_index"]: r for r in groups["original"]}
    for condition, condition_rows in groups.items():
        for fold in [*range(n_folds), None]:
            selected = [r for r in condition_rows if fold is None or r["fold"] == fold]
            result["paired_changes"].append(_paired_changes(selected, thresholds, condition, fold))
            for method in METHODS:
                result["condition_changes"].append(
                    _original_changes(original_index, selected, method, thresholds, condition, fold))
        for method, key in METHODS.items():
            predictions, fold_records = [], []
            for row in condition_rows:
                threshold = thresholds[method][str(row["fold"])]
                prediction = int(row[key] >= threshold) if row["status"] == "scored" else None
                predictions.append({**row, "method": method, "score_key": key, "score": row[key],
                                    "threshold": threshold, "prediction": prediction,
                                    "correct": int(prediction == row["label"]) if prediction is not None else None})
            for fold in range(n_folds):
                selected = [r for r in predictions if r["fold"] == fold]
                curve = _curve(selected, key)
                record = {"condition": condition, "method": method, "fold": fold,
                          "threshold": thresholds[method][str(fold)],
                          "calibration_pairs": calibration_counts[method][str(fold)],
                          **_metrics(selected), "auc": curve["auc"] if curve else None,
                          "eer_interpolated": curve["eer_interpolated"] if curve else None}
                fold_records.append(record)
            aggregate = {"condition": condition, "method": method, **_metrics(predictions)}
            for metric in ("accuracy", "fmr", "fnmr", "auc", "eer_interpolated"):
                values = [r[metric] for r in fold_records if r[metric] is not None]
                all_valid = len(values) == n_folds
                aggregate[f"valid_folds_{metric}"] = len(values)
                aggregate[f"mean_{metric}"] = float(np.mean(values)) if all_valid else None
                aggregate[f"sd_{metric}"] = float(np.std(values, ddof=1)) if all_valid else None
                aggregate[f"se_{metric}"] = float(aggregate[f"sd_{metric}"] / np.sqrt(n_folds)) if all_valid else None
            for metric in ("accuracy", "fmr", "fnmr"):
                aggregate[f"pooled_{metric}"] = aggregate[metric]
            curve = _curve(condition_rows, key)
            aggregate["pooled_auc"] = curve["auc"] if curve else None
            aggregate["pooled_eer_interpolated"] = curve["eer_interpolated"] if curve else None
            if curve:
                for i, threshold in enumerate(curve["threshold"]):
                    result["curves"].append({"condition": condition, "method": method, "point": i,
                        "threshold": float(threshold) if np.isfinite(threshold) else None,
                        "decision_rule": "score >= threshold" if np.isfinite(threshold) else "reject_all",
                        **{metric: float(curve[metric][i]) for metric in ("fmr", "tpr", "fnmr")}})
            result["folds"].extend(fold_records)
            result["predictions"].extend(predictions)
            result["aggregates"].append(aggregate)
        for field in (*METHODS.values(), "reference_quality", "probe_quality"):
            for label, name in ((1, "genuine"), (0, "impostor")):
                values = [r[field] for r in condition_rows if r["status"] == "scored" and r["label"] == label]
                result["distributions"].append({"condition": condition, "field": field,
                                                "pair_type": name, **_describe(values)})
    return result


def make_plots(result, output_dir):
    """Write standalone PNG diagnostics; requires matplotlib only at call time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = list(dict.fromkeys(r["condition"] for r in result["aggregates"]))
    names = {"magface_cosine": "MagFace cosine", "qmagface": "QMagFace"}
    paths = []

    def save(fig, name):
        path = output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(conditions) * 1.3), 5.5))
    for ax, metric in zip(axes, ("accuracy", "fmr", "fnmr")):
        for method in METHODS:
            lookup = {r["condition"]: r for r in result["aggregates"] if r["method"] == method}
            values = [lookup[c][f"mean_{metric}"] for c in conditions]
            ax.plot(range(len(conditions)), [np.nan if v is None else 100 * v for v in values],
                    marker="o", label=names[method])
        ax.set(xticks=range(len(conditions)), xticklabels=conditions,
               ylabel=f"Mean fold {metric.upper()} (%)", title="Original-fold calibration")
        ax.tick_params(axis="x", rotation=65)
        ax.grid(alpha=.2)
    axes[0].legend()
    save(fig, "project2_fixed_threshold_metrics.png")
    columns = min(3, len(conditions))
    rows = (len(conditions) + columns - 1) // columns
    normal = NormalDist()
    for kind in ("roc", "det"):
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
        for ax, condition in zip(axes.flat, conditions):
            for method in METHODS:
                curve = [r for r in result["curves"] if r["condition"] == condition and r["method"] == method]
                if kind == "roc":
                    ax.plot([r["fmr"] for r in curve], [r["tpr"] for r in curve], label=names[method])
                    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="FMR", ylabel="True match rate")
                else:
                    interior = [r for r in curve if 0 < r["fmr"] < 1 and 0 < r["fnmr"] < 1]
                    ax.plot([normal.inv_cdf(r["fmr"]) for r in interior],
                            [normal.inv_cdf(r["fnmr"]) for r in interior], marker=".", label=names[method])
                    ticks = [.001, .01, .05, .2, .5, .8, .95, .99]
                    positions = [normal.inv_cdf(v) for v in ticks]
                    labels = [f"{100*v:g}" for v in ticks]
                    ax.set(xticks=positions, xticklabels=labels, yticks=positions, yticklabels=labels,
                           xlabel="FMR (%) · probit scale", ylabel="FNMR (%) · probit scale")
                ax.set_title(condition)
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
        for ax in list(axes.flat)[len(conditions):]:
            ax.set_visible(False)
        save(fig, f"project2_{kind}.png")
    for kind, fields in (("scores", tuple(METHODS.values())),
                         ("qualities", ("reference_quality", "probe_quality"))):
        fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows), squeeze=False)
        for ax, condition in zip(axes.flat, conditions):
            selected = [r for r in result["predictions"] if r["condition"] == condition
                        and r["method"] == "magface_cosine" and r["status"] == "scored"]
            for field in fields:
                for label, label_name in ((1, "genuine"), (0, "impostor")):
                    values = [r[field] for r in selected if r["label"] == label]
                    if values:
                        ax.hist(values, bins=min(40, max(5, int(np.sqrt(len(values))))),
                                density=True, histtype="step", linewidth=1.5, label=f"{field} · {label_name}")
            ax.set(title=condition, xlabel="Raw embedding norm" if kind == "qualities" else "Score", ylabel="Density")
            if selected:
                ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(conditions):]:
            ax.set_visible(False)
        save(fig, f"project2_{kind}_distributions.png")
    return paths
