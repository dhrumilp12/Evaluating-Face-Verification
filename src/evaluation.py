"""Verification metrics and threshold fitting using only calibration scores."""
import numpy as np


def checked_arrays(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or not scores.size:
        raise ValueError("Expected nonempty, equally sized score and label vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be 0 or 1")
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("Both genuine and impostor comparisons are required")
    return scores, labels.astype(int)


def select_threshold(scores, labels):
    """Maximize calibration accuracy; ties choose the largest threshold.

    Candidate thresholds are midpoints of distinct calibration scores plus
    finite endpoints accepting all/rejecting all. Accept when score >= threshold.
    """
    scores, labels = checked_arrays(scores, labels)
    distinct = np.unique(scores)
    candidates = np.r_[np.nextafter(distinct[0], -np.inf),
                       distinct[:-1] + np.diff(distinct) / 2,
                       np.nextafter(distinct[-1], np.inf)]
    genuine = np.sort(scores[labels == 1])
    impostor = np.sort(scores[labels == 0])
    true_accepts = len(genuine) - np.searchsorted(genuine, candidates, side="left")
    true_rejects = np.searchsorted(impostor, candidates, side="left")
    correct = true_accepts + true_rejects
    return float(candidates[np.flatnonzero(correct == correct.max())[-1]])


def verification_metrics(scores, labels, threshold):
    scores, labels = checked_arrays(scores, labels)
    if not np.isfinite(threshold):
        raise ValueError("Expected a finite threshold")
    accept = scores >= threshold
    tp = int(np.sum(accept & (labels == 1)))
    tn = int(np.sum(~accept & (labels == 0)))
    fp = int(np.sum(accept & (labels == 0)))
    fn = int(np.sum(~accept & (labels == 1)))
    return {"scored_pairs": len(scores), "genuine_pairs": tp + fn, "impostor_pairs": tn + fp,
            "true_accepts": tp, "true_rejects": tn, "false_matches": fp, "false_nonmatches": fn,
            "accuracy": (tp + tn) / len(scores), "fmr": fp / (fp + tn), "fnmr": fn / (fn + tp)}


def cross_validate(rows, n_folds=10):
    """Return fold metrics and one out-of-fold prediction per requested pair.

    Failure rows have status != scored and no similarity. They are excluded from
    conditional recognition metrics, and retained in coverage counts. A separate
    reject-on-failure policy is also reported for all requested attempts.
    """
    if n_folds < 2 or {int(r["fold"]) for r in rows} != set(range(n_folds)):
        raise ValueError("Expected all predefined folds")
    if len({r["pair_index"] for r in rows}) != len(rows):
        raise ValueError("Duplicate pair indices")
    if any(r["label"] not in (0, 1) for r in rows):
        raise ValueError("Invalid pair label")
    folds, predictions = [], []
    for fold in range(n_folds):
        calibration = [r for r in rows if r["fold"] != fold and r["status"] == "scored"]
        requested = [r for r in rows if r["fold"] == fold]
        test = [r for r in requested if r["status"] == "scored"]
        threshold = select_threshold([r["cosine_similarity"] for r in calibration], [r["label"] for r in calibration])
        result = verification_metrics([r["cosine_similarity"] for r in test], [r["label"] for r in test], threshold)
        excluded_genuine = sum(r["label"] == 1 and r["status"] != "scored" for r in requested)
        excluded_impostor = sum(r["label"] == 0 and r["status"] != "scored" for r in requested)
        result.update({"fold": fold, "threshold": threshold, "calibration_pairs": len(calibration),
                       "requested_pairs": len(requested), "excluded_genuine_pairs": excluded_genuine,
                       "excluded_impostor_pairs": excluded_impostor, "coverage": len(test) / len(requested),
                       "genuine_coverage": result["genuine_pairs"] / (result["genuine_pairs"] + excluded_genuine),
                       "impostor_coverage": result["impostor_pairs"] / (result["impostor_pairs"] + excluded_impostor)})
        result["reject_on_failure_accuracy"] = (result["true_accepts"] + result["true_rejects"] + excluded_impostor) / len(requested)
        result["reject_on_failure_genuine_rejection_rate"] = (result["false_nonmatches"] + excluded_genuine) / (result["genuine_pairs"] + excluded_genuine)
        folds.append(result)
        for row in requested:
            scored = row["status"] == "scored"
            prediction = int(row["cosine_similarity"] >= threshold) if scored else None
            predictions.append({**row, "threshold": threshold, "prediction": prediction,
                                "correct": int(prediction == row["label"]) if scored else None})
    predictions.sort(key=lambda r: r["pair_index"])
    aggregate = {}
    for key in ("accuracy", "fmr", "fnmr"):
        values = np.array([f[key] for f in folds])
        aggregate[f"mean_{key}"] = float(values.mean())
        aggregate[f"sd_{key}"] = float(values.std(ddof=1))
        aggregate[f"se_{key}"] = float(values.std(ddof=1) / np.sqrt(n_folds))
    totals = {key: sum(f[key] for f in folds) for key in
              ("true_accepts", "true_rejects", "false_matches", "false_nonmatches", "scored_pairs", "genuine_pairs", "impostor_pairs")}
    aggregate.update({"requested_pairs": len(rows), "scored_pairs": totals["scored_pairs"],
                      "excluded_pairs": len(rows) - totals["scored_pairs"],
                      "coverage": totals["scored_pairs"] / len(rows),
                      "pooled_accuracy": (totals["true_accepts"] + totals["true_rejects"]) / totals["scored_pairs"],
                      "pooled_fmr": totals["false_matches"] / totals["impostor_pairs"],
                      "pooled_fnmr": totals["false_nonmatches"] / totals["genuine_pairs"],
                      "confusion_counts": totals})
    return folds, predictions, aggregate
