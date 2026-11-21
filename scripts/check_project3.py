#!/usr/bin/env python3
"""Independently check saved Project 3 models, predictions and aggregate metrics.

No model fitting, embedding inference or project evaluation imports are needed.
Source and output hashes identify the checked run. A complete check additionally
reconstructs each saved score from the source comparison and serialized model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
COUNT_KEYS = ("requested_pairs", "scored_pairs", "genuine_pairs", "impostor_pairs",
              "excluded_pairs", "excluded_genuine_pairs", "excluded_impostor_pairs",
              "true_accepts", "true_rejects", "false_matches", "false_nonmatches")
RATE_KEYS = ("accuracy", "fmr", "fnmr", "auc", "eer_interpolated")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def inside(base, relative):
    base = Path(base).resolve()
    need(isinstance(relative, str) and not Path(relative).is_absolute(),
         f"Expected relative artifact path: {relative}")
    target = (base / relative).resolve()
    need(target.is_relative_to(base), f"Artifact path escapes directory: {relative}")
    return target


def verify_hashes(base, hashes, required=()):
    need(isinstance(hashes, dict) and hashes, "Missing artifact hashes")
    need(set(required).issubset(hashes), "Required files missing from artifact hashes")
    for relative, expected in hashes.items():
        need(isinstance(expected, str) and len(expected) == 64,
             f"Invalid SHA-256 for {relative}")
        need(digest(inside(base, relative)) == expected, f"Hash mismatch: {relative}")
    return len(hashes)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
             f"Missing or repeated CSV columns: {path}")
        yield from reader


def near(saved, computed, label, tolerance=2e-11):
    if computed is None:
        need(saved in (None, ""), f"Expected undefined value: {label}")
    elif isinstance(computed, (int, np.integer)):
        need(str(saved) == str(computed), f"Count mismatch: {label}")
    else:
        actual = float(saved)
        need(math.isfinite(actual) and math.isclose(actual, computed,
             abs_tol=tolerance, rel_tol=tolerance),
             f"Numerical mismatch: {label} (saved {saved}, computed {computed})")


def index_hash(indices):
    payload = json.dumps(sorted(set(indices)), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def raw_features(rows, method):
    if method in ("logistic_interaction_c0.1", "logistic_interaction_c10"):
        method = "logistic_interaction"
    similarity = np.array([r["cosine_similarity"] for r in rows], dtype=float)
    if method == "logistic_score":
        return similarity[:, None]
    qmin = np.array([min(r["reference_quality"], r["probe_quality"]) for r in rows])
    if method == "logistic_quality":
        return np.column_stack((similarity, qmin))
    need(method == "logistic_interaction", f"Unknown learned method: {method}")
    return np.column_stack((similarity, qmin, similarity*qmin))


def reconstructed_scores(model, rows):
    method = model["method"]
    if method == "cosine_threshold":
        return np.array([r["cosine_similarity"] for r in rows], dtype=float)
    if method == "qmagface_threshold":
        return np.array([r["qmagface_score"] for r in rows], dtype=float)
    x = raw_features(rows, method)
    mean = np.asarray(model["scaler_mean"])
    scale = np.asarray(model["scaler_scale"])
    weights = np.asarray(model["coefficients"])
    need(mean.shape == scale.shape == weights.shape == (x.shape[1],),
         "Serialized scaler or coefficient dimensions differ")
    need(np.isfinite(mean).all() and np.isfinite(scale).all() and
         np.isfinite(weights).all() and (scale > 0).all(), "Invalid saved model")
    return ((x-mean)/scale) @ weights + float(model["intercept"])


def independent_threshold(scores, labels):
    """Brute threshold-count calculation using sorted genuine/impostor scores."""
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf),
                       unique[:-1]+np.diff(unique)/2,
                       np.nextafter(unique[-1], np.inf)]
    genuine = np.sort(scores[labels == 1])
    impostor = np.sort(scores[labels == 0])
    correct = len(genuine)-np.searchsorted(genuine, candidates, side="left")
    correct += np.searchsorted(impostor, candidates, side="left")
    return float(candidates[np.flatnonzero(correct == correct.max())[-1]])


def independent_metrics(rows):
    """Metrics from decoded predictions, with ROC implemented by scikit-learn."""
    scored = [r for r in rows if r["prediction"] is not None]
    y = np.asarray([r["label"] for r in scored], dtype=int)
    predicted = np.asarray([r["prediction"] for r in scored], dtype=int)
    scores = np.asarray([r["score"] for r in scored], dtype=float)
    genuine, impostor = int(np.sum(y == 1)), int(np.sum(y == 0))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp, fn = impostor-tn, genuine-tp
    excluded_genuine = sum(r["label"] == 1 and r["prediction"] is None for r in rows)
    excluded_impostor = sum(r["label"] == 0 and r["prediction"] is None for r in rows)
    auc, eer = None, None
    if genuine and impostor:
        auc = float(roc_auc_score(y, scores))
        fmr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
        fnmr = 1-tpr
        d = fmr-fnmr
        upper = int(np.flatnonzero(d >= 0)[0])
        if d[upper] == 0:
            eer = float(fmr[upper])
        else:
            lower = upper-1
            w = -d[lower]/(d[upper]-d[lower])
            eer = float(fmr[lower]+w*(fmr[upper]-fmr[lower]))
    return {
        "requested_pairs": len(rows), "scored_pairs": len(scored),
        "genuine_pairs": genuine, "impostor_pairs": impostor,
        "excluded_pairs": len(rows)-len(scored),
        "excluded_genuine_pairs": excluded_genuine,
        "excluded_impostor_pairs": excluded_impostor,
        "true_accepts": tp, "true_rejects": tn, "false_matches": fp,
        "false_nonmatches": fn, "coverage": len(scored)/len(rows),
        "accuracy": (tp+tn)/len(scored) if scored else None,
        "fmr": fp/impostor if impostor else None,
        "fnmr": fn/genuine if genuine else None,
        "auc": auc, "eer_interpolated": eer,
        "reject_on_failure_accuracy": (tp+tn+excluded_impostor)/len(rows),
        "reject_on_failure_fmr": fp/(impostor+excluded_impostor),
        "reject_on_failure_fnmr": (fn+excluded_genuine)/(genuine+excluded_genuine),
    }


def check_run(run_dir, source_dir, root=ROOT):
    run_dir, source_dir, root = Path(run_dir), Path(source_dir), Path(root)
    manifest = read_json(run_dir / "manifest.json")
    need(manifest.get("status") == "completed", "Project 3 run is not completed")
    required = ("models.json", "predictions.csv.gz", "folds.csv", "aggregates.csv", "paired_changes.csv")
    output_hash_count = verify_hashes(run_dir, manifest["output_sha256"], required)
    source_hash_count = verify_hashes(root, manifest["source_sha256"],
                                     ("src/project3_calibration.py", "scripts/project3_run.py"))
    need(digest(source_dir / "manifest.json") == manifest["source"]["manifest_sha256"],
         "Project 2 source manifest differs")
    need(digest(source_dir / "pair_scores.jsonl") == manifest["source"]["pair_scores_sha256"],
         "Project 2 source scores differ")
    prior_manifest = read_json(source_dir / "manifest.json")
    need(prior_manifest.get("status") == "completed", "Project 2 source is not completed")
    need(prior_manifest["output_sha256"]["pair_scores.jsonl"] == manifest["source"]["pair_scores_sha256"],
         "Project 2 and Project 3 provenance disagree")
    with (source_dir / "pair_scores.jsonl").open(encoding="utf-8") as stream:
        source_rows = [json.loads(line) for line in stream if line.strip()]
    source = {(r["condition"], r["pair_index"]): r for r in source_rows}
    need(len(source) == len(source_rows) == 60000, "Duplicate or missing source comparisons")
    original = sorted((r for r in source_rows if r["condition"] == "original"),
                      key=lambda r: r["pair_index"])
    pair_ids = [r["pair_index"] for r in original]
    need(len(pair_ids) == len(set(pair_ids)) == 6000, "Expected 6,000 original pairs")
    labels = np.array([r["label"] for r in original], dtype=int)
    eligible = np.array([r["status"] == "scored" for r in original])
    need(int(eligible.sum()) == 5999, "Expected 5,999 successful Project 2 pairs")
    conditions = manifest["coverage"]["conditions"]
    need(set(conditions) == {r["condition"] for r in source_rows}, "Condition list differs from source")
    n_folds = manifest["settings"]["n_folds"]
    need(n_folds == 10, "Expected all ten predefined folds")
    need(manifest["settings"]["primary_c"] == 1.0, "Primary C differs from the declared protocol")
    methods = ("cosine_threshold", "qmagface_threshold", "logistic_score", "logistic_quality", "logistic_interaction")
    need(tuple(manifest["settings"]["primary_methods"]) == methods, "Primary method set differs")
    regimes = {"original_only": (["original"], conditions), "mixed": (conditions, conditions)}
    for family in ("resolution", "blur", "brightness"):
        withheld = [c for c in conditions if c.startswith(family+"_")]
        regimes[f"leave_{family}_out"] = ([c for c in conditions if c not in withheld],
                                          ["original", *withheld])
    expected_models = set()
    for regime in regimes:
        variants = list(methods)
        if manifest["settings"]["sensitivity"] and regime in ("mixed", "leave_resolution_out"):
            variants += ["logistic_interaction_c0.1", "logistic_interaction_c10"]
        expected_models.update((regime, method, fold) for method in variants for fold in range(n_folds))

    models = read_json(run_dir / "models.json")
    model_index = {(m["regime"], m["method"], m["fold"]): m for m in models}
    need(len(model_index) == len(models) == manifest["fitted_models"], "Duplicate or missing fitted models")
    need(set(model_index) == expected_models, "Fitted model coverage differs from protocol")
    # Reconstruct the exact allowed training membership and scaler statistics.
    for regime, (train_conditions, test_conditions) in regimes.items():
        for fold in range(n_folds):
            training = [source[(condition, row["pair_index"])] for condition in train_conditions
                        for row in original if row["fold"] != fold and row["status"] == "scored"]
            train_ids = {r["pair_index"] for r in training}
            heldout_ids = {r["pair_index"] for r in original if r["fold"] == fold}
            need(train_ids.isdisjoint(heldout_ids), "Training/heldout pair overlap")
            for method in [m for r, m, f in model_index if r == regime and f == fold]:
                model = model_index[(regime, method, fold)]
                location = f"{regime}/{method}/fold{fold}"
                need(model["train_conditions"] == list(train_conditions), f"Unexpected training conditions: {location}")
                need(model["test_conditions"] == list(test_conditions), f"Unexpected test conditions: {location}")
                need(model["training_folds"] == [f for f in range(n_folds) if f != fold], f"Training fold leakage: {location}")
                need(model["training_pair_indices_sha256"] == index_hash(train_ids), f"Training membership differs: {location}")
                need(model["heldout_pair_indices_sha256"] == index_hash(heldout_ids), f"Heldout membership differs: {location}")
                near(model["train_rows"], len(training), location+" training rows")
                near(model["training_rows"], len(training), location+" fitted training rows")
                near(model["train_unique_pairs"], len(train_ids), location+" unique training pairs")
                near(model["training_weight_sum"], float(len(train_ids)), location+" total training weight")
                near(model["n_conditions"], len(train_conditions), location+" training condition count")
                if method.endswith("_threshold"):
                    scores = reconstructed_scores(model, training)
                    y = np.asarray([r["label"] for r in training])
                    near(model["threshold"], independent_threshold(scores, y), location+" threshold")
                else:
                    x = raw_features(training, method)
                    mean, scale = x.mean(axis=0), x.std(axis=0)
                    scale[scale == 0] = 1.
                    need(np.allclose(model["scaler_mean"], mean, rtol=1e-11, atol=1e-11),
                         f"Training-only scaler mean mismatch: {location}")
                    need(np.allclose(model["scaler_scale"], scale, rtol=1e-11, atol=1e-11),
                         f"Training-only scaler scale mismatch: {location}")
                    near(model["threshold"], 0., location+" fixed logit threshold")
                    c = .1 if method.endswith("_c0.1") else 10. if method.endswith("_c10") else 1.
                    near(model["c"], c, location+" regularization")
                    need(0 < model["n_iter"] < manifest["settings"]["max_iter"],
                         f"Unconverged classifier: {location}")

    computed_folds, computed_aggregates, predicted_arrays = {}, {}, {}
    expected_groups = {(m["regime"], m["method"], c) for m in models for c in m["test_conditions"]}
    seen_groups = set()
    prediction_count, scored_prediction_count = 0, 0
    maximum_score_difference = 0.
    rows = read_csv(run_dir / "predictions.csv.gz")
    for key, group in itertools.groupby(rows, key=lambda r: (r["regime"], r["method"], r["condition"])):
        need(key in expected_groups and key not in seen_groups, f"Unexpected/repeated prediction group: {key}")
        seen_groups.add(key)
        saved = list(group)
        need([int(r["pair_index"]) for r in saved] == pair_ids, f"Pair coverage/order mismatch: {key}")
        decoded = []
        for r in saved:
            raw = source[(key[2], int(r["pair_index"]))]
            need(r["status"] == raw["status"] and int(r["fold"]) == raw["fold"] and int(r["label"]) == raw["label"],
                 f"Prediction metadata differs from source: {key}/{r['pair_index']}")
            if raw["status"] == "scored":
                value, prediction = float(r["score"]), int(r["prediction"])
                need(math.isfinite(value) and prediction in (0, 1), "Invalid prediction score/label")
                near(r["correct"], int(prediction == raw["label"]), "saved correctness")
                scored_prediction_count += 1
            else:
                need(all(r[k] == "" for k in ("score", "prediction", "correct")),
                     "Failed input must retain empty score, prediction and correctness")
                value, prediction = None, None
            decoded.append({"label": raw["label"], "fold": raw["fold"], "score": value,
                            "prediction": prediction, "pair_index": raw["pair_index"]})
        predicted_arrays[key] = np.array([r["prediction"] if r["prediction"] is not None else -1 for r in decoded], dtype=np.int8)
        condition_folds = []
        for fold in range(n_folds):
            model = model_index[(key[0], key[1], fold)]
            positions = [i for i, r in enumerate(decoded) if r["fold"] == fold]
            selected = [decoded[i] for i in positions]
            successful = [i for i in positions if decoded[i]["prediction"] is not None]
            raw = [source[(key[2], decoded[i]["pair_index"])] for i in successful]
            expected_scores = reconstructed_scores(model, raw)
            actual_scores = np.array([decoded[i]["score"] for i in successful])
            difference = float(np.max(np.abs(actual_scores-expected_scores)))
            maximum_score_difference = max(maximum_score_difference, difference)
            need(np.allclose(actual_scores, expected_scores, rtol=1e-11, atol=1e-11),
                 f"Saved model cannot reconstruct predictions: {key}/fold{fold}")
            for i in positions:
                near(saved[i]["threshold"], model["threshold"], "prediction threshold")
            need(np.array_equal(np.array([decoded[i]["prediction"] for i in successful]),
                                (expected_scores >= model["threshold"]).astype(int)),
                 f"Decision rule mismatch: {key}/fold{fold}")
            values = independent_metrics(selected)
            values["threshold"] = model["threshold"]
            computed_folds[(*key, fold)] = values
            condition_folds.append(values)
        total = {k: sum(r[k] for r in condition_folds) for k in COUNT_KEYS}
        total["coverage"] = total["scored_pairs"]/total["requested_pairs"]
        total["pooled_accuracy"] = (total["true_accepts"]+total["true_rejects"])/total["scored_pairs"]
        total["pooled_fmr"] = total["false_matches"]/total["impostor_pairs"]
        total["pooled_fnmr"] = total["false_nonmatches"]/total["genuine_pairs"]
        total["reject_on_failure_accuracy"] = (total["true_accepts"]+total["true_rejects"]+total["excluded_impostor_pairs"])/total["requested_pairs"]
        total["reject_on_failure_fmr"] = total["false_matches"]/(total["impostor_pairs"]+total["excluded_impostor_pairs"])
        total["reject_on_failure_fnmr"] = (total["false_nonmatches"]+total["excluded_genuine_pairs"])/(total["genuine_pairs"]+total["excluded_genuine_pairs"])
        for metric in RATE_KEYS:
            values = np.array([r[metric] for r in condition_folds])
            total[f"mean_{metric}"] = float(values.mean())
            total[f"sd_{metric}"] = float(values.std(ddof=1))
        computed_aggregates[key] = total
        prediction_count += len(saved)
    need(seen_groups == expected_groups, "Prediction groups missing")
    need(prediction_count == manifest["prediction_rows"], "Prediction count differs from run manifest")

    for name, expected in (("folds", computed_folds), ("aggregates", computed_aggregates)):
        seen = set()
        for row in read_csv(run_dir / f"{name}.csv"):
            key = (row["regime"], row["method"], row["condition"])
            if name == "folds":
                key += (int(row["fold"]),)
            need(key in expected and key not in seen, f"Unexpected or duplicate {name} row: {key}")
            seen.add(key)
            for metric, value in expected[key].items():
                need(metric in row, f"Missing metric: {name}/{metric}")
                near(row[metric], value, f"{name}/{key}/{metric}")
        need(seen == set(expected), f"Missing {name} rows")

    paired_count, seen_paired = 0, set()
    expected_paired = set()
    for regime, method, condition in expected_groups:
        for reference in ("cosine_threshold", "qmagface_threshold"):
            if method != reference:
                expected_paired.add((regime, condition, reference, method))
    for row in read_csv(run_dir / "paired_changes.csv"):
        key = (row["regime"], row["condition"], row["reference_method"], row["method"])
        need(key in expected_paired and key not in seen_paired, f"Unexpected or duplicate paired comparison: {key}")
        seen_paired.add(key)
        a = predicted_arrays[(key[0], key[2], key[1])] == labels
        b = predicted_arrays[(key[0], key[3], key[1])] == labels
        corrected = int(np.sum(eligible & ~a & b))
        introduced = int(np.sum(eligible & a & ~b))
        expected = {"scored_pairs": int(eligible.sum()), "corrected_errors": corrected,
                    "introduced_errors": introduced, "net_correct_gain": corrected-introduced,
                    "both_correct": int(np.sum(eligible & a & b)),
                    "both_wrong": int(np.sum(eligible & ~a & ~b))}
        for metric, value in expected.items():
            near(row[metric], value, f"paired/{key}/{metric}")
        paired_count += 1
    need(seen_paired == expected_paired, "Missing paired comparisons")

    return {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Input/source/output hashes, train/test membership, training scalers and thresholds, serialized model scores, all predictions and metrics",
        "run_manifest_sha256": digest(run_dir / "manifest.json"),
        "checker_sha256": digest(Path(__file__)),
        "source_sha256": manifest["source_sha256"],
        "execution_git_commit": manifest["execution_git_commit"],
        "verified_output_files": output_hash_count, "verified_source_files": source_hash_count,
        "verified_models": len(models), "verified_predictions": prediction_count,
        "verified_scored_predictions": scored_prediction_count,
        "verified_fold_rows": len(computed_folds), "verified_aggregate_rows": len(computed_aggregates),
        "verified_paired_rows": paired_count,
        "maximum_reconstructed_score_difference": maximum_score_difference,
        "environment": {"python": platform.python_version(),
                        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn")}},
        "limitations": "This checks saved-score analysis and does not rerun image alignment or embedding inference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/project3/calibration")
    parser.add_argument("--source", type=Path, default=ROOT / "results/project2/robustness")
    parser.add_argument("--output", type=Path, help="Optional JSON receipt outside the measured run directory")
    args = parser.parse_args()
    if args.output is not None:
        need(not args.output.resolve().is_relative_to(args.run.resolve()),
             "Keep the checker receipt outside the measured run directory")
    try:
        result = check_run(args.run, args.source)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 1
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
