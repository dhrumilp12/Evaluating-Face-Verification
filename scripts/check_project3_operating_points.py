#!/usr/bin/env python3
"""Independently verify Project 3 target-FMR thresholds and held-out decisions.

This checker does not import either project evaluation engine or fit a model.
It reuses only the independently implemented numerical/file helpers from the
earlier saved-result checker. Calibration constraints concern calibration data;
the checker does not require the held-out FMR to equal or remain below 1%.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_project3 import (  # Independent helpers, not evaluation code.
    COUNT_KEYS, RATE_KEYS, digest, index_hash, independent_metrics, near, need,
    raw_features, read_csv, read_json, reconstructed_scores, verify_hashes,
)

METHODS = ("cosine_threshold", "qmagface_threshold", "logistic_interaction")
POLICIES = ("pooled", "condition_cap")


def threshold_from_counts(scores, target=.01):
    """Enumerate unique-score rejection boundaries and choose the first feasible."""
    ordered = np.sort(np.asarray(scores, dtype=float))
    need(ordered.ndim == 1 and len(ordered) and np.isfinite(ordered).all(),
         "Expected finite calibration impostor scores")
    budget = int(Fraction(str(target))*len(ordered))
    candidates = np.nextafter(np.unique(ordered), np.inf)
    counts = len(ordered)-np.searchsorted(ordered, candidates, side="left")
    eligible = np.flatnonzero(counts <= budget)
    need(len(eligible) > 0, "No feasible calibration threshold")
    position = int(eligible[0])
    need(math.isfinite(float(candidates[position])), "Nonfinite calibration threshold")
    return float(candidates[position]), budget, int(counts[position])


def compare_fields(saved, expected, label):
    for field, value in expected.items():
        need(field in saved, f"Missing field: {label}/{field}")
        if isinstance(value, bool):
            actual = saved[field]
            if isinstance(actual, str):
                need(actual in ("True", "False", "0", "1"), f"Invalid boolean: {label}/{field}")
                actual = actual in ("True", "1")
            elif isinstance(actual, int) and actual in (0, 1):
                actual = bool(actual)
            need(isinstance(actual, bool) and actual == value, f"Boolean mismatch: {label}/{field}")
        else:
            near(saved[field], value, f"{label}/{field}")


def check_run(run_dir, source_dir, root=ROOT):
    run_dir, source_dir, root = Path(run_dir), Path(source_dir), Path(root)
    manifest = read_json(run_dir / "manifest.json")
    need(manifest.get("status") == "completed", "Operating-point run is incomplete")
    required = ("models.json", "thresholds.json", "calibration.csv", "folds.csv",
                "aggregates.csv", "predictions.csv.gz")
    outputs_checked = verify_hashes(run_dir, manifest["output_sha256"], required)
    sources_checked = verify_hashes(root, manifest["source_sha256"],
                                   ("src/project3_operating_points.py", "scripts/project3_operating_run.py"))
    need(digest(source_dir / "pair_scores.jsonl") == manifest["source"]["pair_scores_sha256"],
         "Source scores have changed")
    need(digest(source_dir / "manifest.json") == manifest["source"]["manifest_sha256"],
         "Source manifest has changed")
    prior = read_json(source_dir / "manifest.json")
    need(prior.get("status") == "completed" and
         prior["output_sha256"]["pair_scores.jsonl"] == manifest["source"]["pair_scores_sha256"],
         "Source provenance is inconsistent")
    with (source_dir / "pair_scores.jsonl").open(encoding="utf-8") as stream:
        source_rows = [json.loads(line) for line in stream if line.strip()]
    source = {(r["condition"], r["pair_index"]): r for r in source_rows}
    need(len(source) == len(source_rows) == 60000, "Expected 60,000 unique source rows")
    original = sorted((r for r in source_rows if r["condition"] == "original"),
                      key=lambda r: r["pair_index"])
    pair_ids = [r["pair_index"] for r in original]
    need(len(pair_ids) == 6000, "Expected 6,000 original pairs")
    need(sum(r["status"] == "scored" for r in original) == 5999,
         "Expected the common 5,999-pair eligible set")
    conditions = prior["settings"]["conditions"]
    regimes = {
        "original_only": (["original"], conditions),
        "mixed": (conditions, conditions),
        "leave_resolution_out": ([c for c in conditions if not c.startswith("resolution_")],
                                 ["original", *[c for c in conditions if c.startswith("resolution_")]]),
    }
    target = manifest["settings"]["target_fmr"]
    need(target == .01, "This study specifies the 1% calibration target")
    need(tuple(manifest["settings"]["methods"]) == METHODS, "Method set differs from protocol")
    need(tuple(manifest["settings"]["policies"]) == POLICIES, "Threshold policy set differs from protocol")

    models = read_json(run_dir / "models.json")
    model_index = {(m["regime"], m["method"], m["fold"]): m for m in models}
    expected_models = {(r, m, f) for r in regimes for m in METHODS for f in range(10)}
    need(len(models) == len(model_index) == 90 and set(model_index) == expected_models,
         "Fitted-score object coverage differs")
    thresholds = read_json(run_dir / "thresholds.json")
    threshold_index = {(t["regime"], t["method"], t["policy"], t["fold"]): t for t in thresholds}
    expected_thresholds = {(r, m, p, f) for r, m, f in expected_models for p in POLICIES}
    need(len(thresholds) == len(threshold_index) == 180 and set(threshold_index) == expected_thresholds,
         "Operating threshold coverage differs")
    computed_calibration = {}
    learned_fits = 0
    for (regime, method, fold), model in model_index.items():
        train_conditions, test_conditions = regimes[regime]
        calibration_folds = [(fold+1)%10, (fold+2)%10]
        fit_folds = [f for f in range(10) if f not in [fold, *calibration_folds]]
        fit_pairs = [r for r in original if r["fold"] in fit_folds and r["status"] == "scored"]
        calibration_pairs = [r for r in original if r["fold"] in calibration_folds]
        test_pairs = [r for r in original if r["fold"] == fold]
        fit_ids = {r["pair_index"] for r in fit_pairs}
        calibration_ids = {r["pair_index"] for r in calibration_pairs}
        test_ids = {r["pair_index"] for r in test_pairs}
        need(not (fit_ids & calibration_ids or fit_ids & test_ids or calibration_ids & test_ids),
             "Pair roles overlap")
        location = f"{regime}/{method}/fold{fold}"
        need(model["model_id"] == f"{regime}/{method}/fold_{fold}", "Model identifier differs from its coordinates")
        need(model["train_conditions"] == train_conditions and model["test_conditions"] == test_conditions,
             f"Model uses unexpected condition roles: {location}")
        need(model["fit_folds"] == fit_folds and model["calibration_folds"] == calibration_folds and
             model["test_folds"] == [fold], f"Fold role mismatch: {location}")
        for field, ids in (("fit_pair_indices_sha256", fit_ids),
                           ("calibration_requested_pair_indices_sha256", calibration_ids),
                           ("heldout_requested_pair_indices_sha256", test_ids)):
            need(model[field] == index_hash(ids), f"Pair membership mismatch: {location}/{field}")
        fit_rows = [source[(c, r["pair_index"])] for c in train_conditions for r in fit_pairs]
        compare_fields(model, {"fit_rows_available": len(fit_rows),
                               "fit_unique_pairs_available": len(fit_ids)}, location)
        need("threshold" not in model, "Model must not carry an ambiguous decision threshold")
        if method == "logistic_interaction":
            learned_fits += 1
            x = raw_features(fit_rows, method)
            mean, scale = x.mean(axis=0), x.std(axis=0)
            scale[scale == 0] = 1.
            need(np.allclose(model["scaler_mean"], mean, atol=1e-11, rtol=1e-11),
                 f"Fitting-only scaler mean mismatch: {location}")
            need(np.allclose(model["scaler_scale"], scale, atol=1e-11, rtol=1e-11),
                 f"Fitting-only scaler scale mismatch: {location}")
            compare_fields(model, {"fit_rows_used": len(fit_rows), "c": 1.0}, location)
            compare_fields(model, {"training_rows": len(fit_rows), "n_conditions": len(train_conditions),
                                   "training_weight_sum": float(len(fit_ids))}, location)
            need(0 < model["n_iter"] < 2000, f"Classifier failed to converge: {location}")
        else:
            compare_fields(model, {"fit_rows_used": 0}, location)

        by_condition = {}
        negative_pair_ids = {r["pair_index"] for r in calibration_pairs
                             if r["status"] == "scored" and r["label"] == 0}
        need(len(negative_pair_ids) == 600, "Expected 600 unique calibration impostor pairs")
        for condition in train_conditions:
            rows = [source[(condition, r["pair_index"])] for r in calibration_pairs
                    if r["status"] == "scored"]
            scores = reconstructed_scores(model, rows)
            labels = np.asarray([r["label"] for r in rows])
            negatives = scores[labels == 0]
            threshold, budget, _ = threshold_from_counts(negatives, target)
            by_condition[condition] = {"scores": scores, "labels": labels, "negatives": negatives,
                                       "threshold": threshold, "budget": budget}
        pooled_negatives = np.concatenate([v["negatives"] for v in by_condition.values()])
        pooled_threshold, pooled_budget, _ = threshold_from_counts(pooled_negatives, target)
        cap_threshold = max(v["threshold"] for v in by_condition.values())
        need(cap_threshold >= pooled_threshold, "Condition cap is below pooled threshold")
        for policy, operating_threshold in (("pooled", pooled_threshold), ("condition_cap", cap_threshold)):
            record = threshold_index[(regime, method, policy, fold)]
            observed = int(np.sum(pooled_negatives >= operating_threshold))
            expected = {
                "threshold": operating_threshold, "target_fmr": target,
                "n_calibration_impostor_variants": len(pooled_negatives),
                "n_calibration_unique_impostor_pairs": len(negative_pair_ids),
                "allowed_false_matches": pooled_budget,
                "sum_condition_budgets": sum(v["budget"] for v in by_condition.values()),
                "observed_false_matches": observed, "observed_fmr": observed/len(pooled_negatives),
                "calibration_constraint_met": True,
            }
            compare_fields(record, expected, location+"/"+policy)
            need(record["model_id"] == model["model_id"], "Threshold linked to wrong fitted score")
            need(record["threshold_id"] == model["model_id"]+"/"+policy, "Threshold identifier differs from its coordinates")
            need(record["target_fmr_exact"] == "1/100", "Exact calibration target differs")
            need(record["calibration_conditions"] == train_conditions, "Threshold uses unexpected conditions")
            need(record["calibration_folds"] == calibration_folds, "Threshold uses unexpected folds")
            need(record["calibration_impostor_pair_indices_sha256"] == index_hash(negative_pair_ids),
                 "Calibration impostor membership differs")
            need(set(record["condition_thresholds"]) == set(by_condition), "Per-condition thresholds missing")
            for condition, detail in by_condition.items():
                near(record["condition_thresholds"][condition], detail["threshold"], "condition threshold")
                fp = int(np.sum(detail["negatives"] >= operating_threshold))
                genuine_scores = detail["scores"][detail["labels"] == 1]
                fn = int(np.sum(genuine_scores < operating_threshold))
                values = {
                    "n_impostors": len(detail["negatives"]), "allowed_false_matches": detail["budget"],
                    "observed_false_matches": fp, "observed_fmr": fp/len(detail["negatives"]),
                    "target_fmr": target, "threshold": operating_threshold,
                    "condition_minimum_threshold": detail["threshold"],
                    "condition_constraint_met": fp <= detail["budget"],
                    "n_genuine": len(genuine_scores), "observed_false_nonmatches": fn,
                    "observed_fnmr": fn/len(genuine_scores),
                }
                if policy == "condition_cap":
                    need(fp <= detail["budget"], "Per-condition cap exceeded on calibration data")
                computed_calibration[(regime, method, policy, fold, condition)] = values
        if regime == "original_only":
            need(pooled_threshold == cap_threshold, "Original-only threshold policies differ")
    need(learned_fits == 30, "Expected 30 learned fits and 60 fixed-score objects")

    seen = set()
    for row in read_csv(run_dir / "calibration.csv"):
        key = (row["regime"], row["method"], row["policy"], int(row["fold"]), row["condition"])
        need(key in computed_calibration and key not in seen, f"Unexpected calibration row: {key}")
        seen.add(key)
        compare_fields(row, computed_calibration[key], "calibration/"+str(key))
    need(seen == set(computed_calibration), "Missing calibration rows")

    computed_folds, computed_aggregates, decisions = {}, {}, {}
    expected_groups = {(regime, method, policy, condition) for regime, (_, tests) in regimes.items()
                       for method in METHODS for policy in POLICIES for condition in tests}
    seen_groups, prediction_count, scored_count = set(), 0, 0
    maximum_difference = 0.
    for key, group in itertools.groupby(read_csv(run_dir / "predictions.csv.gz"),
                                       key=lambda r: (r["regime"], r["method"], r["policy"], r["condition"])):
        need(key in expected_groups and key not in seen_groups, f"Unexpected prediction group: {key}")
        seen_groups.add(key)
        saved = list(group)
        need([int(r["pair_index"]) for r in saved] == pair_ids, f"Pair coverage/order differs: {key}")
        decoded = []
        for row in saved:
            raw = source[(key[3], int(row["pair_index"]))]
            need(row["status"] == raw["status"] and int(row["label"]) == raw["label"] and int(row["fold"]) == raw["fold"],
                 "Prediction metadata differs from source")
            if raw["status"] == "scored":
                value, prediction = float(row["score"]), int(row["prediction"])
                need(math.isfinite(value) and prediction in (0, 1), "Invalid prediction")
                near(row["correct"], int(prediction == raw["label"]), "prediction correctness")
                scored_count += 1
            else:
                need(all(row[k] == "" for k in ("score", "prediction", "correct")),
                     "Excluded pair must remain explicit with empty decision fields")
                value, prediction = None, None
            decoded.append({"label": raw["label"], "fold": raw["fold"], "score": value,
                            "prediction": prediction, "pair_index": raw["pair_index"]})
        decisions[key] = np.asarray([r["prediction"] if r["prediction"] is not None else -1 for r in decoded])
        condition_folds = []
        for fold in range(10):
            model = model_index[(key[0], key[1], fold)]
            threshold = threshold_index[(key[0], key[1], key[2], fold)]["threshold"]
            positions = [i for i, r in enumerate(decoded) if r["fold"] == fold]
            successful = [i for i in positions if decoded[i]["prediction"] is not None]
            raw = [source[(key[3], decoded[i]["pair_index"])] for i in successful]
            scores = reconstructed_scores(model, raw)
            actual = np.asarray([decoded[i]["score"] for i in successful])
            maximum_difference = max(maximum_difference, float(np.max(np.abs(actual-scores))))
            need(np.allclose(scores, actual, rtol=1e-11, atol=1e-11), "Model cannot reconstruct held-out scores")
            need(np.array_equal((scores >= threshold).astype(int),
                                np.array([decoded[i]["prediction"] for i in successful])),
                 "Held-out decision disagrees with calibrated threshold")
            for i in positions:
                near(saved[i]["threshold"], threshold, "applied threshold")
            values = independent_metrics([decoded[i] for i in positions])
            values["threshold"] = threshold
            computed_folds[(*key, fold)] = values
            condition_folds.append(values)
        total = {k: sum(r[k] for r in condition_folds) for k in COUNT_KEYS}
        total.update({
            "coverage": total["scored_pairs"]/total["requested_pairs"],
            "pooled_accuracy": (total["true_accepts"]+total["true_rejects"])/total["scored_pairs"],
            "pooled_fmr": total["false_matches"]/total["impostor_pairs"],
            "pooled_fnmr": total["false_nonmatches"]/total["genuine_pairs"],
            "reject_on_failure_accuracy": (total["true_accepts"]+total["true_rejects"]+total["excluded_impostor_pairs"])/total["requested_pairs"],
            "reject_on_failure_fmr": total["false_matches"]/(total["impostor_pairs"]+total["excluded_impostor_pairs"]),
            "reject_on_failure_fnmr": (total["false_nonmatches"]+total["excluded_genuine_pairs"])/(total["genuine_pairs"]+total["excluded_genuine_pairs"]),
        })
        for metric in RATE_KEYS:
            values = np.array([r[metric] for r in condition_folds])
            total[f"mean_{metric}"] = float(values.mean())
            total[f"sd_{metric}"] = float(values.std(ddof=1))
        computed_aggregates[key] = total
        prediction_count += len(saved)
    need(seen_groups == expected_groups and prediction_count == manifest["prediction_rows"],
         "Prediction coverage differs from run manifest")
    for regime, (_, tests) in regimes.items():
        for method in METHODS:
            for condition in tests:
                pooled = decisions[(regime, method, "pooled", condition)]
                cap = decisions[(regime, method, "condition_cap", condition)]
                need(np.all(cap <= pooled), "Condition-cap decisions accept additional pairs")
                if regime == "original_only":
                    need(np.array_equal(cap, pooled), "Original-only decisions differ by threshold policy")
    for name, expected in (("folds", computed_folds), ("aggregates", computed_aggregates)):
        seen = set()
        for row in read_csv(run_dir / f"{name}.csv"):
            key = (row["regime"], row["method"], row["policy"], row["condition"])
            if name == "folds":
                key += (int(row["fold"]),)
            need(key in expected and key not in seen, f"Unexpected {name} row: {key}")
            seen.add(key)
            compare_fields(row, expected[key], name+"/"+str(key))
        need(seen == set(expected), f"Missing {name} rows")

    return {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Hashes, disjoint fitting/calibration/test roles, fitting-only scalers, calibration-only integer-budget thresholds, all held-out scores, decisions, counts and metrics",
        "run_manifest_sha256": digest(run_dir / "manifest.json"),
        "checker_sha256": digest(Path(__file__)),
        "independent_helper_sha256": digest(root / "scripts/check_project3.py"),
        "source_sha256": manifest["source_sha256"],
        "execution_git_commit": manifest["execution_git_commit"],
        "verified_source_files": sources_checked, "verified_output_files": outputs_checked,
        "verified_score_objects": len(models), "verified_learned_fits": learned_fits,
        "verified_threshold_configurations": len(thresholds),
        "verified_calibration_rows": len(computed_calibration),
        "verified_predictions": prediction_count, "verified_scored_predictions": scored_count,
        "verified_fold_rows": len(computed_folds), "verified_aggregate_rows": len(computed_aggregates),
        "maximum_reconstructed_score_difference": maximum_difference,
        "environment": {"python": platform.python_version(),
                        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scikit-learn")}},
        "limitations": "Calibration constraints are empirical. Held-out FMR may exceed the 1% target. Repeated degradation variants and overlapping images are not independent observations. No image inference was rerun.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/project3/operating_points")
    parser.add_argument("--source", type=Path, default=ROOT / "results/project2/robustness")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        need(not args.output.resolve().is_relative_to(args.run.resolve()),
             "Keep the checker receipt outside the measured run")
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
