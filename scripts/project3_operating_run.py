#!/usr/bin/env python3
"""Run target-1%-FMR verification with separate fitting and calibration folds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.project3_run import BASE_COMMIT, load_saved_scores, sha256, write_csv, write_json
from src.project3_calibration import experiment_regimes
from src.project3_operating_points import METHODS, POLICIES, PREDICTION_FIELDS, evaluate_operating_regime

SOURCE_FILES = ("src/project3_operating_points.py", "scripts/project3_operating_run.py",
                "src/project3_calibration.py", "scripts/project3_run.py",
                "src/project2_evaluation.py", "src/evaluation.py", "src/score_analysis.py")


def _print_progress(message):
    print(message, flush=True)


def run(source, output, *, progress=_print_progress):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be separate from the Project 2 source directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty output directory; existing results are never overwritten")
    groups, source_hashes = load_saved_scores(source)
    started = time.perf_counter()
    source_fingerprints = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True, capture_output=True, check=True).stdout.strip()
    manifest = {
        "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Target-1%-FMR calibration on frozen scores with disjoint fitting, calibration, and test pair folds",
        "project2_repository_commit": BASE_COMMIT, "execution_git_commit": git_commit,
        "source": {"directory": str(source), **source_hashes}, "source_sha256": source_fingerprints,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("numpy", "scipy", "scikit-learn", "threadpoolctl")}},
        "settings": {"n_folds": 10, "fit_fold_count": 7, "calibration_fold_count": 2,
                     "test_fold_count": 1, "calibration_fold_rule": "(test_fold+1)%10, (test_fold+2)%10",
                     "methods": list(METHODS), "policies": list(POLICIES), "primary_policy": "pooled",
                     "regimes": ["original_only", "mixed", "leave_resolution_out"],
                     "target_fmr": .01, "target_fmr_exact": "1/100", "primary_c": 1.0,
                     "acceptance_rule": "score >= threshold",
                     "threshold_rule": "nextafter(descending_calibration_impostors[floor(exact_target*N)], +infinity)",
                     "condition_cap_rule": "maximum of per-condition target-FMR thresholds",
                     "training_variant_weight": "1 / number_of_fitting_conditions",
                     "refit_after_calibration": False, "solver": "lbfgs", "max_iter": 2000,
                     "tol": 1e-8, "blas_threads": 1},
        "coverage": {"requested_pairs_per_condition": len(groups["original"]),
                     "scored_pairs_per_condition": sum(row["status"] == "scored" for row in groups["original"]),
                     "conditions": list(groups)},
        "limitations": ["Calibration constraints are empirical and do not guarantee held-out FMR.",
                        "Pooled calibration counts repeated, correlated degradation variants of the same impostor pairs.",
                        "LFW pair folds are not identity- or image-disjoint.",
                        "Mixed calibration sees the evaluation degradation types in other pairs.",
                        "The condition-cap policy protects observed calibration conditions only.",
                        "Learned decision scores are not validated probabilities."]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    tables = {name: [] for name in ("calibration", "folds", "aggregates")}
    models, thresholds, prediction_count = [], [], 0
    try:
        with (output / "predictions.csv.gz").open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
                    writer.writeheader()
                    regimes = experiment_regimes(groups)
                    for regime in manifest["settings"]["regimes"]:
                        progress(f"Evaluating target 1% FMR: {regime}")
                        with threadpool_limits(limits=1):
                            result = evaluate_operating_regime(groups, regime=regime, n_folds=10,
                                                              target_fmr=.01, progress=progress, **regimes[regime])
                        for key in tables:
                            tables[key].extend(result[key])
                        models.extend(result["models"])
                        thresholds.extend(result["thresholds"])
                        for record in result["predictions"]:
                            writer.writerow(record)
                            prediction_count += 1
        for key, rows in tables.items():
            write_csv(output / f"{key}.csv", rows)
        write_json(output / "models.json", models)
        write_json(output / "thresholds.json", thresholds)
        if any(sha256(ROOT / name) != digest for name, digest in source_fingerprints.items()):
            raise ValueError("Analysis source changed during execution")
        if sha256(source / "pair_scores.jsonl") != source_hashes["pair_scores_sha256"]:
            raise ValueError("Source scores changed during execution")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        score_objects=len(models), learned_fits=sum(model["model_type"] == "learned_logistic" for model in models),
                        fixed_score_definitions=sum(model["model_type"] == "fixed_score" for model in models),
                        threshold_configurations=len(thresholds), prediction_rows=prediction_count,
                        output_sha256={path.name: sha256(path) for path in sorted(output.iterdir())
                                       if path.is_file() and path.name != "manifest.json"})
    except BaseException as error:
        manifest.update(status="failed", elapsed_seconds=time.perf_counter() - started,
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/project2/robustness")
    parser.add_argument("--output", type=Path, default=ROOT / "results/project3/operating_points")
    args = parser.parse_args()
    manifest = run(args.source, args.output)
    print(json.dumps({key: manifest[key] for key in
                      ("status", "score_objects", "learned_fits", "fixed_score_definitions",
                       "threshold_configurations", "prediction_rows", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
