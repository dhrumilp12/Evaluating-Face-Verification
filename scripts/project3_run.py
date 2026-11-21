#!/usr/bin/env python3
"""Run the Project 3 calibration study from verified, saved Project 2 scores."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
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

from src.project3_calibration import (CONDITIONS, METHODS, PREDICTION_FIELDS,
                                      evaluate_regime, experiment_regimes, validate_records)

BASE_COMMIT = "81e5d0668ffc32bf8271fb89fcae2bc7de53e51f"
SOURCE_FILES = ("src/project3_calibration.py", "scripts/project3_run.py",
                "src/project2_evaluation.py", "src/evaluation.py", "src/score_analysis.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty result table")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_saved_scores(source):
    source = Path(source).resolve()
    manifest_file, score_file = source / "manifest.json", source / "pair_scores.jsonl"
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Project 2 source manifest must have status completed")
    expected = manifest.get("output_sha256", {}).get("pair_scores.jsonl")
    if not isinstance(expected, str) or len(expected) != 64 or sha256(score_file) != expected:
        raise ValueError("Project 2 score file does not match its completed manifest")
    with score_file.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    groups = validate_records(records, n_folds=10)
    if set(groups) != set(CONDITIONS) or len(groups["original"]) != 6000:
        raise ValueError("Expected the original Project 2 run: ten conditions and 6,000 requested pairs")
    # The published condition order gives output files a predictable progression.
    groups = {name: groups[name] for name in CONDITIONS}
    return groups, {"pair_scores_sha256": expected, "manifest_sha256": sha256(manifest_file)}


def _print_progress(message):
    print(message, flush=True)


def run(source, output, *, sensitivity=True, progress=_print_progress):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be separate from the Project 2 source directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new or empty output directory; existing results are never overwritten")
    groups, source_hashes = load_saved_scores(source)
    start = time.perf_counter()
    source_fingerprints = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True, capture_output=True, check=True).stdout.strip()
    manifest = {
        "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "New comparison calibration experiments on frozen Project 2 scores; no image inference",
        "project2_repository_commit": BASE_COMMIT, "execution_git_commit": git_commit,
        "source": {"directory": str(source), **source_hashes},
        "source_sha256": source_fingerprints,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("numpy", "scipy", "scikit-learn", "threadpoolctl")}},
        "settings": {"n_folds": 10, "primary_c": 1.0, "sensitivity": bool(sensitivity),
                     "sensitivity_c_values": [.1, 1.0, 10.0] if sensitivity else [],
                     "sensitivity_regimes": ["mixed", "leave_resolution_out"] if sensitivity else [],
                     "training_variant_weight": "1 / number_of_training_conditions",
                     "solver": "lbfgs", "max_iter": 2000, "tol": 1e-8, "blas_threads": 1,
                     "learned_decision_threshold": 0.0, "acceptance_rule": "score >= threshold",
                     "threshold_baselines": "maximum training accuracy, largest tied candidate threshold",
                     "primary_methods": list(METHODS)},
        "coverage": {"requested_pairs_per_condition": len(groups["original"]),
                     "scored_pairs_per_condition": sum(r["status"] == "scored" for r in groups["original"]),
                     "conditions": list(groups)},
        "limitations": ["LFW pair folds are not identity- or image-disjoint.",
                        "This is a retrospective extension of a previously inspected benchmark.",
                        "Mixed calibration observes the tested degradation types in other folds.",
                        "Mean AUC/EER are foldwise; learned scales differ across fitted folds.",
                        "Learned logits are decision scores, not validated probabilities."]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    tables = {name: [] for name in ("folds", "aggregates", "paired_changes")}
    models = []
    prediction_count = 0
    try:
        with (output / "predictions.csv.gz").open("wb") as binary:
            with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
                    writer.writeheader()
                    for name, conditions in experiment_regimes(groups).items():
                        methods = METHODS
                        if sensitivity and name in ("mixed", "leave_resolution_out"):
                            methods += ("logistic_interaction_c0.1", "logistic_interaction_c10")
                        progress(f"Evaluating {name}: {len(methods)} methods, ten held-out folds")
                        with threadpool_limits(limits=1):
                            result = evaluate_regime(groups, regime=name, n_folds=10,
                                                     c=1.0, methods=methods, progress=progress, **conditions)
                        for key in tables:
                            tables[key].extend(result[key])
                        models.extend(result["models"])
                        for record in result["predictions"]:
                            writer.writerow(record)
                            prediction_count += 1
        for key, rows in tables.items():
            write_csv(output / f"{key}.csv", rows)
        write_json(output / "models.json", models)
        # Detect source changes during a run instead of attributing mixed versions.
        if any(sha256(ROOT / name) != digest for name, digest in source_fingerprints.items()):
            raise ValueError("Analysis source changed during execution")
        if sha256(source / "pair_scores.jsonl") != source_hashes["pair_scores_sha256"]:
            raise ValueError("Source scores changed during execution")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - start,
                        fitted_models=len(models), prediction_rows=prediction_count,
                        output_sha256={path.name: sha256(path) for path in sorted(output.iterdir())
                                       if path.is_file() and path.name != "manifest.json"})
    except BaseException as error:
        manifest.update(status="failed", elapsed_seconds=time.perf_counter() - start,
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/project2/robustness")
    parser.add_argument("--output", type=Path, default=ROOT / "results/project3/calibration")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Run only the five primary methods")
    args = parser.parse_args()
    manifest = run(args.source, args.output, sensitivity=not args.skip_sensitivity)
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                      "fitted_models": manifest["fitted_models"],
                      "prediction_rows": manifest["prediction_rows"],
                      "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
