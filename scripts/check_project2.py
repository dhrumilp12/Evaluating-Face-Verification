"""Read-only Project 2 verification from saved scores; no inference or downloads.

The default needs only NumPy and the submitted sources/outputs. It does not
require the machine's package versions to match the inference environment.
Use --include-caches for source images, alignment bytes, checkpoint provenance,
raw/unit/quality caches, and independent reconstruction of every saved score.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.project2_evaluation import evaluate_conditions
from src.qmagface_scoring import QMAGFACE_100_ALPHA, QMAGFACE_100_BETA, qmagface_scores, score_embedding_pairs

TABLES = ("folds", "aggregates", "paired_changes", "predictions", "curves", "distributions", "condition_changes")
REQUIRED_OUTPUTS = {"pair_scores.jsonl", "evaluation.json", "alignment_failures.json", *(f"{t}.csv" for t in TABLES)}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def inside(base, relative):
    need(isinstance(relative, str) and bool(relative) and not Path(relative).is_absolute(),
         f"Expected a relative artifact path: {relative}")
    base = Path(base).resolve()
    path = (base / relative).resolve()
    need(path.is_relative_to(base), f"Artifact path leaves its directory: {relative}")
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def compare(actual, expected, location="value"):
    """Compare JSON-like structures with strict categories and numerical tolerance."""
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f"Different keys: {location}")
        for key, value in expected.items():
            compare(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f"Different row count: {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        need(isinstance(actual, (int, float)) and not isinstance(actual, bool)
             and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
             f"Numerical mismatch: {location} (saved {actual}, recomputed {expected})")
    else:
        need(type(actual) is type(expected) and actual == expected, f"Value mismatch: {location}")


def verify_hashes(base, hashes, required=()):
    need(isinstance(hashes, dict) and hashes, "Missing output hashes")
    need(set(required).issubset(hashes), f"Required artifacts lack hashes: {sorted(set(required) - set(hashes))}")
    for relative, expected in hashes.items():
        need(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid SHA-256: {relative}")
        need(digest(inside(base, relative)) == expected, f"Hash mismatch: {relative}")
    return len(hashes)


def compare_csv(path, expected):
    fields = list(dict.fromkeys(key for row in expected for key in row))
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        need(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames))
             and set(reader.fieldnames) == set(fields), f"CSV schema mismatch: {path.name}")
        actual = list(reader)
    need(len(actual) == len(expected), f"CSV row count mismatch: {path.name}")
    for index, (saved, computed) in enumerate(zip(actual, expected)):
        for key in fields:
            value, cell = computed.get(key), saved[key]
            location = f"{path.name}[{index}].{key}"
            if value is None:
                need(cell == "", f"Expected empty failure/undefined cell: {location}")
                continue
            if isinstance(value, bool):
                parsed = {"True": True, "False": False}.get(cell)
            elif isinstance(value, int):
                need(bool(re.fullmatch(r"-?[0-9]+", cell)), f"Expected integer: {location}")
                parsed = int(cell)
            elif isinstance(value, float):
                parsed = float(cell)
            elif isinstance(value, (dict, list)):
                parsed = ast.literal_eval(cell)
            else:
                parsed = cell
            compare(parsed, value, location)


def validate_pair_scores(rows, settings, completed_conditions):
    need(settings["alpha"] == QMAGFACE_100_ALPHA and settings["beta"] == QMAGFACE_100_BETA,
         "Primary run must use the published QMagFace-100 coefficients")
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    compare(conditions, settings["conditions"], "conditions")
    compare(completed_conditions, conditions, "completed_conditions")
    result = evaluate_conditions(rows)  # Checks all ten folds, paired metadata, and both methods' eligibility.
    selected = [row for row in rows if row["status"] == "scored"]
    need(selected, "No scored pairs")
    adjusted = qmagface_scores([r["cosine_similarity"] for r in selected],
                              [r["reference_quality"] for r in selected], [r["probe_quality"] for r in selected],
                              alpha=settings["alpha"], beta=settings["beta"])
    need(np.allclose(adjusted, [r["qmagface_score"] for r in selected], rtol=0, atol=1e-12),
         "Saved QMagFace scores disagree with cosine/quality formula")
    original = {r["pair_index"]: r for r in rows if r["condition"] == "original"}
    for row in rows:
        baseline = original[row["pair_index"]]
        need(row["status"] == baseline["status"], "Post-alignment conditions changed the fixed pair eligibility mask")
        if row["status"] == "scored":
            compare(row["reference_quality"], baseline["reference_quality"], "fixed reference quality")
    return result


def verify_caches(root, manifest, rows, run_dir):
    """Read the existing cache; reconstruct all scores without calling a network."""
    settings = manifest["settings"]
    alignment_path = inside(root, manifest["alignment"]["path"])
    need(digest(alignment_path) == manifest["alignment"]["sha256"], "Alignment manifest hash mismatch")
    alignment = read_json(alignment_path)
    identity = alignment["identity"]
    fingerprint = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
    need(alignment_path.parent.name == fingerprint(identity), "Alignment identity directory mismatch")
    compare(identity["adapter_sha256"], settings["source_sha256"]["src/magface_embedding.py"], "alignment adapter provenance")
    compare(identity["selection_sha256"], settings["source_sha256"]["src/face_embedding.py"], "alignment selection provenance")
    compare(identity["environment"], settings["environment"], "alignment environment provenance")
    compare(identity["threads"], settings["threads"], "alignment threads")
    need(identity["device"] == "cpu" and set(identity["detector_sha256"]) == {"pnet.pt", "rnet.pt", "onet.pt"},
         "Missing CPU detector provenance")
    original = [r for r in rows if r["condition"] == "original"]
    paths = sorted({r[key] for r in original for key in ("reference_path", "probe_path")})
    records = alignment["records"]
    for relative in paths:
        record = records[relative]
        source = inside(root / "data/lfw_home/lfw_funneled", relative)
        need(digest(source) == record["source_sha256"], f"Source image hash mismatch: {relative}")
        if record["status"] == "ok":
            need(digest(inside(root, record["crop_path"])) == record["crop_sha256"], f"Aligned crop hash mismatch: {relative}")
    failures = {p: records[p] for p in paths if records[p]["status"] != "ok"}
    compare(read_json(run_dir / "alignment_failures.json"), failures, "alignment_failures")
    compare(manifest["alignment"]["requested_images"], len(paths), "requested alignment images")
    compare(manifest["alignment"]["failed_images"], len(failures), "failed alignment images")
    pairs_path = root / "data/lfw_home/pairs.txt"
    need(digest(pairs_path) == settings["pairs_sha256"], "Original pairs file hash mismatch")
    from src.lfw_dataset import parse_pairs
    pairs = parse_pairs(pairs_path)
    compare([{k: r[k] for k in ("pair_index", "fold", "label", "reference_path", "probe_path")} for r in original],
            [{"pair_index": p["pair_index"], "fold": p["fold"], "label": p["label"],
              "reference_path": p["reference"], "probe_path": p["probe"]} for p in pairs], "original pair definitions")
    checkpoint = Path(settings["checkpoint_path"])
    checkpoint = checkpoint if checkpoint.is_absolute() else inside(root, settings["checkpoint_path"])
    need(digest(checkpoint) == settings["checkpoint_sha256"], "Checkpoint hash mismatch")
    directory = inside(root, manifest["feature_cache"])
    expected_identity = {"checkpoint": settings["checkpoint_sha256"], "device": settings["device"],
                         "adapter": settings["source_sha256"]["src/magface_embedding.py"],
                         "architecture": settings["source_sha256"]["src/vendor/magface_iresnet.py"],
                         "transforms": settings["source_sha256"]["src/project2_degradation.py"],
                         "environment": settings["environment"], "batch_size": settings["batch_size"], "threads": settings["threads"]}
    compare(read_json(directory / "identity.json"), expected_identity, "feature cache identity")
    need(directory.name == fingerprint(expected_identity), "Feature identity directory mismatch")
    features = {}

    def feature(condition, relative):
        key = condition, relative
        if key not in features:
            path = inside(directory, f"{condition}/{Path(relative).with_suffix('.npz').as_posix()}")
            with np.load(path, allow_pickle=False) as cache:
                raw, unit, quality = cache["raw"], cache["unit"], float(cache["quality"].item())
                need(cache["input_sha256"].item() == records[relative]["crop_sha256"], f"Feature input provenance mismatch: {path}")
            need(raw.dtype == np.float32 and raw.shape == unit.shape == (512,)
                 and np.isfinite(raw).all() and np.isfinite(unit).all(), f"Invalid raw/unit feature: {path}")
            norm = float(np.linalg.norm(raw.astype(np.float64)))
            need(math.isfinite(quality) and quality > 0 and math.isclose(quality, norm, rel_tol=1e-12, abs_tol=1e-12),
                 f"Cached raw quality mismatch: {path}")
            need(np.allclose(unit, raw.astype(np.float64) / quality, rtol=1e-12, atol=1e-12), f"Cached unit vector mismatch: {path}")
            features[key] = raw
        return features[key]

    for row in rows:
        ref, probe = row["reference_path"], row["probe_path"]
        eligible = records[ref]["status"] == records[probe]["status"] == "ok"
        need(eligible == (row["status"] == "scored"), f"Pair eligibility differs from alignment: {row['pair_index']}")
        if eligible:
            scores = score_embedding_pairs(feature("original", ref)[None], feature(row["condition"], probe)[None],
                                            alpha=settings["alpha"], beta=settings["beta"])
            for key in ("cosine_similarity", "qmagface_score", "reference_quality", "probe_quality"):
                compare(row[key], float(getattr(scores, key)[0]), f"raw cache reconstruction/{row['condition']}/{row['pair_index']}/{key}")
    return {"images": len(paths), "feature_files": len(features), "reconstructed_scored_pairs": sum(r["status"] == "scored" for r in rows)}


def verify_benchmark(directory, root=ROOT):
    summary = read_json(directory / "summary.json")
    need(summary.get("status") == "completed", "Author benchmark is not completed")
    count = verify_hashes(directory, summary["outputs"], ("pair_scores.csv", "fold_metrics.csv", "excluded_pairs.csv"))
    verify_hashes(root, summary["project_code_sha256"], ("scripts/project2_benchmark.py", "src/qmagface_scoring.py"))
    with (directory / "pair_scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    scored = [r for r in rows if r["status"] == "scored"]
    compare(len(rows), summary["requested_pairs"], "benchmark requested pairs")
    compare(len(scored), summary["scored_pairs"], "benchmark scored pairs")
    compare(len(rows) - len(scored), summary["excluded_pairs"], "benchmark excluded pairs")
    compare(len(scored) / len(rows), summary["coverage"], "benchmark coverage")
    labels = np.array([int(r["is_same"]) for r in scored])
    need(np.isin(labels, [0, 1]).all(), "Invalid benchmark labels")
    compare(int(labels.sum()), summary["scored_genuine"], "benchmark genuine count")
    compare(int((labels == 0).sum()), summary["scored_impostor"], "benchmark impostor count")
    adjusted = qmagface_scores([float(r["magface_cosine_score"]) for r in scored],
                              [float(r["reference_quality"]) for r in scored], [float(r["probe_quality"]) for r in scored],
                              alpha=summary["alpha"], beta=summary["beta"])
    need(np.allclose(adjusted, [float(r["qmagface_score"]) for r in scored], rtol=0, atol=1e-12), "Benchmark QMagFace formula mismatch")
    need(summary["alpha"] == QMAGFACE_100_ALPHA and summary["beta"] == QMAGFACE_100_BETA,
         "Benchmark coefficients differ from published QMagFace-100 parameters")
    need(len(scored) >= 10, "Insufficient benchmark pairs for ten folds")
    fold_ids = np.empty(len(scored), dtype=int)
    for fold, indices in enumerate(np.array_split(np.arange(len(scored)), 10)):
        fold_ids[indices] = fold
    need(np.array_equal(fold_ids, [int(r["upstream_fold"]) for r in scored]), "Benchmark post-exclusion fold assignments differ")
    expected_folds, decisions = [], {}
    for method, metrics in summary["methods"].items():
        scores = np.array([float(r[f"{method}_score"]) for r in scored])
        predicted = np.array([int(r[f"{method}_prediction"]) for r in scored])
        computed = np.array([float(r[f"{method}_score"]) > float(r[f"{method}_threshold"]) for r in scored])
        need(np.array_equal(predicted, computed), f"Benchmark prediction/threshold mismatch: {method}")
        compare(float(np.mean(predicted == labels)), metrics["pooled_accuracy"], f"benchmark {method} accuracy")
        compare(int(np.sum((predicted == 1) & (labels == 0))), metrics["false_matches"], f"benchmark {method} false matches")
        compare(int(np.sum((predicted == 0) & (labels == 1))), metrics["false_nonmatches"], f"benchmark {method} false nonmatches")
        need(0 <= metrics["independent_max_absolute_score_error"] <= 1e-12, f"Official scorer parity failed: {method}")
        compare(metrics["accuracy_percent"], metrics["independent_accuracy_percent"], f"official metric parity/{method}")
        # Preserve the benchmark's documented global grid, strict > decision,
        # first-maximum tie rule, and KFold-after-exclusion behavior.
        grid, accuracies = np.linspace(scores.min(), scores.max(), 600), []
        for fold in range(10):
            train, test = fold_ids != fold, fold_ids == fold
            accuracy = np.mean((scores[train, None] > grid[None, :]) == labels[train, None], axis=0)
            threshold = float(grid[np.argmax(accuracy)])
            saved_thresholds = np.array([float(r[f"{method}_threshold"]) for r, keep in zip(scored, test) if keep])
            need(np.allclose(saved_thresholds, threshold, rtol=0, atol=1e-12), f"Benchmark calibration mismatch: {method}/{fold}")
            truth, decision = labels[test], predicted[test]
            tp, fp = int(np.sum((truth == 1) & (decision == 1))), int(np.sum((truth == 0) & (decision == 1)))
            tn, fn = int(np.sum((truth == 0) & (decision == 0))), int(np.sum((truth == 1) & (decision == 0)))
            accuracy = (tp + tn) / int(test.sum())
            accuracies.append(accuracy)
            expected_folds.append({"method": method, "fold": fold, "threshold": threshold,
                                   "pairs": int(test.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                                   "accuracy": accuracy, "fmr": fp / (fp + tn), "fnmr": fn / (fn + tp)})
        compare(float(np.mean(accuracies) * 100), metrics["accuracy_percent"], f"benchmark mean fold accuracy/{method}")
        compare(float(np.std(np.array(accuracies) * 100)), metrics["fold_accuracy_sd_percentage_points"], f"benchmark fold SD/{method}")
        compare(metrics["accuracy_percent"] - metrics["published_accuracy_percent"], metrics["gap_percentage_points"], f"benchmark published gap/{method}")
        decisions[method] = predicted == labels
    compare_csv(directory / "fold_metrics.csv", expected_folds)
    cosine, adjusted = decisions["magface_cosine"], decisions["qmagface"]
    compare(summary["paired_decision_changes"], {
        "qmagface_corrects_cosine_errors": int(np.sum(~cosine & adjusted)),
        "qmagface_introduces_errors": int(np.sum(cosine & ~adjusted)),
        "both_wrong": int(np.sum(~cosine & ~adjusted)),
    }, "benchmark paired changes")
    return f"{count} output hashes, source hashes, score formulas and upstream-policy recalibration verified"


def check(root=ROOT, run="results/project2/robustness", *, include_caches=False):
    root = Path(root).resolve()
    checks = []
    report = {"checks": checks, "include_caches": include_caches,
              "scope": "Read-only saved-output and numerical verification; no inference, downloads, or runtime-version matching"}

    def perform(name, fn):
        try:
            checks.append({"check": name, "status": "pass", "detail": fn()})
        except Exception as error:
            checks.append({"check": name, "status": "fail", "detail": f"{type(error).__name__}: {error}"})

    try:
        directory = inside(root, run)
        manifest = read_json(directory / "manifest.json")
        need(manifest.get("status") == "completed", "Robustness manifest is not completed")
        checks.append({"check": "completed_manifest", "status": "pass", "detail": "Completed robustness run"})
    except Exception as error:
        checks.append({"check": "completed_manifest", "status": "fail", "detail": f"{type(error).__name__}: {error}"})
        manifest = None
    if manifest is not None:
        perform("output_hashes", lambda: f"{verify_hashes(directory, manifest['output_sha256'], REQUIRED_OUTPUTS)} output hashes match")
        perform("source_hashes", lambda: f"{verify_hashes(root, manifest['settings']['source_sha256'])} executed source hashes match")
        rows = None
        try:
            rows = [json.loads(line) for line in (directory / "pair_scores.jsonl").read_text().splitlines()]
            recomputed = validate_pair_scores(rows, manifest["settings"], manifest["completed_conditions"])
            checks.append({"check": "paired_scores_and_calibration", "status": "pass", "detail": f"{len(rows)} records; ten-fold original-only calibration and paired eligibility verified"})
        except Exception as error:
            checks.append({"check": "paired_scores_and_calibration", "status": "fail", "detail": f"{type(error).__name__}: {error}"})
            recomputed = None
        if recomputed is not None:
            def summary():
                compare(read_json(directory / "evaluation.json"), {k: v for k, v in recomputed.items() if k not in ("predictions", "curves")}, "evaluation.json")
                return "Saved summary matches metrics recomputed from pair scores"
            perform("numerical_summary", summary)
            for table in TABLES:
                perform(f"csv:{table}", lambda table=table: (compare_csv(directory / f"{table}.csv", recomputed[table]) or f"{len(recomputed[table])} rows match recomputation"))
        if include_caches and rows is not None:
            perform("local_caches", lambda: verify_caches(root, manifest, rows, directory))
        elif not include_caches:
            checks.append({"check": "local_caches", "status": "skipped", "detail": "Use --include-caches to revalidate images, checkpoint and raw embeddings"})
    benchmark = root / "results/project2/author_benchmark"
    if (benchmark / "summary.json").is_file():
        perform("author_benchmark", lambda: verify_benchmark(benchmark, root))
    else:
        checks.append({"check": "author_benchmark", "status": "skipped", "detail": "No author benchmark output in this snapshot"})
    for name in ("pass", "fail", "skipped"):
        report[name] = sum(c["status"] == name for c in checks)
    report["status"] = "failed" if report["fail"] else "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", default="results/project2/robustness")
    parser.add_argument("--include-caches", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report to stdout")
    args = parser.parse_args()
    report = check(args.root, args.run, include_caches=args.include_caches)
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        for item in report["checks"]:
            print(f"{item['status'].upper():7s} {item['check']}: {item['detail']}")
        print(f"Project 2 check: {report['status'].upper()} ({report['pass']} passed, {report['fail']} failed, {report['skipped']} skipped)")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
