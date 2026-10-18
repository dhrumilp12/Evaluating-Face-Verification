#!/usr/bin/env python3
"""Reproduce the author's LFW feature benchmark and audit score agreement.

Run in the benchmark environment described in docs/project2/artifact_audit.md.
This intentionally preserves upstream's post-exclusion KFold and threshold grid.
It must not be used as the calibration protocol for the robustness investigation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qmagface_scoring import (  # noqa: E402
    QMAGFACE_100_ALPHA,
    QMAGFACE_100_BETA,
    score_embedding_pairs,
)

UPSTREAM_COMMIT = "395797145da1b1b263cbe7fb1564c6ea91f55b23"
FEATURE_URL = "https://drive.usercontent.google.com/download?id=1ElwkUKFs6-4JEwRnsKh6fJLp00x_FOvS&export=download&confirm=t"
FEATURE_ARCHIVE_SHA256 = "d228033c7bc4a2d4d1e92ba987cda9e6fb4423f219c97e98364820accd614f7f"
SOURCE_HASHES = {
    "similarity/base.py": "ebac233af73fbd8120948ebde3a58abd6d436de3a2a6cd08b974d27c27fbc138",
    "similarity/cosine.py": "1affd9f871111f0ebb645b4d9a2b43b1ae4350e558968bd36ff2a7efdd7eac9b",
    "similarity/qmagface.py": "7c4ebcb8e51d98b8e41dbb5d0481f79c469478079c75082be35410ddeff872bc",
    "evaluation/metrics.py": "eefea475d864ab886cd54efcd8a876c3fac65b2c52cabf8aeb81c8533eeac035",
}
FEATURE_HASHES = {
    "lfw_pairs.txt": "55506bb9b62f8968b0f3e5b5b8fff91604659b7dec61c6ee185e4606ee9b74f3",
    "embeddings_lfw.npy": "7788292d42b4d8fdfdc1e372bafc54aa9c949d470b48758041d5e851216c38da",
    "filenames_lfw.npy": "19f15d4fb900879f59233c70470abb96a5dda92e400d4f05723ddfc4361cc5a2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(directory: Path, expected: dict[str, str]) -> dict[str, Any]:
    records = {}
    for name, expected_hash in expected.items():
        path = directory / name
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(f"Unexpected artifact contents: {path} (SHA-256 {observed})")
        records[name] = {"sha256": observed, "bytes": path.stat().st_size}
    return records


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    curl = shutil.which("curl")
    if curl:
        # macOS system curl uses the configured certificate trust store; some
        # python.org installations do not yet have a working CA bundle.
        subprocess.run([curl, "--fail", "--location", "--silent", "--show-error",
                        "--retry", "3", "--connect-timeout", "30", "--output", str(temporary), url], check=True)
    else:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(8 << 20), b""):
                stream.write(block)
    temporary.replace(destination)


def download_artifacts(feature_dir: Path, source_dir: Path) -> None:
    """Download exact public files; extract only selected inert LFW data members."""
    if not all((feature_dir / name).exists() for name in FEATURE_HASHES):
        archive = ROOT / "data/project2/artifacts/author_features_download.zip"
        if not archive.exists():
            print("Downloading the author feature archive (about 6.2 GB).", flush=True)
            download_file(FEATURE_URL, archive)
        if sha256(archive) != FEATURE_ARCHIVE_SHA256:
            raise ValueError("Downloaded author archive does not match the audited artifact")
        feature_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive) as zipped:
            for filename in FEATURE_HASHES:
                prefix = "_data/pairs/" if filename.endswith(".txt") else "_data/single_images/magface100/"
                if zipped.getinfo(prefix + filename).file_size > 100_000_000:
                    raise ValueError(f"Unexpectedly large archive member: {filename}")
                # A fixed local basename is used; archive paths are never extracted.
                (feature_dir / filename).write_bytes(zipped.read(prefix + filename))
    for name in SOURCE_HASHES:
        if not (source_dir / name).exists():
            url = f"https://raw.githubusercontent.com/pterhoer/QMagFace/{UPSTREAM_COMMIT}/{name}"
            download_file(url, source_dir / name)
    verify_files(feature_dir, FEATURE_HASHES)
    verify_files(source_dir, SOURCE_HASHES)


def load_official(source_dir: Path) -> tuple[Any, Any, Any]:
    """Execute only the four hash-verified upstream files, no package startup code."""
    verify_files(source_dir, SOURCE_HASHES)
    if "similarity" in sys.modules:
        raise RuntimeError("The official scorer must run in a fresh Python process")
    package = types.ModuleType("similarity")
    package.__path__ = []
    sys.modules["similarity"] = package
    loaded = {}
    for name, relative in (
        ("similarity.base", "similarity/base.py"),
        ("similarity.cosine", "similarity/cosine.py"),
        ("similarity.qmagface", "similarity/qmagface.py"),
        ("project2_official_metrics", "evaluation/metrics.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, source_dir / relative)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load audited upstream module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return (
        loaded["similarity.cosine"].Cosine,
        loaded["similarity.qmagface"].QMagFace,
        loaded["project2_official_metrics"],
    )


def load_pairs(feature_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Resolve names without upstream's silent exclusions, retaining all attempts."""
    features = np.load(feature_dir / "embeddings_lfw.npy", allow_pickle=False)
    filenames = np.load(feature_dir / "filenames_lfw.npy", allow_pickle=False)
    if features.ndim != 2 or features.shape[1] != 512 or not np.isfinite(features).all():
        raise ValueError("Expected finite N x 512 raw embeddings")
    if filenames.ndim != 1 or filenames.dtype.kind not in "US" or len(filenames) != len(features):
        raise ValueError("Expected one string filename per embedding")
    names = [str(name).replace("\\", "/") for name in filenames]
    indices = {name: index for index, name in enumerate(names)}
    if len(indices) != len(names):
        raise ValueError("Duplicate feature filenames would make pair mapping ambiguous")
    prefix = "/".join(names[0].split("/")[:-2])
    pairs, labels, records = [], [], []
    lines = (feature_dir / "lfw_pairs.txt").read_text().splitlines()
    if len(lines) != 6000:
        raise ValueError("Expected the authors' headerless 6,000-pair LFW file")
    for index, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) == 3:
            name_a, id_a, id_b = tokens
            name_b, same = name_a, True
        elif len(tokens) == 4:
            name_a, id_a, name_b, id_b = tokens
            same = False
        else:
            raise ValueError(f"Malformed pair row {index}: {line!r}")
        a = f"{prefix}/{name_a}/{name_a}_{int(id_a):04d}.jpg"
        b = f"{prefix}/{name_b}/{name_b}_{int(id_b):04d}.jpg"
        missing = [name for name in (a, b) if name not in indices]
        records.append({
            "pair_index": index,
            "original_fold": index // 600,
            "is_same": int(same),
            "reference_path": a,
            "probe_path": b,
            "status": "excluded" if missing else "scored",
            "exclusion_reason": "missing_author_features: " + " | ".join(missing) if missing else "",
        })
        if not missing:
            pairs.append((indices[a], indices[b]))
            labels.append(same)
    return features, np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=bool), records


def fold_details(scores: np.ndarray, labels: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Audit the actual upstream policy and expose its thresholds/predictions."""
    from sklearn.model_selection import KFold

    thresholds = np.linspace(np.min(scores), np.max(scores), 600)
    folds, predictions, pair_folds = [], np.zeros(len(scores), dtype=bool), np.zeros(len(scores), dtype=int)
    for fold, (train, test) in enumerate(KFold(n_splits=10, shuffle=False).split(scores)):
        train_accuracy = np.mean((scores[train, None] > thresholds[None, :]) == labels[train, None], axis=0)
        threshold = thresholds[np.argmax(train_accuracy)]
        predictions[test] = scores[test] > threshold
        pair_folds[test] = fold
        truth, predicted = labels[test], predictions[test]
        tp, fp = int(np.sum(truth & predicted)), int(np.sum(~truth & predicted))
        tn, fn = int(np.sum(~truth & ~predicted)), int(np.sum(truth & ~predicted))
        folds.append({"fold": fold, "threshold": float(threshold), "pairs": len(test),
                      "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                      "accuracy": (tp + tn) / len(test), "fmr": fp / (fp + tn), "fnmr": fn / (fn + tp)})
    return folds, predictions, pair_folds


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/project2/author_lfw")
    parser.add_argument("--official-source", type=Path, default=ROOT / "data/project2/vendor/QMagFace-main")
    parser.add_argument("--output", type=Path, default=ROOT / "results/project2/author_benchmark")
    parser.add_argument("--download", action="store_true", help="Fetch missing pinned public artifacts (about 6.2 GB)")
    args = parser.parse_args()
    if args.download:
        download_artifacts(args.features, args.official_source)
    artifacts = verify_files(args.features, FEATURE_HASHES)
    source_files = verify_files(args.official_source, SOURCE_HASHES)
    features, pairs, labels, records = load_pairs(args.features)
    Cosine, QMagFace, metrics = load_official(args.official_source)
    official_scores = {
        "magface_cosine": Cosine.similarity(features, pairs),
        "qmagface": QMagFace(QMAGFACE_100_ALPHA, QMAGFACE_100_BETA).similarity(features, pairs),
    }
    ours = score_embedding_pairs(features[pairs[:, 0]], features[pairs[:, 1]],
                                 alpha=QMAGFACE_100_ALPHA, beta=QMAGFACE_100_BETA)
    independent = {"magface_cosine": ours.cosine_similarity, "qmagface": ours.qmagface_score}
    summary: dict[str, Any] = {
        "status": "completed", "dataset": "author_precomputed_lfw_magface100",
        "started_at_utc": started_at,
        "upstream_commit": UPSTREAM_COMMIT, "alpha": QMAGFACE_100_ALPHA, "beta": QMAGFACE_100_BETA,
        "feature_artifacts": artifacts, "executed_upstream_files": source_files,
        "project_code_sha256": {str(path): sha256(ROOT / path) for path in
                                (Path("scripts/project2_benchmark.py"), Path("src/qmagface_scoring.py"))},
        "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
        "requested_pairs": len(records), "scored_pairs": len(pairs), "excluded_pairs": len(records) - len(pairs),
        "requested_genuine": sum(row["is_same"] == 1 for row in records),
        "requested_impostor": sum(row["is_same"] == 0 for row in records),
        "scored_genuine": int(np.sum(labels)), "scored_impostor": int(np.sum(~labels)),
        "coverage": len(pairs) / len(records),
        "protocol": {"folds": "sklearn KFold(10, shuffle=False) AFTER missing-feature exclusions",
                     "thresholds": "600 evenly spaced points from global score min to max",
                     "decision": "score > threshold", "tie_break": "first/smallest threshold",
                     "caveat": "Global grid sees held-out score extrema; post-exclusion folds differ from original LFW folds."},
        "methods": {},
    }
    summary["coverage_by_original_fold"] = [
        {"original_fold": fold,
         "requested": sum(row["original_fold"] == fold for row in records),
         "scored": sum(row["original_fold"] == fold and row["status"] == "scored" for row in records),
         "excluded_genuine": sum(row["original_fold"] == fold and row["status"] == "excluded" and row["is_same"] == 1 for row in records),
         "excluded_impostor": sum(row["original_fold"] == fold and row["status"] == "excluded" and row["is_same"] == 0 for row in records)}
        for fold in range(10)
    ]
    import scipy
    import sklearn
    summary["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    all_folds, accepted = [], [record for record in records if record["status"] == "scored"]
    qualities = np.linalg.norm(features, axis=1)
    summary["raw_quality"] = {"minimum": float(qualities.min()), "mean": float(qualities.mean()),
                              "maximum": float(qualities.max())}
    predictions_by_method = {}
    for method, scores in official_scores.items():
        error = float(np.max(np.abs(scores - independent[method])))
        np.testing.assert_allclose(scores, independent[method], rtol=0, atol=1e-12)
        # This is an execution of the upstream metric implementation, not a paraphrase.
        official_metrics = metrics.evaluate_metrics(scores, labels, ["acc", "acc_std"])
        folds, predicted, pair_folds = fold_details(scores, labels)
        predictions_by_method[method] = predicted
        accuracy = float(np.mean([row["accuracy"] for row in folds]))
        np.testing.assert_allclose(accuracy * 100, official_metrics["acc"], rtol=0, atol=1e-12)
        own_accuracy = metrics.evaluate_metrics(independent[method], labels, ["acc"])["acc"]
        np.testing.assert_allclose(own_accuracy, official_metrics["acc"], rtol=0, atol=1e-12)
        target = 99.73 if method == "magface_cosine" else 99.83
        fp, fn = int(np.sum(predicted & ~labels)), int(np.sum(~predicted & labels))
        summary["methods"][method] = {
            "accuracy_percent": float(official_metrics["acc"]),
            "fold_accuracy_sd_percentage_points": float(official_metrics["acc_std"]),
            "published_accuracy_percent": target, "gap_percentage_points": float(official_metrics["acc"]) - target,
            "independent_max_absolute_score_error": error,
            "independent_accuracy_percent": float(own_accuracy),
            "pooled_accuracy": float(np.mean(predicted == labels)), "false_matches": fp, "false_nonmatches": fn,
            "pooled_fmr": fp / int(np.sum(~labels)), "pooled_fnmr": fn / int(np.sum(labels)),
        }
        for row in folds:
            all_folds.append({"method": method, **row})
        for index, row in enumerate(accepted):
            row["upstream_fold"] = int(pair_folds[index])
            row["reference_quality"] = float(ours.reference_quality[index])
            row["probe_quality"] = float(ours.probe_quality[index])
            row[method + "_score"] = float(scores[index])
            row[method + "_threshold"] = folds[int(pair_folds[index])]["threshold"]
            row[method + "_prediction"] = int(predicted[index])
    cos_right = predictions_by_method["magface_cosine"] == labels
    qmf_right = predictions_by_method["qmagface"] == labels
    summary["paired_decision_changes"] = {
        "qmagface_corrects_cosine_errors": int(np.sum(~cos_right & qmf_right)),
        "qmagface_introduces_errors": int(np.sum(cos_right & ~qmf_right)),
        "both_wrong": int(np.sum(~cos_right & ~qmf_right)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "pair_scores.csv", records)
    write_csv(args.output / "fold_metrics.csv", all_folds)
    write_csv(args.output / "excluded_pairs.csv", [row for row in records if row["status"] == "excluded"])
    summary["outputs"] = {name: sha256(args.output / name) for name in ("pair_scores.csv", "fold_metrics.csv", "excluded_pairs.csv")}
    summary["elapsed_seconds"] = perf_counter() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"coverage": summary["coverage"], "methods": summary["methods"]}, indent=2))


if __name__ == "__main__":
    main()
