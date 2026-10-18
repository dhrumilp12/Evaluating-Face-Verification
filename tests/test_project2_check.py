"""Saved-result audits catch corruption even when a changed file is rehashed."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.project2_evaluation import evaluate_conditions
from src.qmagface_scoring import QMAGFACE_100_ALPHA, QMAGFACE_100_BETA, score_embedding_pairs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_project2.py"
spec = importlib.util.spec_from_file_location("project2_check", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root, caches=False):
    output = root / "results/project2/robustness"
    output.mkdir(parents=True)
    source_names = ("magface_embedding.py", "face_embedding.py", "vendor/magface_iresnet.py", "project2_degradation.py", "project2_evaluation.py")
    sources = {}
    for name in source_names:
        path = root / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Synthetic fixture source: {name}\n")
        sources[f"src/{name}"] = checker.digest(path)
    settings = {"alpha": QMAGFACE_100_ALPHA, "beta": QMAGFACE_100_BETA,
                "conditions": ["original", "blur_2.1"], "source_sha256": sources,
                "environment": {"python": "saved-machine-version", "packages": {"numpy": "saved-version"}},
                "device": "cpu", "batch_size": 2, "threads": 2,
                "checkpoint_path": "models/project2/test.pth"}
    original_features, degraded_features, pairs = {}, {}, []
    for fold in range(10):
        for label in (1, 0):
            index = 2 * fold + (1 - label)
            ref, probe = f"person{index}/ref.jpg", f"person{index}/probe.jpg"
            pairs.append({"pair_index": index, "fold": fold, "label": label, "reference": ref, "probe": probe})
            first, second, degraded = (np.zeros(512, dtype=np.float32) for _ in range(3))
            first[0] = 20
            second[:2] = [18, 2] if label else [2, 18]
            degraded[:2] = [10, 8] if label else [8, 10]
            original_features[ref], original_features[probe], degraded_features[probe] = first, second, degraded
    rows = []
    for condition in settings["conditions"]:
        for pair in pairs:
            ref, probe = pair["reference"], pair["probe"]
            row = {"pair_index": pair["pair_index"], "fold": pair["fold"], "label": pair["label"],
                   "reference_path": ref, "probe_path": probe, "condition": condition,
                   "status": "scored", "error": ""}
            features = original_features if condition == "original" else degraded_features
            score = score_embedding_pairs(original_features[ref][None], features[probe][None],
                                           alpha=settings["alpha"], beta=settings["beta"])
            row.update({key: float(getattr(score, key)[0]) for key in
                        ("cosine_similarity", "qmagface_score", "reference_quality", "probe_quality")})
            rows.append(row)
    (output / "pair_scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = evaluate_conditions(rows)
    write_json(output / "evaluation.json", {k: v for k, v in result.items() if k not in ("predictions", "curves")})
    write_json(output / "alignment_failures.json", {})
    for name in checker.TABLES:
        write_csv(output / f"{name}.csv", result[name])
    manifest = {"status": "completed", "settings": settings, "completed_conditions": settings["conditions"],
                "output_sha256": {p.name: checker.digest(p) for p in output.iterdir()}}
    if caches:
        identity = {"adapter_sha256": sources["src/magface_embedding.py"], "selection_sha256": sources["src/face_embedding.py"],
                    "environment": settings["environment"], "threads": 2, "device": "cpu",
                    "detector_sha256": {p: "0" * 64 for p in ("pnet.pt", "rnet.pt", "onet.pt")}}
        fingerprint = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]
        alignment_dir = root / "data/processed/project2/alignment" / fingerprint(identity)
        records = {}
        for relative in original_features:
            source = root / "data/lfw_home/lfw_funneled" / relative
            crop = alignment_dir / "crops" / Path(relative).with_suffix(".png")
            for path, text in ((source, "source"), (crop, "crop")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"Synthetic {text} bytes for {relative}")
            records[relative] = {"status": "ok", "source_sha256": checker.digest(source),
                                 "crop_sha256": checker.digest(crop), "crop_path": crop.relative_to(root).as_posix()}
        alignment_path = alignment_dir / "records.json"
        write_json(alignment_path, {"identity": identity, "records": records})
        manifest["alignment"] = {"path": alignment_path.relative_to(root).as_posix(), "sha256": checker.digest(alignment_path),
                                 "requested_images": len(records), "failed_images": 0}
        pairs_path = root / "data/lfw_home/pairs.txt"
        pairs_path.write_text("synthetic pair-definition fixture; parser supplied by test\n")
        settings["pairs_sha256"] = checker.digest(pairs_path)
        checkpoint = root / settings["checkpoint_path"]
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint; checker never unpickles this")
        settings["checkpoint_sha256"] = checker.digest(checkpoint)
        cache_identity = {"checkpoint": settings["checkpoint_sha256"], "device": "cpu", "adapter": sources["src/magface_embedding.py"],
                          "architecture": sources["src/vendor/magface_iresnet.py"], "transforms": sources["src/project2_degradation.py"],
                          "environment": settings["environment"], "batch_size": 2, "threads": 2}
        feature_dir = root / "data/processed/project2/features" / fingerprint(cache_identity)
        write_json(feature_dir / "identity.json", cache_identity)
        manifest["feature_cache"] = feature_dir.relative_to(root).as_posix()
        for condition, features in (("original", original_features), ("blur_2.1", degraded_features)):
            for relative, raw in features.items():
                target = feature_dir / condition / Path(relative).with_suffix(".npz")
                target.parent.mkdir(parents=True, exist_ok=True)
                quality = np.linalg.norm(raw.astype(np.float64))
                np.savez(target, raw=raw, unit=raw.astype(np.float64) / quality, quality=quality,
                         input_sha256=records[relative]["crop_sha256"])
    write_json(output / "manifest.json", manifest)
    return output, manifest, rows, pairs


def make_benchmark_fixture(root, original_rows):
    directory = root / "results/project2/author_benchmark"
    directory.mkdir(parents=True)
    source_hashes = {}
    for relative in ("scripts/project2_benchmark.py", "src/qmagface_scoring.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic benchmark source fixture\n")
        source_hashes[relative] = checker.digest(path)
    scores = {"magface_cosine": [r["cosine_similarity"] for r in original_rows],
              "qmagface": [r["qmagface_score"] for r in original_rows]}
    rows = []
    for source in original_rows:
        row = {"pair_index": source["pair_index"], "original_fold": source["fold"],
               "is_same": source["label"], "status": "scored", "upstream_fold": source["fold"],
               "reference_quality": source["reference_quality"], "probe_quality": source["probe_quality"]}
        for method, values in scores.items():
            row[f"{method}_score"] = values[source["pair_index"]]
            row[f"{method}_threshold"] = min(values)
            row[f"{method}_prediction"] = source["label"]
        rows.append(row)
    folds = [{"method": method, "fold": fold, "threshold": min(values), "pairs": 2,
              "tp": 1, "fp": 0, "tn": 1, "fn": 0, "accuracy": 1., "fmr": 0., "fnmr": 0.}
             for method, values in scores.items() for fold in range(10)]
    write_csv(directory / "pair_scores.csv", rows)
    write_csv(directory / "fold_metrics.csv", folds)
    (directory / "excluded_pairs.csv").write_text("pair_index\n")
    summary = {"status": "completed", "alpha": QMAGFACE_100_ALPHA, "beta": QMAGFACE_100_BETA,
               "project_code_sha256": source_hashes, "requested_pairs": 20, "scored_pairs": 20,
               "excluded_pairs": 0, "coverage": 1., "scored_genuine": 10, "scored_impostor": 10,
               "methods": {method: {"pooled_accuracy": 1., "false_matches": 0, "false_nonmatches": 0,
                                     "independent_max_absolute_score_error": 0., "accuracy_percent": 100.,
                                     "independent_accuracy_percent": 100., "fold_accuracy_sd_percentage_points": 0.,
                                     "published_accuracy_percent": 99.83, "gap_percentage_points": 100 - 99.83}
                           for method in scores},
               "paired_decision_changes": {"qmagface_corrects_cosine_errors": 0, "qmagface_introduces_errors": 0, "both_wrong": 0},
               "outputs": {p.name: checker.digest(p) for p in directory.iterdir()}}
    write_json(directory / "summary.json", summary)
    return directory, summary


class Project2CheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def rehash(self, output, manifest, filename):
        manifest["output_sha256"][filename] = checker.digest(output / filename)
        write_json(output / "manifest.json", manifest)

    def assert_failed(self, report, check, text):
        self.assertEqual(report["status"], "failed")
        item = next(c for c in report["checks"] if c["check"] == check)
        self.assertEqual(item["status"], "fail", item)
        self.assertIn(text, item["detail"])

    def test_portable_valid_run_passes_without_inference_packages_or_caches(self):
        make_fixture(self.root)
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = checker.check(self.root)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["skipped"], 2)
        self.assertFalse((self.root / "data").exists())
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_incomplete_and_missing_runs_fail_without_writes(self):
        report = checker.check(self.root)
        self.assert_failed(report, "completed_manifest", "FileNotFoundError")
        self.assertEqual(list(self.root.iterdir()), [])
        output, manifest, *_ = make_fixture(self.root)
        manifest["status"] = "started"
        write_json(output / "manifest.json", manifest)
        self.assert_failed(checker.check(self.root), "completed_manifest", "not completed")

    def test_output_byte_corruption_and_unhashed_required_output_fail(self):
        output, manifest, *_ = make_fixture(self.root)
        (output / "evaluation.json").write_text('{}\n')
        self.assert_failed(checker.check(self.root), "output_hashes", "Hash mismatch")
        del manifest["output_sha256"]["evaluation.json"]
        write_json(output / "manifest.json", manifest)
        self.assert_failed(checker.check(self.root), "output_hashes", "lack hashes")

    def test_source_drift_and_path_escape_fail(self):
        output, manifest, *_ = make_fixture(self.root)
        (self.root / "src/magface_embedding.py").write_text("changed preprocessing")
        self.assert_failed(checker.check(self.root), "source_hashes", "Hash mismatch")
        manifest["output_sha256"]["../../../../outside"] = "0" * 64
        write_json(output / "manifest.json", manifest)
        self.assert_failed(checker.check(self.root), "output_hashes", "leaves its directory")

    def test_rehashed_wrong_adjusted_score_cannot_pass(self):
        output, manifest, rows, _ = make_fixture(self.root)
        rows[0]["qmagface_score"] -= 0.1
        (output / "pair_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.rehash(output, manifest, "pair_scores.jsonl")
        self.assert_failed(checker.check(self.root), "paired_scores_and_calibration", "formula")

    def test_rehashed_changed_fold_cannot_pass(self):
        output, manifest, rows, _ = make_fixture(self.root)
        rows[-1]["fold"] = 0
        (output / "pair_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.rehash(output, manifest, "pair_scores.jsonl")
        self.assert_failed(checker.check(self.root), "paired_scores_and_calibration", "changed pair metadata")

    def test_rehashed_test_selected_threshold_fails_recalibration(self):
        output, manifest, *_ = make_fixture(self.root)
        saved = checker.read_json(output / "evaluation.json")
        saved["thresholds"]["magface_cosine"]["0"] = .99
        write_json(output / "evaluation.json", saved)
        self.rehash(output, manifest, "evaluation.json")
        self.assert_failed(checker.check(self.root), "numerical_summary", "Numerical mismatch")

    def test_rehashed_prediction_and_curve_corruption_fail(self):
        output, manifest, *_ = make_fixture(self.root)
        for name, field, value in (("predictions", "prediction", "0"), ("curves", "fmr", ".75")):
            path = output / f"{name}.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0][field] = value
            write_csv(path, rows)
            self.rehash(output, manifest, path.name)
            self.assert_failed(checker.check(self.root), f"csv:{name}", "mismatch")

    def test_valid_local_cache_reconstructs_every_pair_without_loading_model(self):
        _, _, _, pairs = make_fixture(self.root, caches=True)
        with patch("src.lfw_dataset.parse_pairs", return_value=pairs):
            report = checker.check(self.root, include_caches=True)
        self.assertEqual(report["status"], "passed", report)
        detail = next(c["detail"] for c in report["checks"] if c["check"] == "local_caches")
        self.assertEqual(detail, {"images": 40, "feature_files": 60, "reconstructed_scored_pairs": 40})

    def test_corrupted_cache_quality_is_rejected(self):
        _, manifest, _, pairs = make_fixture(self.root, caches=True)
        path = next((self.root / manifest["feature_cache"] / "original").rglob("*.npz"))
        with np.load(path) as cache:
            values = dict(cache)
        values["quality"] *= 2
        np.savez(path, **values)
        with patch("src.lfw_dataset.parse_pairs", return_value=pairs):
            self.assert_failed(checker.check(self.root, include_caches=True), "local_caches", "quality mismatch")

    def test_coherently_changed_raw_embedding_is_caught_by_score_reconstruction(self):
        _, manifest, _, pairs = make_fixture(self.root, caches=True)
        relative = Path(pairs[0]["probe"]).with_suffix(".npz")
        path = self.root / manifest["feature_cache"] / "original" / relative
        with np.load(path) as cache:
            values = dict(cache)
        values["raw"] = np.roll(values["raw"], 1)
        values["unit"] = np.roll(values["unit"], 1)
        np.savez(path, **values)
        with patch("src.lfw_dataset.parse_pairs", return_value=pairs):
            self.assert_failed(checker.check(self.root, include_caches=True), "local_caches", "raw cache reconstruction")

    def test_changed_source_image_fails_even_with_untouched_embeddings(self):
        _, _, _, pairs = make_fixture(self.root, caches=True)
        (self.root / "data/lfw_home/lfw_funneled" / pairs[0]["reference"]).write_bytes(b"different source")
        with patch("src.lfw_dataset.parse_pairs", return_value=pairs):
            self.assert_failed(checker.check(self.root, include_caches=True), "local_caches", "Source image hash mismatch")

    def test_json_cli_returns_failure_and_never_creates_report_file(self):
        completed = subprocess.run([sys.executable, str(SCRIPT), "--root", str(self.root), "--json"],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(json.loads(completed.stdout)["status"], "failed")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_benchmark_saved_outputs_reproduce_author_calibration_policy(self):
        _, _, rows, _ = make_fixture(self.root)
        make_benchmark_fixture(self.root, [r for r in rows if r["condition"] == "original"])
        report = checker.check(self.root)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["skipped"], 1)

    def test_rehashed_benchmark_threshold_is_recalibrated(self):
        _, _, rows, _ = make_fixture(self.root)
        directory, summary = make_benchmark_fixture(self.root, [r for r in rows if r["condition"] == "original"])
        path = directory / "pair_scores.csv"
        with path.open(newline="") as stream:
            pairs = list(csv.DictReader(stream))
        for pair in pairs:
            pair["magface_cosine_threshold"] = float(pair["magface_cosine_threshold"]) + 0.01
        write_csv(path, pairs)
        summary["outputs"][path.name] = checker.digest(path)
        write_json(directory / "summary.json", summary)
        self.assert_failed(checker.check(self.root), "author_benchmark", "calibration mismatch")


if __name__ == "__main__":
    unittest.main()
