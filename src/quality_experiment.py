"""Independent probe blur/brightness using Commit 6 baseline validation and scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

from baseline_verification import write_json, write_csv
from lfw_dataset import sha256
from face_embedding import FaceEmbedder, validate_embedding, cosine_similarity
from resolution_experiment import inside, load_baseline, evaluate_fixed, verify_model
from quality_degradation import CONDITIONS, QUALITY_CONFIG, apply_condition


def evaluate_condition(rows, scores, thresholds, condition):
    # Reuse the existing evaluator; replace its resolution-specific output label.
    aggregate, folds, predictions = evaluate_fixed(rows, scores, thresholds, 160)
    for record in [aggregate, *folds, *predictions]:
        del record["resolution"]
        record.update(condition)
    return aggregate, folds, predictions


def run_quality(root):
    root = Path(root).resolve()
    metric_dir = root / "results/metrics"
    start = time.perf_counter()
    report_path = metric_dir / "quality_summary.json"
    report = {"status": "started", "run_at_utc": datetime.now(timezone.utc).isoformat(),
              "degradation": QUALITY_CONFIG, "conditions": [],
              "scope": "Fixed original reference embeddings, original face crops, baseline eligible pairs, and saved baseline fold thresholds. No detection or threshold calibration on degraded inputs."}
    write_json(report_path, report)
    try:
        print("Verifying baseline results and cached crops ...", flush=True)
        baseline, saved, rows, cache, records, reference_embeddings = load_baseline(root)
        report["baseline_summary_sha256"] = sha256(metric_dir / "baseline_summary.json")
        report["baseline_thresholds_sha256"] = sha256(metric_dir / "baseline_thresholds.json")
        report["model_fingerprint"] = baseline["model_fingerprint"]
        report["baseline_cache"] = saved["cache_directory"]
        report["source_sha256"] = {p: sha256(root / "src" / p) for p in
                                    ("quality_degradation.py", "quality_experiment.py", "resolution_experiment.py",
                                     "degradation.py", "baseline_verification.py", "lfw_dataset.py", "face_embedding.py", "evaluation.py")}
        try:
            report["git_head_before_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            report["git_head_before_commit"] = None
        probe_paths = sorted({r["probe"] for r in rows if r["status"] == "scored"})
        report["unique_probes"] = len(probe_paths)
        # Instantiate the existing model; never call its crop/detect methods.
        pipeline = FaceEmbedder(root)
        verify_model(root, baseline["model_fingerprint"])
        # Small actual forward check prevents accidentally reusing incompatible weights/code.
        first_record = records[probe_paths[0]]
        with np.load(inside(cache, first_record["payload"]), allow_pickle=False) as data:
            check = validate_embedding(pipeline.embed(apply_condition(data["crop"], CONDITIONS[0])))
        max_diff = float(np.max(np.abs(check - reference_embeddings[probe_paths[0]])))
        report["control_embedding_max_abs_difference"] = max_diff
        if not np.allclose(check, reference_embeddings[probe_paths[0]], atol=1e-6, rtol=0):
            raise ValueError("Original forward check differs from the baseline embedding")
        signature = hashlib.sha256(json.dumps({"baseline": report["baseline_thresholds_sha256"],
            "model": baseline["model_fingerprint"], "code": report["source_sha256"], "degradation": QUALITY_CONFIG}, sort_keys=True).encode()).hexdigest()
        output_cache = root / "data/processed/quality" / signature
        output_cache.mkdir(parents=True, exist_ok=True)
        report["cache_directory"] = output_cache.relative_to(root).as_posix()
        all_folds, all_predictions = [], []
        for condition in CONDITIONS:
            name = condition["condition"]
            condition_start = time.perf_counter()
            reused = 0
            if condition["family"] == "control":
                probe_embeddings = reference_embeddings
            else:
                probe_embeddings = {}
                for index, relative in enumerate(probe_paths, 1):
                    record = records[relative]
                    stem = hashlib.sha256((relative + record["payload_sha256"]).encode()).hexdigest()
                    directory = output_cache / name
                    directory.mkdir(exist_ok=True)
                    payload = directory / (stem + ".npy")
                    receipt = directory / (stem + ".json")
                    vector = None
                    if payload.is_file() and receipt.is_file():
                        try:
                            if json.loads(receipt.read_text())["sha256"] == sha256(payload):
                                vector = validate_embedding(np.load(payload, allow_pickle=False))
                                reused += 1
                        except (OSError, ValueError, KeyError):
                            vector = None
                    if vector is None:
                        with np.load(inside(cache, record["payload"]), allow_pickle=False) as data:
                            degraded = apply_condition(data["crop"], condition)
                        vector = validate_embedding(pipeline.embed(degraded))
                        temporary = payload.with_suffix(".tmp")
                        with temporary.open("wb") as stream:
                            np.save(stream, vector, allow_pickle=False)
                        temporary.replace(payload)
                        write_json(receipt, {"path": relative, **condition, "sha256": sha256(payload),
                                             "baseline_payload_sha256": record["payload_sha256"]})
                    probe_embeddings[relative] = vector
                    if index % 100 == 0 or index == len(probe_paths):
                        print(f"{name}: {index}/{len(probe_paths)} probes | reused {reused}", flush=True)
            scores = {r["pair_index"]: cosine_similarity(reference_embeddings[r["reference"]], probe_embeddings[r["probe"]])
                      for r in rows if r["status"] == "scored"}
            aggregate, folds, predictions = evaluate_condition(rows, scores, saved["thresholds"], condition)
            if condition["family"] == "control":
                for key in ("mean_accuracy", "mean_fmr", "mean_fnmr"):
                    if not np.isclose(aggregate[key], baseline["aggregate"][key], atol=1e-12, rtol=0):
                        raise ValueError("Original control metrics do not reproduce baseline")
                for key in ("false_matches", "false_nonmatches"):
                    if aggregate[key] != baseline["aggregate"]["confusion_counts"][key]:
                        raise ValueError("Original control confusion counts differ from baseline")
            for key in ("accuracy", "fmr", "fnmr"):
                aggregate[f"delta_{key}_pp"] = 100 * (aggregate[f"mean_{key}"] - baseline["aggregate"][f"mean_{key}"])
            aggregate["reused_probe_embeddings"] = reused
            aggregate["elapsed_seconds"] = round(time.perf_counter() - condition_start, 3)
            report["conditions"].append(aggregate)
            all_folds.extend(folds)
            all_predictions.extend(predictions)
            write_json(report_path, report)
            print(f"{name}: accuracy {100*aggregate['mean_accuracy']:.3f}%, FMR {100*aggregate['mean_fmr']:.3f}%, FNMR {100*aggregate['mean_fnmr']:.3f}%", flush=True)
        write_csv(metric_dir / "quality_comparison.csv", report["conditions"])
        write_csv(metric_dir / "quality_folds.csv", all_folds)
        write_csv(metric_dir / "quality_predictions.csv", all_predictions)
        report["status"] = "completed"
        report["output_sha256"] = {name: sha256(metric_dir / name) for name in
                                    ("quality_comparison.csv", "quality_folds.csv", "quality_predictions.csv")}
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        write_json(report_path, report)
    return report


if __name__ == "__main__":
    run_quality(Path(__file__).resolve().parents[1])
