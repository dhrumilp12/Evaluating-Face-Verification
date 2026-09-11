"""Probe-only resolution experiment with frozen baseline eligibility/thresholds."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from baseline_verification import write_json, write_csv
from lfw_dataset import sha256
from face_embedding import CONFIG, FaceEmbedder, validate_embedding, cosine_similarity
from evaluation import verification_metrics
from degradation import RESOLUTIONS, RESOLUTION_CONFIG, reduce_resolution


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Expected a path within the project/cache")
    return path


def load_baseline(root):
    metrics = Path(root) / "results/metrics"
    summary = json.loads((metrics / "baseline_summary.json").read_text())
    if summary.get("status") != "completed":
        raise ValueError("Complete Commit 5 before the resolution experiment")
    for name in ("baseline_scores.csv", "baseline_thresholds.json", "baseline_images.json"):
        if sha256(metrics / name) != summary["output_sha256"][name]:
            raise ValueError(f"Baseline output hash mismatch: {name}")
    thresholds = json.loads((metrics / "baseline_thresholds.json").read_text())
    if thresholds["model_fingerprint"] != summary["model_fingerprint"] or thresholds["cache_directory"] != summary["cache_directory"]:
        raise ValueError("Baseline model/cache records disagree")
    with (metrics / "baseline_scores.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("pair_index", "fold", "label"):
            row[key] = int(row[key])
        row["cosine_similarity"] = float(row["cosine_similarity"]) if row["cosine_similarity"] else None
    eligible = [r["pair_index"] for r in rows if r["status"] == "scored"]
    if len(rows) != summary["requested_pairs"] or len({r["pair_index"] for r in rows}) != len(rows):
        raise ValueError("Baseline pair count/index mismatch")
    if eligible != thresholds["eligible_pair_indices"] or len(eligible) != summary["aggregate"]["scored_pairs"]:
        raise ValueError("Baseline eligibility mismatch")
    if {r["fold"] for r in rows} != set(range(10)) or set(thresholds["thresholds"]) != {str(n) for n in range(10)}:
        raise ValueError("Expected the 10 supplied folds")
    cache = inside(root, thresholds["cache_directory"])
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["fingerprint"] != summary["model_fingerprint"]:
        raise ValueError("Baseline cache fingerprint mismatch")
    records = json.loads((metrics / "baseline_images.json").read_text())
    records = {r["path"]: r for r in records}
    paths = sorted({r[k] for r in rows if r["status"] == "scored" for k in ("reference", "probe")})
    embeddings = {}
    for relative in paths:
        record = records[relative]
        if record["status"] != "ok" or manifest["images"].get(relative) != record:
            raise ValueError(f"Baseline image record mismatch: {relative}")
        payload = inside(cache, record["payload"])
        if sha256(payload) != record["payload_sha256"]:
            raise ValueError(f"Baseline crop/embedding cache is corrupt: {relative}")
        with np.load(payload, allow_pickle=False) as data:
            embeddings[relative] = validate_embedding(data["embedding"]).copy()
            reduce_resolution(data["crop"], 160)  # validates shape/range without modifying the source
    # Recompute control scores from cached embeddings, never from rounded metrics.
    for row in rows:
        if row["status"] == "scored":
            score = cosine_similarity(embeddings[row["reference"]], embeddings[row["probe"]])
            if not np.isclose(score, row["cosine_similarity"], atol=1e-7, rtol=0):
                raise ValueError("Cached 160px control score differs from the baseline")
    return summary, thresholds, rows, cache, records, embeddings


def evaluate_fixed(rows, scores, thresholds, size):
    """Use saved thresholds only; never call select_threshold/cross_validate."""
    eligible = {r["pair_index"] for r in rows if r["status"] == "scored"}
    if set(scores) != eligible:
        raise ValueError("Every condition must score exactly the baseline eligible pairs")
    folds, predictions = [], []
    for fold in range(10):
        requested = [r for r in rows if r["fold"] == fold]
        selected = [r for r in requested if r["status"] == "scored"]
        threshold = float(thresholds[str(fold)])
        result = verification_metrics([scores[r["pair_index"]] for r in selected], [r["label"] for r in selected], threshold)
        result.update({"resolution": size, "fold": fold, "threshold": threshold,
                       "requested_pairs": len(requested), "excluded_pairs": len(requested) - len(selected),
                       "coverage": len(selected) / len(requested)})
        folds.append(result)
        for row in requested:
            score = scores.get(row["pair_index"])
            prediction = int(score >= threshold) if score is not None else None
            predictions.append({"resolution": size, **row, "cosine_similarity": score,
                                "threshold": threshold, "prediction": prediction,
                                "correct": int(prediction == row["label"]) if prediction is not None else None})
    aggregate = {"resolution": size, "requested_pairs": len(rows), "scored_pairs": len(eligible),
                 "excluded_pairs": len(rows) - len(eligible), "coverage": len(eligible) / len(rows)}
    for key in ("accuracy", "fmr", "fnmr"):
        values = np.array([r[key] for r in folds])
        aggregate[f"mean_{key}"] = float(values.mean())
        aggregate[f"sd_{key}"] = float(values.std(ddof=1))
        aggregate[f"se_{key}"] = float(values.std(ddof=1) / np.sqrt(10))
    for key in ("false_matches", "false_nonmatches"):
        aggregate[key] = sum(r[key] for r in folds)
    return aggregate, folds, sorted(predictions, key=lambda r: r["pair_index"])


def verify_model(root, fingerprint):
    import facenet_pytorch
    actual = {"config": CONFIG, "helper_sha256": sha256(Path(root) / "src/face_embedding.py"),
              "checkpoint_sha256": sha256(Path(root) / "models/checkpoints/20180402-114759-vggface2.pt"),
              "packages": {p: importlib.metadata.version(p) for p in fingerprint["packages"]},
              "python": platform.python_version(), "platform": platform.platform(),
              "detector_weights": {p.name: sha256(p) for p in sorted((Path(facenet_pytorch.__file__).parent / "data").glob("*net.pt"))}}
    if actual != fingerprint:
        raise ValueError("Environment/model differs from the baseline. Restore the baseline environment before comparing conditions.")


def run_resolution(root):
    root = Path(root).resolve()
    metric_dir = root / "results/metrics"
    start = time.perf_counter()
    report_path = metric_dir / "resolution_summary.json"
    report = {"status": "started", "run_at_utc": datetime.now(timezone.utc).isoformat(),
              "degradation": RESOLUTION_CONFIG, "conditions": [],
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
                                    ("degradation.py", "resolution_experiment.py", "face_embedding.py", "evaluation.py")}
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
            check = validate_embedding(pipeline.embed(reduce_resolution(data["crop"], 160)))
        max_diff = float(np.max(np.abs(check - reference_embeddings[probe_paths[0]])))
        report["control_embedding_max_abs_difference"] = max_diff
        if not np.allclose(check, reference_embeddings[probe_paths[0]], atol=1e-6, rtol=0):
            raise ValueError("160px forward check differs from the baseline embedding")
        signature = hashlib.sha256(json.dumps({"baseline": report["baseline_thresholds_sha256"],
            "model": baseline["model_fingerprint"], "code": report["source_sha256"], "degradation": RESOLUTION_CONFIG}, sort_keys=True).encode()).hexdigest()
        output_cache = root / "data/processed/resolution" / signature
        output_cache.mkdir(parents=True, exist_ok=True)
        report["cache_directory"] = output_cache.relative_to(root).as_posix()
        all_folds, all_predictions = [], []
        for size in RESOLUTIONS:
            condition_start = time.perf_counter()
            reused = 0
            if size == 160:
                probe_embeddings = reference_embeddings
            else:
                probe_embeddings = {}
                for index, relative in enumerate(probe_paths, 1):
                    record = records[relative]
                    stem = hashlib.sha256((relative + record["payload_sha256"]).encode()).hexdigest()
                    directory = output_cache / str(size)
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
                            degraded = reduce_resolution(data["crop"], size)
                        vector = validate_embedding(pipeline.embed(degraded))
                        temporary = payload.with_suffix(".tmp")
                        with temporary.open("wb") as stream:
                            np.save(stream, vector, allow_pickle=False)
                        temporary.replace(payload)
                        write_json(receipt, {"path": relative, "resolution": size, "sha256": sha256(payload),
                                             "baseline_payload_sha256": record["payload_sha256"]})
                    probe_embeddings[relative] = vector
                    if index % 100 == 0 or index == len(probe_paths):
                        print(f"{size}px: {index}/{len(probe_paths)} probes | reused {reused}", flush=True)
            scores = {r["pair_index"]: cosine_similarity(reference_embeddings[r["reference"]], probe_embeddings[r["probe"]])
                      for r in rows if r["status"] == "scored"}
            aggregate, folds, predictions = evaluate_fixed(rows, scores, saved["thresholds"], size)
            if size == 160:
                for key in ("mean_accuracy", "mean_fmr", "mean_fnmr"):
                    if not np.isclose(aggregate[key], baseline["aggregate"][key], atol=1e-12, rtol=0):
                        raise ValueError("160px control metrics do not reproduce baseline")
                for key in ("false_matches", "false_nonmatches"):
                    if aggregate[key] != baseline["aggregate"]["confusion_counts"][key]:
                        raise ValueError("160px control confusion counts differ from baseline")
            for key in ("accuracy", "fmr", "fnmr"):
                aggregate[f"delta_{key}_pp"] = 100 * (aggregate[f"mean_{key}"] - baseline["aggregate"][f"mean_{key}"])
            aggregate["reused_probe_embeddings"] = reused
            aggregate["elapsed_seconds"] = round(time.perf_counter() - condition_start, 3)
            report["conditions"].append(aggregate)
            all_folds.extend(folds)
            all_predictions.extend(predictions)
            write_json(report_path, report)
            print(f"{size}px: accuracy {100*aggregate['mean_accuracy']:.3f}%, FMR {100*aggregate['mean_fmr']:.3f}%, FNMR {100*aggregate['mean_fnmr']:.3f}%", flush=True)
        write_csv(metric_dir / "resolution_comparison.csv", report["conditions"])
        write_csv(metric_dir / "resolution_folds.csv", all_folds)
        write_csv(metric_dir / "resolution_predictions.csv", all_predictions)
        report["status"] = "completed"
        report["output_sha256"] = {name: sha256(metric_dir / name) for name in
                                    ("resolution_comparison.csv", "resolution_folds.csv", "resolution_predictions.csv")}
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        write_json(report_path, report)
    return report


if __name__ == "__main__":
    run_resolution(Path(__file__).resolve().parents[1])
