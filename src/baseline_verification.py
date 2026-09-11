"""Resumable original-quality LFW verification using the frozen Commit 4 model."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image

from lfw_dataset import SOURCES, parse_pairs, sha256
from face_embedding import CONFIG, MODEL_URL, FaceEmbedder, cosine_similarity, validate_embedding
from evaluation import cross_validate


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        # Match Git's text normalization so recorded hashes survive a checkout.
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(root):
    root = Path(root).resolve()
    metric_dir = root / "results/metrics"
    prerequisites = [(metric_dir / "dataset_summary.json", "all_checks_passed"),
                     (metric_dir / "embedding_smoke_test.json", "checks_passed")]
    previous = {}
    for path, flag in prerequisites:
        if not path.exists():
            raise RuntimeError(f"Missing {path.name}; finish Commits 3 and 4 first.")
        previous[path.name] = json.loads(path.read_text())
        if previous[path.name].get(flag) is not True:
            raise RuntimeError(f"Resolve failed prerequisite checks in {path.name}.")
    smoke = previous["embedding_smoke_test.json"]
    if smoke.get("status") != "completed" or smoke.get("config") != CONFIG:
        raise RuntimeError("Current preprocessing differs from the completed Commit 4 run. Validate it on development images first.")
    helper_hash = sha256(root / "src/face_embedding.py")
    recorded_hash = smoke.get("source_sha256", {}).get("face_embedding.py")
    if recorded_hash is not None and recorded_hash != helper_hash:
        raise RuntimeError("face_embedding.py changed after Commit 4. Rerun the development pipeline check.")
    pair_path = root / "data/lfw_home/pairs.txt"
    if sha256(pair_path) != SOURCES["pairs.txt"][1]:
        raise RuntimeError("pairs.txt checksum failed; restore the verified original pair file.")
    pairs = parse_pairs(pair_path)
    if len(pairs) != 6000 or {p["fold"] for p in pairs} != set(range(10)):
        raise RuntimeError("Expected 6,000 evaluation pairs in the supplied 10 folds")
    paths = sorted({p[k] for p in pairs for k in ("reference", "probe")})
    image_root = root / "data/lfw_home/lfw_funneled"
    start = time.perf_counter()
    report_path = metric_dir / "baseline_summary.json"
    report = {"status": "started", "run_at_utc": datetime.now(timezone.utc).isoformat(),
              "condition": "original LFW funneled images, no synthetic degradation",
              "config": CONFIG, "requested_pairs": len(pairs), "unique_images": len(paths),
              "pair_file_sha256": sha256(pair_path), "python": platform.python_version(),
              "platform": platform.platform(), "model_source_url": MODEL_URL,
              "source_sha256": {p: sha256(root / "src" / p) for p in
                                ("lfw_dataset.py", "face_embedding.py", "evaluation.py", "baseline_verification.py")},
              "prerequisite_sha256": {p.name: sha256(p) for p, _ in prerequisites},
              "protocol": "Each fold is tested once; threshold maximizes scored-pair accuracy on the other nine folds. Accept when cosine >= threshold. Ties choose the largest candidate threshold.",
              "limitations": ["Metrics condition on successful preprocessing; coverage is reported separately.",
                              "Externally pretrained model; training-data overlap has not been independently audited.",
                              "Pair folds are not asserted to be identity-disjoint.",
                              "Fold standard errors are descriptive; shared training folds and repeated identities limit independence."]}
    try:
        report["git_head_before_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_head_before_commit"] = None
    write_json(report_path, report)
    try:
        print("Loading the fixed face model on CPU ...", flush=True)
        pipeline = FaceEmbedder(root)
        import facenet_pytorch
        detector_files = sorted((Path(facenet_pytorch.__file__).parent / "data").glob("*net.pt"))
        report["packages"] = {p: importlib.metadata.version(p) for p in
                              ("facenet-pytorch", "torch", "torchvision", "numpy", "Pillow", "matplotlib")}
        checkpoint = root / "models/checkpoints/20180402-114759-vggface2.pt"
        fingerprint = {"config": CONFIG, "helper_sha256": helper_hash,
                       "checkpoint_sha256": sha256(checkpoint), "packages": report["packages"],
                       "python": platform.python_version(), "platform": platform.platform(),
                       "detector_weights": {p.name: sha256(p) for p in detector_files}}
        for key, actual in (("checkpoint_sha256", fingerprint["checkpoint_sha256"]),
                            ("detector_weight_sha256", fingerprint["detector_weights"])):
            if key in smoke and smoke[key] != actual:
                raise RuntimeError("Model weights differ from Commit 4; validate on development images first.")
        signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        cache = root / "data/processed/baseline" / signature
        cache.mkdir(parents=True, exist_ok=True)
        manifest_path = cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"fingerprint": fingerprint, "images": {}}
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Cache fingerprint mismatch")
        report["cache_directory"] = cache.relative_to(root).as_posix()
        report["model_fingerprint"] = fingerprint
        report["cache_reused_images"] = 0
        report["freshly_processed_images"] = 0
        embeddings, records = {}, []
        for index, relative in enumerate(paths, 1):
            image_path = image_root / relative
            source_hash = sha256(image_path) if image_path.is_file() else None
            record = manifest["images"].get(relative)
            reused = False
            if record is not None and record.get("input_sha256") == source_hash:
                if record["status"] != "ok":
                    reused = True
                else:
                    payload_path = cache / record["payload"]
                    if payload_path.exists() and sha256(payload_path) == record["payload_sha256"]:
                        try:
                            with np.load(payload_path, allow_pickle=False) as data:
                                vector = validate_embedding(data["embedding"])
                                crop = data["crop"]
                                if crop.shape != (3, 160, 160) or not np.isfinite(crop).all() or crop.min() < 0 or crop.max() > 255:
                                    raise ValueError("Invalid cached crop")
                            embeddings[relative] = vector.copy()
                            reused = True
                        except (OSError, ValueError, KeyError):
                            print(f"Recomputing invalid cached payload: {relative}", flush=True)
            if reused:
                report["cache_reused_images"] += 1
            else:
                report["freshly_processed_images"] += 1
                record = {"path": relative, "input_sha256": source_hash}
                if source_hash is None:
                    record["status"] = "missing_image"
                else:
                    try:
                        with Image.open(image_path) as image:
                            image.load()
                            rgb = image.convert("RGB")
                    except (OSError, ValueError) as error:
                        record.update({"status": "image_read_error", "error": str(error)})
                    else:
                        raw, detection = pipeline.crop(rgb)
                        record.update(detection)
                        if raw is not None:
                            vector = validate_embedding(pipeline.embed(raw))
                            embeddings[relative] = vector
                            filename = hashlib.sha256(relative.encode()).hexdigest() + ".npz"
                            payload_path = cache / filename
                            temporary = payload_path.with_suffix(".tmp")
                            with temporary.open("wb") as stream:
                                np.savez_compressed(stream, crop=raw, embedding=vector)
                            temporary.replace(payload_path)
                            record.update({"payload": filename, "payload_sha256": sha256(payload_path)})
                manifest["images"][relative] = record
            records.append(record)
            # Checkpoint every 25 images; a hard interruption may redo up to 24.
            if index % 25 == 0 or index == len(paths):
                write_json(manifest_path, manifest)
            if index % 50 == 0 or index == len(paths):
                elapsed = time.perf_counter() - start
                print(f"{index}/{len(paths)} images | reused {report['cache_reused_images']} | elapsed {elapsed / 60:.1f} min", flush=True)
        # Cache manifests may contain older paths; export only this run's inputs.
        write_json(metric_dir / "baseline_images.json", records)
        report["successful_images"] = len(embeddings)
        report["failed_images"] = len(paths) - len(embeddings)
        report["image_status_counts"] = dict(Counter(r["status"] for r in records))
        score_rows = []
        for pair in pairs:
            ok = pair["reference"] in embeddings and pair["probe"] in embeddings
            score_rows.append({**pair, "status": "scored" if ok else "preprocessing_failed",
                               "cosine_similarity": cosine_similarity(embeddings[pair["reference"]], embeddings[pair["probe"]]) if ok else None})
        write_csv(metric_dir / "baseline_scores.csv", score_rows)
        folds, predictions, aggregate = cross_validate(score_rows)
        write_csv(metric_dir / "baseline_folds.csv", folds)
        write_csv(metric_dir / "baseline_predictions.csv", predictions)
        write_json(metric_dir / "baseline_thresholds.json", {
            "condition": "original", "rule": "accept if cosine >= threshold", "model_fingerprint": fingerprint,
            "cache_directory": cache.relative_to(root).as_posix(), "pair_file_sha256": sha256(pair_path),
            "thresholds": {str(r["fold"]): r["threshold"] for r in folds},
            "eligible_pair_indices": [r["pair_index"] for r in score_rows if r["status"] == "scored"]})
        report.update({"status": "completed", "aggregate": aggregate, "folds": folds,
                       "interpretation": "Conditional recognition metrics on successfully scored pairs; all requested attempts and preprocessing failures are retained in the exports."})
        for file in ("baseline_images.json", "baseline_scores.csv", "baseline_folds.csv", "baseline_predictions.csv", "baseline_thresholds.json"):
            report.setdefault("output_sha256", {})[file] = sha256(metric_dir / file)
    except BaseException as error:
        report.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        write_json(report_path, report)
        print("Saved results/metrics/baseline_summary.json", flush=True)
    return report


if __name__ == "__main__":
    result = run_baseline(Path(__file__).resolve().parents[1])
    print(json.dumps(result["aggregate"], indent=2))
