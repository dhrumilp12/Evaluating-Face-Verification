"""Exercise the embedding pipeline on a fixed, balanced development subset."""
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image

from lfw_dataset import parse_pairs, sha256, save_json, SOURCES
from face_embedding import FaceEmbedder, CONFIG, MODEL_URL, cosine_similarity


def select_development_pairs(pairs, per_class=10):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    same = [p for p in pairs if p["label"] == 1][:per_class]
    different = [p for p in pairs if p["label"] == 0][:per_class]
    if len(same) != per_class or len(different) != per_class:
        raise ValueError("Insufficient development pairs")
    return same + different


def run_smoke_test(root):
    root = Path(root).resolve()
    summary_file = root / "results/metrics/dataset_summary.json"
    if not summary_file.exists():
        raise RuntimeError("Complete Commit 3: dataset_summary.json is missing.")
    dataset_summary = json.loads(summary_file.read_text())
    if dataset_summary.get("all_checks_passed") is not True:
        raise RuntimeError("Commit 3 dataset checks did not pass. Resolve them first.")
    home = root / "data/lfw_home"
    pair_file = home / "pairsDevTrain.txt"
    if sha256(pair_file) != SOURCES["pairsDevTrain.txt"][1]:
        raise ValueError("Development-pair checksum differs from the official source")
    pairs = select_development_pairs(parse_pairs(pair_file))
    paths = sorted({p[role] for p in pairs for role in ("reference", "probe")})
    for relative in paths:
        if not (home / "lfw_funneled" / relative).is_file():
            raise FileNotFoundError(f"Missing image: {relative}; rerun Commit 3 validation.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    cache = root / "data/processed/commit4" / run_id
    cache.mkdir(parents=True)
    report_path = root / "results/metrics/embedding_smoke_test.json"
    report = {"run_id": run_id, "status": "started", "config": CONFIG,
              "subset": "first 10 same-person and first 10 different-person development-training pairs",
              "requested_pairs": len(pairs), "unique_images": len(paths),
              "dataset_summary_sha256": sha256(summary_file), "pair_file_sha256": sha256(pair_file),
              "cache_directory": cache.relative_to(root).as_posix(),
              "python": platform.python_version(), "platform": platform.platform(),
              "model_source_url": MODEL_URL, "images": [], "pairs": [],
              "training_overlap": "VGGFace2/LFW identity or image overlap has not been independently audited.",
              "source_sha256": {name: sha256(root / "src" / name)
                                 for name in ("lfw_dataset.py", "face_embedding.py", "embedding_smoke_test.py")}}
    start = time.perf_counter()
    try:
        report["git_head_before_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_head_before_commit"] = None
    save_json(report_path, report)
    try:
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ("facenet-pytorch", "torch", "torchvision", "numpy", "Pillow")}
        print("Loading MTCNN and VGGFace2-pretrained FaceNet on CPU ...", flush=True)
        pipeline = FaceEmbedder(root)
        checkpoint = root / "models/checkpoints/20180402-114759-vggface2.pt"
        report["checkpoint_sha256"] = sha256(checkpoint)
        # This records observed bytes, not a comparison to a publisher-supplied hash.
        import facenet_pytorch
        package_data = Path(facenet_pytorch.__file__).parent / "data"
        report["detector_weight_sha256"] = {p.name: sha256(p) for p in sorted(package_data.glob("*net.pt"))}
        embeddings = {}
        first_raw = first_embedding = None
        for number, relative in enumerate(paths):
            path = home / "lfw_funneled" / relative
            record = {"path": relative, "input_sha256": sha256(path)}
            try:
                with Image.open(path) as picture:
                    picture.load()
                    rgb = picture.convert("RGB")
            except (OSError, ValueError) as error:
                record.update({"status": "image_read_error", "error": str(error)})
                report["images"].append(record)
                continue
            raw, detection = pipeline.crop(rgb)
            record.update(detection)
            if raw is not None:
                vector = pipeline.embed(raw)
                crop_path = cache / f"crop_{number:03d}.npy"
                np.save(crop_path, raw, allow_pickle=False)
                embeddings[relative] = vector
                record.update({"crop_path": crop_path.relative_to(root).as_posix(),
                               "embedding_norm": float(np.linalg.norm(vector))})
                if first_raw is None:
                    first_raw, first_embedding = raw.copy(), vector.copy()
            report["images"].append(record)
            print(f"{number + 1}/{len(paths)}: {record['status']}", flush=True)
        for pair in pairs:
            row = dict(pair)
            eligible = pair["reference"] in embeddings and pair["probe"] in embeddings
            row["status"] = "scored" if eligible else "preprocessing_failed"
            row["cosine_similarity"] = cosine_similarity(embeddings[pair["reference"]], embeddings[pair["probe"]]) if eligible else None
            report["pairs"].append(row)

        names = list(embeddings)
        np.save(cache / "embeddings.npy", np.stack([embeddings[n] for n in names]) if names else np.empty((0, 512), dtype=np.float32), allow_pickle=False)
        save_json(cache / "embedding_paths.json", names)
        scored = [p for p in report["pairs"] if p["status"] == "scored"]
        report.update({"successful_images": len(embeddings), "failed_images": len(paths) - len(embeddings),
                       "scored_pairs": len(scored), "excluded_pairs": len(pairs) - len(scored),
                       "scored_same_pairs": sum(p["label"] == 1 for p in scored),
                       "scored_different_pairs": sum(p["label"] == 0 for p in scored)})
        if first_raw is None:
            raise RuntimeError("No usable face crops; inspect detection records.")
        repeat = pipeline.embed(first_raw)
        report["repeat_max_abs_difference"] = float(np.max(np.abs(first_embedding - repeat)))
        report["self_similarity"] = cosine_similarity(first_embedding, first_embedding)
        report["checks"] = {
            "finite_unit_512d_embeddings": len(embeddings) > 0,
            "repeat_embedding_close": bool(np.allclose(first_embedding, repeat, atol=1e-6, rtol=0)),
            "self_similarity_near_one": bool(abs(report["self_similarity"] - 1) < 1e-5),
            "both_pair_classes_scored": report["scored_same_pairs"] > 0 and report["scored_different_pairs"] > 0,
        }
        report["checks_passed"] = all(report["checks"].values())
        if not report["checks_passed"]:
            raise RuntimeError("Embedding checks failed. Inspect the report before continuing.")
        report["status"] = "completed"
        report["interpretation"] = "Small pipeline check only. No decision threshold, accuracy, FMR, or FNMR estimated. Inspect the selected crops visually."
    except Exception as error:
        report.update({"status": "failed", "checks_passed": False,
                       "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        save_json(report_path, report)
        csv_path = report_path.with_name("embedding_smoke_pairs.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            fields = ["pair_index", "fold", "reference", "probe", "label", "status", "cosine_similarity"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report["pairs"])
        print("Saved results/metrics/embedding_smoke_test.json", flush=True)
    return report


if __name__ == "__main__":
    result = run_smoke_test(Path(__file__).resolve().parents[1])
    print(json.dumps({key: result[key] for key in ("status", "successful_images", "failed_images", "scored_pairs", "excluded_pairs", "checks")}, indent=2))
