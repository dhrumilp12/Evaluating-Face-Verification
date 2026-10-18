"""Reproducible MagFace/QMagFace robustness experiment, with resumable raw caches.

Run from the repository root: python -m src.project2_experiment --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from .lfw_dataset import parse_pairs, sha256
from .project2_degradation import CONDITIONS, MINIMUM_CONDITIONS, CONFIG as TRANSFORMS, apply_condition
from .qmagface_scoring import QMAGFACE_100_ALPHA, QMAGFACE_100_BETA, score_embedding_pairs


ROOT = Path(__file__).resolve().parents[1]
# Observed SHA-256 of the authors' linked checkpoint, not a publisher checksum.
CHECKPOINT_SHA256 = "cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "Pillow", "torch", "torchvision", "facenet-pytorch", "matplotlib")}}


def unique_paths(pairs):
    return sorted({pair[role] for pair in pairs for role in ("reference", "probe")})


def prepare_alignment(root, paths, *, threads=2):
    """Revalidate source/crop bytes before reusing a completed alignment record."""
    from .magface_embedding import MagFaceAligner
    import facenet_pytorch

    identity = {"adapter_sha256": sha256(Path(__file__).with_name("magface_embedding.py")),
                "selection_sha256": sha256(Path(__file__).with_name("face_embedding.py")),
                "detector_sha256": {name: sha256(Path(facenet_pytorch.__file__).parent / "data" / name)
                                    for name in ("pnet.pt", "rnet.pt", "onet.pt")},
                "environment": environment(), "threads": threads, "device": "cpu"}
    directory = root / "data/processed/project2/alignment" / fingerprint(identity)[:16]
    record_file = directory / "records.json"
    saved = json.loads(record_file.read_text()) if record_file.exists() else {"identity": identity, "records": {}}
    records = saved["records"]
    detector = None
    started = time.perf_counter()
    reused = 0
    try:
        for index, relative in enumerate(paths):
            source = root / "data/lfw_home/lfw_funneled" / relative
            source_hash = sha256(source)
            previous = records.get(relative)
            crop_file = directory / "crops" / Path(relative).with_suffix(".png")
            valid = previous and previous.get("source_sha256") == source_hash
            if valid and previous["status"] == "ok":
                valid = crop_file.is_file() and sha256(crop_file) == previous.get("crop_sha256")
            if valid:
                reused += 1
                continue
            if detector is None:
                detector = MagFaceAligner(device="cpu", threads=threads)
            with Image.open(source) as picture:
                crop, record = detector.align(picture.convert("RGB"))
            record = {**record, "source_sha256": source_hash}
            if crop is not None:
                crop_file.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_file, format="PNG")
                record.update(status="ok", crop_sha256=sha256(crop_file),
                              crop_path=crop_file.relative_to(root).as_posix())
            records[relative] = record
            if (index + 1) % 100 == 0:
                write_json(record_file, saved)
                print(f"Aligned {index + 1}/{len(paths)} images ({time.perf_counter() - started:.1f}s)", flush=True)
    finally:
        write_json(record_file, saved)
    print(f"Alignment ready: {len(paths)} requested, {reused} reused, "
          f"{sum(records[p]['status'] != 'ok' for p in paths)} failures", flush=True)
    return {path: records[path] for path in paths}, record_file


def read_feature(path, expected):
    """Reject mismatched provenance and validate raw/unit/quality consistency."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["input_sha256"].item()) != expected:
            return None
        raw = np.asarray(saved["raw"], dtype=np.float32)
        unit = np.asarray(saved["unit"], dtype=np.float64)
        quality = float(saved["quality"].item())
    if raw.shape != (512,) or unit.shape != (512,) or not np.isfinite(raw).all():
        raise ValueError(f"Corrupt feature cache: {path}")
    measured = float(np.linalg.norm(raw.astype(np.float64)))
    if not np.isfinite(quality) or quality <= 0 or not np.isclose(quality, measured, rtol=1e-10):
        raise ValueError(f"Corrupt cached quality: {path}")
    if not np.allclose(unit, raw.astype(np.float64) / quality, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Corrupt cached unit embedding: {path}")
    return raw


def save_feature(path, raw, expected):
    raw = np.asarray(raw, dtype=np.float32)
    quality = float(np.linalg.norm(raw.astype(np.float64)))
    if raw.shape != (512,) or not np.isfinite(raw).all() or not np.isfinite(quality) or quality <= 0:
        raise ValueError("Cannot cache an invalid raw embedding")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, raw=raw, quality=quality, unit=raw.astype(np.float64) / quality,
                            input_sha256=np.asarray(expected))
    temporary.replace(path)


def extract_condition(root, model, records, paths, condition, directory, *, batch_size=32):
    features, pending, images = {}, [], []
    computed = 0
    started = time.perf_counter()

    def flush():
        nonlocal computed
        if not images:
            return
        outputs = model.embed_batch(images)
        if outputs.shape != (len(images), 512):
            raise ValueError("Unexpected inference batch shape")
        for (relative, target, expected), raw in zip(pending, outputs):
            save_feature(target, raw, expected)
            features[relative] = raw
        computed += len(images)
        pending.clear()
        images.clear()
        if computed % (batch_size * 10) == 0:
            print(f"{condition}: inferred {computed} new images ({time.perf_counter() - started:.1f}s)", flush=True)

    for relative in paths:
        record = records[relative]
        if record["status"] != "ok":
            continue
        target = directory / condition / Path(relative).with_suffix(".npz")
        expected = record["crop_sha256"]
        cached = read_feature(target, expected)
        if cached is not None:
            features[relative] = cached
            continue
        with Image.open(root / record["crop_path"]) as crop:
            images.append(apply_condition(crop.convert("RGB"), condition))
        pending.append((relative, target, expected))
        if len(images) == batch_size:
            flush()
    flush()
    elapsed = time.perf_counter() - started
    print(f"{condition}: {len(features)} features, {computed} new, {elapsed:.1f}s", flush=True)
    return features, {"condition": condition, "requested_images": len(paths), "features": len(features),
                      "new_features": computed, "seconds": elapsed}


def pair_rows(pairs, records, reference_features, probe_features, condition):
    rows = []
    for pair in pairs:
        ref, probe = pair["reference"], pair["probe"]
        row = {"pair_index": pair["pair_index"], "fold": pair["fold"], "label": pair["label"],
               "reference_path": ref, "probe_path": probe, "condition": condition,
               "status": "failed", "error": "", "cosine_similarity": None, "qmagface_score": None,
               "reference_quality": None, "probe_quality": None}
        if ref not in reference_features or probe not in probe_features:
            row["error"] = "; ".join(f"{role}:{records[path]['status']}" for role, path in
                                     (("reference", ref), ("probe", probe)) if records[path]["status"] != "ok")
            if not row["error"]:
                raise ValueError("An eligible embedding is missing")
        else:
            score = score_embedding_pairs(reference_features[ref][None], probe_features[probe][None],
                                           alpha=QMAGFACE_100_ALPHA, beta=QMAGFACE_100_BETA)
            row.update(status="scored", **{name: float(getattr(score, name)[0]) for name in
                                          ("cosine_similarity", "qmagface_score", "reference_quality", "probe_quality")})
        rows.append(row)
    return rows


def source_hashes():
    names = ("project2_experiment.py", "project2_degradation.py", "project2_evaluation.py",
             "magface_embedding.py", "qmagface_scoring.py", "lfw_dataset.py", "evaluation.py", "score_analysis.py",
             "face_embedding.py", "vendor/magface_iresnet.py")
    return {f"src/{name}": sha256(Path(__file__).parent / name) for name in names}


def analyze(output):
    from .project2_evaluation import evaluate_conditions, make_plots
    import csv

    rows = [json.loads(line) for line in (output / "pair_scores.jsonl").read_text().splitlines()]
    result = evaluate_conditions(rows)
    write_json(output / "evaluation.json", {key: value for key, value in result.items()
                                           if key not in ("predictions", "curves")})
    for name in ("folds", "aggregates", "paired_changes", "predictions", "curves", "distributions", "condition_changes"):
        values = result[name]
        if values:
            with (output / f"{name}.csv").open("w", newline="") as stream:
                fields = list(dict.fromkeys(key for row in values for key in row))
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
    make_plots(result, output / "figures")
    return result


def run(root, args):
    from .magface_embedding import MagFaceEmbedder, CONFIG as MAGFACE_CONFIG

    output = (root / args.output).resolve()
    pairs_file = root / "data/lfw_home/pairs.txt"
    pairs = parse_pairs(pairs_file)
    conditions = CONDITIONS if args.grid == "full" else MINIMUM_CONDITIONS
    settings = {"conditions": list(conditions), "transforms": TRANSFORMS,
                "model_and_alignment": MAGFACE_CONFIG,
                "alpha": QMAGFACE_100_ALPHA, "beta": QMAGFACE_100_BETA,
                "device": args.device, "batch_size": args.batch_size, "threads": args.threads,
                "pairs_sha256": sha256(pairs_file), "source_sha256": source_hashes(),
                "environment": environment()}
    manifest = {"status": "started", "started_at_utc": utc_now(), "settings": settings}
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    try:
        records, alignment_file = prepare_alignment(root, unique_paths(pairs), threads=args.threads)
        manifest["alignment"] = {"path": alignment_file.relative_to(root).as_posix(), "sha256": sha256(alignment_file),
                                 "requested_images": len(records), "failed_images": sum(r["status"] != "ok" for r in records.values())}
        model = MagFaceEmbedder(root, checkpoint=args.checkpoint, device=args.device, threads=args.threads)
        checkpoint = Path(model.checkpoint)
        settings["checkpoint_sha256"] = sha256(checkpoint)
        if settings["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError("Published QMagFace-100 coefficients require the audited author checkpoint")
        settings["checkpoint_path"] = checkpoint.relative_to(root).as_posix() if checkpoint.is_relative_to(root) else str(checkpoint)
        cache_identity = {"checkpoint": settings["checkpoint_sha256"], "device": args.device,
                          "adapter": settings["source_sha256"]["src/magface_embedding.py"],
                          "architecture": settings["source_sha256"]["src/vendor/magface_iresnet.py"],
                          "transforms": settings["source_sha256"]["src/project2_degradation.py"],
                          "environment": settings["environment"], "batch_size": args.batch_size, "threads": args.threads}
        cache = root / "data/processed/project2/features" / fingerprint(cache_identity)[:16]
        write_json(cache / "identity.json", cache_identity)
        manifest["feature_cache"] = cache.relative_to(root).as_posix()
        all_rows, timings = [], []
        original, timing = extract_condition(root, model, records, unique_paths(pairs), "original", cache, batch_size=args.batch_size)
        timings.append(timing)
        probe_paths = sorted({pair["probe"] for pair in pairs})
        for condition in conditions:
            if condition == "original":
                probe = original
            else:
                probe, timing = extract_condition(root, model, records, probe_paths, condition, cache, batch_size=args.batch_size)
                timings.append(timing)
            all_rows.extend(pair_rows(pairs, records, original, probe, condition))
            manifest["completed_conditions"] = list(dict.fromkeys(r["condition"] for r in all_rows))
            manifest["timings"] = timings
            write_json(output / "manifest.json", manifest)
        with (output / "pair_scores.jsonl").open("w") as stream:
            for row in all_rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "alignment_failures.json", {p: r for p, r in records.items() if r["status"] != "ok"})
        analyze(output)
        manifest.update(status="completed", completed_at_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
        manifest["output_sha256"] = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*"))
                                     if p.is_file() and p.name not in ("manifest.json", "manifest.json.part")}
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return manifest


def pilot(root, args):
    """Time development images disjoint from the evaluation image paths; no labels used."""
    from .magface_embedding import MagFaceEmbedder

    evaluation_paths = set(unique_paths(parse_pairs(root / "data/lfw_home/pairs.txt")))
    development_paths = unique_paths(parse_pairs(root / "data/lfw_home/pairsDevTrain.txt"))
    paths = [p for p in development_paths if p not in evaluation_paths][:16]
    if len(paths) != 16:
        raise ValueError("Need sixteen development images outside the test image set")
    records, alignment_file = prepare_alignment(root, paths, threads=args.threads)
    model = MagFaceEmbedder(root, checkpoint=args.checkpoint, device=args.device, threads=args.threads)
    crops = []
    for path in paths:
        if records[path]["status"] == "ok":
            with Image.open(root / records[path]["crop_path"]) as crop:
                crops.append(crop.convert("RGB"))
    if not crops:
        raise ValueError("All development alignments failed")
    model.embed_batch(crops[:1])  # warm up before timing
    started = time.perf_counter()
    features = model.embed_batch(crops)
    elapsed = time.perf_counter() - started
    quality = np.linalg.norm(features.astype(np.float64), axis=1)
    report = {"status": "completed", "at_utc": utc_now(), "device": args.device,
              "images": paths, "test_image_overlap": 0, "features_shape": list(features.shape),
              "quality_min": float(quality.min()), "quality_max": float(quality.max()),
              "inference_seconds": elapsed, "seconds_per_image": elapsed / len(crops),
              "checkpoint_sha256": sha256(model.checkpoint), "environment": environment(),
              "alignment_manifest": alignment_file.relative_to(root).as_posix(),
              "scope": "Development-only compatibility/throughput check; no verification accuracy or parameter tuning"}
    if args.device == "mps":
        cpu = MagFaceEmbedder(root, checkpoint=args.checkpoint, device="cpu", threads=args.threads)
        cpu_features = cpu.embed_batch(crops[:2])
        delta = np.max(np.abs(cpu_features.astype(np.float64) - features[:2]))
        report["cpu_mps_max_absolute_difference"] = float(delta)
        report["cpu_mps_allclose_rtol_1e3_atol_1e4"] = bool(np.allclose(cpu_features, features[:2], rtol=1e-3, atol=1e-4))
        if not report["cpu_mps_allclose_rtol_1e3_atol_1e4"]:
            raise ValueError("CPU/MPS pilot disagrees beyond the documented tolerance")
    write_json(root / "results/project2/pilot.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("align", "pilot", "run", "analyze"), default="run")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--grid", choices=("full", "minimum"), default="full")
    parser.add_argument("--output", default="results/project2/robustness")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.stage == "align":
        prepare_alignment(ROOT, unique_paths(parse_pairs(ROOT / "data/lfw_home/pairs.txt")), threads=args.threads)
    elif args.stage == "pilot":
        pilot(ROOT, args)
    elif args.stage == "analyze":
        analyze(ROOT / args.output)
    else:
        run(ROOT, args)


if __name__ == "__main__":
    main()
