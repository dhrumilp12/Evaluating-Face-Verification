#!/usr/bin/env python3
"""Reanalyze verified Project 2 scores into a new directory, preserving the run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.project2_experiment import analyze, write_json  # noqa: E402
from src.lfw_dataset import sha256  # noqa: E402


def reanalyze(source, output):
    """Require a completed, hash-consistent run and an empty separate destination."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Reanalysis output must be outside the source run directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Reanalysis output must be a new or empty directory; choose another --output")
    manifest_file, scores_file = source / "manifest.json", source / "pair_scores.jsonl"
    source_manifest_bytes = manifest_file.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    if source_manifest.get("status") != "completed":
        raise ValueError("Source inference manifest must have status='completed'")
    expected = source_manifest.get("output_sha256", {}).get("pair_scores.jsonl")
    if not isinstance(expected, str) or len(expected) != 64 or sha256(scores_file) != expected:
        raise ValueError("Saved pair_scores.jsonl does not match the completed inference manifest")
    started = time.perf_counter()
    analysis_sources = ("scripts/project2_reanalyze.py", "src/project2_experiment.py",
                        "src/project2_evaluation.py", "src/evaluation.py", "src/score_analysis.py")
    manifest = {
        "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved-score evaluation only; no image alignment or embedding inference",
        "source_run_directory": str(source),
        "source_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "source_pair_scores_sha256": expected,
        "analysis_source_sha256": {name: sha256(ROOT / name) for name in analysis_sources},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("numpy", "matplotlib", "Pillow", "certifi")}},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis_manifest.json", manifest)
    try:
        (output / "source_manifest.json").write_bytes(source_manifest_bytes)
        shutil.copyfile(scores_file, output / "pair_scores.jsonl")
        if sha256(output / "pair_scores.jsonl") != expected:
            raise ValueError("Source scores changed during the copy; reanalysis was not performed")
        analyze(output)
        # A source edit during execution would invalidate attribution to this snapshot.
        if any(sha256(ROOT / name) != digest for name, digest in manifest["analysis_source_sha256"].items()):
            raise ValueError("Analysis source files changed while reanalysis was running")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started)
        manifest["output_sha256"] = {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in ("analysis_manifest.json", "analysis_manifest.json.part")
        }
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        write_json(output / "analysis_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/project2/robustness",
                        help="Completed inference run containing manifest.json and pair_scores.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results/project2/reanalysis",
                        help="New or empty output directory outside the source run")
    args = parser.parse_args()
    manifest = reanalyze(args.source, args.output)
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()),
                      "source_pair_scores_sha256": manifest["source_pair_scores_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
