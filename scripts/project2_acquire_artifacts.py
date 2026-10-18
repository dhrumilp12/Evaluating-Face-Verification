#!/usr/bin/env python3
"""Fetch and verify the exact public QMagFace reproduction artifacts.

The defaults retrieve about 6.5 GB, with only LFW members extracted from the
author feature archive. Model and datasets remain under gitignored directories.
"""
from __future__ import annotations

import argparse
import json

from project2_benchmark import (
    ROOT, FEATURE_HASHES, SOURCE_HASHES, download_artifacts, download_file, verify_files,
)

CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H&export=download&confirm=t"
CHECKPOINT_HASH = {"magface_epoch_00025.pth": "cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint-only", action="store_true")
    mode.add_argument("--features-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Check existing artifacts without network access")
    args = parser.parse_args()
    observed = {}
    if not args.features_only:
        folder = ROOT / "models/project2"
        checkpoint = folder / "magface_epoch_00025.pth"
        if not args.verify_only and not checkpoint.exists():
            print("Downloading the MagFace-100 MS1MV2 checkpoint (283 MB).", flush=True)
            download_file(CHECKPOINT_URL, checkpoint)
        observed["checkpoint"] = verify_files(folder, CHECKPOINT_HASH)
    if not args.checkpoint_only:
        features = ROOT / "data/project2/author_lfw"
        source = ROOT / "data/project2/vendor/QMagFace-main"
        if not args.verify_only:
            download_artifacts(features, source)
        observed["features"] = verify_files(features, FEATURE_HASHES)
        observed["official_source"] = verify_files(source, SOURCE_HASHES)
    print(json.dumps(observed, indent=2))


if __name__ == "__main__":
    main()
