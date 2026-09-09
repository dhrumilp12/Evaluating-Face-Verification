"""Download and inspect LFW funneled images without loading all pairs into RAM.

Source URLs and SHA-256 values: scikit-learn/sklearn/datasets/_lfw.py.
No model training, face recognition, or verification scoring is performed here.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import ssl
import subprocess
import tarfile
import urllib.request

import certifi
from PIL import Image

SOURCES = {
    "lfw-funneled.tgz": ("5976015", "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a"),
    "pairsDevTrain.txt": ("5976012", "1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa"),
    "pairsDevTest.txt": ("5976009", "7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c"),
    "pairs.txt": ("5976006", "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"),
}
PROTOCOL = {"pairsDevTrain.txt": (1, 1100), "pairsDevTest.txt": (1, 500), "pairs.txt": (10, 300)}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def download_lfw(root):
    """Cache verified inputs. Re-running skips downloads with matching hashes."""
    home = Path(root) / "data" / "lfw_home"
    home.mkdir(parents=True, exist_ok=True)
    manifest_path = home / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Supplement system roots for Python.org macOS installs without a CA bundle.
    tls_context = ssl.create_default_context()
    tls_context.load_verify_locations(cafile=certifi.where())
    for name, (file_id, expected) in SOURCES.items():
        target = home / name
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        downloaded_at = manifest.get(name, {}).get("downloaded_at_utc")
        if not target.exists() or sha256(target) != expected:
            temporary = target.with_name(target.name + ".part")
            print(f"Downloading {name} ...", flush=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "BiometricsCourseProject/1.0"})
                with urllib.request.urlopen(request, timeout=60, context=tls_context) as response, temporary.open("wb") as output:
                    received = 0
                    next_update = 25 * 1024 * 1024
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                        if received >= next_update:
                            print(f"  {received / 1024**2:.0f} MiB received", flush=True)
                            next_update += 25 * 1024 * 1024
                if sha256(temporary) != expected:
                    raise ValueError(f"Checksum mismatch for {name}; file was not accepted.")
                temporary.replace(target)
                downloaded_at = utc_now()
            except Exception as error:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Download failed for {name}: {error}. Check the connection and rerun.") from error
        manifest[name] = {"url": url, "sha256": expected, "bytes": target.stat().st_size,
                          "downloaded_at_utc": downloaded_at, "verified_at_utc": utc_now()}
        save_json(manifest_path, manifest)
        print(f"Verified {name}", flush=True)

    # Extract individual regular files instead of trusting arbitrary archive paths.
    # Existing, modified, or truncated archive members are restored on each run.
    print("Extracting verified archive ...", flush=True)
    with tarfile.open(home / "lfw-funneled.tgz", "r:gz") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "lfw_funneled":
                raise ValueError(f"Unexpected archive path: {member.name}")
            target = home.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(home.resolve()):
                raise ValueError("Archive path escapes the dataset directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")
    return home


def image_name(identity, index):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", identity) or identity in (".", ".."):
        raise ValueError(f"Invalid identity folder: {identity}")
    number = int(index)
    if number < 1:
        raise ValueError("Image indices must be positive")
    return f"{identity}/{identity}_{number:04d}.jpg"


def parse_pairs(path):
    """Keep source order; labels are 1=same, 0=different; folds are 0-based."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty pair file: {path.name}")
    header = list(map(int, lines[0].split()))
    if len(header) == 1:
        folds, per_class = 1, header[0]
    elif len(header) == 2:
        folds, per_class = header
    else:
        raise ValueError("Expected one or two integers in the pair header")
    if folds < 1 or per_class < 1 or len(lines) - 1 != folds * 2 * per_class:
        raise ValueError(f"Pair count disagrees with header in {path.name}")
    if path.name in PROTOCOL and (folds, per_class) != PROTOCOL[path.name]:
        raise ValueError(f"Unexpected official protocol header: {path.name}")
    pairs = []
    for row, line in enumerate(lines[1:]):
        fields = line.split()
        expected_same = row % (2 * per_class) < per_class
        if len(fields) == 3:
            identity, first, second = fields
            reference, probe = image_name(identity, first), image_name(identity, second)
            label = 1
            if reference == probe:
                raise ValueError("A genuine pair must contain two different images")
        elif len(fields) == 4:
            first_id, first, second_id, second = fields
            if first_id == second_id:
                raise ValueError("An impostor pair must contain different identities")
            reference, probe = image_name(first_id, first), image_name(second_id, second)
            label = 0
        else:
            raise ValueError(f"Malformed pair at line {row + 2} in {path.name}")
        if label != int(expected_same):
            raise ValueError(f"Unexpected class order at line {row + 2} in {path.name}")
        pairs.append({"pair_index": row, "fold": row // (2 * per_class),
                      "reference": reference, "probe": probe, "label": label})
    return pairs


def inspect_lfw(root):
    """Decode one image at a time; write measured results before enforcing gates."""
    root = Path(root)
    home = root / "data" / "lfw_home"
    images_root = home / "lfw_funneled"
    files = sorted(images_root.glob("*/*.jpg"))
    if not files:
        raise FileNotFoundError("No LFW JPEGs found. Run download_lfw(ROOT) first.")
    counts, dimensions, modes = Counter(), Counter(), Counter()
    unreadable = []
    for number, path in enumerate(files, 1):
        counts[path.parent.name] += 1
        try:
            with Image.open(path) as picture:
                picture.load()
                dimensions[f"{picture.width}x{picture.height}"] += 1
                modes[picture.mode] += 1
        except Exception as error:
            unreadable.append({"path": path.relative_to(images_root).as_posix(), "error": str(error)})
        if number % 3000 == 0:
            print(f"Inspected {number}/{len(files)} images", flush=True)
    pair_sets, pair_summary, missing = {}, {}, []
    for name in PROTOCOL:
        pairs = parse_pairs(home / name)
        pair_sets[name] = pairs
        for pair in pairs:
            for role in ("reference", "probe"):
                if not (images_root / pair[role]).is_file():
                    missing.append({"file": name, "pair_index": pair["pair_index"], "path": pair[role]})
        pair_summary[name] = {"total": len(pairs), "same": sum(p["label"] for p in pairs),
                              "different": sum(1 - p["label"] for p in pairs),
                              "folds": len({p["fold"] for p in pairs})}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    versions = {}
    for package in ("Pillow", "matplotlib", "jupyterlab", "ipykernel", "certifi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    checks = {"expected_image_count": len(files) == 13233,
              "expected_identity_count": len(counts) == 5749,
              "all_images_250x250": dimensions == {"250x250": len(files)},
              "all_images_readable": not unreadable, "all_pair_paths_exist": not missing}
    result = {"run_at_utc": utc_now(), "image_version": "LFW funneled",
              "git_head_before_commit": head, "python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "image_count": len(files), "identity_count": len(counts),
              "identities_with_two_or_more_images": sum(n >= 2 for n in counts.values()),
              "images_per_identity_min": min(counts.values()), "images_per_identity_max": max(counts.values()),
              "dimensions": dict(dimensions), "color_modes": dict(modes), "pairs": pair_summary,
              "unreadable_images": unreadable, "missing_pair_references": missing,
              "checks": checks, "all_checks_passed": all(checks.values()),
              "helper_sha256": sha256(Path(__file__))}
    manifest = home / "download_manifest.json"
    if manifest.exists():
        result["download_manifest"] = json.loads(manifest.read_text())
    output = root / "results" / "metrics" / "dataset_summary.json"
    save_json(output, result)
    print(json.dumps({k: result[k] for k in ("image_count", "identity_count", "dimensions", "color_modes", "pairs", "checks", "all_checks_passed")}, indent=2))
    print("Saved results/metrics/dataset_summary.json")
    if not all(checks.values()):
        raise ValueError("Dataset validation failed. Inspect dataset_summary.json before continuing.")
    return result, counts, pair_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.download:
        download_lfw(project_root)
    inspect_lfw(project_root)
