# Dataset Setup

## Selected Dataset
Labeled Faces in the Wild (LFW), funneled image version.

Research and selection rationale:
[Dataset research](../docs/dataset_research.md)

## Current Status
The LFW funneled archive and all three pair files have been downloaded and
verified against the published SHA-256 checksums. All dataset checks passed:
13,233 readable 250 x 250 RGB images, 5,749 identities, and no missing pair paths.

Implemented download and validation instructions:
[Commit 3 setup](../docs/commit3_setup.md).

Measured results: [dataset summary](../results/metrics/dataset_summary.json).

## Download and Local Layout
The helper in src/lfw_dataset.py uses the Figshare resources and SHA-256 checksums
referenced by scikit-learn. Files are stored in data/lfw_home/; extracted images
are under `data/lfw_home/lfw_funneled/<identity>/`.

data/lfw_home/download_manifest.json records actual download URLs, timestamps,
filenames, sizes, and checksums. A copy is included in the measured summary.

Selected archive: lfw-funneled.tgz

Required pair files:
- pairsDevTrain.txt
- pairsDevTest.txt
- pairs.txt

## Loading and Validation
The implementation reads one JPEG at a time with Pillow, preserving source
dimensions and color mode. It does not call fetch_lfw_pairs or apply automatic
cropping, resizing, or grayscale conversion. Display examples are converted to
RGB only for plotting. Model-specific preprocessing remains a later stage.

The notebook checks image and identity counts, image readability and dimensions,
pair counts and class order, and every referenced image path. It preserves the
supplied evaluation folds. The observed pair totals are 2,200 for development
training, 1,000 for development testing, and 6,000 for evaluation across 10 folds.

Run notebooks/01_dataset_exploration.ipynb to reproduce the checks and figures.
The downloader reuses files whose checksums match and restores archive members
on each run. For future model evaluation, read image paths in batches and cache
embeddings for repeated images.

## Git and Usage
Downloaded images and caches stay under data/ and are excluded by
the repository's .gitignore.

Review source usage notices and retain dataset citations.
Do not assume a mirror grants unrestricted rights to the photographs.
