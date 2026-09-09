# Dataset Setup

## Selected Dataset
Labeled Faces in the Wild (LFW), funneled image version.

Research and selection rationale:
[Dataset research](../docs/dataset_research.md)

## Current Status
Dataset selection is documented.
The archive has not yet been downloaded or verified locally.

## Planned Download Route
Use the LFW download resources referenced by scikit-learn.
Record the actual download URL, date, archive filename, and checksum
when downloading.

Selected archive: lfw-funneled.tgz

Required pair files:
- pairsDevTrain.txt
- pairsDevTest.txt
- pairs.txt

## Loading Requirements
Preserve full image dimensions and RGB channels during initial loading.

The [scikit-learn loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html)
describes defaults that crop, resize, and convert images to grayscale.
If using fetch_lfw_pairs, explicitly configure:
- funneled=True
- color=True
- resize=1.0
- slice_=(slice(0, 250), slice(0, 250))
- data_home="data"

The pair loader materializes the selected subset; it is not a batch
iterator. Do not load all full-resolution pairs into memory at once.
For model evaluation, plan to read image paths in batches and cache
embeddings for repeated images.

The exact executable setup procedure will be added in Commit #3.

## Validation Planned for Commit #3
- Count image files and identity folders.
- Inspect image dimensions and color modes.
- Parse pair files and verify every referenced image exists.
- Check pair counts and same-person/different-person labels.
- Preserve the supplied evaluation folds.
- Record download errors or missing files.

## Git and Usage
Downloaded images and caches stay under data/ and are excluded by
the repository's .gitignore.

Review source usage notices and retain dataset citations.
Do not assume a mirror grants unrestricted rights to the photographs.
