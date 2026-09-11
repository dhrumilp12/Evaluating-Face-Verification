# Biometrics Project 1
## Evaluating Face Verification Under Degraded Image Quality

### Project Status
The original-quality LFW baseline is complete. The fixed VGGFace2-pretrained
embedding pipeline processed 7,700 of 7,701 unique evaluation images and scored
5,999 of 6,000 pairs across all 10 supplied folds.

| Metric | Measured result |
|---|---:|
| Mean fold accuracy | 99.167% |
| Accuracy standard error | 0.188 percentage points |
| Mean fold FMR | 0.433% |
| Mean fold FNMR | 1.234% |
| Pair coverage | 99.983% |

Recognition rates condition on successful preprocessing. One image failed the
fixed detection-confidence cutoff, excluding one genuine pair. Across scored
pairs, there were 13 false matches and 37 false non-matches. Results and
observations are saved in [the baseline notebook](notebooks/03_baseline_verification.ipynb)
and [experiment log](docs/experiment_log.md). Controlled degradation experiments
have not yet been run.

Setup instructions: [Setup and execution](#setup-and-execution).
Baseline instructions: [Original-quality baseline](#original-quality-baseline).

### Problem Statement
This project studies how image quality affects face verification.

Research question:
How do low resolution, blur, and reduced brightness affect a system's
ability to determine whether two face images belong to the same person?

This is a 1:1 verification task. The approach uses a fixed,
pretrained face recognition model to generate embeddings and compare
pairs of images.

### Biometric Modality
The project uses the face, a physiological biometric modality.

### Pipeline and Scope
Dataset images → face preprocessing → embeddings → matching → decision

The project focuses on:
- Image quality: apply controlled degradation to the probe image.
- Representation: generate embeddings with a pretrained model.
- Matching: compare the reference and probe embeddings.
- Decision: apply a threshold to predict match or non-match.

Face detection is not the main research focus. Preprocessing choices and detection
outcomes are recorded in the embedding report and reviewed before full evaluation.

### Why This Problem Matters
Face verification systems may receive blurry, dim, or low-resolution
images. These conditions can cause a system to reject the correct person
or incorrectly accept a different person. Measuring these errors helps
us understand the system's limitations.

### Candidate Datasets
- Labeled Faces in the Wild (LFW), funneled images: selected primary dataset;
  local download and validation are complete.
- SCface: possible extension using surveillance images.
- QMUL-SurvFace: possible extension using low-resolution surveillance faces.

Dataset sources, access requirements, selection rationale, and the
planned verification protocol are documented in
[Dataset Research](docs/dataset_research.md).

### Planned Experiments
1. Inspect the selected dataset and verification protocol (completed).
2. Establish performance using original images (completed).
3. Reduce probe-image resolution.
4. Apply Gaussian blur to probe images.
5. Reduce probe-image brightness.
6. Compare results and document limitations.

Keep the reference image and recognition model fixed across conditions.
Use development pairs to finalize the model, degradation levels, and
threshold-selection rule. During 10-fold evaluation, calibrate each
threshold on original-condition scores from the other nine folds, then
keep it fixed across the held-out original and degraded conditions.
Never tune thresholds on the held-out fold. Preserve the supplied folds
and record the exact protocol before running experiments.

Brightness reduction is a controlled simulation; it does not reproduce
all effects of real low-light camera capture.

### Planned Evaluation
- Verification accuracy
- False Match Rate (FMR): fraction of different-person comparisons accepted
- False Non-Match Rate (FNMR): fraction of same-person comparisons rejected
- ROC and DET curves

### Repository Organization
- docs/: research notes and experiment log
- notebooks/: dataset exploration and experimental notebooks
- src/: reusable Python code
- tests/: regression checks for preprocessing, embeddings, downloads, and evaluation
- data/: local datasets and caches, excluded from Git
- models/: downloaded recognition weights, excluded from Git
- results/figures/: generated plots
- results/metrics/: generated evaluation results
- requirements.txt: pinned Python dependencies

### Setup and Execution

Use Python 3.13 (validated with 3.13.7). Stop project Jupyter servers and kernels
before updating their environment. From the repository root on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --sys-prefix --name biometrics-project1 --display-name "Biometrics Project 1"
python -m jupyterlab
```

The named kernel is installed inside `.venv`, so launch Jupyter from that
environment. Run [the dataset notebook](notebooks/01_dataset_exploration.ipynb)
first, then [the embedding notebook](notebooks/02_face_embedding_pipeline.ipynb),
then [the baseline notebook](notebooks/03_baseline_verification.ipynb).
Select **Biometrics Project 1** and run each notebook's cells in order. The LFW download is
approximately 232 MiB; allow several GB of disk space for the archive,
extracted images, model weights, cached crops, and outputs. The recognition checkpoint is
downloaded on the first embedding run and reused later.

For Windows Git Bash, create the environment with `python -m venv .venv`
and activate with `source .venv/Scripts/activate`. In PowerShell, activate with
`.venv\Scripts\Activate.ps1`.

To execute and save the notebooks from an activated terminal:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=biometrics-project1 notebooks/01_dataset_exploration.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=biometrics-project1 notebooks/02_face_embedding_pipeline.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=biometrics-project1 notebooks/03_baseline_verification.ipynb
```

To run the data validation, embedding check, and baseline without notebook figures:

```bash
python src/lfw_dataset.py --download
python src/embedding_smoke_test.py
python src/baseline_verification.py
```

Files are stored under `data/lfw_home/`. The helper verifies SHA-256 checksums,
reuses verified downloads, and restores extracted archive members on each run.
This directory contains `lfw-funneled.tgz`, `pairsDevTrain.txt`,
`pairsDevTest.txt`, `pairs.txt`, and `download_manifest.json`. Extracted images
are under `data/lfw_home/lfw_funneled/<identity>/`.

The manifest records download URLs, timestamps, file sizes, and checksums;
a copy is included in the dataset summary. JPEGs are read individually with
Pillow, preserving source dimensions and color mode during dataset validation.
The embedding pipeline converts to RGB and applies the preprocessing below.

Validation checks image and identity counts, readability, dimensions, pair
counts and class order, and every referenced image path. The supplied evaluation
folds and pair order are preserved. For future model evaluation, read images in
batches and cache embeddings for repeated images.

Review source usage notices and retain dataset citations. A download mirror
does not establish unrestricted rights to the photographs. See
[Dataset Research](docs/dataset_research.md) for sources and selection rationale.

The notebook saves [dataset checks](results/metrics/dataset_summary.json) and the
[identity distribution plot](results/figures/lfw_identity_distribution.png).
After a rerun, update its observations cell, save the notebook, and record actual
findings and any errors in [the experiment log](docs/experiment_log.md).

If setup or execution fails:

- Missing module or wrong Python: activate `.venv`, install `requirements.txt`,
  and select the **Biometrics Project 1** kernel.
- Connection or checksum failure: record the error and rerun. Failed partial
  downloads are removed; persistent checksum mismatches need investigation.
- Certificate error: confirm the pinned `certifi` package is installed in the
  active environment. Both downloaders add its CA bundle to Python's default
  trust roots while retaining certificate and hostname verification.
- Missing or corrupt images: rerun the download cell to restore archive members.
  Unexpected extra JPEGs need inspection; the helper does not delete them.

### Pretrained Embedding Pipeline

The [facenet-pytorch model](https://github.com/timesler/facenet-pytorch) is
InceptionResnetV1 pretrained on VGGFace2. It runs on CPU in evaluation mode, with
gradients disabled, seed 42, deterministic algorithms, and two PyTorch threads.
We have not independently audited overlap between its training data and LFW.

The tested Python 3.13 combination is `facenet-pytorch==2.5.3`, `torch==2.8.0`,
`torchvision==0.23.0`, `numpy==2.5.3`, and `Pillow==11.2.1`. FaceNet 2.6.0's
dependency limits require an older PyTorch generation without Python 3.13 wheels
for this Mac. Version 2.5.3 permits the tested combination through normal pip
resolution. See the [2.6.0 dependency metadata](https://pypi.org/pypi/facenet-pytorch/2.6.0/json),
[2.5.3 dependencies](https://github.com/timesler/facenet-pytorch/blob/v2.5.3/setup.py),
and [PyTorch version pairs](https://pytorch.org/get-started/previous-versions/#v2-8-0).

Preprocessing and scoring use these fixed choices:

- MTCNN proposals: minimum face size 20, thresholds `[0.6, 0.7, 0.7]`, factor 0.709.
- Select the highest-probability valid box containing the image center, with a
  detector score of at least 0.90. Missing or unsuitable detections are failures.
- Crop with margin 0 and resize to 160 x 160 using PIL bilinear interpolation.
  No additional landmark rotation is applied to the already funneled images.
- Retain unstandardized float32 RGB crops; apply `(pixel - 127.5) / 128.0` once
  before embedding. Later synthetic degradations will be applied before this step.
- Validate finite, unit-length 512-dimensional embeddings and compare them using
  cosine similarity. Detector scores and cosine similarities are not match probabilities.

The functionality check uses the first 10 same-person and first 10 different-person
development-training pairs. This is a fixed debugging subset. It does not choose
a decision threshold or measure accuracy, FMR, or FNMR. Check the displayed source
images and crops manually; numerical validation alone cannot confirm face selection.

Saved results are [the pipeline report](results/metrics/embedding_smoke_test.json),
[pair scores](results/metrics/embedding_smoke_pairs.csv), and
[the similarity plot](results/figures/embedding_smoke_scores.png). The report records
failures, exclusions, package versions, source and image hashes, numerical checks,
and observed checkpoint fingerprints. Model fingerprints are not comparisons to
publisher-supplied checksums.

Recognition weights stay under `models/checkpoints/`. Crops and embeddings stay
under `data/processed/commit4/<run-id>/`, with array rows mapped by
`embedding_paths.json`. Each run creates a fresh cache and replaces the JSON/CSV
reports. A failed run is recorded as failed; an older plot is not evidence of success.

Run the regression checks with:

```bash
python -m unittest discover -s tests -v
```

### Original-Quality Baseline

Run [the baseline notebook](notebooks/03_baseline_verification.ipynb) after the
dataset validation and development embedding check pass. Use the same Python
3.13 environment; this stage adds no dependencies. To check the evaluation code:

```bash
python -m pip check
python -m unittest discover -s tests -p test_evaluation.py -v
```

The notebook evaluates all 6,000 pairs from the verified `pairs.txt`, preserving
the supplied 10 folds of 300 same-person and 300 different-person comparisons.
It uses the frozen model and preprocessing from the development check on original
funneled images, with no synthetic degradation or model training. The runner
checks the preprocessing configuration, helper source, and model weight hashes
against the completed development run before processing evaluation images.

Following the [LFW ten-fold calibration approach](https://people.cs.umass.edu/~elm/papers/lfw.pdf),
each fold's threshold maximizes accuracy on successfully scored pairs from the
other nine folds. Candidate thresholds are midpoints between distinct calibration
scores and endpoints accepting or rejecting all scores. Ties choose the largest
candidate threshold. A pair is accepted when cosine similarity is at least the
threshold. A fold's test scores and labels never select its own threshold.

Recognition metrics condition on successful preprocessing: accuracy is correct
decisions divided by scored pairs, FMR is accepted different-person pairs divided
by scored different-person pairs, and FNMR is rejected same-person pairs divided
by scored same-person pairs. Reports include unweighted fold means, sample SD
(`ddof=1`), and SE (`SD / sqrt(10)`), plus separate pooled out-of-fold metrics.
The run stops if either class lacks usable calibration or test scores in a fold.

Coverage is scored pairs divided by requested pairs, with separate genuine and
impostor coverage per fold. Failures remain in image and pair exports, with no
score or prediction for excluded pairs. The fold report also describes a separate
hypothetical policy that rejects failed comparisons; its genuine rejection rate
includes preprocessing failures and is distinct from conditional FNMR.

Processing prints progress every 50 images and checkpoints every 25. Rerun the
baseline cell to resume an interruption; up to 24 images may be recomputed. Reuse
requires matching input hashes, model/preprocessing/environment fingerprints,
and verified crop/embedding payloads. Failed detections are cached too; restoring
a missing or changed image triggers reprocessing. Keep the computer awake and
run only one baseline process against a cache at a time.

The local cache is `data/processed/baseline/<fingerprint>/`, excluded from Git.
Retain its original crops and embeddings for degradation experiments. Saved
results include the [summary](results/metrics/baseline_summary.json),
[image outcomes](results/metrics/baseline_images.json),
[pair scores](results/metrics/baseline_scores.csv),
[fold metrics](results/metrics/baseline_folds.csv),
[held-out predictions](results/metrics/baseline_predictions.csv),
[thresholds and eligible pair indices](results/metrics/baseline_thresholds.json),
and [fold plot](results/figures/baseline_fold_metrics.png).

Reruns replace current reports. Only use a summary with `status=completed` and
matching exported-file hashes; an interrupted run may leave older CSVs or a plot.
After a successful run, update the notebook observations, README results, and
[experiment log](docs/experiment_log.md). Git preserves committed experiments.

This is evaluation with an externally pretrained VGGFace2 model, not a claim of
training only on LFW's restricted pairs. Training-data overlap remains unaudited,
and the supplied pair folds are not asserted to be identity-disjoint. Fold SE is
descriptive because calibration sets overlap and images/identities recur.

For the next experiment, reduce probe-crop resolution while keeping references,
original crops, model, baseline fold thresholds, and baseline eligible pairs
fixed. Document any later model or preprocessing revision as evaluation reuse.

### Reproducibility

Dependency changes are tracked in Git. Python and main package versions are recorded with each experiment summary.
Indirect dependencies are resolved during installation and may vary between runs.
Actual dataset checks are saved in [results/metrics/dataset_summary.json](results/metrics/dataset_summary.json).
Reruns replace the current summary and plot; Git retains committed versions.
The summary's `git_head_before_commit` identifies HEAD at execution time, and
`helper_sha256` records the helper file used, including any uncommitted changes.
Datasets, model weights, and virtual environments will not be committed.
Experiment records will describe configurations, results, failures, and
decisions. Git history will track actual changes as the project develops.
