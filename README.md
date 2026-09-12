# Biometrics Project 1
## Evaluating Face Verification Under Degraded Image Quality

[Reproduce the work](docs/reproducibility.md) · [Repository guide](docs/repository_guide.md) · [Experiment history](docs/experiment_log.md)

[Final findings](docs/final_report.md) · [Submission instructions](docs/submission.md)

### Project Status
The original-quality LFW baseline, resolution, Gaussian blur, and brightness
experiments, and score-distribution/ROC/DET analysis are complete.
The fixed VGGFace2-pretrained embedding pipeline processed 7,700 of 7,701 unique evaluation images and scored
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
and [experiment log](docs/experiment_log.md).

Resolution comparison (unweighted means across the same 10 folds):

| Probe resolution | Accuracy | FMR | FNMR | Accuracy change (pp) |
|---|---:|---:|---:|---:|
| 160 x 160 | 99.167% | 0.433% | 1.234% | +0.000 |
| 80 x 80 | 99.117% | 0.433% | 1.334% | -0.050 |
| 40 x 40 | 98.833% | 0.467% | 1.867% | -0.333 |
| 20 x 20 | 89.982% | 0.967% | 19.071% | -9.185 |

All four conditions score the same 5,999 pairs at the saved baseline thresholds.
At 20 x 20, accuracy decreased by 9.185 percentage points; false non-matches
increased from 37 to 572, while false matches increased from 13 to 29.
Measured outputs and crop examples are in [the resolution notebook](notebooks/04_resolution_degradation.ipynb).

Blur and brightness comparison (mean fold rates at the same baseline thresholds):

| Condition | Accuracy | FMR | FNMR | Accuracy change (pp) |
|---|---:|---:|---:|---:|
| Original | 99.167% | 0.433% | 1.234% | +0.000 |
| Blur σ = 1 px | 99.117% | 0.433% | 1.334% | -0.050 |
| Blur σ = 2 px | 98.917% | 0.433% | 1.734% | -0.250 |
| Blur σ = 3 px | 97.816% | 0.533% | 3.834% | -1.350 |
| Brightness × 0.75 | 99.217% | 0.367% | 1.200% | +0.050 |
| Brightness × 0.50 | 99.200% | 0.333% | 1.267% | +0.033 |
| Brightness × 0.25 | 98.566% | 0.433% | 2.434% | -0.600 |

Each condition independently uses the original probe crop and the same 5,999
eligible pairs. Blur at sigma 3 caused the largest decrease in this run, while
brightness at 75% and 50% produced small accuracy increases on these pairs.
These increases do not establish a general benefit from dimming. Different
quality scales are not severity-matched. Results and examples are saved in
[the blur and brightness notebook](notebooks/05_blur_brightness.ipynb).

Pooled score analysis across the same 5,999 eligible pairs:

| Condition | Pooled AUC | Pooled interpolated EER |
|---|---:|---:|
| Original | 0.999116 | 0.900% |
| Resolution 80 × 80 | 0.999039 | 0.867% |
| Resolution 40 × 40 | 0.998535 | 1.267% |
| Resolution 20 × 20 | 0.984612 | 6.002% |
| Blur σ = 1 | 0.999051 | 0.834% |
| Blur σ = 2 | 0.998729 | 1.233% |
| Blur σ = 3 | 0.997686 | 1.867% |
| Brightness × 0.75 | 0.999161 | 0.867% |
| Brightness × 0.50 | 0.999098 | 0.867% |
| Brightness × 0.25 | 0.998527 | 1.333% |

Resolution 20 × 20 shows the largest loss of score separation among these
settings: the mean genuine cosine score falls from 0.751540 to 0.536432,
while the mean impostor score changes from 0.022740 to 0.032296. These
threshold-sweep summaries complement the earlier fixed-threshold tables,
which remain the primary operational comparison. AUC is not accuracy, and
interpolated EER does not establish a deployable threshold. Small differences
between mild conditions do not establish a significant improvement.
See [the executed analysis notebook](notebooks/06_score_analysis.ipynb).

Setup instructions: [Setup and execution](#setup-and-execution).
Baseline instructions: [Original-quality baseline](#original-quality-baseline).
Resolution instructions: [Probe resolution experiment](#probe-resolution-experiment).
Blur/brightness instructions: [Blur and brightness experiment](#blur-and-brightness-experiment).
Score-analysis instructions: [Score distributions and ROC/DET analysis](#score-distributions-and-rocdet-analysis).

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

### Experimental Design
1. Inspect the selected dataset and verification protocol (completed).
2. Establish performance using original images (completed).
3. Reduce probe-image resolution (completed).
4. Apply Gaussian blur to probe images (completed).
5. Reduce probe-image brightness (completed).
6. Compare scores with ROC/DET curves and AUC/EER; document limitations (completed).

Keep the reference image and recognition model fixed across conditions.
Use development pairs to finalize the model, degradation levels, and
threshold-selection rule. During 10-fold evaluation, calibrate each
threshold on original-condition scores from the other nine folds, then
keep it fixed across the held-out original and degraded conditions.
Never tune thresholds on the held-out fold. Preserve the supplied folds
and record the exact protocol before running experiments.

Brightness reduction is a controlled simulation; it does not reproduce
all effects of real low-light camera capture.

### Evaluation Metrics
- Verification accuracy
- False Match Rate (FMR): fraction of different-person comparisons accepted
- False Non-Match Rate (FNMR): fraction of same-person comparisons rejected
- ROC and DET curves
- Pooled and per-fold ROC AUC and interpolated EER

### Repository Organization
- docs/: research notes and experiment log
- notebooks/: dataset exploration and experimental notebooks
- src/: reusable Python code
- tests/: regression checks for preprocessing, embeddings, downloads, and evaluation
- data/: local datasets and caches, excluded from Git
- models/: downloaded recognition weights, excluded from Git
- results/figures/: generated plots
- results/metrics/: generated evaluation results
- requirements.txt: full pinned direct Python dependencies
- requirements-analysis.txt: CLI score-analysis dependency subset
- scripts/: repository checks and reporting

### Setup and Execution

The [reproduction guide](docs/reproducibility.md) is the single source for setup,
notebook order, terminal commands, dataset/cache layout, and troubleshooting.
The [repository guide](docs/repository_guide.md) explains file organization and
what must be included in the final submission.

From the existing working environment:

```bash
source .venv/bin/activate
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_repository.py --profile full --output results/metrics/repository_check.json
```

Commit 9 local verification: all 41 tests passed, and the full repository
checker passed 21 checks with zero failures or skips, including installed
dependencies and Git history. See the [saved check report](results/metrics/repository_check.json)
and [Stage 009 record](docs/experiment_log.md#stage-009--reproduction-documentation-and-repository-checks).

To reproduce only the score analysis from committed exports:

```bash
python src/score_analysis.py
```

This does not require images, model weights or inference caches. A fresh analysis
installation can use [requirements-analysis.txt](requirements-analysis.txt);
full inference and Jupyter use [requirements.txt](requirements.txt). See the guide
for the distinction between file validation, score analysis, and full reproduction.

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

The resolution experiment below keeps references, original crops, model,
baseline fold thresholds, and baseline eligible pairs fixed. Document any later
model or preprocessing revision as evaluation reuse.

### Probe Resolution Experiment

Run [the resolution notebook](notebooks/04_resolution_degradation.ipynb) in the
same Python 3.13 environment after completing the baseline. No new packages are
required. Its tests are included in the full regression suite, or run them with:

```bash
python -m unittest discover -s tests -p test_resolution.py -v
```

The four conditions use probe crops at 160, 80, 40, and 20 pixels per side.
The 160-pixel control reuses the original embeddings. Each reduced condition
uses Pillow BOX downsampling followed by BILINEAR enlargement to 160 x 160.
Channels stay in float32 mode F, avoiding extra uint8 quantization; the existing
embedding helper standardizes the transformed crop once. The enlarged crops
have the required input dimensions but retain less spatial detail.

Reference embeddings, original crop selection, model, saved fold thresholds,
and the same 5,999 eligible pairs stay fixed. No detection, recropping, training,
or threshold calibration runs on degraded images. The excluded baseline pair
remains in each condition's predictions with no score or decision. Any invalid
degraded embedding stops the run rather than changing pair eligibility.

The runner verifies baseline export and cache hashes, model/environment
fingerprints, and cached control scores. A fresh forward pass on one original
probe must match its cached embedding, and control metrics and error counts
must reproduce the baseline. A fresh clone needs its baseline reproduced
locally first: Git does not contain the cached crops and embeddings.

Progress prints every 100 unique probes. Each completed degraded embedding is
cached with a hash receipt under `data/processed/resolution/<fingerprint>/`,
excluded from Git. Rerun the notebook cell to reuse verified work; corrupted
degraded payloads are recomputed. Baseline payload corruption stops execution
and should be repaired through the original baseline workflow. Run only one
experiment against the cache at a time. Reruns replace reports; use only a
completed summary with matching output hashes.

Outputs are the [resolution summary](results/metrics/resolution_summary.json),
[condition comparison](results/metrics/resolution_comparison.csv),
[fold metrics](results/metrics/resolution_folds.csv),
[pair predictions](results/metrics/resolution_predictions.csv), and
[comparison plot](results/figures/resolution_comparison.png). The notebook also
displays one illustrative probe under all four transformations. Update its
observations and the experiment log after a rerun.

Rates are unweighted means across the same 10 folds, conditional on baseline
preprocessing success. Changes from the 160-pixel control are in percentage
points. Fold SD/SE are descriptive; means alone do not establish significance.
This experiment measures synthetic detail loss after cropping, not detection
on low-resolution inputs or every effect of a real low-resolution camera.

### Blur and Brightness Experiment

Run [the blur and brightness notebook](notebooks/05_blur_brightness.ipynb) in
the same Python 3.13 environment, retaining the original baseline crop cache.
No dependency changes are required. The six new tests are included in the full
suite and can also be run separately:

```bash
python -m unittest discover -s tests -p test_quality.py -v
```

Seven conditions compare the original crop with Gaussian blur at sigma 1, 2,
and 3 pixels and brightness factors 0.75, 0.50, and 0.25. Each transform starts
independently from the original 160 x 160 probe crop. Degradations are never
combined or accumulated. Reference embeddings, model, crop selection, the same
5,999 eligible pairs, and all ten saved baseline thresholds remain fixed.

Blur uses a normalized sampled Gaussian kernel along each spatial axis, with
radius `ceil(3 * sigma)`. NumPy reflect padding excludes the repeated edge pixel;
channels are filtered separately with float64 accumulation and float32 output.
Brightness directly multiplies the encoded RGB float32 values by the factor,
preserving that fraction of the original values. Neither transform quantizes
to uint8. The unchanged model standardization runs once after transformation.

The runner reuses baseline validation and fixed-threshold scoring from the
resolution helper. It does not run the resolution experiment. Original control
scores, metrics, and error counts must match the baseline, and one new control
forward pass must match its cached vector. Original crops and reference
embeddings remain read-only. No detection or threshold recalibration runs on
degraded inputs; an invalid embedding aborts instead of dropping another pair.

The six degraded conditions each process 4,575 unique eligible probes on the
first run. Progress prints every 100 probes. Hash-verified completed embeddings
are stored under `data/processed/quality/<fingerprint>/<condition>/`, excluded
from Git. Rerun the execution cell to resume; valid entries are reused and
corrupted degraded payloads are recomputed. Keep one process per cache/output.
Use only a completed summary with matching CSV hashes; failed reruns can leave
older plots or exports. A fresh clone must reproduce its baseline cache first.

Saved outputs are the [quality summary](results/metrics/quality_summary.json),
[seven-condition comparison](results/metrics/quality_comparison.csv),
[70 fold rows](results/metrics/quality_folds.csv),
[42,000 pair records](results/metrics/quality_predictions.csv),
[example transformations](results/figures/quality_examples.png), and
[performance plots](results/figures/quality_comparison.png). Every condition
retains the same baseline exclusion as an empty score/prediction. Save actual
observations in the notebook and experiment log after a completed rerun.

These transformations study synthetic quality changes after cropping. RGB
scaling does not simulate exposure, gamma conversion, sensor noise, or other
low-light camera effects. Fold means and SD/SE describe these tested pairs and
settings; they do not establish significance. Resolution, blur, and brightness
levels have different severity scales, so their results do not establish a
universal ranking of degradation types. Training-data overlap remains unaudited.

### Score Distributions and ROC/DET Analysis

Run [the score-analysis notebook](notebooks/06_score_analysis.ipynb) in the
existing Python 3.13 environment, or run `python src/score_analysis.py` from
the repository root. No dependencies change. The nine analysis tests are
included in the full 36-test suite and can also be run separately:

```bash
python -m unittest discover -s tests -p test_score_analysis.py -v
```

This stage reads seven committed files under `results/metrics/`:
`baseline_summary.json`, `baseline_thresholds.json`, `baseline_scores.csv`,
`resolution_summary.json`, `resolution_predictions.csv`, `quality_summary.json`,
and `quality_predictions.csv`. It validates their provenance and export hashes,
pair indices, labels, folds, exclusions, control scores, and saved threshold
decisions. A separate local check reproduced every analysis CSV and PNG in a
temporary directory containing only these inputs and the analysis helper.

The original condition is counted once, giving ten distinct conditions with
2,999 genuine and 3,000 impostor comparisons each. No face inference, model
training, or threshold calibration runs. The earlier fixed-threshold accuracy,
FMR, FNMR, and error counts must reproduce before the analysis proceeds.

The empirical ROC accepts `score >= threshold`, groups equal scores, and
includes reject-all and accept-all endpoints. AUC integrates the full curve.
EER linearly interpolates adjacent empirical points where FMR equals FNMR;
the interpolated point may not be attainable by a deterministic threshold.
Pooled metrics combine all scored pairs; the fold CSV reports each supplied
fold separately, and the comparison also includes unweighted fold means and
sample SD. These summaries describe evaluation scores and do not select a
new operating threshold. Small changes in AUC and EER need not rank conditions
identically.

The ROC plot zooms to FMR 0–5% and true match rate 75–100% for readability;
this zoom does not limit AUC integration. The DET plot uses standard-normal
quantiles of FMR and FNMR, omitting zero/one rates only from display. The curve
CSV retains those endpoints unchanged. Score distributions use common bins
and axes, with each class normalized separately to a density. Method references:
[ROC definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
and [DET interpretation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html).
The helper computes the metrics directly with NumPy; scikit-learn is not a dependency.

Saved outputs are the [analysis summary and fingerprints](results/metrics/analysis_summary.json),
[ten-condition comparison](results/metrics/analysis_comparison.csv),
[100 fold summaries](results/metrics/analysis_folds.csv),
[59,990 empirical curve points](results/metrics/analysis_curves.csv),
[class score statistics](results/metrics/analysis_score_statistics.csv),
[ROC plot](results/figures/analysis_roc.png), [DET plot](results/figures/analysis_det.png),
and [score distributions](results/figures/analysis_score_distributions.png).
Reruns replace these outputs. Require `status=completed` and matching output
hashes, then save notebook observations and update the experiment log. If an
input validation fails, investigate the affected prior export and its source
experiment before rerunning; do not bypass the integrity checks.

With 3,000 impostor pairs, one false match changes pooled FMR by 0.0333
percentage points; lower plot ticks do not create additional measurement
precision. Repeated images/identities and overlapping calibration folds limit
independence, and no significance test was performed. Evaluation scores were
already used in earlier stages. Synthetic post-crop transformations, the one
baseline exclusion, unmatched severity scales, and unaudited training-data
overlap retain their earlier limitations.

### Final Findings and Submission

The [final report](docs/final_report.md) consolidates the problem statement,
dataset choice, methods, ten-condition results, conclusions and limitations.
After committing the final documentation, use the [submission workflow](docs/submission.md)
to create a ZIP of committed project files and the actual `.git` directory.
The packager verifies the extracted history and saves a verification receipt
next to the ZIP. A GitHub source-code ZIP alone does not include Git history.

Commit 10 local review passed all 46 tests and all 23 full-profile repository
checks, with zero failures or skips. The final report's ten-condition table
matches the saved metrics. [Stage 010](docs/experiment_log.md#stage-010--final-findings-and-submission-workflow)
records this review; the external packaging receipt records the archive checks
performed after the commit.

### Reproducibility

Dependency changes are tracked in Git. Python and main package versions are recorded with each experiment summary.
Indirect dependencies are resolved during installation and may vary between runs.
Actual dataset checks are saved in [results/metrics/dataset_summary.json](results/metrics/dataset_summary.json).
Reruns replace the current summary and plot; Git retains committed versions.
The summary's `git_head_before_commit` identifies HEAD at execution time, and
`helper_sha256` or `source_sha256` records the helper code used, including any
uncommitted changes.
Datasets, model weights, and virtual environments are excluded from Git.
Experiment records describe configurations, results, failures, and decisions.
Use the [reproduction guide](docs/reproducibility.md) and repository checker to
verify a local checkout. The final submission must include the real `.git`
directory; a GitHub source ZIP alone does not contain that history.
