# Project and Experiment Log

## Stage 001 — Initial Repository Setup

### Goal
Define the research question and organize the repository.

### Changes
- Drafted the face-verification problem statement.
- Identified the biometric modality and pipeline stages.
- Listed candidate datasets for further research.
- Outlined baseline and image-degradation experiments.
- Added folders and Git ignore rules.

### Current Status
Planning only. No dataset inspection, model execution, or measured
results are available yet.

### Initial Decision
Investigate LFW first for the baseline.
Confirm dataset availability and protocol before implementation.

### Next Step
Research candidate datasets, record sources and access requirements,
and select the baseline dataset and image version.

## Stage 002 — Dataset Research and Selection

### Goal
Compare candidate datasets and select an initial verification dataset.

### Changes
- Documented LFW, SCface, and QMUL-SurvFace.
- Recorded source links and access requirements.
- Selected LFW funneled images for the baseline.
- Defined reference/probe roles and a planned evaluation protocol.
- Documented loading settings and checks needed before experiments.

### Findings
LFW provides predefined verification pairs.
SCface requires an institutional request and a staff-signed agreement.
QMUL-SurvFace offers a larger surveillance collection.

### Decision
Begin with LFW, subject to successful download and validation.
Use original-condition calibration thresholds unchanged across
degradation conditions within each evaluation fold.

### Current Status
Documentation and planning only.
No download, model execution, or measured performance results yet.

### Next Step
Download LFW, inspect its images and pair files in Jupyter, and
record the actual findings.

## Stage 003 — LFW Download and Exploration

### Run time (UTC)
2026-09-09T02:41:47.735201+00:00

### Goal
Download LFW funneled images and validate them for the baseline.

### Changes
Added checksum-verified downloading, pair parsing, image inspection,
and a Jupyter exploration notebook.

### Measured Results
- Images: 13233
- Identities: 5749
- Image dimensions: {'250x250': 13233}
- Source color modes: {'RGB': 13233}
- Identities with two or more images: 1680
- Unreadable images: 0
- Missing pair references: 0
- All dataset validation checks passed: True

### Measured Pair Protocol

| File | Same person | Different people | Total | Folds |
|---|---:|---:|---:|---:|
| pairsDevTrain.txt | 1,100 | 1,100 | 2,200 | 1 |
| pairsDevTest.txt | 500 | 500 | 1,000 | 1 |
| pairs.txt | 3,000 | 3,000 | 6,000 | 10 |

### Saved Outputs
- notebooks/01_dataset_exploration.ipynb
- results/metrics/dataset_summary.json
- results/figures/lfw_identity_distribution.png

### Observations and Problems
The distribution is strongly uneven: 4,069 identities (70.8%) have one image;
1,680 have at least two. Counts range from 1 to 530. The displayed development
pairs show changes in expression, lighting, background, and slight head angle.
Some examples have black or rotated alignment borders and differing apparent
sharpness. These four pairs are illustrative, not a dataset-wide quality sample.

Execution used the Biometrics Project 1 Jupyter kernel in the local Python
3.13.7 virtual environment. The supplied four dependency pins installed
successfully and pip check reported no broken requirements. The first download
failed with CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate.
Both default Python CA paths were absent. Adding certifi==2026.7.22 to the
requirements and supplementing the downloader TLS context with its CA bundle
resolved the error. Certificate and hostname verification stayed enabled.
All four downloads then matched their published SHA-256 checksums. No HTTP 403
occurred locally. All six notebook code cells completed successfully.

Python and main package versions are recorded in results/metrics/dataset_summary.json.
Data and the virtual environment remain excluded from Git. The UTC run date
above corresponds to 2026-09-08 in America/New_York.

The data is ready for baseline implementation; model selection, training-data
overlap review, preprocessing, and embedding generation remain pending.

### Next Step
Select a pretrained model and test the face-preprocessing and embedding
pipeline on development images. No verification metrics exist yet.

## Stage 004 — Pretrained Face Embedding Pipeline

### Run
20260909T214720243224Z (UTC)

### Goal
Check preprocessing and embeddings using 20 development-training pairs.

### Changes
Added MTCNN face selection and a fixed VGGFace2-pretrained InceptionResnetV1.
Used Python 3.13.7 with facenet-pytorch 2.5.3,
PyTorch 2.8.0, torchvision 0.23.0,
NumPy 2.5.3, and Pillow 11.2.1.
Retained original-condition crops for later controlled degradation.

### Measured Results
- Unique images attempted: 33
- Successful images: 33
- Failed images: 0
- Scored pairs: 20 / 20
- Excluded pairs: 0
- Repeated embedding maximum absolute difference: 0.0
- Self-similarity: 1.0
- Pipeline checks passed: True

### Outputs
- notebooks/02_face_embedding_pipeline.ipynb
- results/metrics/embedding_smoke_test.json
- results/metrics/embedding_smoke_pairs.csv
- results/figures/embedding_smoke_scores.png
- Local crops and embeddings under data/processed/commit4/20260909T214720243224Z (excluded from Git)

### Observations and Problems
Visual review covered the six notebook examples and all 33 source/crop pairs in
local contact sheets. Every reviewed crop selected the central target, including
sources with multiple faces. Tight crops retain some blur, shadows, pose variation,
and partial occlusion. This does not establish detection reliability on all LFW.

The 10 same-person scores ranged from 0.6131 to 0.9181; the 10 different-person
scores ranged from -0.2534 to 0.2767. The ranges did not overlap in this debugging
subset. No threshold or verification performance metric was calculated.

The bundle's proposed FaceNet 2.6.0 dependencies required an older PyTorch stack
without Python 3.13 wheels for this Mac. Official package metadata was checked
before installation. FaceNet 2.5.3 permits the newer PyTorch/torchvision pair;
normal dependency resolution and pip check passed with the versions listed above.
Pillow and the Jupyter pins were retained. No dependency constraints were bypassed.

The model downloader supplements default TLS roots with certifi, as needed by this
Python.org macOS installation. It preserves certificate and hostname verification,
uses a timeout, and discards incomplete downloads. The official model downloaded
successfully, and the first full notebook run after adaptation completed without
model-download or inference errors. The checkpoint fingerprint is an observed
hash, not a comparison with a publisher-provided digest.

All five notebook code cells and all eight regression tests passed. The tests
cover central-target selection, invalid vectors, cosine values, balanced pair
selection, verified TLS, incomplete-download cleanup, and refusing to load a
model when dataset validation failed. Setup and dependencies remain consolidated
in README.md and requirements.txt; versions and source hashes are in the report.

### Limitations and Next Step
This was a small development check. No threshold, accuracy, FMR, or FNMR
was estimated. Training-data overlap remains unaudited. Next, run the
original-quality verification baseline with a documented calibration protocol.

## Stage 005 — Original-Quality Verification Baseline

### Run (UTC)
2026-09-11T03:34:32.405142+00:00

### Method
Used the fixed Commit 4 model on LFW's supplied 6,000 pairs in 10 folds.
For each held-out fold, selected a threshold using only scored pairs from
the other nine folds. Accepted similarities at or above the threshold.
No model training or synthetic degradation was performed.

### Measured Results
- Successful images: 7700 / 7701
- Scored pairs: 5999 / 6000
- Excluded pairs: 1
- Pair coverage: 99.98%
- Mean fold accuracy: 99.167%
- Accuracy SD: 0.593 percentage points
- Accuracy SE: 0.188 percentage points
- Mean fold FMR: 0.433%
- Mean fold FNMR: 1.234%

### Saved Outputs
- notebooks/03_baseline_verification.ipynb
- results/metrics/baseline_summary.json
- results/metrics/baseline_images.json
- results/metrics/baseline_scores.csv
- results/metrics/baseline_folds.csv
- results/metrics/baseline_predictions.csv
- results/metrics/baseline_thresholds.json
- results/figures/baseline_fold_metrics.png
- Verified original crops and embeddings under the baseline cache directory recorded
  in the summary (local data/processed/baseline/, excluded from Git).

### Observations and Problems
All 10 folds completed on 2026-09-10 (America/New_York), using Python
3.13.7 on macOS-26.6.2-arm64-arm-64bit-Mach-O. The CPU model and preprocessing were unchanged
from the development check. No packages were added or changed.

Mean fold accuracy was 99.167%, FMR 0.433%, and FNMR
1.234%. Across all scored pairs there were 2,962 true accepts, 2,987
true rejects, 13 false matches, and 37 false non-matches. Pooled accuracy was
99.166528%, pooled FMR 0.433333%, and pooled FNMR
1.233745%; these are separate from unweighted fold means.

One of 7,701 images failed the fixed 0.90 detector-confidence cutoff:
`Princess_Aiko/Princess_Aiko_0001.jpg`. This excluded genuine pair index 818
(zero-based) in displayed fold 2 (internal fold 1). Overall genuine coverage was
2,999/3,000 (99.967%); impostor coverage was 3,000/3,000 (100%). The source image
was visually reviewed: it is soft and includes another partially visible face
at the right edge. This observation does not establish the cause of low detector
confidence. The cutoff and face-selection rule were retained.

Fold accuracy ranged from 98.333% to 99.667%, FMR from 0% to 1%, and FNMR from
0% to 3%. Accuracy SD was 0.593 percentage points and SE was
0.188 percentage points. The fold plot was visually checked. No
model or threshold-selection rule was tuned after inspecting these results.

The first complete inference run started at 2026-09-11T03:28:35.349431+00:00 and took
276.501 seconds (4.61 minutes). Review found that the supplied CSV
writer used CRLF line endings, which Git would normalize and invalidate the
recorded file hashes on checkout. Setting the CSV writer to LF fixed this.
The final notebook rerun took 11.500 seconds, reused all 7,701 verified cached
image outcomes, and processed 0 fresh images. All fold thresholds, metrics,
and failure counts were identical. There were no notebook execution errors.

All 8 evaluation tests passed, alongside the 8 existing embedding tests.
Local synthetic integration checks verified 6,000-pair exports, failure accounting,
cache reuse, corrupt-payload recovery, changed-image invalidation, and recovery
from an injected interruption. An independent audit checked all input/cache/output
hashes, all 6,000 scores and decisions, and recalculated every fold threshold and
confusion count. All six notebook code cells completed successfully.

This establishes an original-quality baseline for subsequent controlled quality
experiments. It does not establish performance on degraded images or other
populations; external model training overlap remains unaudited. Setup remains in
README.md and dependencies in requirements.txt; the report records actual versions.

### Limitations
Recognition rates condition on successful preprocessing; failure coverage
and an explicit reject-on-failure policy are reported separately. Model
training overlap remains unaudited. Fold errors are not fully independent.

### Next Step
Apply controlled resolution degradation to probe crops. Keep the reference
images, model, baseline fold thresholds, and baseline eligible pairs fixed.

## Stage 006 — Probe Resolution Degradation

### Run (UTC)
2026-09-11T15:14:43.285240+00:00

### Method
Reduced probe crops to 80, 40, and 20 pixels with BOX downsampling,
then enlarged to 160 pixels using BILINEAR interpolation. Retained
float32 pixels and applied the existing normalization once afterward.
Kept reference embeddings, original face crops, the model, baseline
eligible pairs, and baseline fold thresholds fixed.

### Measured Results

| Probe pixels | Scored pairs | Accuracy | FMR | FNMR | Accuracy change (pp) |
|---:|---:|---:|---:|---:|---:|
| 160 | 5999 | 99.167% | 0.433% | 1.234% | +0.000 |
| 80 | 5999 | 99.117% | 0.433% | 1.334% | -0.050 |
| 40 | 5999 | 98.833% | 0.467% | 1.867% | -0.333 |
| 20 | 5999 | 89.982% | 0.967% | 19.071% | -9.185 |

### Saved Outputs

- notebooks/04_resolution_degradation.ipynb
- results/metrics/resolution_summary.json
- results/metrics/resolution_comparison.csv
- results/metrics/resolution_folds.csv
- results/metrics/resolution_predictions.csv
- results/figures/resolution_comparison.png
- Local degraded embeddings under data/processed/resolution/ (excluded from Git).

### Observations and Problems
The experiment started on 2026-09-11 at 11:14:43 EDT, using the same
Python 3.13.7 CPU environment and frozen model as the baseline. All four
conditions finished in 205.242 seconds (3.42 minutes). Each reduced condition
processed 4,575 unique probes, producing 13,725 fresh degraded embeddings in total.
The 160px control reused baseline embeddings; its additional real forward check
had maximum absolute difference 0.0 from the cached vector.

The 160px condition exactly reproduced the baseline: 99.167% mean fold accuracy,
0.433% FMR, 1.234% FNMR, 13 false matches, and 37 false non-matches. All four
conditions retained the same 5,999 eligible pairs out of 6,000 (99.983% coverage),
with the same genuine pair excluded in fold 2. There were no additional exclusions.
All reference embeddings and saved fold thresholds stayed fixed.

At 80px, accuracy was 99.117%, a decrease of 0.050 percentage points. FMR remained
0.433%, while FNMR rose to 1.334%; error counts were 13 false matches and 40 false
non-matches. At 40px, accuracy was 98.833%, down 0.333 percentage points, with
0.467% FMR and 1.867% FNMR (14 false matches and 56 false non-matches).

The strongest tested reduction, 20px, had the largest accuracy decrease:
9.185 percentage points, to 89.982%. FMR rose by 0.533 percentage points to 0.967%,
and FNMR rose by 17.838 percentage points to 19.071%. This condition had 29 false
matches and 572 false non-matches. The main deterioration at the fixed operating
thresholds was rejection of genuine comparisons. These are descriptive results;
no significance test was performed and the means do not prove a general cutoff.

Visual inspection of the notebook's example crops showed progressively softer
edges and less detail around the glasses, eyes, and mouth, especially at 20px.
Enlargement restored the input dimensions without restoring that visible detail.
The displayed probe is one illustrative example, not a representative sample.
The comparison chart was reviewed; accuracy labels were moved below the points
to avoid touching the upper axis border. Only the plot cell was re-executed from
the saved metrics, so inference results and metric reports were unchanged.

All 5 resolution tests and all 16 earlier regression tests passed. The supplied
synthetic integration test checked cache reuse, baseline immutability, and
rejection of corrupt baseline payloads. Independent checks verified all 24,000
prediction rows, all 40 fold results, all 13,725 cached degraded embeddings and
receipts, fixed thresholds and eligibility, and the aggregate means, SD/SE,
error counts, and percentage-point changes. Nine additional real-probe checks
used independent block averaging before enlargement and reproduced cached
embeddings within 1e-6. Baseline report, manifest, and original payload hashes
were unchanged. All five notebook code cells completed without execution errors.

No dependency changes, installation fixes, or model/preprocessing changes were
needed. The existing LF CSV writer preserved the recorded output hashes.
Instructions remain in README.md; model/environment and source fingerprints
are in the result JSON. The result measures synthetic loss of spatial detail
after cropping. It does not evaluate detection on degraded photographs or all
camera artifacts; training-data overlap with LFW remains independently unaudited.

### Limitations
The same baseline exclusions apply at every level. This experiment measures
synthetic resolution loss after face cropping, not degraded-image detection.
Training-data overlap remains independently unaudited.

### Next Step
Apply Gaussian blur to probe crops using the same baseline comparison protocol.

## Stage 007 — Gaussian blur and reduced brightness

Run timestamp (UTC): 2026-09-11T16:56:57.204309+00:00

### Goal and method
Evaluate independent Gaussian blur (σ = 1, 2, 3 pixels) and encoded RGB brightness scaling (0.75, 0.50, 0.25) on original probe crops. Keep original references, the model, eligible pairs, and ten baseline fold thresholds fixed. No detection or threshold recalibration on degraded images.

### Results

| Condition | Accuracy | FMR | FNMR | Accuracy change (pp) |
|---|---:|---:|---:|---:|
| Original | 99.167% | 0.433% | 1.234% | +0.000 |
| Blur σ = 1 px | 99.117% | 0.433% | 1.334% | -0.050 |
| Blur σ = 2 px | 98.917% | 0.433% | 1.734% | -0.250 |
| Blur σ = 3 px | 97.816% | 0.533% | 3.834% | -1.350 |
| Brightness × 0.75 | 99.217% | 0.367% | 1.200% | +0.050 |
| Brightness × 0.50 | 99.200% | 0.333% | 1.267% | +0.033 |
| Brightness × 0.25 | 98.566% | 0.433% | 2.434% | -0.600 |

Every condition scored 5999/6000 pairs; 1 baseline exclusion(s) retained. Original control reproduced the baseline. Run completed in 6.07 minutes (cache reuse is recorded in the summary).

### Saved outputs

- notebooks/05_blur_brightness.ipynb
- results/metrics/quality_summary.json
- results/metrics/quality_comparison.csv
- results/metrics/quality_folds.csv
- results/metrics/quality_predictions.csv
- results/figures/quality_examples.png
- results/figures/quality_comparison.png
- Local degraded embeddings under data/processed/quality/ (excluded from Git).

### Observations and problems encountered
The run started on 2026-09-11 at 12:56:57 EDT and completed in
364.024 seconds (6.07 minutes) using the unchanged Python 3.13.7 CPU
environment. Each of the six degraded conditions processed 4,575 original probe
crops, producing 27,450 fresh embeddings. No degraded embeddings were reused
on this first run. The control reused baseline embeddings and its additional
forward check had maximum absolute difference 0.0 from the cached vector.

All seven conditions scored the same 5,999 of 6,000 pairs (99.983% coverage).
The original control reproduced 99.167% mean fold accuracy, 0.433% FMR,
1.234% FNMR, 13 false matches, and 37 false non-matches. The baseline's one
excluded genuine pair remained excluded in every condition; there were no new
failures or exclusions. Reference embeddings and all ten thresholds stayed fixed.

Increasing blur reduced accuracy at each tested step: 99.117%, 98.917%, and
97.816% for sigma 1, 2, and 3. FNMR rose to 1.334%, 1.734%, and 3.834%.
FMR stayed at 0.433% through sigma 2, then rose to 0.533% at sigma 3.
False non-matches numbered 40, 52, and 115; false matches numbered 13, 13,
and 16. Thus most added errors were genuine rejections. Sigma 3 had the largest
accuracy decrease among these six new conditions: 1.350 percentage points.
Changes are computed from unrounded rates, so subtracting displayed rounded
percentages can differ by 0.001 percentage points.

Brightness results were not monotonic relative to the original control.
Retaining 75% of RGB values increased accuracy by 0.050 percentage points to
99.217%, with 11 false matches and 36 false non-matches. At 50%, accuracy was
99.200% (+0.033 percentage points), with 10 false matches and 38 false
non-matches. These conditions had 47 and 48 errors versus 50 at the control.
At 25%, accuracy fell to 98.566% (-0.600 percentage points), with 13 false
matches and 73 false non-matches; FMR was 0.433% and FNMR was 2.434%.
The small increases at 75% and 50% were retained as measured; they do not
establish that dimming generally improves verification. No significance test
or condition-specific threshold tuning was performed.

All blur and brightness conditions had higher accuracy than the earlier
20 x 20 resolution condition (89.982%). These severity scales are not matched,
so this observation applies only to the chosen settings and does not rank the
degradation types universally. Future comparisons should retain this limitation.

Visual review of the example figure showed softer glasses, eye, and mouth edges
with increasing blur. The brightness panels became darker while retaining the
original edge structure. Both rows start from the same original probe; the
display preserves absolute brightness rather than rescaling each panel. This
single probe illustrates the transforms and is not a representative sample.
The performance plots were also reviewed; their vertical axis ranges differ,
so the labeled rates and table should guide comparisons between panels.

All 6 new quality tests and all 21 earlier tests passed. The synthetic integration
test exercised frozen thresholds, pair membership, cache reuse and repair,
invalid-embedding failure, and rejection of corrupt baseline payloads. Independent
checks matched all 42,000 prediction rows, all 70 fold results, all 27,450 cached
vectors and receipts, fixed eligibility and thresholds, error counts, means,
SD/SE, and percentage-point changes. Eighteen additional real-probe forward
checks used independently computed convolutions or pixel scaling and matched
cached vectors within 1e-6. Prior baseline and resolution reports, the baseline
manifest, and all original crop/embedding payload hashes were unchanged.

All six notebook code cells completed without execution errors. No package
installation, dependency changes, model changes, or runtime fixes were needed.
Setup references were consolidated in README.md, and actual model/environment,
source, and output fingerprints are retained in the JSON report. This is a
synthetic post-crop study conditional on baseline preprocessing success; RGB
scaling does not simulate camera exposure or sensor noise, and degraded-input
face detection was not measured. Training-data overlap with LFW remains unaudited.

### Limitations
Synthetic post-crop transformations do not reproduce all real camera effects. Brightness scaling does not simulate exposure or added sensor noise. Rates are conditional on baseline preprocessing successes. Different degradation scales are not severity-matched; comparisons apply to these tested settings. Fold SD/SE are descriptive.

### Next step
Inspect score distributions and ROC/DET curves, retaining the fixed-threshold results as the primary operational comparison.

## Stage 008 — Score distributions and ROC/DET analysis

Analysis run (UTC): 2026-09-11T17:29:28.739298+00:00

### Goal and method
Analyze saved cosine similarities for ten conditions without rerunning the face model. Validate input hashes, common pair membership, labels, exclusions, controls, and fixed baseline threshold decisions. Compute pooled and per-fold ROC AUC and linearly interpolated EER; plot descriptive pooled ROC/DET curves and score distributions.

### Results

| Condition | Pooled AUC | Pooled interpolated EER | Mean fold AUC |
|---|---:|---:|---:|
| Original | 0.999116 | 0.900% | 0.999069 |
| Resolution 80 × 80 | 0.999039 | 0.867% | 0.999006 |
| Resolution 40 × 40 | 0.998535 | 1.267% | 0.998471 |
| Resolution 20 × 20 | 0.984612 | 6.002% | 0.984411 |
| Blur σ = 1 | 0.999051 | 0.834% | 0.999026 |
| Blur σ = 2 | 0.998729 | 1.233% | 0.998680 |
| Blur σ = 3 | 0.997686 | 1.867% | 0.997635 |
| Brightness × 0.75 | 0.999161 | 0.867% | 0.999130 |
| Brightness × 0.50 | 0.999098 | 0.867% | 0.999062 |
| Brightness × 0.25 | 0.998527 | 1.333% | 0.998509 |

Every condition uses 5999/6000 pairs (2999 genuine and 3000 impostor). Existing fixed-threshold metrics were reproduced and preserved.

### Saved outputs

- `notebooks/06_score_analysis.ipynb`
- `results/metrics/analysis_summary.json`
- `results/metrics/analysis_comparison.csv`
- `results/metrics/analysis_folds.csv`
- `results/metrics/analysis_curves.csv`
- `results/metrics/analysis_score_statistics.csv`
- `results/figures/analysis_roc.png`
- `results/figures/analysis_det.png`
- `results/figures/analysis_score_distributions.png`

### Observations

The local analysis completed on Python 3.13.7 in 1.164 seconds using the
committed scores from Stage 007. All ten conditions retained the same 5,999
eligible pairs (2,999 genuine and 3,000 impostor); the original control was
counted once. No images, model weights, or inference caches were loaded. The
original fixed-threshold results reproduced exactly, including 13 false matches
and 37 false non-matches.

The ROC, DET, and common-bin score-distribution plots were inspected. The
20 × 20 condition has the clearest loss of separation: pooled AUC decreases
from 0.999116 to 0.984612 and interpolated EER rises from 0.900% to 6.002%.
Its genuine-score mean shifts from 0.751540 to 0.536432, while the impostor
mean shifts from 0.022740 to 0.032296. The broader genuine distribution and
its lower-score tail overlap the impostor distribution more visibly. Its
ROC is lower and DET error rates are higher over the displayed low-FMR region.

Blur σ = 3 also shifts genuine scores lower (mean 0.669063), with pooled AUC
0.997686 and EER 1.867%. Brightness × 0.25 has a genuine-score mean of 0.706034,
AUC 0.998527, and EER 1.333%. These results support the earlier fixed-threshold
findings for the tested settings. Resolution, blur, and brightness scales are
not severity-matched, so this is not a universal ordering of degradation types.

Mild changes have small, nonmonotonic differences. Brightness × 0.75 has the
highest pooled AUC here (0.999161); blur σ = 1 has the lowest interpolated EER
(0.834%) despite a slightly lower AUC than the original. AUC averages ranking
performance over the full curve, whereas EER describes its equal-error crossing.
Neither observation establishes a significant benefit. Pooled and mean-fold
metrics are reported separately and need not be equal or rank settings alike.

EER is a descriptive threshold-sweep result, not a newly calibrated operating
point. For example, the 20 × 20 condition still has mean fold accuracy 89.982%,
FMR 0.967%, and FNMR 19.071% at the frozen baseline fold thresholds. Its pooled
EER of 6.002% uses a different evaluation summary and does not replace those
operational rates. The ROC display is zoomed, but AUC uses every curve point;
DET omits zero/one rates only from the plot, retaining them in the CSV.

All 36 regression tests passed, including the nine new analysis tests. An
independent audit checked AUC by genuine–impostor score comparisons, every
one of the 59,990 ROC points by class-specific threshold counts, interpolated
EER, all 100 fold summaries, 20 score-statistic rows, and fixed-threshold
metrics. All four CSVs reproduced the supplied analysis within absolute
floating-point tolerance 1e-12. A separate run with only the helper and seven
saved inputs reproduced all four CSVs and three PNGs byte for byte. Source,
input, and output hashes passed, and all 23 earlier result files were unchanged.
All eight notebook code cells completed without errors. No package changes or
runtime fixes were needed; setup instructions remain in README.md.

### Limitations
Pooled AUC/EER are descriptive evaluation summaries, not new validated operating thresholds. Interpolated EER may lie between attainable deterministic points. Finite pair counts limit low-FMR precision, and fold dependence prevents treating fold spread as an independent significance test. Earlier synthetic-degradation and exclusion limitations remain.

### Next step
Review the completed findings and reproduction instructions, then prepare the submission archive including Git history.

## Stage 009 — Reproduction Documentation and Repository Checks

Verification date: 2026-09-11. The saved report records the exact UTC check time.

### Goal

Consolidate reproduction instructions and verify the local repository before
the final conclusions and submission archive.

### Changes

- Moved setup, execution order, cache requirements, and troubleshooting into
  [the reproduction guide](reproducibility.md), retaining measured tables and
  methods in README.md.
- Added [the repository guide](repository_guide.md) and README navigation.
- Added `requirements-analysis.txt` for command-line score analysis and checks,
  pinning NumPy and Matplotlib to the same versions as `requirements.txt`.
  Full inference and Jupyter continue to use the existing full requirements.
- Added `scripts/check_repository.py` and five focused regression tests.
- Kept stage-specific setup instructions out of the repository: the supplied
  `commit9_setup.md` was used as a reference, and its workflow is covered by
  the durable reproduction guide and this entry. The checker requires that
  guide instead of a redundant commit-specific setup document.

### Verification

- Python 3.13.7 on macOS arm64; all nine installed direct dependencies match
  the full requirements. `python -m pip check` found no broken requirements.
- `python -m unittest discover -s tests -v`: all 41 tests passed, including
  the five new checker tests. Download-related unit tests use synthetic
  fixtures; they did not download a model or perform new face inference.
- Full repository profile: **21 passed, 0 failed, 0 skipped**. Details are in
  [repository_check.json](../results/metrics/repository_check.json).
- All 42 required files are present, and relative Markdown file targets pass.
  All six notebooks have saved execution counts and no saved error outputs,
  covering 36 code cells. The notebooks were inspected without re-execution.
- The baseline, resolution, quality, and analysis summaries are completed;
  all 18 recorded output hashes match. Analysis input and source hashes pass.
  Saved-score recomputation reproduces all ten condition summaries on the
  same 5,999 eligible pairs, including the baseline's 13 false matches and
  37 false non-matches.
- `git fsck --full` passed for the real local repository, with 10 commits
  before this commit and HEAD `91cce44b1f7508db7ebbe383f910055d0fab6962`.
  No datasets, model caches, virtual environments, or ZIPs are tracked.
  The report's uncommitted working-tree state is expected before Commit 9.
- A separate SHA-256 comparison confirmed that all 48 pre-existing notebook,
  source, full-requirements, and result files remained unchanged. Earlier
  experiment-log entries were preserved; this entry was appended.

### Problems and adjustments

No tests or full-profile checks failed. Pip reported that its user cache was
not writable in the execution sandbox and disabled that cache; the dependency
check still passed and no installation was attempted. No dependency, model,
or inference-code changes were needed.

The full check has 21 checks rather than the bundle's 22 total checks because
the omitted commit-specific setup document no longer needs a link check.
Dependency and Git checks both ran; neither was skipped. The guide also makes
clear that notebook 06 requires the Jupyter environment, while the minimal
requirements support command-line use.

### Scope and limitations

This stage verified saved artifacts, documentation, installed direct dependency
versions, and local Git objects. It did not reproduce a fresh full installation,
rerun image inference, inspect inference-cache contents, or create a submission
ZIP. Notebook execution counts do not establish freshness. The checker does
not validate external URLs or Markdown anchors, and direct pins do not lock
all indirect dependencies. Earlier experimental limitations remain unchanged.

### Next step

Consolidate final findings and limitations, then create and inspect the
submission ZIP from the real local repository, including its `.git` directory.

## Stage 010 — Final Findings and Submission Workflow

Verification date: 2026-09-11, using the existing Python 3.13.7 environment.

### Goal

Consolidate the measured findings and prepare a verifiable repository submission.

### Changes

- Added [the final report](final_report.md), covering the problem, modality,
  pipeline, dataset selection, method, ten-condition results, conclusions,
  limitations, and possible future experiments.
- Added [submission instructions](submission.md) and linked both documents
  from README.md.
- Added `scripts/package_submission.py` to package committed project files
  with the actual `.git` directory, then verify the extracted repository.
- Added five packaging tests using temporary fixture histories, covering
  archive round trips, ignored-data omission, dirty repositories, existing
  or internal output paths, external object alternates, and symlinks.

### Verification

- `python -m unittest discover -s tests -v`: **46 tests passed**. Packaging
  tests preserved their fixture HEAD and two-commit history, reproduced a
  clean extracted working tree, and rejected unsupported packaging states.
- `python scripts/check_repository.py --profile full`: **23 passed, 0 failed,
  0 skipped**, including all nine direct dependency pins and Python 3.13.7.
  The two additional checks inspect links in the new report and submission
  guide. The saved Commit 9 `repository_check.json` was not overwritten.
- All six saved notebooks have execution counts and no saved error outputs.
  Recorded result hashes match, and all ten saved-score summaries reproduce
  on the same 5,999 eligible pairs. No notebooks or face inference were rerun.
- The final report's ten-row accuracy/FMR/FNMR/AUC/EER table was independently
  compared against full-precision saved metrics at the displayed rounding.
  The findings distinguish frozen-threshold operational results from pooled
  threshold-sweep summaries and retain the earlier limitations.
- Real local Git objects passed verification. Before this commit, HEAD was
  `da4024ee8fe91be4d3089710c43175fffbb24fdf`, with 11 commits. No datasets,
  model caches, virtual environments, or ZIPs are tracked.
- Earlier notebooks, source modules, dependency pins, metric exports, figures,
  and the Commit 9 check report remain unchanged. Earlier experiment-log
  entries and all README result tables were preserved.

### Problems and fixes

No local tests or full-profile checks failed. No package installation,
dependency changes, model changes, or inference-code fixes were required.
The source bundle's preparation results were treated as prior evidence;
the counts above are from this local run.

### Archive status

The submission archive will be generated after this commit and push, once
the working tree is clean. The packager reruns the full checks and tests,
includes the actual `.git` directory, extracts the archive, checks every
archived file hash, and verifies Git objects, HEAD, commit count, and clean
working-tree state. Its external verification receipt records the exact
archived HEAD, ZIP checksum, byte size, and actual verification results.

Ignored datasets, inference caches, model weights, virtual environments, and
downloaded update ZIPs are omitted. Packaging does not reproduce image
inference or establish installation on a second machine. The verified
submission ZIP must be uploaded through the course submission system; the
downloaded Commit 10 update bundle is not that submission.
