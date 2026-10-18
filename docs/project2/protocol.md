# Proposed evaluation protocol - freeze before full experiments

Status: frozen for execution on October 18, 2026, after the development pilot.
The full ten-condition grid below was selected before evaluating test scores.
The 16 pilot images are disjoint from the evaluation image paths; they supplied
throughput and compatibility evidence only. GPU inference took 0.588 seconds
for 16 images, and CPU/GPU raw outputs agreed within rtol=0.001, atol=0.0001.
The source checkpoint SHA-256 is
`cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58`.
See the saved pilot and inference validation under `results/project2/`.

Execution decisions: one BGR /255 network pass, no flip, 512-dimensional raw
output; MTCNN five-point alignment with the fixed ArcFace 112px template;
Pillow affine bilinear interpolation with black border. Resolution uses Pillow
bilinear down/up sampling. Blur uses a normalized sampled separable Gaussian,
radius ceil(3*sigma), and reflect padding. Brightness multiplies encoded RGB.
Blur/brightness use float64 arithmetic, clipping, nearest-even rounding and
uint8 conversion before the shared model input adapter. Detection runs once on
original images and both scoring methods share all successes and failures.
CPU alignment uses two threads; full inference uses MPS, batches of 32.

## 1. Separate benchmark reproduction from our robustness experiment

The benchmark reproduction should use author-provided features/pairs when
available, matching model identity and the official evaluation implementation.
Our scorer must be compared against the official scorer on identical inputs.
Record the published accuracy, measured accuracy, numerical gap, artifacts, pair
coverage, and every known protocol difference. A rerun using our own alignment
is a separate experiment, not automatically an exact paper reproduction.

The robustness experiment will use the existing LFW source-image identities and
pair definitions. Preserve pair order, reference/probe roles, labels, and the ten
provided folds. Apply a degradation only to the probe. Apply each condition
independently to the original aligned crop; do not compound transformations.

## 2. Model, alignment, and quality provenance

Use one compatible MagFace-100 checkpoint for both cosine and QMagFace scoring.
Before inference, document source URL, observed SHA-256, architecture, training
source, package versions, and preprocessing. Publisher checksum verification and
an observed local hash are different claims.

MagFace documentation specifies five-landmark alignment and 112 x 112 images.
Project 1 used a 160 x 160 box crop without additional landmark rotation. Do not
pass those old crops into a new backbone and call that the paper's alignment.
Do not guess RGB/BGR ordering, input standardization, flip augmentation, or
feature aggregation. Verify them against the selected checkpoint's inference
code. For an original/flip combination, verify exactly when norms and unit
normalization are computed; averaging normalized features can destroy the
intended quality interpretation.

Align the original image first and keep the alignment fixed when degrading the
probe crop. This isolates post-alignment verification robustness, not detection
robustness. Save alignment failures and the successful pair eligibility mask.

Store raw embeddings and raw norms alongside unit embeddings; validate finite,
nonzero, 512-dimensional outputs. Independently validate path-to-row mappings.
Do not use image labels, detector confidence, or brightness as substitutes for
the learned quality input.

## 3. Planned degradation grid

For the planned 112 x 112 MagFace crops:

| Family | Planned levels |
|---|---|
| Original | 112 x 112 crop, no synthetic degradation |
| Resolution | Downsample to 56, 28, or 14 square pixels, then restore to 112 |
| Gaussian blur | sigma = 0.7, 1.4, or 2.1 pixels on the 112-pixel crop |
| Brightness | Multiply pixel values by 0.75, 0.50, or 0.25 |

This preserves the relative resolution reductions and relative blur scales of
Project 1's 160-pixel design. It does NOT make a 14-pixel MagFace crop identical
to Project 1's 20-pixel crop, nor does it erase differences in crop geometry.
Document resampling, Gaussian implementation, clipping, dtype, and transform
order in the next increment. Do not silently relabel the new conditions as the
old absolute pixel settings.

First time a development-only pilot. The proposed minimum full-data scope is
original, 14-pixel resolution, sigma=2.1 blur, and brightness x0.25. Expand to the
full grid after the runtime check. Decide the grid before examining full test
results and record any later amendments as exploratory.

## 4. Parameters and thresholds

Keep the published QMagFace-100 alpha/beta fixed in the primary reproduction and
robustness experiment, after compatibility is verified. Do not refit parameters
on LFW's held-out scores or choose them separately for favorable test conditions.

For our robustness experiment, select each method's threshold on original-quality
pairs in the other nine folds, maximizing calibration accuracy with the existing
largest-threshold tie rule. Apply that one threshold to the held-out original and
all degraded conditions for that method. Repeat across the ten folds.

Use separate calibrated thresholds for MagFace-cosine and QMagFace; their score
scales differ. Existing `src/evaluation.py` accepts general finite scores in its
basic helpers, but `cross_validate()` hardcodes the field `cosine_similarity`.
Add a tested score-key interface or a separate Project 2 runner before using it
for adjusted scores. Never write QMagFace values into a misleading cosine column.
Do not change the historical Project 1 outputs while doing this.

A secondary operating-point analysis may select thresholds at a target 1% FMR
using calibration folds only, then report the actually achieved held-out FMR and
FNMR. Treat ROC/EER threshold sweeps as descriptive summaries, not deployable
thresholds chosen from test labels. A lower FNMR with a higher FMR is a tradeoff,
not an unqualified improvement.

## 5. Pair eligibility and failure reporting

Score B and C from identical MagFace embeddings and exactly the same pairs.
Count requested, scored, and excluded pairs by label and fold, with reasons.
Keep failed attempts in exported records; do not remove them from denominators
without reporting conditional metrics and coverage separately.

For any direct cross-system comparison with FaceNet, use a predeclared common
success set and report each system's full coverage as well. The common set may be
smaller than Project 1's 5,999 pairs; its size is not known yet. Keep a separate
published-benchmark run on its original eligible set rather than masking it to
force agreement with our historical pipeline.

## 6. Measurements and interpretation

Primary outputs: per-fold accuracy, FMR, FNMR, error counts, thresholds, coverage,
and paired B-versus-C decision changes. Also save ROC/DET plots, AUC/EER summaries,
raw-quality distributions, genuine/impostor score distributions, and changes from
each method's own original condition.

Use the same folds/pairs for every paired comparison. Fold SD/SE may be shown
descriptively; overlapping calibration folds and repeated subjects mean they are
not proof of independent experimental replicates. Do not treat ten condition
means as independent samples or claim significance from tiny accuracy changes.
Plan an appropriate dependence-aware uncertainty analysis before adding formal
confidence claims. No result is assumed to favor QMagFace.

There are only 3,000 nominal impostor comparisons in the usual 6,000-pair protocol.
One false match corresponds to about 0.0333% pooled FMR (1/3,000). A fold has still
less resolution. Do not claim reliable 1-in-100,000 performance from this sample
or equate zero observed false matches with a true zero error probability.

Synthetic brightness scaling does not reproduce all aspects of real low-light
capture. This benchmark is not evidence of demographic fairness or deployment
safety. Training/evaluation identity overlap remains an audit limitation unless
checked; pretrained models must not be described as trained only on LFW.

## 7. Reproducible outputs

Preserve Project 1's existing files. New runs should write under dedicated
`results/project2/` and `data/processed/project2/` paths, including:

- A run manifest with source hashes, environment, settings, model identity,
  parameter source, status, and output hashes.
- Pair records with pair index, fold, label, reference/probe paths, condition,
  success/failure status, both raw qualities, cosine score, adjusted score,
  method-specific threshold, prediction, and exclusion reason.
- Per-fold summaries, plots, execution time, and written observations.

Large caches, images, and checkpoints remain out of Git. The final repository
must include actual execution instructions and meaningful history. Use the
course's final Project 2 submission requirements when they are confirmed.
