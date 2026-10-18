# Executed Project 2 evaluation protocol

Status: completed on October 18, 2026. This document began as the proposed
protocol in the first working increment. The full ten-condition grid was frozen
after the development pilot and before test-score evaluation. The original
planning context remains in [plan.md](plan.md) and the execution history in
[experiment_log.md](experiment_log.md). This revision records the conventions
actually used; it does not change the saved experiment.

The starter's minimum four-condition scope and optional target-1%-FMR analysis
were planning options. The executed investigation includes all ten conditions;
no separate target-FMR calibration experiment was performed.

## 1. Two experiments with distinct protocols

**Author-feature benchmark.** The reproduction used the authors' released
MagFace-100 LFW embeddings, filename mapping, and pair file. It executed the
hash-verified official cosine scorer, QMagFace scorer, and evaluation code.
Independent scores agreed within `1e-12`, and produced identical official-CV
accuracy. The released features permit 5,994 of 6,000 pairs. Upstream silently
excludes the other six and then repartitions surviving pairs into ten contiguous
KFold splits. Its 600-point threshold grid spans global score extrema; it uses
strict `score > threshold` and the smallest threshold in an accuracy tie. These
behaviors were preserved for reproduction and explicitly documented, including
the use of held-out score extrema to define the grid. See
[artifact_audit.md](artifact_audit.md) for results and exact source identities.

**Robustness investigation.** This experiment used the repository's LFW source
images and original 6,000-pair definitions. It preserved pair order,
reference/probe roles, labels, and all ten original folds. A synthetic degradation
was applied only to the probe, independently from the original aligned crop.
The reference remained at original quality. Its local alignment and calibration
are separate from the author-feature experiment; their accuracies are not
interchangeable measures of exact paper reproduction.

## 2. Model, alignment, and quality provenance

Both methods use the same iResNet100 MagFace checkpoint trained on MS1MV2, with
512-dimensional raw output. The observed checkpoint SHA-256 is
`cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58`.
The primary source, code versions, file hashes, and environment are recorded in
[the artifact manifest](artifact_manifest.json) and
[the run manifest](../../results/project2/robustness/manifest.json). An observed
local hash identifies downloaded bytes; it is not a publisher-signed checksum.
The network was neither retrained nor fitted on LFW in this work.

The executed adapter supplies BGR unsigned-byte pixels divided by 255, with no
additional centering or scaling, one unflipped network pass, and no output L2
normalization before retaining the raw features. These conventions were checked
against official inference code. Raw embeddings, raw norms, and unit features
are stored separately; finite, nonzero, 512-dimensional output and cache
raw/norm/unit consistency are validated. The quality input is the raw MagFace
norm, not detector confidence or an image brightness statistic. Project 1's
normalized FaceNet cache was not reused as a quality estimate.

The local detector is `facenet_pytorch.MTCNN`, with minimum face size 20,
thresholds `[0.6, 0.7, 0.7]`, factor 0.709, and a minimum detection probability
of 0.9. Among detections containing the image center, the highest-probability
box is selected. Five landmarks are mapped by a least-squares similarity
transform to the fixed ArcFace 112-pixel template. Cropping uses Pillow affine
bilinear interpolation, an integer pixel-center convention, and black fill.
This differs from Project 1's 160-pixel box crop and does not establish exact
agreement with the authors' pre-aligned benchmark images.

Detection and alignment run once on each original image. All degradations use
that fixed crop, so the experiment measures post-alignment robustness rather
than detector robustness under degradation. The 16-image development pilot had
no image-path overlap with the evaluation images; this is not a claim of
subject-identity disjointness. It checked compatibility and throughput only.
MPS inference took 0.588 seconds for the pilot's 16 images; the two-image
CPU/MPS comparison passed `rtol=0.001, atol=0.0001`. The full run used MPS batches
of 32 and two CPU threads for alignment, with seed 42.

## 3. Executed degradation grid

All inputs start from a 112 by 112 aligned crop:

| Family | Executed levels | Implementation |
|---|---|---|
| Original | No synthetic degradation | Original aligned crop |
| Resolution | 56, 28, or 14 square pixels, restored to 112 | Pillow bilinear downsample and restore |
| Gaussian blur | sigma 0.7, 1.4, or 2.1 pixels | Normalized sampled separable Gaussian, radius `ceil(3*sigma)`, reflect padding |
| Brightness | Factors 0.75, 0.50, or 0.25 | Multiply encoded RGB pixels without gamma linearization or added sensor noise |

Blur and brightness use float64 arithmetic, clipping to `[0,255]`, nearest-even
rounding with `numpy.rint`, and uint8 conversion before the common BGR network
adapter. The transformations are never compounded.

The resolution and blur levels preserve relative scales from the Project 1
160-pixel design. A 14-pixel downsample here is not the same image operation as
a 20-pixel downsample of Project 1's differently aligned crop. Synthetic
brightness multiplication does not reproduce every property of low-light capture.

## 4. Fixed coefficients and original-quality calibration

The published QMagFace-100 coefficients remain `alpha=0.077428` and
`beta=0.125926` throughout both experiments. They were not refitted on held-out
scores or selected separately by degradation condition.

For each robustness method and original LFW fold, the threshold maximizes
accuracy over successful original-quality pairs in the other nine folds.
Candidates are midpoints between distinct calibration scores plus finite
accept-all/reject-all endpoints. Ties choose the largest threshold, and a match
requires `score >= threshold`. That one method/fold threshold is applied
unchanged to the held-out original condition and every degraded condition.
Calibration never uses a held-out pair's label or a degraded condition's score.
The two methods use separate thresholds because cosine and QMagFace scores have
different scales.

The Project 2 evaluation implementation exports distinct `cosine_similarity`
and `qmagface_score` fields and preserves historical Project 1 outputs. Saved
predictions include the method-specific threshold. ROC, AUC, DET, and
interpolated EER are descriptive score sweeps; their test-score thresholds are
not used to produce held-out predictions. A lower FNMR accompanied by a higher
FMR is reported as a tradeoff.

## 5. Eligibility, coverage, and paired comparisons

The robustness run requested 7,701 unique image paths; 7,700 aligned
successfully. `Princess_Aiko/Princess_Aiko_0001.jpg` failed the detection
probability criterion. This excludes one genuine pair in original zero-based fold 1 from
conditional scoring in every condition. All ten conditions and both methods
therefore score the same 5,999 pairs: 2,999 genuine and 3,000 impostor. Coverage
is 99.9833%. There are 60,000 requested pair-condition records, including ten
failure records, and 120,000 method-specific prediction records.

Failed attempts remain in the exports with null scores and an exclusion reason.
Conditional accuracy, FMR, and FNMR are accompanied by requested/scored/excluded
counts, label-specific coverage, and separate reject-on-failure metrics over all
requested attempts. Undefined metrics are null rather than fabricated zeros.
Missing-fold metrics invalidate the corresponding mean while retaining a count
of valid folds; all ten executed folds have both classes.

Paired cosine/QMagFace comparisons use identical eligible pairs and count both
corrected and newly introduced errors. Changes from a method's own original
condition use the common successful pair set and report its size. The author
benchmark retains its own six exclusions and upstream fold protocol rather than
being restricted to the local robustness success set.

Project 1 FaceNet results are historical whole-system context only. No new
matched-pair FaceNet experiment was executed, so differences across those
systems cannot be attributed solely to quality-aware scoring.

## 6. Measurements and limits of interpretation

Saved measurements include per-fold and pooled accuracy, FMR, FNMR, confusion
counts, thresholds, coverage, paired decision changes, and changes from original
quality. The exports also include ROC/DET curves, AUC and interpolated EER,
raw-quality distributions, and genuine/impostor score distributions. Distribution
summaries are weighted by successful pair occurrences, not unique identities or
unique images. Mean-fold and pooled accuracies are separately labeled.

Fold SD/SE are descriptive. Overlapping calibration sets and repeated subjects
prevent treating them as independent experimental replicates. No formal
significance or dependence-adjusted confidence claim is made. Improved, unchanged,
and degraded outcomes are all retained; for example, introduced errors accompany
some low-resolution gains, and brightness x0.75 gives identical decisions.

With 3,000 impostor comparisons, one false match changes pooled FMR by about
0.0333 percentage points. This sample does not establish reliable
1-in-100,000 operation, and zero observed false matches is not a true-zero-error
guarantee. The experiment also does not establish demographic fairness or
production suitability. Training/evaluation identity overlap was not audited.

## 7. Evidence, verification, and delivery

The completed robustness outputs are under
[results/project2/robustness](../../results/project2/robustness/); the separate
author benchmark is under
[results/project2/author_benchmark](../../results/project2/author_benchmark/).
Manifests record status, environment, model/configuration/source identities,
coverage, timings, and output hashes. Pair records retain paths, fold, label,
condition, success/failure, both raw qualities, and both scores. Prediction
records add each method's threshold and decision. Large crops, raw-feature
caches, images, and checkpoint files remain ignored by Git.

The [reproduction guide](reproduce.md) documents full inference, independent
saved-score reanalysis, and read-only verification. `scripts/check_project2.py`
recomputes evaluation and verifies output hashes; `--include-caches` also checks
local input/crop/checkpoint provenance and reconstructs saved scores from raw
features. Historical Project 1 results remain intact. The confirmed final
format is the repository with its `.git` history plus a written report; slides
are not required.
