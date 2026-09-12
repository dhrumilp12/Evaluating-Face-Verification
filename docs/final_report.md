# Project 1 — Final Findings

## Evaluating Face Verification Under Degraded Image Quality

### Problem statement

This project asks: **How do low resolution, Gaussian blur, and reduced brightness
affect a pretrained face-verification system?** The task is 1:1 verification:
given a reference image and a probe image, determine whether they depict the
same person. The project evaluates an existing system rather than training a new
recognition model.

The biometric modality is the **face**, a physiological biometric. The pipeline
is dataset acquisition, face detection/cropping, embedding representation,
cosine-similarity matching, and a threshold-based decision. The experiments
focus on representation, matching, and decision after controlled probe-image
quality changes. Detection is held fixed using the original crops.

This matters because systems may encounter blurry, dim, or small face images.
A useful evaluation must distinguish incorrectly accepting different people
(false matches) from incorrectly rejecting the same person (false non-matches).

### Dataset selection

LFW funneled images were selected as the primary dataset because the dataset is
manageable and supplies a standard verification-pair protocol. Dataset exploration
validated 13,233 images from 5,749 identities. The evaluation uses the 6,000
supplied pairs in ten folds, with 300 genuine and 300 impostor pairs per fold
before preprocessing exclusions. Development pairs were used for the earlier
small pipeline check.

SCface and QMUL-SurvFace were researched as possible surveillance extensions.
They were not downloaded or evaluated in these experiments. Selection rationale,
sources and access considerations are recorded in [dataset research](dataset_research.md).

### Experimental method

The system uses MTCNN face detection/cropping followed by InceptionResnetV1
pretrained on VGGFace2. Input crops are 160 × 160 RGB images. Existing pixel
standardization is applied before generating unit-normalized 512-dimensional
embeddings. Similarity is the cosine similarity of the two embeddings; a match
is accepted when the score is at least the decision threshold.

For each evaluation fold, the baseline threshold is selected using original
scores from the other nine folds. That threshold is then fixed for the held-out
fold across all degraded conditions. Reference embeddings, original crops,
model configuration, fold membership, and eligible pairs stay fixed. Degradation
runs do not redetect faces or recalibrate thresholds.

Ten distinct conditions were evaluated:

- Original crops.
- Probe resolution reduced to 80, 40, or 20 pixels per side, then enlarged back
  to 160 × 160 for the model.
- Independent Gaussian blur with sigma 1, 2, or 3 pixels on the original crop.
- Independent brightness scaling by 0.75, 0.50, or 0.25 on original encoded RGB values.

Every degraded condition begins with the original probe crop; degradations are
not combined or accumulated. The actual full inference run used Python 3.13.7
and CPU. Exact packages and fingerprints are recorded in the result summaries.

### Coverage and baseline

Of 7,701 unique evaluation images attempted, 7,700 were processed successfully.
One image failed the fixed detection-confidence cutoff, excluding one genuine
pair. Every reported condition uses the same 5,999 eligible pairs: 2,999 genuine
and 3,000 impostor pairs. Coverage is 99.983%.

Original mean fold accuracy is 99.167%, mean FMR is 0.433%, and mean FNMR is
1.234%. There are 13 false matches and 37 false non-matches. These recognition
rates are conditional on preprocessing success; the exclusion is reported
separately rather than silently counted as a scored comparison.

### Results

Accuracy, FMR and FNMR below are unweighted means across the ten folds at their
saved baseline thresholds. AUC and interpolated EER are descriptive pooled-score
summaries across thresholds. They are not additional deployment evaluations.

| Condition | Accuracy | FMR | FNMR | Pooled AUC | Interpolated EER |
|---|---:|---:|---:|---:|---:|
| Original | 99.167% | 0.433% | 1.234% | 0.999116 | 0.900% |
| Resolution 80 × 80 | 99.117% | 0.433% | 1.334% | 0.999039 | 0.867% |
| Resolution 40 × 40 | 98.833% | 0.467% | 1.867% | 0.998535 | 1.267% |
| Resolution 20 × 20 | 89.982% | 0.967% | 19.071% | 0.984612 | 6.002% |
| Blur σ = 1 | 99.117% | 0.433% | 1.334% | 0.999051 | 0.834% |
| Blur σ = 2 | 98.917% | 0.433% | 1.734% | 0.998729 | 1.233% |
| Blur σ = 3 | 97.816% | 0.533% | 3.834% | 0.997686 | 1.867% |
| Brightness × 0.75 | 99.217% | 0.367% | 1.200% | 0.999161 | 0.867% |
| Brightness × 0.50 | 99.200% | 0.333% | 1.267% | 0.999098 | 0.867% |
| Brightness × 0.25 | 98.566% | 0.433% | 2.434% | 0.998527 | 1.333% |

Values are rounded for display. The full-precision values, individual predictions,
fold metrics and provenance are in [the saved metrics](../results/metrics/).

### Main findings

**Severe resolution reduction caused the largest deterioration among the tested
settings.** At 20 × 20, accuracy fell to 89.982%, and FNMR increased to 19.071%.
False non-matches rose from 37 to 572, while false matches rose from 13 to 29.
The pooled EER increased from 0.900% to 6.002%. Mean genuine cosine similarity
fell from 0.751540 to 0.536432; mean impostor similarity changed from 0.022740
to 0.032296. The main shift was genuine pairs becoming less similar.

**Increasing blur reduced verification performance.** At sigma 3, accuracy was
97.816% and FNMR was 3.834%. Pooled AUC was 0.997686 and interpolated EER was
1.867%. Most of the fixed-threshold deterioration again came from genuine
comparisons being rejected.

**Moderate brightness scaling produced small changes.** Factors 0.75 and 0.50
had slightly higher measured accuracy than the original condition. Those small
differences do not establish that dimming generally improves recognition. At
factor 0.25, accuracy fell to 98.566% and FNMR increased to 2.434%.

The score-distribution and ROC/DET analyses broadly support the fixed-threshold
findings. Small AUC/EER differences do not necessarily move together: AUC
summarizes the whole curve, while EER describes its equal-error crossing. Neither
metric replaces the original operating-point results.

### Figures

- [ROC comparisons](../results/figures/analysis_roc.png)
- [DET comparisons](../results/figures/analysis_det.png)
- [Genuine and impostor score distributions](../results/figures/analysis_score_distributions.png)
- [Resolution performance](../results/figures/resolution_comparison.png)
- [Blur and brightness performance](../results/figures/quality_comparison.png)

### Limitations

1. Degradations are synthetic and applied after original face cropping. The
   study does not measure how degraded images affect acquisition or detection.
2. RGB brightness scaling does not simulate all low-light effects, including
   sensor noise, camera exposure, or nonuniform illumination.
3. LFW is not a real surveillance-camera evaluation. Results should not be
   generalized to SCface, QMUL-SurvFace or deployment settings without testing.
4. Performance conditions on the same baseline preprocessing successes. One
   genuine pair remains excluded throughout.
5. The severity scales for resolution, blur and brightness are not matched.
   The strongest effect here applies to the tested settings, not a universal
   ranking of degradation types or a minimum camera-resolution requirement.
6. Training-data overlap with LFW was not independently audited. Demographic
   subgroup performance was not evaluated.
7. Repeated identities and overlapping fold calibration sets limit independence.
   No inferential significance test was performed. Fold SD/SE are descriptive.
8. Pooled EER is interpolated and may lie between attainable deterministic
   operating points. It is not a validated deployment threshold. With only
   3,000 impostor comparisons, very low false-match rates cannot be estimated
   precisely; zero observed errors does not establish zero population risk.
9. Reproduction documentation and direct dependency pins are provided, but they
   are not a full transitive lockfile or evidence of a fresh full installation
   on a second machine.

### Reproduction and documented evolution

The repository contains six executed notebooks, reusable source, saved scores,
metrics, figures, tests and a chronological experiment log. The work progressed
from problem definition and dataset selection through loading, pipeline checks,
full baseline evaluation, independent degradations, curve analysis and
reproduction checks. Actual Git history records these changes.

Commit 9's saved full-profile report records 21 passed checks, zero failures and
zero skips, including local direct dependencies and Git objects. The full test
suite at that stage had 41 passing tests. Commit 10 adds packaging tests; current
local verification passed all 46 tests and 23 full-profile repository checks,
with zero failures or skips. The five added tests use temporary fixture histories
to verify archive round trips and rejection of invalid packaging states. Actual
local outcomes are recorded in [Stage 010](experiment_log.md#stage-010--final-findings-and-submission-workflow).

Use [reproduction instructions](reproducibility.md) for either saved-score
analysis or the full image pipeline. Follow [submission instructions](submission.md)
to package committed files and the real Git history after the final commit.
The external verification receipt records the actual archive verification; this
report does not claim a submission archive was already created.

### Conclusion and future work

For this fixed pretrained system and protocol, severe probe-resolution reduction
and stronger Gaussian blur mainly increased false non-matches. Moderate RGB
brightness scaling had relatively little effect, while stronger darkening reduced
performance. Keeping thresholds, references and pair membership fixed made these
comparisons interpretable and repeatable.

Useful extensions include naturally degraded surveillance datasets, degradation
before detection, camera-noise/illumination models, combined degradations, and
carefully designed demographic and training-overlap audits. Any threshold
adaptation should use separate calibration data and retain an independent test.
