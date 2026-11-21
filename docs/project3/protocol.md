# Project 3: quality-aware calibration

## Question and scope

**Can calibration that accounts for image quality improve face-verification
decisions under degradation, beyond simply changing the decision threshold?**

Project 2 compared cosine and the published QMagFace score on fixed MagFace
embeddings, with thresholds fitted only to original-quality faces. Project 3
changes the comparison and decision stages. It learns a small quality-dependent
score and separates the effect of its features from the effect of exposing
calibration to degraded training examples. The face detector, alignment,
backbone, embeddings, pair definitions, and degradation operations stay fixed.

This document was written on November 21, 2026, before running the new Project 3
experiments. It is a prospective analysis specification for this extension,
not a claim that LFW is a previously unseen benchmark: the Project 2 results
were already available and motivated the question. Any subsequent changes and
their reasons belong in the [experiment log](experiment_log.md).

The new score is a course-project modification, not a claimed novel biometric
method or an exact reproduction of a second paper. QMagFace and MagFace supply
the research context; earlier work on face-score calibration supplies additional
motivation [1–4].

## Data and inherited measurements

The input is the committed
[Project 2 pair-score export](../../results/project2/robustness/pair_scores.jsonl).
Its provenance and image operations are documented in the
[Project 2 protocol](../project2/protocol.md) and
[run manifest](../../results/project2/robustness/manifest.json).
Project 3 reuses these measured scores; it does not claim a new model-inference
run. Record the input SHA-256 and source commit in the new run manifest.

There are 6,000 official LFW pairs in ten original folds. One genuine pair
failed the inherited detection criterion. The same 5,999 eligible pairs
(2,999 genuine, 3,000 impostor) occur in each of ten conditions. The 59,990
successful pair-condition rows are repeated measurements of 5,999 pairs, not
59,990 independent trials. Keep the failed pair and coverage visible in the
provenance; calculate this study's conditional metrics on the common eligible
set. No method may select an easier subset.

| Family | Conditions inherited from Project 2 |
|---|---|
| Original | Original aligned 112 by 112 crop |
| Resolution | Downsample to 56, 28, or 14 pixels, then restore to 112 |
| Blur | Gaussian sigma 0.7, 1.4, or 2.1 pixels |
| Brightness | Multiplication by 0.75, 0.50, or 0.25 |

Only the probe was degraded. The reference and original-image alignment remain
fixed. Transformations are separate, never compounded. The historical Project 1
FaceNet results use a different backbone, training set, crop, and pixel scales;
they provide context, not a controlled causal comparison.

## Five methods

Let `s` be cosine similarity, `r` the published QMagFace score, and
`q = min(reference_quality, probe_quality)`, where each quality is the raw
MagFace embedding norm. The predictor never receives the condition name,
severity, fold number, pair ID, image path, identity, or true label as a feature.

| Saved method name | Features or score | Decision rule |
|---|---|---|
| `cosine_threshold` | `s` | Training-fitted scalar threshold |
| `qmagface_threshold` | `r` | Training-fitted scalar threshold |
| `logistic_score` | `[s]` | Logistic decision score at least zero |
| `logistic_quality` | `[s, q]` | Logistic decision score at least zero |
| `logistic_interaction` | `[s, q, s*q]` | Logistic decision score at least zero |

For each logistic variant, fit a `StandardScaler` to its feature columns on
training rows only, then fit L2-regularized logistic regression with `C=1`,
`solver="lbfgs"`, `tol=1e-8`, and `max_iter=2000`. Check convergence. The intercept
is fitted; no class weighting is needed for these almost balanced pairs.
Save scaler parameters, coefficients, intercept, iteration count, and any
convergence warning. Preprocessing must be fitted within each split [5,6].

Each training row has weight `1/K`, where `K` is the number of included training
conditions. Every eligible underlying pair therefore contributes total weight
one, and augmentation does not multiply its effective regularization weight.
All conditions receive equal weight within a training regime. The logistic
decision threshold remains zero; it is not adjusted after examining results.
This is a learned decision score, not a validated deployment probability or
forensic likelihood ratio.

The two scalar controls maximize accuracy on exactly the same training rows
available to the learned methods. Preserve the Project 2 candidate convention:
distinct-score midpoints and finite accept-all/reject-all endpoints, the largest
threshold in an accuracy tie, and acceptance when `score >= threshold`. QMagFace
keeps the published coefficients `alpha=0.077428` and `beta=0.125926`.

The controls and logistic models optimize different training objectives.
Consequently, isolate the effect of adding quality through the logistic
ablations, and use the recalibrated scalar controls to determine whether gains
can be explained by a simpler threshold change. Do not credit all improvement
over a frozen Project 2 threshold to the quality feature.

## Outer folds and training regimes

For each held-out original LFW fold, train on eligible pairs from the other nine
folds. Exclude **all condition variants of every held-out pair**. Fit scaling,
model parameters, and scalar thresholds from those training rows only. Apply
the resulting model unchanged to the specified held-out conditions.

| Regime | Training conditions in the other nine folds | Held-out evaluation |
|---|---|---|
| Original only | Original | All ten conditions |
| Mixed | All ten conditions | All ten conditions |
| Leave resolution out | Original, all blur, all brightness | Original and all resolution |
| Leave blur out | Original, all resolution, all brightness | Original and all blur |
| Leave brightness out | Original, all resolution, all blur | Original and all brightness |

Original-only cosine and QMagFace predictions must reproduce the saved Project 2
decisions on the eligible set. This is an integration check, not a new historical
result. The mixed regime evaluates held-out pairs under known synthetic
degradation types. Only the family-excluded regimes test transfer to an excluded
degradation family. Neither setting establishes generalization to a new dataset
or to identities absent from training.

The primary hyperparameter is fixed at `C=1`. Repeat the full interaction model with
`C=0.1` and `C=10` for the mixed and leave-resolution-out regimes as sensitivity
analyses. Report all settings and keep `C=1` as the primary result. There is no
outer-test-driven winner selection. Nested hyperparameter cross-validation is
not used because no hyperparameter is selected from these evaluations.

## Measurements, checks, and interpretation

Export predictions with pair, fold, condition, label, score, threshold, and
decision; export each fitted model and its training-condition list. Verify
identical coverage, deterministic fold assignments, valid finite inputs,
training/test pair separation, and consistency of score, threshold, and decision.

Report accuracy, FMR, FNMR, confusion counts, ROC AUC, and interpolated EER for
each condition. Use mean per-fold metrics for primary comparisons. In particular,
calculate AUC and EER within each fold before averaging: separately fitted
logistic scores need not have exactly comparable scales across folds. Any pooled
metrics must be labeled separately. ROC/EER test-score sweeps are descriptive;
their thresholds do not become operational decision thresholds.

Summaries should include original quality, the strongest resolution degradation,
equal-condition macro averages, and corrected/introduced errors on identical
pairs. Report FMR and FNMR together. A method that lowers FNMR by accepting more
impostors has made a tradeoff, not an unconditional improvement. The primary
experiment does not optimize or guarantee a target low FMR. With 3,000 impostor
pairs, one false match changes pooled FMR by approximately 0.0333 percentage
points; the data cannot establish reliable ultra-low-FMR operation.

Inspect the interaction model's effective cosine slope over observed quality
values. An unconstrained interaction can produce a negative slope outside its
training support. Record extrapolation or non-monotonic behavior rather than
silently changing the model after seeing held-out outcomes. Improved thresholded
accuracy need not imply improved ranking, and calibration cannot restore image
information lost to degradation.

Fold variation is descriptive. Folds share calibration data, and image/identity
occurrences may overlap across different pairs. No statistical significance or
independence-adjusted confidence claim is planned. The study is pair-fold
evaluation, not identity-disjoint evaluation. Training/evaluation identity
overlap in the pretrained model has not been audited.

Other limits remain: synthetic brightness is not real noisy low light; fixed
alignment excludes degraded-image detection; quality magnitude is specific to
this pretrained MagFace model; balanced LFW priors and equal costs differ from
many deployments. Quality-dependent gains here do not establish demographic
fairness or production readiness. A negative or modest result is a valid study
outcome and must remain in the report and presentation.

## Sources

1. Terhorst et al. **QMagFace: Simple and Accurate Quality-Aware Face Recognition.**
   WACV 2023. [Paper](https://openaccess.thecvf.com/content/WACV2023/html/Terhorst_QMagFace_Simple_and_Accurate_Quality-Aware_Face_Recognition_WACV_2023_paper.html),
   [author implementation](https://github.com/pterhoer/QMagFace).
2. Meng et al. **MagFace: A Universal Representation for Face Recognition and
   Quality Assessment.** CVPR 2021. [Paper](https://arxiv.org/abs/2103.06627).
3. **Score calibration in face recognition.** IET Biometrics,
   [DOI:10.1049/iet-bmt.2013.0066](https://doi.org/10.1049/iet-bmt.2013.0066).
   Motivation for using auxiliary information in score calibration; this project
   does not reproduce that paper's face representation or categorical protocol.
4. **Calibration of score based likelihood ratio estimation in automated
   forensic facial image comparison.** Forensic Science International 334 (2022),
   111239. [DOI:10.1016/j.forsciint.2022.111239](https://doi.org/10.1016/j.forsciint.2022.111239).
   Related calibration research, not a basis for forensic claims in this study.
5. scikit-learn. [Common pitfalls: data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
6. scikit-learn. [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
   and [decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html).

