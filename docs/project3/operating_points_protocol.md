# Extension: verification at a 1% calibration-FMR target

## Why this extension was added

Project 2 feedback identified the omitted target-FMR experiment as an unfinished
part of the original plan. The first Project 3 study reports FMR but optimizes
training accuracy for scalar controls and uses a zero threshold for learned
scores. Those results do not answer a target-FMR operating-point question.

This addendum is recorded before executing the new experiment. It preserves
the earlier protocol and results, and asks: **What false-nonmatch cost and
achieved test FMR result when all methods use thresholds calibrated for 1%
FMR? Does controlling a mixture conceal a difficult individual condition?**

The experiment reuses the same measured Project 2 comparisons. It is a
feedback-driven extension of a previously inspected benchmark, not a new
untouched validation dataset or a claim of deployment readiness.

## Fixed scope

- Scores: cosine, published QMagFace, and the full learned interaction score
  from Project 3 with fixed `C=1`.
- Conditions and eligibility: the ten existing conditions and inherited
  failures, unchanged from the earlier study.
- Regimes: `original_only`, `mixed`, and `leave_resolution_out`. The fitting
  and calibration conditions are identical within each regime. Original-only
  and mixed evaluate all ten conditions; resolution-excluded evaluates
  original quality and all three resolution conditions.
- Target: **1%** empirical calibration FMR, fixed before measurement. No
  lower target or regularization search is part of this extension.
- Policies: pooled calibration (primary) and a per-condition calibration cap
  (ablation). Neither policy consults the held-out test outcomes.

## Three separate roles for the original pair folds

For each outer test fold `f`, calibration uses `(f+1) % 10` and `(f+2) % 10`.
The other seven folds fit the learned score and its standardization. Every
condition variant of a pair remains in the same role. All three methods use
the same fitting, calibration, and test pair assignments.

Refit the full learned model using only the seven fitting folds, retaining
the original feature definitions, training-only standardization, `C=1`, and
weight `1/K` for each of the `K` condition variants of a training pair. Freeze
this model before scoring calibration data. Do not reuse the earlier nine-fold
models and do not refit after threshold selection. Cosine and QMagFace are
fixed score functions without learned parameters in this extension.

The separation prevents threshold fitting on the same pairs used to learn
the score. It remains pair-level separation; LFW folds are not guaranteed to
be image- or identity-disjoint. See the scikit-learn guidance on
[separate data for model fitting and threshold tuning](https://scikit-learn.org/1.8/modules/classification_threshold.html#important-notes-regarding-the-internal-cross-validation).

## Threshold rule, including tied scores

Use successfully scored **impostor** calibration records only. Let `N` be their
count and `k=floor(0.01*N)` the allowed false-match count. Sort their scores in
descending order and choose the next representable floating-point value above
the score at zero-based index `k`. Accept a pair when `score >= threshold`.

This excludes the entire boundary tie group and yields at most `k` calibration
false matches. It is the smallest representable feasible threshold, so it
maximizes acceptance under that empirical constraint without using genuine
calibration scores. Use exact decimal/rational arithmetic for the integer
budget. Do not interpolate a ROC curve or select a threshold on the test fold.

For the **pooled** policy, apply this rule to all included calibration
condition rows. It constrains the observed mixture, which weights conditions
equally because each condition has the same eligible pairs.

For the **condition-cap** policy, compute the rule separately for each observed
calibration condition and use the maximum of those thresholds. This remains
one global operating threshold, applied to every test condition without
requiring a condition label. It constrains each observed calibration condition
empirically. Original-only calibration makes both policies identical.

The cap threshold cannot be below the pooled threshold. On the same test
scores it cannot increase false matches or reduce false nonmatches. Neither
policy guarantees that held-out FMR meets 1%, especially for an excluded
degradation family.

## Reporting and sample limits

Report **achieved held-out FMR and FNMR at a calibration target of 1%**, together
with counts, coverage, accuracy, fold means and standard deviations. Retain
all specified methods, regimes, and policies regardless of which performs
best. Distinguish achieved calibration FMR from achieved test FMR. A test
result exceeding 1% is an outcome, not a reason to tune again.

Each calibration split has 600 unique impostor pairs, giving a budget of six
false matches per condition. Mixed calibration has 6,000 impostor condition
rows but still only 600 unique impostor pairs; the repeated versions are
correlated. Each held-out fold contains 300 impostors, so one false match
changes its FMR by about 0.333 percentage points. Across the ten held-out
folds there are 3,000 impostor pairs per condition, with possible shared images
or identities. Do not interpret condition rows as independent extra people,
make ultra-low-FMR claims, or add naive significance claims.

These models use seven fitting folds and a new threshold criterion. Their
accuracy is not a controlled one-variable comparison with the earlier
nine-fold, accuracy-oriented experiment. The controlled comparisons are
between methods and threshold policies **within this extension**.

## Evidence and verification

Save the source hash, execution revision, environment, model parameters,
three-way split fingerprints, thresholds, calibration counts by condition,
test predictions, fold results and aggregate results in a separate directory.
Never overwrite `results/project3/calibration/`.

Test the integer budget and ties, three-way isolation, genuine-score
independence of thresholds, policy ordering and inherited failure handling.
Independently reconstruct saved thresholds, predictions and metrics, including
the calibration constraint and the actual held-out errors. Update the results
discussion and presentation from executed outputs. Record design,
implementation, measured results and documentation as the changes actually
occur; do not manufacture activity by splitting trivial edits into commits.
