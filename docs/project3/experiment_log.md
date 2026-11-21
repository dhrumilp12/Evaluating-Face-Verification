# Project 3 experiment log

## November 21, 2026 — design before new experiments

The capstone continues the LFW face-verification problem from Projects 1 and 2.
Its required deliverables are documented, reproducible code and a final
presentation. The extension investigates quality-aware calibration while
preserving the existing detector, alignment, MagFace backbone, and measured
pair-score inputs.

Project 2 had already shown a large failure under severe resolution degradation
when original-quality thresholds were held fixed. This motivated a distinction
between changes to the decision boundary and loss of discriminative information.
The Project 3 dataset is therefore a previously studied benchmark, not an
untouched confirmatory test set.

Before producing new results, the [protocol](protocol.md) fixes:

- Five methods: scalar cosine and QMagFace thresholds, and logistic scores with
  cosine alone, cosine plus minimum raw norm, and an additional score/norm
  interaction.
- Original-only, mixed-condition, and three excluded-family training regimes,
  with every variant of a held-out pair excluded from training.
- Training-only feature standardization, primary `C=1`, a fixed zero logistic
  decision threshold, and equal total training weight per underlying pair.
- `C=0.1` and `C=10` sensitivity runs for mixed and leave-resolution-out training;
  no selection of a winning setting from held-out results.
- Joint reporting of accuracy, FMR, FNMR, ranking metrics, and paired changes.
  Improvements and regressions will both be retained.

The development branch starts from repository commit `81e5d06`. The exact full
source revision and input hash will be recorded with the executed results.
No Project 3 measurements are claimed in this initial entry. Later entries will
record executed commands, observed outcomes, verification, and any protocol
deviations. The final presentation will be built from the committed measured
outputs.

## November 21, 2026 — implementation and measured run

The protocol was committed as `bc036df` before the new measurements. The
implementation commit was `7c3c7e5`. The committed protocol specifies sensitivity
for the full interaction model only; the executed scope follows that choice.

Implementation added training-only scaling, weighted logistic models, scalar
threshold controls, the original/mixed/excluded-family regimes, and provenance
exports. Thirteen Project 3 unit tests passed before the main run, covering the
new calibration behavior. The experiment then ran through
`scripts/project3_run.py`, using the committed Project 2 robustness directory
and writing new results to `results/project3/calibration`.

The [run manifest](../../results/project3/calibration/manifest.json) records a
completed run from 19:57:01 to 19:57:19 UTC, with 290 fitted models and 1,128,000
requested prediction rows. Runtime was 18.818 seconds on the recorded Linux,
Python 3.12.14 environment with one BLAS thread. This is runtime for the new
saved-score experiment, excluding the earlier image-inference work. The source
score SHA-256 is
`0aefe1fabb3baaa79e0e60d939d1fe913dc661dcd8f754a69dc82ea05c928bbc`.

The measurements changed the expected story. At 14 pixels, recalibrating the
unchanged QMagFace score on mixed training conditions reaches 85.748% accuracy,
above the full learned model's 84.798%. The full model also has higher FMR and
FNMR than that control. Quality features help relative to score-only logistic
regression, but the interaction does not consistently improve on the simpler
quality model. Excluding resolution from training reduces full-model 14-pixel
accuracy to 67.478%. These findings are retained rather than selecting a
favorable method, condition, or regularization setting after evaluation.

The [results discussion](results.md) reports all ten mixed-calibration
conditions, both types of error, all three excluded-family investigations,
feature ablations, and the planned sensitivity settings. Primary `C=1` and
the prediction thresholds were not changed after the run. No new inference
or additional data collection was performed.

## November 21, 2026 — verification and reproducibility

The [independent saved-result audit](validation.json) passed. It checked source
and output hashes, training/test membership, scalers and scalar thresholds, all
290 fitted models, and all 1,128,000 prediction rows (1,127,812 scored). It also
reconstructed the 1,880 fold rows, 188 aggregate rows, and 312 paired-comparison
rows. The maximum reconstructed-score difference was zero.

One fresh CLI rerun used
`python scripts/project3_run.py --output results/project3/reproduction_check`.
It took 19.076 seconds and reproduced all five payload files byte for byte;
timestamps and elapsed-time metadata were intentionally excluded from equality.
The [reproducibility receipt](reproducibility.json) records that comparison and
the preservation check: all 134 existing tracked files outside `README.md` and
`.gitignore` matched the Project 2 base revision. The temporary repeat outputs
were removed after comparison. No additional experiment settings were tested.

The [analysis summary](../../results/project3/analysis/summary.json) and four
figures were generated from the measured outputs. The planned quality-support
check found only four out-of-range quality values among the 5,999 held-out
14-pixel pairs under leave-resolution-out training. All observed learned-score
cosine slopes remained positive. These descriptive checks do not establish
that training and evaluation joint distributions match.

The executed experiment follows the committed prospective protocol with no
deviations. The main run remains the 18.818-second run; the later run is
reproducibility evidence. Neither verification nor analysis reran image
alignment or embeddings. The final presentation uses the saved measurements
and includes the losses, limitations, and unsuccessful comparisons.

## November 21, 2026 — final presentation

The completed presentation contains twelve main slides and three appendix
slides. Its charts and tables use the measured results, with speaker notes
and answers to likely questions supplied alongside the slides. The editable
PowerPoint and its PDF copy were checked slide by slide for readable content
and consistent numbers. The intended speaking time is about 10–12 minutes;
that is a planning choice, not an instructor-specified limit.

The submission package includes the full repository and Git history, the
reproduction instructions, measured outputs, and both presentation formats.
The concise AI-assistance statement records the support used in this work.

## November 21, 2026 — extension prompted by Project 2 feedback

After the first Project 3 package was prepared, the Project 2 feedback became
available in this work session. It identified the omitted target-FMR study and
limited commit history as the remaining weaknesses. The earlier Project 3
experiment reports errors but does not calibrate to a target FMR, so reporting
those errors alone does not address the missing investigation.

The [operating-point addendum](operating_points_protocol.md) now fixes a 1%
calibration-FMR target, seven fitting folds, two separate calibration folds,
and one test fold. It compares cosine, QMagFace and the full learned score
under original-only, mixed and resolution-excluded conditions. Pooled
calibration is the primary policy; a per-condition cap is an ablation. Both
use a single global threshold at test time. The extension retains the existing
accuracy-oriented results and will report achieved test FMR even when it
exceeds the calibration target. This entry precedes its new measurements.

## November 21, 2026 — measured target-FMR extension

The addendum was committed as `538fca2`. Fourteen new tests passed before
measurement, including an example where pooled calibration hides an individual
condition's false-match tail. Implementation and tests were committed as
`fd12327`, the execution revision recorded by the new
[manifest](../../results/project3/operating_points/manifest.json).

`python scripts/project3_operating_run.py` completed in 9.913 seconds. It
created 30 learned fits and 60 fixed score definitions, 180 threshold
configurations, 1,080 calibration rows, 1,440 fold rows, 144 aggregate rows,
and 864,000 requested predictions (863,856 scored). It retained the original
source-score hash and all specified methods, regimes, and policies.

The new result clarifies the missing operating-point question. With mixed
calibration, the learned score's pooled policy has mean 14-pixel FMR 3.667%
and FNMR 27.575%, despite satisfying the 1% calibration-mixture constraint.
The condition-cap policy lowers these test false matches from 110 to 33
(FMR 1.100%) while increasing false nonmatches from 827 to 1,255
(FNMR 41.846%). Capped QMagFace has fewer errors of both types at this condition:
23 false matches and 1,180 false nonmatches. These results are retained without
retuning. The [discussion](operating_points.md) includes the original-only and
resolution-excluded results and the limits of the empirical constraint.

The [independent audit](operating_points_validation.json) passed for every
threshold and saved decision, with maximum reconstructed-score difference zero.
One fresh CLI run to an ignored output directory reproduced all six payloads
byte for byte in 10.387 seconds. The
[reproduction receipt](operating_points_reproducibility.json) records this
comparison and confirms that all 116 earlier tracked source, test, script,
requirements, and result files checked against the first Project 3 package
remain unchanged. Temporary repeat outputs were removed after comparison.

The figure and analysis were generated from these measured outputs. There were
no deviations from the operating-point addendum. The target constrains
calibration data; observed test FMR is reported even when it exceeds 1%.
