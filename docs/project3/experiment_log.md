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
