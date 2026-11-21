# Project 3: Quality-aware calibration for face verification

**Dhrumil Patel — Biometrics capstone**

This project asks whether image quality helps a learned decision rule beyond
simply recalibrating cosine or QMagFace thresholds. It continues the same LFW
face-verification problem and uses the measured comparisons from Project 2.
The extension changes the comparison and decision stages, with the original
embeddings and probe degradations held fixed.

[First-study results](results.md) · [Target-FMR results](operating_points.md) ·
[Protocol](protocol.md) · [Target-FMR addendum](operating_points_protocol.md) ·
[Experiment log](experiment_log.md) · [AI assistance](ai_usage.md)

## Main finding

At 14-pixel probe resolution, QMagFace accuracy increases from **65.411%** with
original-only calibration to **85.748%** with mixed-condition calibration.
The new interaction model reaches **84.798%** on the same calibration data.
It reaches **67.478%** when calibration excludes every resolution degradation.
All figures are mean accuracies over the ten original pair folds.

Calibration data explains much of the improvement. Quality helps relative to a
cosine-only logistic model, but the interaction adds little and does not beat
the recalibrated QMagFace control at 14 pixels. More false matches accompany
the reduction in false non-matches. These are measured findings, including
negative results, rather than a claim of a universally better recognizer.

The feedback-driven extension calibrates all methods to a **1% empirical FMR
target**, using seven fitting folds, two separate threshold-calibration folds,
and one test fold. With mixed conditions, the learned score's 14-pixel test FMR
is **3.667%** under pooled calibration and **1.100%** under a per-condition cap;
FNMR rises from **27.575% to 41.846%**. A target met on calibration data does not
guarantee the same error rate on held-out pairs. The
[operating-point analysis](operating_points.md) reports both errors and all
methods, including the stronger QMagFace result at the capped operating point.

## Reproduce the experiments

Run from the repository root with Python 3.12. The recorded environment used
Python 3.12.14. This stage needs the committed Project 2 score files, which a
normal clone includes. It does not require downloading LFW, model weights, or
using a GPU. Follow the [Project 2 instructions](../project2/reproduce.md) if
you also want to regenerate the original image embeddings.

```bash
python3.12 -m venv .venv-project3
source .venv-project3/bin/activate
python -m pip install -r requirements-project3.txt

python -m unittest discover -s tests -p 'test_project3_*.py' -v
python scripts/check_project3.py

python scripts/project3_run.py --output results/project3/reproduction
python scripts/check_project3.py --run results/project3/reproduction
python scripts/project3_summarize.py --run results/project3/reproduction --output results/project3/reproduction_analysis

python scripts/check_project3_operating_points.py
python scripts/project3_operating_run.py --output results/project3/reproduction_operating
python scripts/check_project3_operating_points.py --run results/project3/reproduction_operating
python scripts/project3_operating_plot.py --run results/project3/reproduction_operating --output results/project3/reproduction_operating_analysis
```

On Windows, activate with `.venv-project3\Scripts\activate`. Use a new output
directory for another run. The runner refuses to overwrite existing results.
The first saved-score study took about 19 seconds and the target-FMR extension
about 10 seconds in the recorded environment; hardware and library versions
affect runtime. Each source manifest stores the environment,
input hashes, source hashes, settings, and exact execution revision.

The classifier uses fixed `C=1`. Sensitivity runs at `C=0.1` and `C=10` remain
separate results, without selecting a winner from held-out performance. All
variants of a held-out pair stay out of training, including when training
uses multiple image conditions. This preserves pair folds, not disjoint people
or disjoint images across folds.

The target-FMR extension fits fresh seven-fold models and keeps threshold
calibration separate. Its comparisons are controlled within that new split;
differences from the first study cannot be attributed solely to threshold policy.

## Repository map

| Path | Purpose |
|---|---|
| `src/project3_calibration.py` | Feature construction, fitting, grouped evaluation |
| `scripts/project3_run.py` | Run all primary experiments and sensitivity checks |
| `scripts/check_project3.py` | Independently reconstruct scores and verify saved metrics |
| `scripts/project3_summarize.py` | Regenerate scientific figures and summary data |
| `tests/test_project3_calibration.py` | Leakage, grouping, serialization, and inherited-threshold checks |
| `src/project3_operating_points.py` | Separate fitting/calibration/test roles and empirical FMR thresholds |
| `scripts/project3_operating_run.py` | Run the target-FMR extension |
| `scripts/check_project3_operating_points.py` | Independently verify operating thresholds and held-out decisions |
| `scripts/project3_operating_plot.py` | Reproduce the target-FMR tradeoff figure |
| `tests/test_project3_operating_points.py` | Target budgets, ties, split isolation, and policy checks |
| `results/project3/calibration/` | Fitted models, predictions, fold results, paired changes, provenance |
| `results/project3/analysis/` | Figures and summary data derived from the measured results |
| `results/project3/operating_points/` | Target-FMR models, thresholds, calibration and test evidence |
| `results/project3/operating_points_analysis/` | Target-FMR figure and provenance |
| `docs/project3/` | Study documentation and final presentation |

There are 6,000 requested pairs and 5,999 eligible pairs per condition. Repeated
conditions and fitted models do not increase the number of independent pairs.
The independent checker verifies every saved decision and the reported fold
statistics. It does not rerun the face detector or embedding network.

## Final presentation

- [Editable presentation](presentation.pptx)
- [PDF presentation](presentation.pdf)
- [Speaker notes and questions](speaker_notes.md)

The deck has thirteen main slides and three appendix slides, designed for
approximately 11–13 minutes plus questions. Adjust that
length if the instructor provides a specific time limit. The required capstone
deliverables are this documented, reproducible repository and the final
presentation. This Markdown documentation supplies the written analysis.

Retain the complete Git history when submitting the repository. A GitHub source
ZIP does not include `.git`; the supplied Project 3 submission ZIP does.
