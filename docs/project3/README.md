# Project 3: Quality-aware calibration for face verification

**Dhrumil Patel — Biometrics capstone**

This project asks whether image quality helps a learned decision rule beyond
simply recalibrating cosine or QMagFace thresholds. It continues the same LFW
face-verification problem and uses the measured comparisons from Project 2.
The extension changes the comparison and decision stages, with the original
embeddings and probe degradations held fixed.

[Results](results.md) · [Protocol](protocol.md) ·
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

python -m unittest discover -s tests -p 'test_project3_calibration.py' -v
python scripts/check_project3.py

python scripts/project3_run.py --output results/project3/reproduction
python scripts/check_project3.py --run results/project3/reproduction
python scripts/project3_summarize.py --run results/project3/reproduction --output results/project3/reproduction_analysis
```

On Windows, activate with `.venv-project3\Scripts\activate`. Use a new output
directory for another run. The runner refuses to overwrite existing results.
The saved-score run took about 19 seconds in the recorded environment; hardware
and library versions affect runtime. The source manifest stores the environment,
input hashes, source hashes, settings, and exact execution revision.

The classifier uses fixed `C=1`. Sensitivity runs at `C=0.1` and `C=10` remain
separate results, without selecting a winner from held-out performance. All
variants of a held-out pair stay out of training, including when training
uses multiple image conditions. This preserves pair folds, not disjoint people
or disjoint images across folds.

## Repository map

| Path | Purpose |
|---|---|
| `src/project3_calibration.py` | Feature construction, fitting, grouped evaluation |
| `scripts/project3_run.py` | Run all primary experiments and sensitivity checks |
| `scripts/check_project3.py` | Independently reconstruct scores and verify saved metrics |
| `scripts/project3_summarize.py` | Regenerate scientific figures and summary data |
| `tests/test_project3_calibration.py` | Leakage, grouping, serialization, and inherited-threshold checks |
| `results/project3/calibration/` | Fitted models, predictions, fold results, paired changes, provenance |
| `results/project3/analysis/` | Figures and summary data derived from the measured results |
| `docs/project3/` | Study documentation and final presentation |

There are 6,000 requested pairs and 5,999 eligible pairs per condition. Repeated
conditions and fitted models do not increase the number of independent pairs.
The independent checker verifies every saved decision and the reported fold
statistics. It does not rerun the face detector or embedding network.

## Final presentation

- [Editable presentation](presentation.pptx)
- [PDF presentation](presentation.pdf)
- [Speaker notes and questions](speaker_notes.md)

The deck is designed for approximately 10–12 minutes plus questions. Adjust that
length if the instructor provides a specific time limit. The required capstone
deliverables are this documented, reproducible repository and the final
presentation. This Markdown documentation supplies the written analysis.

Retain the complete Git history when submitting the repository. A GitHub source
ZIP does not include `.git`; the supplied Project 3 submission ZIP does.
