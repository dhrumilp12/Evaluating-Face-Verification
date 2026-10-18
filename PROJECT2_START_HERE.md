# Project 2: Quality-aware face verification under image degradation

The QMagFace reproduction and ten-condition robustness investigation are
complete. Project 2 extends the existing face-verification repository and keeps
Project 1's code, results, and Git history. The deliverables are the repository
with its `.git` history and the written report.

## Read the completed work

- [Final written report](docs/project2/final_report.md) and [report PDF](output/pdf/project2_report.pdf): methods, results, interpretation, and limitations.
- [Reproduction instructions](docs/project2/reproduce.md): environments, verified downloads, inference, saved-score analysis, and checks.
- [Executed protocol](docs/project2/protocol.md): alignment, degradations, calibration, failure handling, and scope.
- [Artifact and benchmark audit](docs/project2/artifact_audit.md): source links, hashes, official-code agreement, and benchmark differences.
- [Experiment log](docs/project2/experiment_log.md) and [AI usage record](docs/project2/ai_usage.md): development and assistance provenance.

## Measured results

On the authors' released LFW features, the official evaluation produces
**99.749722% MagFace-cosine accuracy and 99.833139% QMagFace accuracy** over
5,994 of 6,000 requested pairs. QMagFace corrects five cosine errors and introduces
none. The independently implemented scorer agrees with the official scorer to
within `4.8e-15`. The paper reports 99.73% and 99.83%; the audit explains the six
missing-feature pairs and the released evaluator's fold and threshold behavior.
See [the benchmark summary](results/project2/author_benchmark/summary.json).

The separate robustness experiment evaluates ten conditions with identical
embeddings and pair eligibility for both scoring methods. Each method's
threshold is calibrated on original-quality pairs in the other nine original
LFW folds and then held fixed for degraded probes. All conditions score
**5,999 of 6,000 pairs** (99.9833% coverage).

| Selected condition | MagFace cosine accuracy | QMagFace accuracy |
|---|---:|---:|
| Original | 99.800% | 99.817% |
| Resolution: 14 to 112 pixels | 58.827% | 65.411% |
| Gaussian blur: sigma 2.1 pixels | 99.500% | 99.733% |
| Brightness: x0.25 | 99.783% | 99.800% |

These are mean held-out-fold accuracies from
[evaluation.json](results/project2/robustness/evaluation.json). At 14-pixel
resolution, QMagFace corrects 398 cosine errors and introduces three. Both
methods still fail frequently: pooled FNMR is 82.36% for cosine and 69.16% for
QMagFace, while pooled FMR changes from 0% to 0.0333%. Brightness x0.75 produces
identical decisions. These findings show conditional benefits and remaining
failure modes, without a claim of universal improvement or statistical
significance.

Our local alignment is a documented five-landmark pipeline, not evidence of
identical crops to the authors' pre-aligned benchmark. The historical FaceNet
results remain a system reference; this work's controlled comparison changes
only the scoring rule on the same MagFace features.

## Verify and reproduce

From the repository root, with the existing inference environment available:

```bash
.venv/bin/python scripts/check_project2.py
.venv/bin/python -m unittest discover -s tests -v
```

The read-only checker validates saved output hashes and recomputes evaluation
from the exported scores. With local images, weights, and raw-feature caches,
add `--include-caches` to also check their provenance and reconstruct scores.
The complete [reproduction guide](docs/project2/reproduce.md) covers fresh
setup and reruns; a saved-score rerun does not require model inference.

The full robustness command used Apple Silicon MPS:

```bash
.venv/bin/python -m src.project2_experiment --stage run --device mps --grid full --batch-size 32 --threads 2
```

Use `--device cpu` on systems without MPS. A device/configuration change creates
a separate feature-cache fingerprint; use a separate `--output` directory to
retain another run. Downloaded images, checkpoint files, and caches are ignored
by Git and acquired using the verified commands in the reproduction guide.

The submission evidence is stored under
[results/project2/author_benchmark](results/project2/author_benchmark/) and
[results/project2/robustness](results/project2/robustness/), including manifests,
pair scores, predictions, exclusions, fold metrics, paired changes, and figures.
