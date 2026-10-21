# Reproduce Project 2

Run all commands from the repository root. There are two experiments. The
author-feature benchmark checks the published scoring and evaluation code using
the authors' saved features. The robustness experiment aligns LFW images and
degrades the probes before extracting new embeddings. Both runs are complete;
their manifests and summaries record coverage, environment and measurements.

To check the saved results without repeating face detection or inference, use
the saved-score analysis command below. The [report](final_report.md) discusses
the findings. This guide also covers [deliverables](#deliverables),
[verification and packaging](#verify-and-package), and
[report rebuilding](#rebuild-the-written-report).

## Deliverables

The submission package contains the complete repository, including its `.git`
history, and a written report in [PDF](../../output/pdf/project2_report.pdf) and
[Markdown](final_report.md). This follows the requested scope; slides are not
included.
The supplied Project 2 instruction slide does not itself specify a written-report
requirement, report page limit or slides format; any additional formatting rules
must come from the professor's instructions or submission page.

Alongside the Project 1 history, the repository contains Project 2's code, tests,
protocol, scores, per-fold predictions, plots and report generator. Source and
artifact hashes identify the inputs; the AI-assistance record documents how the
work was prepared. Downloaded face images, author feature arrays, checkpoints
and local environments are excluded from Git.
The [artifact acquisition](#obtain-and-verify-model-and-author-features) and
[LFW preparation](#prepare-lfw-images-for-robustness-inference) sections below
explain how to obtain and verify the experiment inputs.

## Environments

Use the full environment for inference, plots and tests:

```bash
source .venv/bin/activate
python --version
python -m pip check
```

Direct dependencies are pinned in [requirements.txt](../../requirements.txt).
For a fresh checkout, create `.venv` with Python 3.13, install that file, and run
`pip check`. The reference repository uses Python 3.13.7. These are direct pins,
not a complete transitive lockfile; actual versions are recorded with each run.
Project 1's `requirements-analysis.txt` does not provide all dependencies needed
by the Project 2 command-line runner.

The author benchmark also needs SciPy and scikit-learn. Install these in a
separate environment to preserve the existing inference setup:

```bash
python3 -m venv .venv-project2
.venv-project2/bin/python -m pip install -r requirements-project2-benchmark.txt
.venv-project2/bin/python -m pip check
```

The benchmark commands below name their interpreter explicitly. Its dependencies
are listed in [requirements-project2-benchmark.txt](../../requirements-project2-benchmark.txt).
The local benchmark environment may reuse the existing NumPy installation;
the fresh-setup command above installs the declared NumPy version directly.

## Obtain and verify model and author features

The acquisition script downloads the audited author checkpoint, the author's
feature archive, and the small upstream scorer/metric source files:

```bash
.venv-project2/bin/python scripts/project2_acquire_artifacts.py
.venv-project2/bin/python scripts/project2_acquire_artifacts.py --verify-only
```

The full download is approximately 6.5 GB, mostly the author archive, which
contains benchmarks beyond the LFW subset used here. Allow additional disk
space for extracted data, the checkpoint, and inference caches. To prepare
only the robustness experiment, use `--checkpoint-only`. To prepare only the
author-feature benchmark, use `--features-only`. `--verify-only` checks local
artifacts without fetching replacements.

Expected local paths are:

| Artifact | Path |
|---|---|
| MagFace-100 checkpoint | `models/project2/magface_epoch_00025.pth` |
| LFW raw features | `data/project2/author_lfw/embeddings_lfw.npy` |
| Feature-to-filename mapping | `data/project2/author_lfw/filenames_lfw.npy` |
| Author pair list | `data/project2/author_lfw/lfw_pairs.txt` |
| Pinned official scorer and metric files | `data/project2/vendor/QMagFace-main/` |

The expected hashes were computed during the artifact audit; they are not
publisher-issued checksums. The runner rejects a checkpoint whose bytes differ
from the audited checkpoint associated with the
fixed QMagFace-100 coefficients. Read [the artifact audit](artifact_audit.md)
for source URLs, exact hashes, upstream commits, and benchmark differences.

## Rerun the author-feature benchmark

```bash
.venv-project2/bin/python scripts/project2_benchmark.py
```

`--download` also obtains missing author-feature/scorer inputs. `--features`,
`--official-source`, and `--output` can select explicit directories; the default
output is `results/project2/author_benchmark/`.

This command verifies the feature/mapping/pair files, executes four hash-pinned
upstream source files, checks agreement with our independent scorer, and writes
`summary.json`, `pair_scores.csv`, `fold_metrics.csv`, and `excluded_pairs.csv`.
The summary records the measured scores, differences from the paper, exclusions,
package versions and output hashes.

The official evaluator repartitions the successful pairs into ten unshuffled
KFold splits after exclusions. Its threshold grid spans global score extrema,
uses strict `score > threshold`, and resolves ties with the first threshold.
This benchmark retains those choices to reproduce the released evaluator. The
robustness experiment below instead preserves the original folds for calibration.
The report discusses the limitations of each protocol.

## Prepare LFW images for robustness inference

Reuse the existing validated `data/lfw_home/` dataset. The required files are
`pairs.txt`, `pairsDevTrain.txt`, and the original images under
`lfw_funneled/<identity>/`. The development pair file is needed for the pilot.

If the dataset is absent, acquire it through the existing verified downloader:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.lfw_dataset import download_lfw
download_lfw(Path.cwd())
PY
```

This calls the downloader directly, preserving the historical Project 1 dataset
report. It verifies the archive and pair files and extracts the source images.

## Pilot and full robustness run

The pilot checks compatibility and throughput using sixteen development image
paths that do not appear in the evaluation set. It does not tune scores,
coefficients or thresholds using test outcomes:

```bash
.venv/bin/python -m src.project2_experiment --stage pilot --device cpu
```

On an Apple Silicon machine with PyTorch MPS support, use `--device mps`. The MPS
pilot also compares two embeddings with CPU output and records the observed
difference and the tolerance check. Its output is `results/project2/pilot.json`.
Separate image paths do not guarantee separate subject identities.

Run the complete ten-condition grid on CPU:

```bash
.venv/bin/python -m src.project2_experiment --stage run --device cpu --grid full --batch-size 32 --threads 2
```

Or select MPS explicitly on a supported Mac:

```bash
.venv/bin/python -m src.project2_experiment --stage run --device mps --grid full --batch-size 32 --threads 2
```

Use one of these commands for a given run. To retain separate device experiments,
add distinct directories such as `--output results/project2/robustness_cpu` and
`--output results/project2/robustness_mps`. The default destination is
`results/project2/robustness/`. Runtime depends on the device and whether caches
already exist; use the pilot's measured throughput for planning.

The full grid is original quality; downsample to 56, 28, or 14 pixels and restore
to 112; Gaussian blur with sigma 0.7, 1.4, or 2.1; and brightness factors 0.75,
0.50, or 0.25. `--grid minimum` selects original, resolution 14, blur 2.1, and
brightness 0.25. Choose the grid before inspecting test results and record any
later change as exploratory.

Alignment runs on the original image on CPU. Each probe transformation starts
from that original 112-pixel aligned crop; the reference embedding stays fixed.
Inference uses BGR pixels divided by 255, without flip augmentation, and saves
the raw 512-dimensional embedding before unit normalization. The raw norm is
the quality input. These preprocessing choices define the robustness experiment;
they need not match the preparation of the author's pre-aligned benchmark images.

Alignment can also be prepared without loading recognition weights:

```bash
.venv/bin/python -m src.project2_experiment --stage align --threads 2
```

## Caches, failures, and calibration

Alignment records and PNG crops live under
`data/processed/project2/alignment/<fingerprint>/`. Reuse checks the source
image hash and successful crop hash. Detection/alignment failure records remain
available, with reasons. The same original alignment determines eligibility
for every robustness condition.

Raw features, unit features, qualities, and crop hashes live under
`data/processed/project2/features/<fingerprint>/<condition>/`. The fingerprint
includes checkpoint, adapter, architecture, transformation code, device,
environment, batch size, and thread settings. Reuse checks input hashes and
raw/norm/unit consistency. Rerunning the same command resumes work from validated
caches. Changing the device or a relevant configuration setting starts a separate
feature cache. After an interruption, check the manifest and rerun the command
to finish the remaining work.

The pair export retains every requested pair for each condition, including
failure rows with null scores. MagFace cosine and QMagFace share embeddings,
pair identities, and success eligibility. QMagFace keeps the published
`alpha=0.077428` and `beta=0.125926` fixed.

For each of the ten original folds, each method's threshold maximizes accuracy
on successful original-quality pairs from the other nine folds. Ties choose
the largest threshold. Acceptance is `score >= threshold`. That method/fold
threshold is reused unchanged for the held-out original and degraded pairs.
The two methods have separate thresholds because their score scales differ.

Conditional accuracy, FMR, and FNMR are accompanied by requested/scored/excluded
counts, label-specific coverage, and a separate reject-on-failure policy.
Missing test classes produce null metrics, not zero error rates. Fold means are
null when any fold lacks that metric, with valid-fold counts exported. Direct
condition changes use the same successful pairs in original and degraded
conditions and report the size of that common set. Paired B/C decision changes
separately count corrected and introduced errors.

ROC, AUC, and interpolated EER are descriptive score sweeps. Their thresholds
are not used for held-out predictions. Fold SD/SE describe variation across folds;
they do not establish sample independence or statistical significance. Quality
summaries weight successful pair occurrences, so
repeated subjects or images can contribute more than once.

## Reanalyze saved robustness scores

Once `pair_scores.jsonl` is available in a completed run directory, regenerate
metrics and figures without images, weights, detection, or model inference:

```bash
source .venv/bin/activate
python scripts/project2_reanalyze.py
```

This verifies the `pair_scores.jsonl` hash recorded in the completed inference
manifest, copies the verified scores and original manifest, and writes new
reports under `results/project2/reanalysis/`. Original inference outputs remain
untouched. An existing nonempty destination is rejected to preserve previous
reanalyses. For another saved run or another analysis output, pass both paths:

```bash
python scripts/project2_reanalyze.py --source results/project2/robustness_cpu --output results/project2/reanalysis_cpu
```

`analysis_manifest.json` records the original manifest and input-score hashes,
the current analysis source hashes and environment, analysis status/timing, and
new output hashes. `source_manifest.json` is an exact copy of the inference
manifest. Analysis validates pair identity across conditions, score/quality
validity, failure eligibility, and original-fold calibration. PNG bytes can
depend on the plotting environment even when numeric results agree.

The runner's `--stage analyze` entry point is an internal export helper that
rewrites its target directory. Use `scripts/project2_reanalyze.py` for public
saved-score reproduction so input hashes are checked and inference provenance
is preserved.

The output files are:

| File | Contents |
|---|---|
| `manifest.json` | Full-run status, configuration, input/source identities, timing, and output hashes |
| `pair_scores.jsonl` | Pair metadata, both raw qualities, both scores, and retained failures |
| `alignment_failures.json` | Image paths and alignment failure records |
| `evaluation.json` | Thresholds and compact summary tables; large prediction/curve tables are separate CSVs |
| `folds.csv`, `aggregates.csv` | Coverage, confusion counts, fixed-threshold metrics, AUC/EER, descriptive fold statistics |
| `predictions.csv` | One method-specific prediction record per requested pair and condition |
| `paired_changes.csv` | Paired cosine/QMagFace decision changes |
| `condition_changes.csv` | Paired changes from each method's original condition |
| `curves.csv` | ROC/DET points with explicit reject-all endpoints |
| `distributions.csv` | Score and raw-quality distribution summaries |
| `figures/` | Fixed-threshold metrics, ROC, DET, score distributions, and quality distributions |

The separate reanalysis directory contains the copied scores and source
manifest, a new analysis manifest, and regenerated tables and figures. Alignment
failure records remain in the original run directory; inference is not repeated.

In paired-change CSVs, an empty `fold` denotes the pooled record. Null scores,
decisions, or undefined rates also serialize as empty CSV cells. JSON preserves
these values as `null`.

## Checks

Run the project tests in the full environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The focused evaluation checks can be run independently:

```bash
.venv/bin/python -m unittest tests.test_project2_evaluation -v
.venv/bin/python -m unittest tests.test_project2_reanalyze -v
```

These tests cover calibration leakage, separate method scales, paired
denominators, failure coverage, missing classes, tied ROC/EER and plot generation.
They use synthetic inputs to check the implementation. Recognition results come
from the benchmark and full inference runs. Large data, weights and caches stay
outside Git; Project 1's historical result files are preserved.

Check saved Project 2 results and the complete repository separately:

```bash
.venv/bin/python scripts/check_project2.py
.venv/bin/python scripts/check_repository.py --profile full
```

## Verify and package

From the repository root, with intended work committed:

```bash
.venv/bin/python scripts/check_project2.py
.venv/bin/python scripts/package_submission.py --output /tmp/project2_submission_revised.zip
```

Choose a new output filename if it already exists. The packager requires a
complete, clean, standalone Git repository. It runs the original repository
checks and complete test suite, includes every tracked working file and the
actual `.git` directory, extracts to an isolated temporary directory, compares
every file hash, and checks Git integrity, HEAD, commit count and cleanliness.
It writes an adjacent `.verification.json` receipt. The source/output manifests
retain the exact exported CSV bytes across Git checkouts through `.gitattributes`.

The revised delivery copies are `dist/project2_submission_revised.zip` and
`dist/project2_submission_revised.verification.json`. The receipt records the
archive's HEAD and complete commit count. The report is included in the ZIP;
the ZIP and its receipt stay outside the tracked files.
After packaging, extract the archive and rerun its Project 2 numerical checks
without the local data or model caches. Record these new checks separately from
the historical computational evidence.

Packaging does not push to GitHub or submit to the course portal.

## Rebuild the written report

Use an environment with the full project plotting dependencies and the optional
PDF dependencies in [requirements-project2-report.txt](../../requirements-project2-report.txt).
For example, after installing the full requirements into a separate environment:

```bash
.venv-project2/bin/python -m pip install -r requirements-project2-report.txt
.venv-project2/bin/python scripts/build_project2_report.py
```

The generator checks the measured results before producing the Markdown report,
two figures and `output/pdf/project2_report.pdf`. The report manifest records
input and output hashes. [validation.json](validation.json) and
[cleanup_validation.json](cleanup_validation.json) retain the October 18 checks
and fingerprints for the report version in commit `2b9281b`. The October 21
revision is checked in [editorial_validation.json](editorial_validation.json).
Local preview PNGs are ignored. Rebuild the report
when its text, generator or figures change. Edits to other documentation alone
do not require a rebuild if the report inputs and measured outputs are unchanged.
