# Reproducing the project

Use this page as the single setup and execution guide. Methods and measured
results remain in the [README](../README.md), the six executed notebooks, and
the historical [experiment log](experiment_log.md).

## Choose the scope

| Goal | Required local inputs | Environment | Action |
|---|---|---|---|
| Inspect the work | Committed repository files | None for reading | Read README, log, notebooks, figures |
| Verify saved results | Committed repository and Git history | Existing full environment, or analysis dependencies | Run repository checker |
| Recompute ROC/DET and score summaries | Committed score exports | NumPy and Matplotlib for CLI; full environment for Jupyter | Run score_analysis.py or notebook 06 |
| Reproduce face inference and all experiments | LFW download, model weights, regenerated local caches | Full pinned environment | Run notebooks 01–06 in order |

Score-only reproduction is a useful independent check of exported results. It
does not demonstrate that image preprocessing or face inference was reproduced.
The repository checker also does not run those stages.

## Existing working environment

For the author’s current repository, retain the working environment. From the
repository root on macOS/Linux:

```bash
source .venv/bin/activate
python -m pip check
python -m unittest discover -s tests -v
python scripts/check_repository.py --profile full --output results/metrics/repository_check.json
```

The full reference run used Python 3.13.7, CPU, InceptionResnetV1 pretrained on
VGGFace2, and 512-dimensional embeddings. Direct dependency pins are in
[requirements.txt](../requirements.txt); actual inference versions and platform
are recorded in the baseline summary. Do not change them solely for this cleanup.

## Fresh setup for saved-score analysis

In a fresh clone, select Python 3.13.7 if reproducing the recorded environment.
The score-analysis source was also exercised in Python 3.12 during preparation;
that is not a claim of matching the full inference environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip check
python -m unittest discover -s tests -p test_score_analysis.py -v
python -m unittest discover -s tests -p test_repository_check.py -v
python scripts/check_repository.py --profile analysis
python src/score_analysis.py
```

The two analysis pins match the numerical packages in the full environment.
This route supports command-line analysis and checks. Jupyter and the other
experiment tests are provided by the full environment below. The existing
requirements are direct pins, not a complete transitive lockfile. A fresh install
on another machine has not been established by a saved-score rerun.

Required score-analysis inputs under `results/metrics/` are the baseline summary,
thresholds and scores; the resolution summary and predictions; and the quality
summary and predictions. No local `data/` or `models/` directory is required.
Analysis reruns overwrite only `analysis_*` outputs; inspect changes before
committing. PNG bytes may vary with the plotting environment even when metrics
agree, and the new analysis summary records the resulting file hashes.

## Fresh setup for full reproduction

Use Python 3.13.7 for the reference configuration; verify the interpreter before
creating the environment. The `python3` command must resolve to that interpreter
if exact environment reproduction is intended.

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --sys-prefix --name biometrics-project1 --display-name "Biometrics Project 1"
python -m jupyterlab
```

For Windows Git Bash use `python -m venv .venv` and
`source .venv/Scripts/activate`. In PowerShell activate with
`.venv\Scripts\Activate.ps1`. The recorded full run was on macOS; installation
instructions for another platform are not evidence of an identical result.

Stop active project kernels before changing their environment. Launch Jupyter
from the activated environment, select **Biometrics Project 1**, and run all
cells in each notebook in this order:

| Notebook | Purpose | Important dependency |
|---|---|---|
| [01_dataset_exploration](../notebooks/01_dataset_exploration.ipynb) | Download/validate LFW and explore images/pairs | Internet on first download |
| [02_face_embedding_pipeline](../notebooks/02_face_embedding_pipeline.ipynb) | Check pretrained embeddings on development pairs | Validated data and model download |
| [03_baseline_verification](../notebooks/03_baseline_verification.ipynb) | Original-quality ten-fold baseline | Working model and dataset |
| [04_resolution_degradation](../notebooks/04_resolution_degradation.ipynb) | Probe resolution reduction | Original baseline crops/embeddings/thresholds |
| [05_blur_brightness](../notebooks/05_blur_brightness.ipynb) | Independent probe blur and brightness | Same original baseline cache |
| [06_score_analysis](../notebooks/06_score_analysis.ipynb) | ROC/DET and score statistics | Completed saved score exports |

Save notebooks with outputs and add actual observations to the experiment log.
A fresh full run can replace source reports that later reports refer to by hash;
continue through all downstream notebooks to obtain one internally consistent set.
Do not combine outputs from different baselines or bypass provenance checks.

For terminal execution and saved notebook outputs, use this command for each
notebook in the table, changing the final path:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=biometrics-project1 notebooks/01_dataset_exploration.ipynb
```

Helper entry points are also available, in order:

```bash
python src/lfw_dataset.py --download
python src/embedding_smoke_test.py
python src/baseline_verification.py
python src/resolution_experiment.py
python src/quality_experiment.py
python src/score_analysis.py
```

The helper commands save metrics. The score-analysis helper also saves its
figures. Other notebook-specific visualizations and notebook execution records
require running the notebooks; CLI helpers do not mark notebook cells as executed.

## Dataset and cache layout

The loader stores inputs in `data/lfw_home/`: `lfw-funneled.tgz`,
`pairsDevTrain.txt`, `pairsDevTest.txt`, `pairs.txt`, and `download_manifest.json`.
Extracted images are in `data/lfw_home/lfw_funneled/<identity>/`.

The archive is approximately 232 MiB. Allow several GB for extracted images,
model weights, original crops, embeddings, and degraded embedding caches.
The downloader uses the Figshare mirror URLs and expected SHA-256 values embedded
in `src/lfw_dataset.py`; it records URLs, sizes and checksums in the manifest.
Reruns reuse verified downloads and restore archive members. Dataset checks cover
image counts, identities, readability, dimensions and pair references. Sources
and usage considerations are in [dataset research](dataset_research.md).

Original crops/embeddings are under `data/processed/baseline/<fingerprint>/`;
resolution and quality caches are under `data/processed/resolution/` and
`data/processed/quality/`. Recognition weights are under `models/checkpoints/`.
These directories are intentionally excluded from Git. Recreate them through
full reproduction; a GitHub clone alone cannot run degraded-image inference
using the original cache until it has been regenerated.

Degradation runs keep the original reference embeddings and ten baseline fold
thresholds fixed. Their model/environment fingerprints must match their own
baseline. On another machine or environment, reproduce that baseline and all
subsequent conditions together and document differences. Existing committed
results remain accessible in Git history.

## Repository checks

```bash
python scripts/check_repository.py --profile full --output results/metrics/repository_check.json
```

The full profile checks direct dependency pins and Python 3.13.7. The analysis
profile checks only NumPy and Matplotlib pins. Both validate files, notebooks,
relative Markdown file links, saved output hashes, common pair eligibility,
threshold decisions, and the ten recorded analysis summaries. Both check local
Git objects/history by default and report working-tree cleanliness. An uncommitted
cleanup is expected before making Commit 9 and does not fail the history check.
The saved report records the HEAD and working tree at check time, before the
commit containing that report. Run `pip check` and the test suite separately
as shown above; the repository checker does not invoke those commands.

The file-only profile and `--skip-git` exist for reviewing downloaded snapshots:

```bash
python scripts/check_repository.py --profile files --skip-git
```

This still needs NumPy to validate numerical results, but deliberately skips
installed-version comparisons and Git checks. Its report identifies those skips;
it does not establish submission readiness or environment reproduction.

The checker leaves experiment artifacts unchanged and writes a JSON report only
when the optional report path is supplied. It only
allows `results/metrics/repository_check.json` as that destination, protecting
existing code, documentation and measured result files. It does not test external
URLs, Markdown anchor correctness, current dataset availability, inference cache
contents, notebook freshness, or ZIP contents. A recorded execution count with no
saved error is a notebook completeness check, not a new execution.

## Troubleshooting

- Wrong interpreter or missing module: activate the intended environment, check
  `python --version`, install its requirements and choose its Jupyter kernel.
- Certificate error: confirm the pinned certifi package is installed. Existing
  downloaders supplement default trust roots while retaining TLS verification.
- Download/checksum failure: record the error and rerun. Persistent mismatches
  require investigating the source; do not replace expected hashes to force success.
- Missing/corrupt original cache: reproduce notebook 03, then downstream stages.
- Model/environment mismatch: restore the baseline environment, or establish and
  document a new baseline followed by its downstream experiments.
- Export hash mismatch: identify which file changed and regenerate the affected
  stage and dependent reports. Do not manually alter hashes to pass the checker.
- Notebook with a saved error/unexecuted cell: resolve it, rerun and save. Do not
  delete error outputs merely to present a passing notebook.

Record failures and fixes in the log. The repository checker report is a local
verification record, not a claim that the entire image pipeline ran again.
