# Commit 3 — LFW download and exploration

## What this stage adds
- src/lfw_dataset.py: download, SHA-256 checks, extraction, pair parsing, image inspection.
- notebooks/01_dataset_exploration.ipynb: runnable exploration and observations.
- requirements-commit3.txt: pinned direct dependencies.

The saved notebook contains outputs from a successful local run on Python 3.13.7.
The measured dataset summary and distribution plot are committed; downloaded
dataset files remain local under data/. No recognition model or performance
measurement is included.

The repository's requirements.txt includes requirements-commit3.txt. Run the
commands below from the repository root.

## Environment
Use Python 3.13 (validated with 3.13.7). On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name biometrics-project1 --display-name "Biometrics Project 1"
python -m jupyterlab
```

For Windows Git Bash, create the environment with `python -m venv .venv`
and activate with `source .venv/Scripts/activate`. For PowerShell use
`.venv\Scripts\Activate.ps1`.

The named kernel is installed inside .venv, so launch Jupyter from that environment.
Open notebooks/01_dataset_exploration.ipynb and select the Biometrics Project 1
kernel. Run the cells in order. The notebook downloads approximately 232 MiB;
allow at least 1 GB of disk space for the archive, extracted images, and outputs.

The supplied Pillow, matplotlib, JupyterLab, and ipykernel pins installed on
Python 3.13 without changes. certifi was added for trusted download certificates.
The resolved packages are captured in docs/environment_commit3.txt. To recreate
that package set in a fresh Python 3.13 environment, install with
`python -m pip install -r docs/environment_commit3.txt`.

The downloader supplements Python's default TLS trust roots with certifi's CA
bundle. Certificate and hostname verification remain enabled, and every download
must match its published SHA-256 checksum. This avoids a machine-wide certificate
change on Python.org macOS installations that lack default CA paths.

To execute and save the complete notebook using Jupyter without opening its UI:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=biometrics-project1 notebooks/01_dataset_exploration.ipynb
```

## Data layout
The notebook uses data/lfw_home/:
- lfw-funneled.tgz
- lfw_funneled/<identity>/<identity>_0001.jpg
- pairsDevTrain.txt
- pairsDevTest.txt
- pairs.txt
- download_manifest.json

The manifest stores source URLs, SHA-256 values, bytes, verification times, and
actual download timestamps where known. Existing verified files have an unknown
download timestamp if they were obtained before the manifest was created.

This replaces the tentative scikit-learn loader approach in Commit 2. We use the
same published source files and checksums, but read JPEGs individually with Pillow
without automatic cropping, resizing, or grayscale conversion. This keeps memory
use small and avoids unintended changes to image quality.

No pretrained-model input preprocessing has been implemented in this stage.

## Expected protocol checks
Published expectations, to be verified by your run:

| File | Same-person | Different-person | Total | Folds |
|---|---:|---:|---:|---:|
| pairsDevTrain.txt | 1,100 | 1,100 | 2,200 | 1 |
| pairsDevTest.txt | 500 | 500 | 1,000 | 1 |
| pairs.txt | 3,000 | 3,000 | 6,000 | 10 |

Within each fold, genuine pairs precede impostor pairs. The parser checks this
order and preserves pair indices. Labels are 1=same and 0=different. Folds are
numbered 0 through 9 in code. They are not identity-disjoint folds.

Only development-training examples are displayed. Checking evaluation metadata
and file integrity does not compute performance or select model hyperparameters.

## Outputs and observations
- results/metrics/dataset_summary.json: measured counts, checks, versions, source
  manifest, run timestamp, helper checksum, and Git HEAD at run time.
- results/figures/lfw_identity_distribution.png: count distribution.
- Executed notebook: displayed development pairs and your written observations.

The summary is overwritten on a rerun. Save prior runs under distinct filenames
if comparing different data versions. Git retains committed prior versions.
Git HEAD in the summary is the commit before your new changes are committed;
it is not claimed to describe the entire uncommitted working tree.

After successful execution:
1. Fill in the notebook's observations cell and save the notebook.
2. Copy the generated Stage 003 entry into docs/experiment_log.md.
3. Replace its observations placeholder with what you actually noticed.
4. Add a link to this setup document from README.md and data/README.md.
5. Update README.md status to say dataset validation is complete only if it passed.
6. Capture the installed environment:

```bash
python -m pip freeze > docs/environment_commit3.txt
```

Record any error and the steps that fixed it. Never fill in expected counts as
measured results before running the notebook.

## Commit
Review the changes and ensure downloaded files remain ignored:

```bash
git check-ignore data/lfw_home/lfw-funneled.tgz
git status --short
```

Stage only the intended files:

```bash
git add requirements.txt requirements-commit3.txt src/lfw_dataset.py notebooks/01_dataset_exploration.ipynb docs/commit3_setup.md docs/environment_commit3.txt docs/experiment_log.md README.md data/README.md results/metrics/dataset_summary.json results/figures/lfw_identity_distribution.png
git diff --cached --stat
git commit -m "Add LFW dataset loading and exploration"
git log --oneline -3
git status
```

## Troubleshooting
- HTTP/connection error: retain the exact error and retry later. Failed partial
  downloads are removed; successful verified files are reused. Do not disable TLS
  or skip checksums. If the mirror is unavailable, research another documented
  source before changing URLs.
- Checksum failure: file is rejected. Rerun; investigate persistent mismatches.
- Missing/corrupt images: rerun the download cell to restore archive members.
  Unexpected extra JPEGs are not deleted automatically; inspect them explicitly.
- Missing module: check the kernel uses the .venv environment, then reinstall
  requirements from an activated terminal.
- macOS certificate error: the downloader now loads the pinned certifi CA bundle
  in addition to default trust roots. Confirm the active environment has the
  requirements installed. Investigate any remaining trust error without disabling
  certificate verification.
- Download interrupted: rerun. This downloader restarts the affected file rather
  than resuming a partial stream.

Terminal alternative (validates data but does not generate notebook figures):

```bash
python src/lfw_dataset.py --download
```

## Sources and validation scope
- LFW technical report: https://people.cs.umass.edu/~elm/papers/lfw.pdf
- Source metadata: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py
- Archive mirror: https://figshare.com/articles/dataset/lfw-funneled_tgz/3829980

The bundle's preparation notes reported synthetic checks and an HTTP 403 from
that earlier environment. The local run on this Mac did not encounter HTTP 403.
Its initial attempt failed with `CERTIFICATE_VERIFY_FAILED: unable to get local
issuer certificate`; both default certificate paths were absent. Adding certifi
to the downloader's verified TLS context resolved that failure.

The full notebook then completed on Python 3.13.7, with the validation timestamp
2026-09-09T02:41:47.735201+00:00 (2026-09-08 in America/New_York):

- All four downloaded files matched the published SHA-256 checksums.
- 13,233 images and 5,749 identities were measured.
- All images decoded successfully as 250 x 250 RGB images.
- Development pair totals were 2,200 and 1,000; evaluation had 6,000 pairs in
  10 folds, with equal same-person and different-person counts in every fold.
- No unreadable images or missing pair references were found.
- The distribution plot and four development-pair displays were inspected.

The actual observations and certificate fix are recorded in the notebook and
Stage 003 of docs/experiment_log.md. No verification performance has been measured.
