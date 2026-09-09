# Project and Experiment Log

## Stage 001 — Initial Repository Setup

### Goal
Define the research question and organize the repository.

### Changes
- Drafted the face-verification problem statement.
- Identified the biometric modality and pipeline stages.
- Listed candidate datasets for further research.
- Outlined baseline and image-degradation experiments.
- Added folders and Git ignore rules.

### Current Status
Planning only. No dataset inspection, model execution, or measured
results are available yet.

### Initial Decision
Investigate LFW first for the baseline.
Confirm dataset availability and protocol before implementation.

### Next Step
Research candidate datasets, record sources and access requirements,
and select the baseline dataset and image version.

## Stage 002 — Dataset Research and Selection

### Goal
Compare candidate datasets and select an initial verification dataset.

### Changes
- Documented LFW, SCface, and QMUL-SurvFace.
- Recorded source links and access requirements.
- Selected LFW funneled images for the baseline.
- Defined reference/probe roles and a planned evaluation protocol.
- Documented loading settings and checks needed before experiments.

### Findings
LFW provides predefined verification pairs.
SCface requires an institutional request and a staff-signed agreement.
QMUL-SurvFace offers a larger surveillance collection.

### Decision
Begin with LFW, subject to successful download and validation.
Use original-condition calibration thresholds unchanged across
degradation conditions within each evaluation fold.

### Current Status
Documentation and planning only.
No download, model execution, or measured performance results yet.

### Next Step
Download LFW, inspect its images and pair files in Jupyter, and
record the actual findings.

## Stage 003 — LFW Download and Exploration

### Run time (UTC)
2026-09-09T02:41:47.735201+00:00

### Goal
Download LFW funneled images and validate them for the baseline.

### Changes
Added checksum-verified downloading, pair parsing, image inspection,
and a Jupyter exploration notebook.

### Measured Results
- Images: 13233
- Identities: 5749
- Image dimensions: {'250x250': 13233}
- Source color modes: {'RGB': 13233}
- Identities with two or more images: 1680
- Unreadable images: 0
- Missing pair references: 0
- All dataset validation checks passed: True

### Measured Pair Protocol

| File | Same person | Different people | Total | Folds |
|---|---:|---:|---:|---:|
| pairsDevTrain.txt | 1,100 | 1,100 | 2,200 | 1 |
| pairsDevTest.txt | 500 | 500 | 1,000 | 1 |
| pairs.txt | 3,000 | 3,000 | 6,000 | 10 |

### Saved Outputs
- notebooks/01_dataset_exploration.ipynb
- docs/environment_commit3.txt
- results/metrics/dataset_summary.json
- results/figures/lfw_identity_distribution.png

### Observations and Problems
The distribution is strongly uneven: 4,069 identities (70.8%) have one image;
1,680 have at least two. Counts range from 1 to 530. The displayed development
pairs show changes in expression, lighting, background, and slight head angle.
Some examples have black or rotated alignment borders and differing apparent
sharpness. These four pairs are illustrative, not a dataset-wide quality sample.

Execution used the Biometrics Project 1 Jupyter kernel in the local Python
3.13.7 virtual environment. The supplied four dependency pins installed
successfully and pip check reported no broken requirements. The first download
failed with CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate.
Both default Python CA paths were absent. Adding certifi==2026.7.22 to the
requirements and supplementing the downloader TLS context with its CA bundle
resolved the error. Certificate and hostname verification stayed enabled.
All four downloads then matched their published SHA-256 checksums. No HTTP 403
occurred locally. All six notebook code cells completed successfully.

The installed environment is captured in docs/environment_commit3.txt.
Data and the virtual environment remain excluded from Git. The UTC run date
above corresponds to 2026-09-08 in America/New_York.

The data is ready for baseline implementation; model selection, training-data
overlap review, preprocessing, and embedding generation remain pending.

### Next Step
Select a pretrained model and test the face-preprocessing and embedding
pipeline on development images. No verification metrics exist yet.
