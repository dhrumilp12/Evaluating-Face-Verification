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
