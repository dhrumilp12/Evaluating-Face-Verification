# Biometrics Project 1
## Evaluating Face Verification Under Degraded Image Quality

### Project Status
Initial project setup and experimental planning.
No dataset has been downloaded and no experiments have been run yet.

### Problem Statement
This project studies how image quality affects face verification.

Research question:
How do low resolution, blur, and reduced brightness affect a system's
ability to determine whether two face images belong to the same person?

This is a 1:1 verification task. The planned approach uses a fixed,
pretrained face recognition model to generate embeddings and compare
pairs of images.

### Biometric Modality
The project uses the face, a physiological biometric modality.

### Pipeline and Scope
Dataset images → face preprocessing → embeddings → matching → decision

The project focuses on:
- Image quality: apply controlled degradation to the probe image.
- Representation: generate embeddings with a pretrained model.
- Matching: compare the reference and probe embeddings.
- Decision: apply a threshold to predict match or non-match.

Face detection is not the main research focus. The preprocessing method
and handling of failed detections will be documented before evaluation.

### Why This Problem Matters
Face verification systems may receive blurry, dim, or low-resolution
images. These conditions can cause a system to reject the correct person
or incorrectly accept a different person. Measuring these errors helps
us understand the system's limitations.

### Candidate Datasets
- Labeled Faces in the Wild (LFW): proposed primary dataset.
- SCface: possible extension using surveillance images.
- QMUL-SurvFace: possible extension using low-resolution surveillance faces.

Dataset sources, access requirements, and the final selection will be
documented in docs/dataset_research.md during the next stage.

### Planned Experiments
1. Inspect the selected dataset and verification protocol.
2. Establish performance using original images.
3. Reduce probe-image resolution.
4. Apply Gaussian blur to probe images.
5. Reduce probe-image brightness.
6. Compare results and document limitations.

Keep the reference image and recognition model fixed across conditions.
Select decision thresholds using development data, without tuning them
on the held-out evaluation pairs. Record the exact split before running
experiments.

Brightness reduction is a controlled simulation; it does not reproduce
all effects of real low-light camera capture.

### Planned Evaluation
- Verification accuracy
- False Match Rate (FMR): fraction of different-person comparisons accepted
- False Non-Match Rate (FNMR): fraction of same-person comparisons rejected
- ROC and DET curves

### Repository Organization
- docs/: research notes and experiment log
- notebooks/: dataset exploration and experimental notebooks
- src/: reusable Python code
- data/: dataset setup instructions and local data
- results/figures/: generated plots
- results/metrics/: generated evaluation results
- requirements.txt: dependencies, added as the implementation develops

### Reproducibility
Setup and execution instructions will be added when working code exists.
Datasets, model weights, and virtual environments will not be committed.
Experiment records will describe configurations, results, failures, and
decisions. Git history will track actual changes as the project develops.
