# Dataset Research and Selection

## Goal
Choose a dataset for studying how low resolution, blur, and reduced
brightness affect face verification.

This stage documents research and experimental planning.
No dataset has been downloaded or experimentally inspected yet.

## Candidate Comparison

| Dataset | Reported images | Identities | Planned role |
|---|---:|---:|---|
| LFW | 13,233 | 5,749 | Primary baseline |
| SCface | 4,160 | 130 | Possible surveillance extension |
| QMUL-SurvFace | 463,507 | 15,573 | Possible larger extension |

These are published counts, not counts measured by our code. Sources:
[LFW technical report](https://people.cs.umass.edu/~elm/papers/lfw.pdf),
[SCface official site](https://www.scface.org/), and the
[QMUL-SurvFace project website](https://qmul-survface.github.io/).
QMUL-SurvFace counts above follow its project website.

## 1. Labeled Faces in the Wild (LFW)

LFW contains face photographs collected from the web. It includes
variation in pose, expression, lighting, and image quality.

The images are provided as 250 x 250 pixel JPEG files. The dataset
includes predefined same-person and different-person pairs.
[LFW technical report](https://people.cs.umass.edu/~elm/papers/lfw.pdf).

### Why it fits this project
- Its verification pairs match our 1:1 comparison task.
- Its scale is manageable for an initial experiment.
- We can compare each original probe with controlled degraded versions.
- It provides established development and evaluation protocols.

### Selected image version
Use the funneled version: lfw-funneled.tgz.

Funneling is an alignment procedure. This is different from the
deep-funneled version. Record the exact archive used and do not mix
image versions across experiments.

Here, "original condition" means the selected funneled images with
no additional synthetic degradation.

### Access
The historical UMass LFW homepage at vis-www.cs.umass.edu/lfw/
could not be retrieved during the source check on 2026-09-08.
The original technical report on people.cs.umass.edu was accessible.
The [scikit-learn LFW implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py)
provides a download route using hosted dataset files.

The [Figshare page for the funneled archive](https://figshare.com/articles/dataset/lfw-funneled_tgz/3829980)
was accessible. The download and archive contents still need to be
verified locally.

### Limitations
LFW is based on web photographs rather than surveillance footage.
Its population and capture conditions do not represent every setting.
Synthetic degradation cannot reproduce every effect of a real camera.

## 2. SCface

SCface contains visible-light and infrared face images captured
using cameras of different qualities and at different distances.

### Why it is useful
It could test whether findings from synthetic degradation also
appear in actual surveillance images.

### Access requirements
The official site requires an institutional cover letter and a
completed release agreement. A full-time staff member must sign
the agreement; a student signature is not sufficient.
[SCface access requirements](https://www.scface.org/).

### Decision
Keep SCface as a possible extension. Access has not been requested.
If used, begin with visible-light images so that infrared comparisons
do not introduce a separate research question.

## 3. QMUL-SurvFace

QMUL-SurvFace contains naturally low-resolution face images from
surveillance scenes. The project releases data and evaluation code.

### Why it is useful
It could extend the study to difficult, naturally degraded images.

### Access and usage
The project website lists Google Drive and Baidu download links.
It states that the dataset is for research purposes and that image
copyright remains with the original owners.
[QMUL-SurvFace download and usage notice](https://qmul-survface.github.io/).

The linked archive has not been downloaded or inspected here.

### Decision
Keep it as a later extension because its much larger image collection
would expand the initial implementation and evaluation workload.

## Selected Dataset

Use LFW funneled images for the initial baseline, subject to a
successful download and integrity checks during the next stage.

This choice allows the project to begin with a defined verification
task and change one image-quality factor at a time.

## Planned Verification Protocol

LFW provides:
- pairsDevTrain.txt: 2,200 development-training pairs.
- pairsDevTest.txt: 1,000 development-test pairs.
- pairs.txt: 6,000 evaluation pairs arranged in 10 folds.

Each evaluation fold contains 300 same-person pairs and
300 different-person pairs.

### Development
Use the development pairs to debug preprocessing and scoring.
Finalize the model, degradation levels, and threshold-selection
rule before evaluating the 10-fold set.

The development and final-evaluation views reuse some data; they are
not independent datasets. Keep the pretrained embedding model fixed
and calibrate each evaluation threshold only within its nine training
folds. See Section III.B of the
[LFW technical report](https://people.cs.umass.edu/~elm/papers/lfw.pdf).

### Final evaluation
Preserve the supplied folds and pair order.

For each held-out fold:
1. Select a decision threshold using original-condition scores
   from the other nine folds.
2. Apply that threshold to the held-out original-condition pairs.
3. Keep the threshold fixed when evaluating degraded versions of
   those same held-out pairs.

Repeat for all 10 folds and report fold-level results and their mean.
Also report the standard error of mean accuracy, as requested in the
[LFW reporting protocol](https://people.cs.umass.edu/~elm/papers/lfw.pdf).

This measures performance changes without recalibrating the threshold
for every degradation condition. Never select a threshold using the
fold on which it is evaluated.

These are pair-based folds; do not describe them as identity-disjoint.

## Planned Image Handling

- Treat the first image in each pair as the reference.
- Treat the second image as the probe.
- Keep the reference unchanged.
- Apply one degradation type at a time to the probe.
- Preserve color and source resolution during initial loading.
- Document model-specific resizing and normalization separately.

For the main representation study, determine face crops from the
undegraded images and then degrade the probe crop. This isolates
recognition sensitivity after preprocessing.

Record preprocessing failures and report how many pairs remain.
Use the same eligible pairs across conditions. This experiment will
not measure detection performance on degraded full images.

## Remaining Checks

- Verify download access, image counts, dimensions, and pair paths.
- Review dataset notices before use or redistribution.
- Select and document the pretrained model and training-data source.
- Investigate possible overlap between model training data and LFW.
- Report unknown training overlap as a limitation.
- Describe this as evaluation using an externally pretrained model,
  rather than claiming training used only the LFW pairs.

## Sources

Source pages checked on 2026-09-08. Dataset downloads and local
integrity checks remain pending.

1. [LFW original technical report](https://people.cs.umass.edu/~elm/papers/lfw.pdf)

2. [scikit-learn LFW pair loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html)

3. [scikit-learn LFW download implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py)

4. [Funneled archive hosted for scikit-learn](https://figshare.com/articles/dataset/lfw-funneled_tgz/3829980)

5. [SCface official website and access requirements](https://www.scface.org/)

6. [QMUL-SurvFace official project website](https://qmul-survface.github.io/)
