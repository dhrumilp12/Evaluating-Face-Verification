# Project 2 experiment log

## Increment P2-001 - scope and mathematical scorer

Prepared: October 12, 2026.

The initial review covered the remote Project 1 README, latest commit,
source listing and evaluation helper, along with the Project 2 instruction slide
and relevant Week 1 notes. Review of the QMagFace paper, authors' scoring code
and MagFace documentation led to a standalone NumPy scorer. All 29 synthetic
tests passed.

At this stage, the complete repository test suite had not been run. No MagFace
weights or author embeddings had been downloaded, and neither model inference
nor the official QMagFace library had been executed. Benchmark reproduction,
parameter fitting and full-pipeline CPU timing had not been attempted. The
GitHub repository had not been changed or committed.

The Git clone failed because the container could not resolve GitHub. The GitHub
connection allowed source inspection, but the tests used a separate starter
directory rather than a complete local checkout. See
[starter_validation.json](starter_validation.json) and
[starter_validation.txt](starter_validation.txt).

QMagFace needs raw embedding magnitudes, so the normalized FaceNet cache could
not supply its quality values. The starter rejects all-unit-norm raw features
unless a testing override is explicitly enabled. The Project 1 runner's
`cosine_similarity` field also needed a separate counterpart for adjusted scores.

The next steps were to confirm access to compatible artifacts, reproduce scores
on the authors' features and run a small MagFace pilot before full experiments.
Local execution and student review were still pending at this point.

### Original selection rationale and plan

The October 12 working title was **Quality-Aware Face Verification Under Image
Degradation: Reproducing and Investigating QMagFace**. The research question was
whether model-specific quality in the comparison score improves one-to-one face
verification under reduced resolution, blur and brightness relative to cosine
scoring of the same embeddings.

The Project 2 slide called for reproducing and investigating a method from a
modern biometric journal or conference paper while retaining Project 1's
problem. The deadline was October 23. It gave no publication-year cutoff and
did not specify full network retraining or a submission format. The Week 1 notes
emphasized iteration, reproducibility, version control and recording AI prompts
and their use. The plan therefore focused on the method and benchmark. Project
1's repository and `.git` requirements informed continuity between the projects;
they did not establish additional Project 2 requirements. The deliverables
confirmed on October 18 were repository history and a written report.

The starting point was the Project 1 README and evaluation helper inspected at
[commit `860e896d9c18619139e7d529009dc3ea9c2bb563`](https://github.com/dhrumilp12/Evaluating-Face-Verification/tree/860e896d9c18619139e7d529009dc3ea9c2bb563).
Its [preserved results](../../README.md) used VGGFace2-pretrained
InceptionResnetV1 and covered the 5,999/6,000 pairs that passed preprocessing.
Those results provided background; a direct comparison with MagFace would also
change the model, training data and preprocessing.

Terhorst et al.'s *QMagFace: Simple and Accurate Quality-Aware Face Recognition*
(WACV 2023) provided a published comparison rule, coefficients, code and artifact
links. This made it possible to study scoring without repeating backbone
training. Comparing MagFace cosine with QMagFace on the same raw features would
isolate the effect of quality-aware scoring. The original method and source
research is preserved in [the artifact audit](artifact_audit.md).

The plan was to compare the new scorer with the authors' implementation, inspect
their evaluator and attempt the LFW benchmark with compatible artifacts and fixed
published coefficients. Any gap from the published result would be reported
without tuning parameters to close it. The investigation would extract raw
MagFace features from images with controlled probe degradations. Pair eligibility
and calibration rules would stay the same for both scorers, with improvements
and regressions reported together. MagFace retraining, QMagFace parameter fitting
and reproduction of every dataset in the paper were outside the scope.

Before running the experiments, the plan required checking artifact access,
model and training-source compatibility, dataset and checkpoint terms, and file
hashes. Author features would come first, followed by a development pilot to
check alignment, channels, normalization, raw norms, model loading, CPU throughput
and cache restarts. A separate environment would preserve Project 1's working
dependencies. The iResNet18 CASIA-WebFace release was not a suitable drop-in
replacement: its model name alone did not justify using MagFace-100 coefficients.

The original schedule placed scope and scorer work on October 12, artifact
checks and a pilot on October 13–14, controlled experiments on October 15–19,
error analysis on October 20–21, reproduction and packaging on October 22, and
final review and submission on October 23. These were targets. The entries below
record the work actually completed on October 18. The plan also called for
preserving Project 1's outputs and Git history.

## Unused local review template

The starter's blank local-review template was never completed. This note records
that omission; the completed experiments and checks appear below.

## October 18, 2026: integrated execution

The clean repository matched commit `860e896`. Adding the starter preserved
Project 1's files, and all 75 initial tests passed in Python 3.13.7. Work then
covered the author artifacts, an independent inference check and paired
evaluation with tests for calibration leakage.

The development pilot used 16 images absent from the evaluation image paths.
MagFace returned finite raw 512-dimensional features with norms 26.15–33.04.
MPS inference took 0.588 seconds; CPU/MPS maximum absolute feature difference
was 0.0000179. The independent pinned author network and adapter were bitwise
identical on two numerical input examples. No evaluation labels or accuracies
were used for selecting the full grid, parameters, or preprocessing.

The ten-condition grid was fixed before evaluation inference began. Alpha and
beta used the published values. Each method's held-out predictions were
calibrated only on original-quality scores from the other nine original LFW
folds.

## Completed measurements and validation

The complete run took 731.44 seconds on MPS. Of 7,701 unique evaluation images,
7,700 aligned successfully; the remaining image had a low-confidence detection.
Each of the ten conditions scored 5,999/6,000 requested pairs: 2,999 genuine and
3,000 impostor. The saved records retain all 60,000 attempts, including the ten
unscored comparisons. The run computed and cached 48,884 raw embeddings and
produced 59,990 scored comparisons.

The author-feature reproduction measured 99.749722% cosine and 99.833139%
QMagFace on 5,994 pairs. The robustness experiment's mean-fold original accuracy
was 99.800000% cosine and 99.816667% QMagFace. At 14-pixel resolution it was
58.826739% and 65.411324%, respectively: 398 corrected errors and three introduced
errors. AUC decreased and EER increased at that severity. The accuracy gain
therefore applies to the calibrated thresholds; overall score discrimination
became worse. The [final report](final_report.md) gives the results for every
condition and discusses their limits.

All 131 tests passed. The saved-run verifier passed 14 checks, including
reconstruction of all 59,990 scored pairs from raw caches. A separate invocation
of saved-score reanalysis reproduced all seven CSV tables byte for byte and the
evaluation JSON numerically, while preserving the original outputs. The
Project 1 repository checks also passed. The seven-page PDF was rendered and
visually reviewed; no clipped text, missing glyphs or broken tables remained.
See [validation.json](validation.json), [the test log](validation.txt),
[cache verification](saved_run_verification.json), and
[reanalysis verification](reanalysis_verification.json).

Commit `1d2556b` contains the tested Project 2 pipeline and author benchmark.
The next commit added the final measurements, report and packaging instructions.
Both continue Project 1's existing history. No remote push or course submission
was performed. The agreed deliverables were repository history and a written
report. Slides were outside that scope, and the course instruction slide itself did
not specify a report requirement, page limit or slides format.

## October 18, 2026: documentation review corrections

Following review feedback, the report generator and rebuilt report
were updated to use Dhrumil Patel in the subtitle, footer and PDF metadata. The
starter entry was linked to its original 29-test evidence, and the unused
local-review template was labelled as historical. The note above retains that
clarification. Submission instructions were also revised to distinguish the
requested deliverables from the requirements in the course slide.

The code, experimental results and conclusions stayed unchanged. No image
alignment or MagFace inference was rerun. The rebuilt report and package received
updated verification records. The external review code and evidence mentioned
in the feedback were not available locally.

## October 18, 2026: documentation consolidation

Starting from `4639bda`, navigation from the separate Project 2 overview moved
into the README. The kickoff rationale and plan moved into this log, and the
method and source cautions moved into the artifact audit. Instructions for
reproduction, report building and packaging were combined in the reproduction
guide. The four former documents were removed after their unique content had
been transferred; their original versions remain in Git history. The shortened
AI-assistance record and the wording change in the kickoff entry were
preserved during this cleanup.

The cleanup checks passed: 131 tests, 13 saved-result checks (local caches
skipped), and all 23 full-repository checks. All remaining Markdown files,
including nested Project 2 documents, passed local path and heading-anchor
checks. External URLs were not rechecked. All eight report-manifest fingerprints
matched. The code, measured outputs, generated reports, historical verification
records and Project 1 files were unchanged. No image inference or PDF rebuild
was performed, and local data/model caches were retained.

These checks are recorded in [cleanup_validation.json](cleanup_validation.json)
and [cleanup_validation.txt](cleanup_validation.txt), separately from the
original experiment evidence. The cleanup ZIP's receipt records the final
commit, full Git history and extraction checks performed after the documentation
commit.

## October 21, 2026: writing revision

The Project 2 documentation was reviewed for clarity. Long sentences were
shortened, repeated qualifications were combined, and the results discussion
was rewritten to explain each degradation condition without repeating the same
sentence pattern. The AI-assistance record was retained and updated to include
this editing work.

The report generator now produces the revised text in both Markdown and PDF.
Its numerical tables and reproduction commands are unchanged, as are the
experimental code and measured outputs. The earlier JSON and text logs remain
historical evidence for the versions they checked. The report review and
verification for this revision are recorded in
[editorial_validation.json](editorial_validation.json).
