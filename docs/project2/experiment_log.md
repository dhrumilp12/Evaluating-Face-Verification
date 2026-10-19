# Project 2 experiment log

## Increment P2-001 - scope and mathematical scorer

Prepared: October 12, 2026.

**Completed in this assistance session:** inspected the remote Project 1 README,
latest commit, source listing, and evaluation helper; reviewed the uploaded
Project 2 instruction and the relevant Week 1 course notes; researched the
QMagFace paper, authors' scoring source, and MagFace model documentation; wrote
a separate NumPy scorer and synthetic tests; executed those 29 tests successfully.

**Not completed:** running the user's complete existing test suite, downloading
or executing MagFace weights, downloading the authors' precomputed embeddings,
executing the official QMagFace library, reproducing a benchmark accuracy, fitting
quality-weight parameters, measuring full-pipeline CPU throughput, or changing
or committing files in the user's GitHub repository.

The container's attempted Git clone failed because it could not resolve GitHub.
Repository inspection succeeded through the GitHub connection, but this was not
a full local checkout or full regression run. New scorer tests were run in a
standalone additive starter directory. See
[starter_validation.json](starter_validation.json) and
[starter_validation.txt](starter_validation.txt).

**Observation:** the method needs raw embedding magnitudes. Reusing the existing
normalized FaceNet cache would give invalid quality inputs. The starter therefore
rejects all-unit-norm raw feature matrices unless an explicit testing override is
requested. The current evaluation runner also labels its selected score field as
`cosine_similarity`; a future Project 2 interface must distinguish adjusted scores.

**Next gate:** verify artifact access and scope, reproduce scores on author
features, and validate a small development-only MagFace inference run before full
experiments. Local user execution and review remain pending.

### Original selection rationale and plan

The October 12 working title was **Quality-Aware Face Verification Under Image
Degradation: Reproducing and Investigating QMagFace**. The research question was
whether model-specific quality in the comparison score improves one-to-one face
verification under reduced resolution, blur and brightness relative to cosine
scoring of the same embeddings.

The supplied Project 2 slide called for reproducing a method from a modern
biometric journal or conference paper, investigating it, and retaining Project
1's problem, with an October 23 deadline. It did not specify a publication-year
cutoff, full network retraining or a final submission format. The Week 1 notes
emphasized iteration, reproducibility, version control and recording AI prompts
and their use. The kickoff therefore proposed a bounded method/benchmark
reproduction; Project 1's repository and `.git` requirements were continuity
guidance, not evidence of additional Project 2 format requirements. The user
confirmed repository history plus a written report on October 18.

The starting point was the Project 1 README and evaluation helper inspected at
[commit `860e896d9c18619139e7d529009dc3ea9c2bb563`](https://github.com/dhrumilp12/Evaluating-Face-Verification/tree/860e896d9c18619139e7d529009dc3ea9c2bb563).
Its [preserved results](../../README.md) came from VGGFace2-pretrained
InceptionResnetV1, conditional on successful preprocessing of 5,999/6,000 pairs.
They were historical context, not new Project 2 measurements or a matched-pair
comparison with MagFace.

Terhorst et al.'s *QMagFace: Simple and Accurate Quality-Aware Face Recognition*
(WACV 2023) was selected because its published comparison rule, coefficients,
code and artifact links permitted studying the scoring stage without repeating
backbone training. MagFace cosine and QMagFace on exactly the same raw features
were the principal comparison; a FaceNet-to-MagFace difference also changes
training data and preprocessing and cannot isolate quality-aware scoring.
The original method/source research is preserved in
[the artifact audit](artifact_audit.md).

The reproduction plan was to check agreement with the authors' scorer and
attempt their LFW benchmark using compatible artifacts and fixed published
coefficients. It required inspecting the official evaluator before claiming
protocol agreement and recording numerical gaps instead of tuning toward a
published accuracy. The investigation would independently extract raw MagFace
features for controlled probe degradations, hold pair eligibility and calibration
rules fixed across scorers, and report improvements, neutral findings and
regressions. Retraining MagFace, fitting QMagFace parameters and reproducing every
paper dataset were outside the planned scope.

Feasibility gates were artifact access, model/training-source compatibility,
dataset and checkpoint terms, and recorded hashes. Precomputed features were to
precede a development-only pilot checking alignment, channels, normalization,
raw norms, model loading, CPU throughput and cache restart behavior. A separate
integration environment would preserve Project 1's working dependencies instead
of replacing them with an old CUDA/MXNet stack. A smaller model was not an
interchangeable fallback: in particular, the documented iResNet18 CASIA-WebFace
release could not inherit MagFace-100 coefficients based only on its name.

The proposed milestones were scope/scorer work on October 12; artifact access,
benchmark reproduction and a pilot on October 13–14; controlled experiments on
October 15–19; error analysis on October 20–21; clean-run reproduction and
packaging on October 22; and final review/submission on October 23. These were
targets, not a record of work on each date or a required number of commits.
The October 18 entries below record the actual execution and supersede the
planned timing. Project 1 outputs and the real Git history were to be preserved.

## Local review entry - UNUSED HISTORICAL TEMPLATE

The starter included a blank template for a future local review. It was not
completed and is retained here only as a historical note, not as evidence of a
student review. The executed work and validation are recorded below. Planned
measurements must not be filled with expected or published results.

## October 18, 2026: integrated execution

The existing clean repository matched commit `860e896`. The starter was added
without overwriting Project 1 files. All 75 initial tests passed in the existing
Python 3.13.7 environment. Parallel work audited author artifacts, independently
validated inference, and implemented leakage-tested paired evaluation.

The development pilot used 16 images absent from the evaluation image paths.
MagFace returned finite raw 512-dimensional features with norms 26.15–33.04.
MPS inference took 0.588 seconds; CPU/MPS maximum absolute feature difference
was 0.0000179. The independent pinned author network and adapter were bitwise
identical on two numerical input examples. No evaluation labels or accuracies
were used for selecting the full grid, parameters, or preprocessing.

The full ten-condition grid was frozen before starting evaluation inference.
Published alpha/beta remain fixed. Each method's held-out predictions use only
original-quality calibration scores from the other nine original LFW folds.

## Completed measurements and validation

The complete run finished in 731.44 seconds on MPS. It requested 7,701 unique
evaluation images, aligned 7,700 and retained one low-confidence detection
failure. Each of ten conditions scores 5,999/6,000 requested pairs (2,999 genuine
and 3,000 impostor), for 59,990 scored comparisons and 60,000 retained attempts.
All 48,884 required raw embeddings were computed and cached in this run.

The author-feature reproduction measured 99.749722% cosine and 99.833139%
QMagFace on 5,994 pairs. The robustness experiment's mean-fold original accuracy
was 99.800000% cosine and 99.816667% QMagFace. At 14-pixel resolution it was
58.826739% and 65.411324%, respectively: 398 corrected errors and three introduced
errors. AUC decreased and EER increased at that severity, so this is an
operating-point improvement rather than uniformly better discrimination.
The [final report](final_report.md) includes every condition, count and caveat.

All 131 tests passed. The full saved-run verifier passed 14 checks, including
reconstruction of all 59,990 scored pairs from raw caches. Independent invocation
of the safe saved-score reanalysis reproduced all seven CSV tables byte for byte
and the evaluation JSON numerically, while preserving original outputs. The
Project 1 repository checks also passed. The seven-page PDF was rendered and
visually reviewed; no clipped text, missing glyphs or broken tables remained.
See [validation.json](validation.json), [the test log](validation.txt),
[cache verification](saved_run_verification.json), and
[reanalysis verification](reanalysis_verification.json).

The first genuine Project 2 commit is `1d2556b`, covering the tested pipeline and
author benchmark. The final measurements, report and packaging instructions
form the next increment. The history continues from the real Project 1 commits;
no dates or commits were manufactured. No remote push or course submission was
performed. The user requested repository history plus a written report; slides
are outside the requested scope. The supplied Project 2 instruction slide does
not itself specify a written-report requirement, page limit or slides format.

## October 18, 2026: documentation review corrections

The user requested the full author name, Dhrumil Patel, in the written report's
subtitle, footer and PDF metadata. The report generator and regenerated report
were updated accordingly. The starter entry now links to its original 29-test
evidence, and the unused local-review template is explicitly identified above.
The submission documentation distinguishes the user's requested repository and
written report from the requirements stated in the supplied Project 2 slide.

These are documentation corrections. The implementation, saved experimental
results and scientific conclusions are preserved; no image alignment or MagFace
inference rerun is needed. Updated report and package verification records
accompany the rebuilt deliverables. The user's review feedback prompted these
edits; external review code and evidence were not available locally.

## October 18, 2026: documentation consolidation

Starting from `4639bda`, the README absorbed the extra Project 2 overview's
navigation, this log absorbed the kickoff rationale and planning history, and
the artifact audit absorbed the historical method/source cautions. Reproduction,
report-building and packaging instructions now share the reproduction guide.
The four superseded documents were retired from the working tree after merging
their unique content; their original versions remain in Git history. The
existing shortened AI-assistance record and the user's wording edit in the
kickoff entry were preserved.

The cleanup checks passed: 131 tests, 13 saved-result checks (local caches
skipped), and all 23 full-repository checks. All remaining Markdown files,
including nested Project 2 documents, passed local path and heading-anchor
checks. External URLs were not rechecked. All eight report-manifest fingerprints
still match, and the code, measured outputs, generated reports, historical
verification records and Project 1 files remain unchanged. No image inference
or PDF rebuild was performed, and local data/model caches were retained.

These new checks are recorded in [cleanup_validation.json](cleanup_validation.json)
and [cleanup_validation.txt](cleanup_validation.txt), separately from the
original experiment evidence. The newly named cleanup ZIP's adjacent receipt
records its final commit, full Git history and extraction integrity; it is
generated after this documentation commit.
