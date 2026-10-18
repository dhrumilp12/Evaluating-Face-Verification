# Project 2 experiment log

## Increment P2-001 - scope and mathematical scorer

Prepared: October 12, 2026.

**Completed in this assistance session:** inspected the remote Project 1 README,
latest commit, source listing, and evaluation helper; reviewed the uploaded
Project 2 instruction image and the relevant Week 1 course notes; researched the
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
standalone additive starter directory. See `validation.json` and `validation.txt`.

**Observation:** the method needs raw embedding magnitudes. Reusing the existing
normalized FaceNet cache would give invalid quality inputs. The starter therefore
rejects all-unit-norm raw feature matrices unless an explicit testing override is
requested. The current evaluation runner also labels its selected score field as
`cosine_similarity`; a future Project 2 interface must distinguish adjusted scores.

**Next gate:** verify artifact access and scope, reproduce scores on author
features, and validate a small development-only MagFace inference run before full
experiments. Local user execution and review remain pending.

## Local review entry - fill after performing the work

Date/time:
Local branch and commit:
Python/NumPy versions:
New test suite output:
Complete repository test output:
Code reviewed and understood:
Changes made after review:
Known failures or discrepancies:
Next decision:

Do not fill planned measurements with expected or published results.
# October 18, 2026: integrated execution

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
