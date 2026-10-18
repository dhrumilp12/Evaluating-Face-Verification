# Project 2 - first working increment

**Quality-Aware Face Verification Under Image Degradation: Reproducing and Investigating QMagFace**

This is an additive starter for the existing `Evaluating-Face-Verification`
repository. It is NOT a complete Project 2 submission or a replacement repository.
It contains no `.git` directory, images, model weights, or fabricated experiments.

## What is implemented now

`src/qmagface_scoring.py` implements QMagFace's comparison equation, including raw
embedding-magnitude preservation and input validation. The 29 synthetic unit tests
passed in the assistant's Python 3.13.5 / NumPy 2.3.5 Linux environment. This does
not establish a face-recognition accuracy or validate your entire repository.

Read [the project plan](docs/project2/plan.md),
[method notes](docs/project2/method_notes.md), and
[the evaluation protocol](docs/project2/protocol.md) before running experiments.

## Apply to your existing repository

From a terminal inside your current repository, first inspect:

```bash
git status --short
git log -1 --oneline
```

Resolve any existing changes before proceeding. The remote version inspected for
this starter was `860e896d9c18619139e7d529009dc3ea9c2bb563`, whose message is
`Document final findings and add verified submission packaging`. A newer local
commit is not automatically a problem; inspect its changes before applying this
increment. Do not reset or discard local work to force a match.

Create a working branch, then extract the ZIP **into this repository root**:

```bash
git switch -c project2-qmagface
unzip -n "$HOME/Downloads/project2_commit01.zip" -d .
```

The ZIP contains only new paths. `unzip -n` does not overwrite existing files.
When the browser saved it elsewhere, use that actual ZIP path. When a
`project2-qmagface` branch already exists, inspect it rather than trying to
recreate it. Do not replace the existing `src`, `tests`, or `docs` directories
with folders from the archive.

Activate your existing Project 1 environment for these NumPy-only tests:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_qmagface_scoring.py' -v
python -m unittest discover -s tests -v
```

The first command should report 29 passing tests. The second also checks your
existing tests; it has NOT been run against your complete repository by this
starter. No dependency changes are needed for this increment.

After reviewing the code, explaining the equation in your own words, and
checking the local test output, commit this genuine increment:

```bash
git add PROJECT2_START_HERE.md docs/project2 src/qmagface_scoring.py tests/test_qmagface_scoring.py
git diff --cached --stat
git diff --cached --check
git commit -m "Start Project 2: document QMagFace reproduction and test scoring"
```

Do not backdate commits or mark planned experiments as completed. Keep the
existing Project 1 history and results intact. The next increment is an artifact
and environment feasibility check, followed by an official-embedding benchmark
reproduction and a MagFace inference smoke test. It is not yet time to launch
all degradation experiments.

## Small mathematical demonstration

```python
from src.qmagface_scoring import qmagface_scores

score = qmagface_scores(
    [0.5], [2.0], [3.0], alpha=0.08, beta=0.12
)
print(score)  # [0.46]; synthetic arithmetic, not a real face comparison
```

The demonstration's coefficients are deliberately simple toy values. For the
planned QMagFace-100 experiment, the published coefficients are alpha=0.077428
and beta=0.125926, subject to verifying the compatible checkpoint and feature
pipeline. Never substitute normalized Project 1 FaceNet embeddings.
