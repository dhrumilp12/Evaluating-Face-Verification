# Project 2 submission

The user-requested deliverables are the complete repository with its real `.git`
history and a written report. The report is available as
[PDF](../../output/pdf/project2_report.pdf) and
[Markdown](final_report.md). Slides are not part of the confirmed scope.
The supplied Project 2 instruction slide does not itself specify a written-report
requirement, report page limit or slides format; any additional formatting rules
must come from the professor's instructions or submission page.

The repository includes all Project 1 history, Project 2 code and tests,
artifact/source hashes, the executed protocol, measured benchmark and robustness
scores, per-fold predictions, plots, the report generator, an AI-assistance
record, and reproduction instructions. Downloaded face images, author feature
arrays, checkpoints and local environments are deliberately omitted from Git;
[reproduce.md](reproduce.md) explains how to obtain and verify them.

## Verify and package

From the repository root, with intended work committed:

```bash
.venv/bin/python scripts/check_project2.py
.venv/bin/python scripts/package_submission.py --output /tmp/project2_submission.zip
```

Choose a new output filename if it already exists. The packager requires a
complete, clean, standalone Git repository. It runs the original repository
checks and complete test suite, includes every tracked working file and the
actual `.git` directory, extracts to an isolated temporary directory, compares
every file hash, and checks Git integrity, HEAD, commit count and cleanliness.
It writes an adjacent `.verification.json` receipt. The source/output manifests
retain the exact exported CSV bytes across Git checkouts through `.gitattributes`.

The prepared delivery copies are `dist/project2_submission.zip` and
`dist/project2_submission.verification.json`. The archive's receipt records the
actual final HEAD and complete commit count; neither artifact is recursively
included inside itself. The report is also inside the archive. The package has
been extracted and its Project 2 numerical checks rerun without the local data
or model caches.

No GitHub push or course-portal submission is performed by this packaging step.

## Rebuild the written report

Use an environment with the full project plotting dependencies and the optional
PDF dependencies in [requirements-project2-report.txt](../../requirements-project2-report.txt).
For example, after installing the full requirements into a separate environment:

```bash
.venv-project2/bin/python -m pip install -r requirements-project2-report.txt
.venv-project2/bin/python scripts/build_project2_report.py
```

The generator first verifies measured results, then creates the Markdown report,
two scientific figures and `output/pdf/project2_report.pdf`. The report manifest
records input and output hashes. Rendering/visual-review evidence is retained in
[validation.json](validation.json); local preview PNGs are ignored.
