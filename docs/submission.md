# Final review and submission

The professor requires the repository, actual `.git` history, reproducible
instructions, technical functionality and documented experimentation.
The final findings are in [final_report.md](final_report.md); methods, results,
notebooks and earlier log entries remain in their existing locations.

## Review and make the final commit

Apply the Commit 10 bundle in the repository root. It updates README.md and adds
this guide, the final report, a packaging script and five packaging tests. Merge
any README edits made locally after the preceding commit. It preserves all
existing inference code, dependency pins, experiment records and measured results.

Review the final report against your actual work. The report describes measured
results and states the limits of the evaluation. It does not claim the final ZIP
already exists or that full inference was rerun during documentation cleanup.

Run the tests from the existing environment:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/check_repository.py --profile full
```

No package changes or face inference are needed. The checker is run without an
output path so that the earlier repository_check.json remains a truthful record
of Commit 9's pre-commit validation. The final packaging step checks the current
repository again and writes a separate receipt outside the repository.

Append a Stage 010 entry to [experiment_log.md](experiment_log.md), using your
actual local test/check outcomes. Suggested text:

```markdown
## Stage 010 — Final Findings and Submission Workflow

### Goal
Consolidate the measured findings and prepare a verifiable repository submission.

### Changes
- Added docs/final_report.md with the problem, modality, pipeline, dataset choice,
  methods, measured results, conclusions and limitations.
- Linked final findings and submission instructions from README.md.
- Added a packager that includes tracked project files and the real .git directory.
- Added five tests for archive round trips and rejection of invalid packaging states.

### Verification
- Local tests: [insert actual count and outcome].
- Full repository check: [insert actual passed/failed/skipped counts].
- Problems and fixes: [record actual issues, or state none if true].
- Existing image inference and experimental results were not rerun in this stage.

### Archive Status
The submission archive will be generated after this commit. Its external
verification receipt records the exact archived HEAD, preserved commit count,
ZIP checksum, extracted-file verification and Git checks.
```

Then stage only the final work and commit:

```bash
git add README.md docs/final_report.md docs/submission.md docs/experiment_log.md
git add scripts/package_submission.py tests/test_package_submission.py
git diff --cached --stat
git commit -m "Document final findings and add verified submission packaging"
git push origin main
git status
```

The working tree must be clean before packaging. Resolve unintended untracked
files or changes yourself; the packager never deletes them. Ignored datasets,
caches, downloaded weights, virtual environments and downloaded ZIP bundles do
not make the tree dirty and are omitted from the submission files.

## Create the actual submission ZIP

After the final commit, from the same repository and activated environment:

```bash
python scripts/package_submission.py
```

The script:

1. Requires a standalone clean repository with committed history.
2. Runs the full repository checks and complete test suite.
3. Archives every Git-tracked working file and all files in the actual `.git`
   directory, including objects, refs and history metadata.
4. Extracts the ZIP to a temporary location and checks every archived file hash.
5. Runs Git object checks on the extracted copy and verifies matching HEAD,
   commit count, working-tree location and clean working-tree state.
6. Publishes the verified ZIP and a JSON receipt in the repository's parent folder.

The filenames contain the final commit prefix, for example:

```text
Evaluating-Face-Verification_submission_<commit>.zip
Evaluating-Face-Verification_submission_<commit>.verification.json
```

The printed paths are authoritative; the name uses your actual local directory
name. The ZIP has one top-level project folder containing project files and `.git`.
The receipt includes its SHA-256, byte size, archived HEAD, commit count, Git file
count and verification results. The ZIP itself contains the committed project;
the receipt stays outside it to avoid a self-referential checksum or extra commit.

To choose another filename outside the repository:

```bash
python scripts/package_submission.py --output ../biometrics-project1-final.zip
```

Existing ZIPs or verification receipts are never overwritten. Choose a new name
for a deliberate repeat. Close active Git operations before packaging; changes
or lock files during the snapshot cause the script to stop rather than accept an
inconsistent archive.

Submit the **verified submission ZIP** through the course submission system.
Keep the verification receipt; include it as an extra file if the submission
system permits. No tool in this workflow uploads to the course system for you.
The downloaded Commit 10 update bundle is not the final submission ZIP.

## Scope and remaining checks

The archive contains the version-controlled project and actual Git history.
Ignored image datasets, caches and environments are recreated using
[reproducibility.md](reproducibility.md). It is not a backup of every local file.
A GitHub source ZIP or `git archive` alone omits `.git` and does not satisfy the
professor's explicit history requirement.

The packager supports ordinary standalone repositories. It rejects linked
worktrees, shallow/partial clones, external object alternates, symlinks, tracked
submodule directories, and busy Git lock files because they can prevent a
self-contained, verifiable extracted repository. It does not rewrite or repair
Git history. Use your original full local repository.

Packaging verifies saved artifacts and Git integrity. It does not perform new
face inference, establish that all future machines can install the environment,
or check external dataset links or course-upload limits. Check any additional
syllabus or submission-portal requirements that are not provided in this repository.

Preparation checked the actual committed files and exercised packaging using
explicit temporary test repositories. Those fixture histories are never included
in the user's submission. The real final archive can only be confirmed after the
command above runs against your own committed repository.

Preparation validation for this bundle: all 46 tests passed. The updated file
snapshot passed 21 repository checks with zero failures; dependency-version
and Git checks were explicitly skipped here. Commit 9’s actual local full
report records 21 passed, zero failed and zero skipped before these additions.
Run the final full checks on your computer as instructed above.
