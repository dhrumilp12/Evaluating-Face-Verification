# Repository guide

Start with the [README](../README.md) for the question, scope and measured results.
Use [reproducibility](reproducibility.md) for setup and execution, and the
[experiment log](experiment_log.md) for the actual development history.

| Location | Role | Submission treatment |
|---|---|---|
| README.md | Project question, modality, pipeline, relevance, methods, results | Include |
| docs/dataset_research.md | Dataset candidates, sources and selection | Include; planned passages describe the earlier research stage |
| docs/experiment_log.md | Trials, measured results, failures and decisions | Preserve prior entries; append new work |
| docs/reproducibility.md | Setup, execution order, cache requirements, troubleshooting | Include |
| notebooks/ | Six experiments/analyses with executed outputs and observations | Include |
| src/ | Reusable loaders, embeddings, transforms, evaluation and analysis | Include |
| tests/ | Focused numerical, validation and synthetic integration checks | Include |
| scripts/check_repository.py | Read-only repository checks with optional report | Include |
| requirements.txt | Full environment direct dependency pins | Include |
| requirements-analysis.txt | Smaller CLI analysis dependency subset | Include |
| results/metrics/ | Saved scores, thresholds, metrics, provenance and check report | Include |
| results/figures/ | Experimental examples and plots | Include |
| data/, models/, .venv/ | Large downloaded inputs, local caches and environment | Recreate from documentation; excluded from Git |
| .git/ | Actual Git history and objects | Must be present in the professor's submission ZIP |

Experiment data flows from dataset validation to the embedding smoke check,
then the original baseline. Resolution and quality experiments independently
consume that baseline. Score analysis consumes the three completed result sets.
Repository checks inspect these saved outputs without rerunning face inference.

No new empty folders or duplicate inference modules are needed. The existing
stage-specific runners are retained because their source hashes identify the
experiments already performed. This cleanup changes documentation and adds checks;
it does not refactor fingerprinted model code or alter experimental results.

## Historical notes versus current instructions

Earlier log entries intentionally retain what was planned or known at that time.
Current reproduction instructions live in reproducibility.md. Git history
preserves previous README versions; do not rewrite old experiment entries to
make their planned steps look completed earlier.

## Preparing for the final submission

The professor requires the entire repository with `.git`, documented evolution,
reproducibility and technical functionality. GitHub's source-code ZIP and
`git archive` omit `.git` and therefore do not meet that explicit history
requirement by themselves.

Commit 9 verifies organization and reproduction documentation. The final stage
will consolidate conclusions and create/inspect the submission archive from the
real local repository. A checker result on a downloaded file snapshot cannot
verify local Git history or the contents of that future archive.
