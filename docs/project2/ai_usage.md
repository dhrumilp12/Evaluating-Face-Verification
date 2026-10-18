# AI assistance record

## October 18, 2026 - implementation and measured experiments

User request: "finish this work", referring to the Project 2 handoff and starter
ZIP. The user subsequently confirmed the deliverables: a complete repository
with its `.git` history, reproducible code, experiments and measured results,
plus a written report; no slides were requested.

The handoff was treated as project context, rather than as independent authority
to perform every command in the attachment. The repository was clean and matched
the stated Project 1 commit. Existing source, data and results were inspected
before integrating the additive starter. No files were pushed to a remote.

AI assistance used: primary paper/code/artifact verification; checkpoint and
feature downloads; MagFace preprocessing and alignment implementation; fixed
threshold evaluation; cache/provenance checks; tests; actual benchmark and GPU
experiments; report, figures and submission packaging. Parallel agents handled
artifact auditing, inference, evaluation and independent saved-result checking.
Their changes were inspected and tested in the shared workspace.

Examples of delegated tasks were to verify the official checkpoint and scoring
protocol, preserve raw feature magnitudes, test that degraded test scores cannot
affect calibration, and independently reconstruct saved metrics. The official
feature benchmark was rerun using pinned upstream source. A disjoint development
pilot compared CPU and MPS output before the full condition grid was frozen.

Outputs were used as repository code and documentation after mathematical,
integration, corruption and execution checks. Numeric report tables are generated
from saved measured results, with explicit coverage and protocol differences.
The full commands, logs, source hashes and output hashes are retained. Generated
prose still requires the student's review and understanding; assistance is not
evidence that the student has independently reviewed or authored every line.

This work does not claim to retrain MagFace, fit the published coefficients,
validate demographic fairness or reproduce every experiment in the paper.

## October 12, 2026 - Project 2 kickoff

User request: start Project 2 using the uploaded instruction slide and the
Project 1 description. The instruction requires reproducing and investigating a
modern biometric method while preserving the Project 1 problem.

Assistance used: repository inspection, paper/code source review, candidate-method
selection, protocol drafting, independent mathematical scoring implementation,
synthetic tests, and packaging of an additive first increment.

Why used: to connect the existing face-verification degradation study to a
reproducible published comparison method and catch implementation/protocol
mistakes before model inference.

Generated work requiring user review: all Project 2 planning text, scorer code,
and tests in this increment. The user remains responsible for understanding the
method, checking the cited sources, reviewing code, running it locally, recording
actual experiments, and making genuine commits. This entry does not claim those
user review steps have already happened.

Limitations: no user-machine inference or benchmark results were generated in
this session; no original upstream model training was reproduced; no project
files were pushed to GitHub. Mathematical tests alone are not evidence of a
recognition-performance result.
