# Project 2 plan

This is the original kickoff plan, retained for provenance. The artifact audit,
frozen protocol, experiment log and reproduction guide record the subsequent
execution. The user confirmed repository history plus a written report as the
deliverables on October 18; slides are outside the requested scope.

## Working title and problem

**Quality-Aware Face Verification Under Image Degradation: Reproducing and Investigating QMagFace**

Does incorporating model-specific image quality into comparison scoring improve
one-to-one face verification under reduced resolution, blur, and reduced
brightness, relative to cosine scoring of the same embeddings?

The biometric modality remains the face. The core experimental comparison changes
the matching stage, while the representation stage supplies a compatible
pretrained MagFace model and its raw feature magnitudes.

## Assignment interpretation

The uploaded Project 2 slide says: reproduce a method from a modern biometric
journal or conference paper, investigate, and keep the same problem as Project 1.
It gives an October 23 deadline. It does not specify a publication-year cutoff,
require full network retraining, or state the final submission format. The plan
therefore uses a clearly bounded method/benchmark reproduction; acceptance of
that scope and any additional Project 2 submission requirements should be checked
early. Project 1's repository and `.git` requirements are a sensible continuity
standard, not proof that Project 2 has no additional deliverables.

The course's Week 1 slides also emphasize iteration, reproducibility, version
control, and documenting AI prompts and their use.

## Project 1 starting point

Source: the repository README inspected on October 12, 2026, at remote commit
`860e896d9c18619139e7d529009dc3ea9c2bb563`.

| Existing condition | Mean fold accuracy | Mean fold FMR | Mean fold FNMR |
|---|---:|---:|---:|
| Original | 99.167% | 0.433% | 1.234% |
| Resolution 20 x 20 | 89.982% | 0.967% | 19.071% |
| Blur sigma=3 px | 97.816% | 0.533% | 3.834% |
| Brightness x0.25 | 98.566% | 0.433% | 2.434% |

These are previously recorded Project 1 values, not newly measured Project 2
results. They condition on successful preprocessing: 5,999 of 6,000 pairs were
scored using VGGFace2-pretrained InceptionResnetV1. Its existing stored embeddings
are unit-normalized and must not be treated as MagFace quality estimates.

## Selected paper and rationale

Terhorst et al., **QMagFace: Simple and Accurate Quality-Aware Face Recognition**,
IEEE/CVF WACV 2023. The authors publish a comparison function, fitted parameters,
code, and links to precomputed embeddings and a MagFace-100 checkpoint. See
[method notes](method_notes.md) for the primary references and access caveats.

This is a useful course-project choice because the new comparison stage can be
implemented and evaluated independently of the expensive backbone training.
The controlled baseline is MagFace with ordinary cosine scoring, not only the
older FaceNet system. A difference between those two scoring rules can therefore
be studied without simultaneously changing their feature extractor.

## Reproduction versus investigation

**Reproduction:** implement the comparison function, verify numerical agreement
with the authors' scoring behavior, and attempt the LFW benchmark with compatible
published artifacts and fixed published coefficients. Inspect the official
benchmark evaluation before claiming its protocol or results were reproduced.
Use the same artifacts for MagFace-cosine and QMagFace. Record numerical gaps
instead of tuning toward a published accuracy.

**Investigation:** extract compatible raw MagFace features for our own controlled
probe degradations. Compare cosine and quality-aware scores using identical
images, pair eligibility, and calibrated-threshold rules. Measure whether image
quality estimates and genuine/impostor separation behave differently across
resolution, blur, and brightness. Report improvements, neutral findings, and
regressions equally.

Not claimed: retraining MagFace, reproducing every dataset in the paper,
reproducing the two-parameter training procedure in this first increment, or
achieving the paper's reported recognition performance.

## Planned comparisons

| Arm | Features | Comparison | Role |
|---|---|---|---|
| A | Existing FaceNet / InceptionResnetV1 | Cosine | Project 1 historical system reference |
| B | Compatible MagFace-100 raw features | Cosine | Controlled scoring baseline |
| C | The exact same MagFace-100 raw features | QMagFace | Reproduced method |

B versus C is the principal experiment. A versus B/C is a whole-system comparison
with training-data and preprocessing differences; do not attribute that entire
difference to quality-aware scoring. A strict new FaceNet comparison would need
its own explicitly matched-pair experiment rather than copying a historical table
and pretending every preprocessing condition is identical.

## Feasibility gates

Before full inference, check that the authors' artifact links are usable, that
weights and feature files match the intended architecture and training source,
and that dataset/model terms permit this coursework. Preserve file hashes.

First use precomputed embeddings to exercise the reproduction with low compute.
Then use a small development-only image set to check landmark alignment, image
channels, normalization, raw norms, model loading, CPU throughput, and cache
restart behavior. Benchmark the local computer before estimating the full run.
Do not install an old CUDA/MXNet environment over the working Project 1 environment.
Create a separate environment for model integration after checking dependencies.

The primary target is MagFace-100 with the matching published coefficients.
A smaller model is not an interchangeable fallback: verify its checkpoint and
training data and document the altered scope. Current MagFace documentation lists
an iResNet18 CASIA-WebFace release; do not blindly attach coefficients to it based
only on the architecture's name.

## Milestones, not manufactured commits

| Target dates | Increment | Evidence to save |
|---|---|---|
| October 12 | Scope, paper notes, independent scorer | This starter and local tests |
| October 13-14 | Artifact access, official benchmark reproduction, inference pilot | Hashes, environment, coverage, scores, timings |
| October 15-19 | Controlled degradation experiments | Configuration, pair scores, raw norms, fold predictions |
| October 20-21 | Error analysis and interpretation | Matched-pair comparisons, plots, honest limitations |
| October 22 | Clean-run reproduction and packaging | Verification log and actual Git history |
| October 23 | Final review and submission | Course-required deliverables |

Commit completed changes when they are ready. Split or combine increments to
reflect real work; the schedule is not a required commit count. Preserve the
old Project 1 results rather than silently replacing them with Project 2 outputs.
