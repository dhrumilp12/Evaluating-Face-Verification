# QMagFace method notes and sources

These are the retained kickoff notes. The October 18 executed artifact and
benchmark audit is in [artifact_audit.md](artifact_audit.md), including the final
conference paper, downloaded artifact identities and measured reproduction.
Historical statements below about unverified downloads refer to October 12.

## Core reproduction

Let `e_ref` and `e_probe` be raw MagFace embeddings. Define `q_ref=||e_ref||_2`,
`q_probe=||e_probe||_2`, and cosine similarity `s`. The comparison equation is:

```text
weight = min(0, beta*s - alpha)
qmagface_score = s + weight * min(q_ref, q_probe)
```

Equations (3)-(4) in the authors' preprint and the official `similarity()`
implementation agree on this scoring form. This starter independently implements
that equation. It does not vendor the official implementation or execute its
training procedure. Its randomized check uses an independent scalar calculation,
not an execution of the upstream library.

The published QMagFace-100 coefficients are `alpha=0.077428` and `beta=0.125926`.
They are not universal constants for every architecture, checkpoint, or feature
aggregation convention. The authors' Table 2 lists LFW accuracies of 99.73% for
MagFace-100 and 99.83% for QMagFace-100. These are comparison targets from the
reviewed preprint, not our results or a guarantee of exact reproduction.

## Implementation consequences

The raw feature magnitudes must survive feature caching. A cache containing only
unit vectors has discarded the input needed for the quality term. Taking the
norm of Project 1's normalized FaceNet cache is not a valid substitute. The code
includes a guard against all-unit-norm feature inputs; that guard is not proof
that other inputs came from the right model.

The adjusted value is a score, not a probability, and can fall below -1. Do not
clip it to the cosine range or reuse FaceNet's numeric threshold. For positive
qualities, the adjustment is nonpositive. At a fixed cosine score below the
weight cutoff, a larger minimum quality produces a larger negative adjustment.
Thus the rule should not be described as simply boosting all blurry-face scores.

The official fitting code uses sigmoid-transformed intermediate scores while its
final inference score does not. Reproducing parameter fitting requires auditing
that procedure separately; do not insert a sigmoid into inference merely because
it appears in the training code.

## Verified sources / provenance

1. **Conference paper identity:** Philipp Terhorst, Malte Ihlefeld, Marco Huber,
   Naser Damer, Florian Kirchbuchner, Kiran Raja, and Arjan Kuijper.
   *QMagFace: Simple and Accurate Quality-Aware Face Recognition*. WACV, 2023.
   Publisher record: https://ieeexplore.ieee.org/document/10030931/
2. **Reviewed method document:** authors' arXiv preprint, version 3,
   March 23, 2022: https://arxiv.org/abs/2111.13475v3
   PDF: https://arxiv.org/pdf/2111.13475v3
   Sections 3.1 and 4.3; Tables 1 and 2. The equations and those tables were
   visually inspected. This is a preprint version of the WACV paper, not a claim
   that every line of the final proceedings version was checked. The CVF
   proceedings link was inaccessible in this research session.
3. **Authors' QMagFace repository:** https://github.com/pterhoer/QMagFace
   README includes artifact links, example evaluation commands, and an
   acknowledgement of alignment reproducibility difficulties.
4. **Scoring source inspected:**
   https://github.com/pterhoer/QMagFace/blob/main/similarity/qmagface.py
   Observed Git blob SHA: `6ea91187361a3ddd7ea2246c52ee2c73d85d7990`.
   A blob identifies this file's content, not the entire upstream repository.
5. **Underlying MagFace implementation/model documentation:**
   https://github.com/IrvingMeng/MagFace
   Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou. *MagFace: A Universal
   Representation for Face Recognition and Quality Assessment*. CVPR, 2021.
6. **Existing project baseline:**
   https://github.com/dhrumilp12/Evaluating-Face-Verification/tree/860e896d9c18619139e7d529009dc3ea9c2bb563
   README and `src/evaluation.py` inspected using the GitHub connection.

Sources accessed October 12, 2026. Artifact links were observed, but checkpoint
and precomputed-embedding downloads were not verified during this increment.
The QMagFace README states CC BY-NC 4.0 for its code; MagFace uses Apache-2.0.
Dataset and checkpoint permissions must be checked separately before reuse or
redistribution. Do not include model or dataset files in the public repository.
