# Project 2 artifact and official benchmark audit

Audit and execution date: October 18, 2026. The public checkpoint, author feature
archive, and official scoring implementation were accessible. This is an executed
feature-benchmark reproduction, separate from our locally aligned robustness run.

## Primary artifacts

The [QMagFace README at the audited commit](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/README.md)
links both Google Drive artifacts below. The [MagFace model table](https://github.com/IrvingMeng/MagFace/blob/99bae614ac2643b9694bf18e0c5645272ff6acfa/README.md)
links the same checkpoint as a distributed-training iResNet100 MagFace model
trained on MS1MV2. Model training was not repeated in this project.

| Artifact | Source | Observed bytes | Observed SHA-256 |
|---|---|---:|---|
| Author features and pairs archive | [Google Drive](https://drive.google.com/file/d/1ElwkUKFs6-4JEwRnsKh6fJLp00x_FOvS/view) | 6,158,445,827 | `d228033c7bc4a2d4d1e92ba987cda9e6fb4423f219c97e98364820accd614f7f` |
| `magface_epoch_00025.pth` | [Google Drive](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view) | 283,040,192 | `cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58` |
| `embeddings_lfw.npy` | Archive: `_data/single_images/magface100/` | 54,177,920 | `7788292d42b4d8fdfdc1e372bafc54aa9c949d470b48758041d5e851216c38da` |
| `filenames_lfw.npy` | Archive: `_data/single_images/magface100/` | 4,550,216 | `19f15d4fb900879f59233c70470abb96a5dda92e400d4f05723ddfc4361cc5a2` |
| `lfw_pairs.txt` | Archive: `_data/pairs/` | 155,328 | `55506bb9b62f8968b0f3e5b5b8fff91604659b7dec61c6ee185e4606ee9b74f3` |

Hashes identify bytes observed in this audit; they are not publisher-signed
checksums. [artifact_manifest.json](artifact_manifest.json) records source-file
hashes and immutable URLs. Upstream commits were:

- QMagFace: `395797145da1b1b263cbe7fb1564c6ea91f55b23`.
- MagFace: `99bae614ac2643b9694bf18e0c5645272ff6acfa`.

Only the three MagFace-100 LFW data files were extracted. NumPy arrays were loaded
with `allow_pickle=False`. Features are 13,227 by 512, `float64`; filename storage
is a fixed-width Unicode array. All features are finite. Raw norm minimum, mean,
and maximum are 17.617511, 28.813498, and 34.338258. Flip feature files in the
archive were not used: the official single-image benchmark loads
`embeddings_lfw.npy` directly.

The QMagFace README identifies its code as CC BY-NC 4.0; MagFace has an Apache-2.0
license. These code licenses are not a separate grant of rights to redistribute
face datasets or weights. Downloaded data, model files, and upstream snapshots
remain ignored by Git. The shareable outputs contain metrics and pair records.

## Inference contract verified from source

The [official embedding entry point](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/preprocessing/embed.py)
uses OpenCV image reads followed by `ToTensor`: BGR channels, unsigned-byte pixels
scaled by 1/255, with no mean subtraction or division by an additional standard
deviation. It outputs raw 512-dimensional features. The corresponding
[inference script](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/preprocessing/magface/gen_feat.py)
constructs a flipped image but executes inference on the unflipped tensor only;
normalization of the output is commented out. No flip aggregation is used here.

The source uses five-landmark alignment to 112 by 112 images. Its
[alignment helper](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/preprocessing/insightface/src/face_preprocess.py)
contains the ArcFace target landmarks. The authors explicitly report alignment
reproduction difficulties and obtaining already aligned ArcFace images. Our
facenet-pytorch landmark detector and independently computed crop therefore
define a separate preprocessing experiment; a compatible input convention does
not establish identical alignment or identical embeddings.

The fixed coefficients are `alpha=0.077428`, `beta=0.125926`. The
[official inference score](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/similarity/qmagface.py)
uses the smaller raw feature norm and a nonpositive quality weight. Its final
comparison score has no sigmoid and is not clipped to the cosine range.

## Exact benchmark protocol and observed exclusions

The benchmark runner executes the hash-verified upstream `Cosine`, `QMagFace`,
and `evaluate_metrics` implementations. An independent filename-to-row mapping
retains every requested pair and reports missing rows; its resulting sequence
was checked against the original upstream `PairDataset` execution.

The [official pair loader](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/datasets/paired.py)
silently skips comparisons with unavailable feature filenames. Of 6,000 requested
pairs, 5,994 are scored: 2,996 genuine and 2,998 impostor, for 99.9% coverage.
The six excluded original pair indices are 2358, 2419, 2786, 3803, 3804, and 5608
(zero based). They are recorded with names and reasons in
[excluded_pairs.csv](../../results/project2/author_benchmark/excluded_pairs.csv).

The [official metric implementation](https://github.com/pterhoer/QMagFace/blob/395797145da1b1b263cbe7fb1564c6ea91f55b23/evaluation/metrics.py)
has the following behavior, preserved for this reproduction:

1. Build 600 equally spaced thresholds from the minimum to maximum score over
   the entire eligible dataset, separately for each method.
2. Apply `KFold(n_splits=10, shuffle=False)` to the surviving sequence. This
   creates four 600-pair folds and six 599-pair folds. Their boundaries differ
   from the original 600-pair LFW folds after exclusions.
3. Maximize calibration-fold accuracy and choose the first, smallest threshold
   in a tie. A match requires `score > threshold`, strictly.
4. Report the unweighted mean of ten held-out fold accuracies.

Thus the global threshold grid uses held-out score extrema, although threshold
selection uses calibration labels only. This is preserved as upstream behavior,
not recommended as the protocol for the robustness experiment. That experiment
keeps original fold identities and fits thresholds using original-quality
calibration pairs, as specified in [protocol.md](protocol.md).

## Measured reproduction

The final [WACV 2023 paper](https://openaccess.thecvf.com/content/WACV2023/papers/Terhorst_QMagFace_Simple_and_Accurate_Quality-Aware_Face_Recognition_WACV_2023_paper.pdf)
was downloaded and Table 2 was visually inspected on PDF page 6, printed page
3489. It reports LFW accuracy of 99.73% for MagFace-100 and 99.83% for QMagFace-100.

| Method | Published accuracy | Measured mean-fold accuracy | Gap (percentage points) | False matches | False nonmatches |
|---|---:|---:|---:|---:|---:|
| MagFace cosine | 99.73% | 99.749722% | +0.019722 | 3 | 12 |
| QMagFace | 99.83% | 99.833139% | +0.003139 | 2 | 8 |

QMagFace corrects five cosine errors, introduces zero, and leaves ten comparisons
wrong for both methods. Its accuracy gain is 0.083417 percentage points in this
run. These counts do not establish statistical significance. The QMagFace result
rounds to the reported number; the cosine result rounds to 99.75%, so we do not
claim exact reproduction of every published value. Missing features, the
released evaluator, rounded published values, and a modern numerical environment
are documented differences; their individual causal effects were not established.

Against the executed official scorer on all 5,994 pairs, the independent scorer's
maximum absolute discrepancy is `1.9984014443252818e-15` for cosine and
`4.773959005888173e-15` for QMagFace. Running the official evaluator on either
set of scores gives identical accuracy. The numerical agreement tolerance is
`1e-12`; this is a real-feature equivalence check, not only a synthetic test.

Complete evidence is in
[summary.json](../../results/project2/author_benchmark/summary.json),
[fold_metrics.csv](../../results/project2/author_benchmark/fold_metrics.csv), and
[pair_scores.csv](../../results/project2/author_benchmark/pair_scores.csv).
The summary includes input and output hashes, versions, fold coverage, and the
two original/official fold identifiers are both retained in pair records.

## Re-run commands and environment

The existing Project 1 environment was left unchanged. This run used a new
`.venv-project2` with a `.pth` reference to the existing environment for NumPy,
and installed sklearn/SciPy into the new environment only. Actual benchmark
versions were Python 3.13.7, NumPy 2.5.3, SciPy 1.16.2, and scikit-learn 1.7.2.
The upstream environment specifies Python 3.8; it was not recreated.

A standalone benchmark environment can be created without that local `.pth`:

```bash
python3 -m venv .venv-project2
.venv-project2/bin/python -m pip install -r requirements-project2-benchmark.txt
.venv-project2/bin/python scripts/project2_acquire_artifacts.py
.venv-project2/bin/python scripts/project2_acquire_artifacts.py --verify-only
.venv-project2/bin/python scripts/project2_benchmark.py
```

The acquisition command downloads the checkpoint first, validates observed
SHA-256 identities, fetches the four required source files at the pinned commit,
and extracts only the three fixed LFW archive members. `--checkpoint-only` and
`--features-only` limit acquisition. The archive is about 6.2 GB; the checkpoint
is 283 MB. A resumed run reuses validated existing artifacts. Partial downloads
are written to `.partial` files and are restarted if interrupted. Downloads use
system `curl` when available, retaining HTTPS certificate verification; otherwise
they use Python's verified HTTPS client. The clean acquisition path was checked
with a separate extraction directory and only the four pinned upstream files,
and produced identical pair/fold outputs.

`project2_benchmark.py --download` can acquire only benchmark inputs. Its
`--features`, `--official-source`, and `--output` flags override directories;
the pinned file identities still must match. Runtime execution does not require
PyMuPDF. PyMuPDF 1.26.4 was installed separately in the new environment only to
inspect the final paper.

No artifact-access blocker remains. This audit does not establish training/test
identity disjointness, replicate model training or parameter fitting, or
reproduce every benchmark in the paper.
