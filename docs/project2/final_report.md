# Quality-Aware Face Verification Under Image Degradation

Reproducing and Investigating QMagFace | Dhrumil Patel | Biometrics Project 2 | October 21, 2026

## Abstract

This study tests whether QMagFace improves face verification when images lose resolution, become blurred or become darker. QMagFace adjusts cosine similarity using raw MagFace feature magnitude as an image-quality estimate. On the authors' released LFW features, accuracy is 99.7497% for cosine scoring and 99.8331% for QMagFace, using 5,994 of 6,000 pairs. The independent scorer agrees with the official implementation within 5e-15. A separate experiment tests ten conditions using the same network and pairs for both scorers. Only probe images are degraded, and thresholds are calibrated on original-quality training folds. Original-image accuracy is 99.800% for cosine and 99.817% for QMagFace, on 5,999 scored pairs. At 14-pixel resolution, QMagFace raises accuracy from 58.827% to 65.411%, but its AUC falls from 0.951571 to 0.947902. QMagFace therefore helps at the original-image decision thresholds, even though overall score separation worsens under severe resolution loss. The study uses a pretrained network.

## 1. Problem and selected paper

Project 1 showed that image degradation can substantially reduce face-verification accuracy. Project 2 asks whether using image quality in the comparison score can reduce that loss. The selected paper is QMagFace: Simple and Accurate Quality-Aware Face Recognition, by Terhorst et al. (IEEE/CVF WACV 2023) [1]. Its released code, pretrained model and features allow the scoring method to be reproduced without retraining the network. The work first reproduces the authors' LFW evaluation, then compares QMagFace with cosine similarity under controlled degradation.

The earlier VGGFace2-pretrained FaceNet system achieved mean-fold accuracy of 99.167% on original images and 89.982% at 20 x 20 probe resolution. Those results provide context, but its network and 160-pixel bounding-box crops differ from this study. To isolate the effect of the scoring rule, the main comparison uses cosine and QMagFace on identical raw MagFace embeddings. Project 1's code, results and Git history remain in the repository.

## 2. Reproduced method and implementation

For raw reference and probe embeddings e_r and e_p, let q_r and q_p be their L2 norms and s the cosine similarity of their unit-normalized versions. QMagFace applies the following comparison function [1,2]:

```bash
q_r = ||e_r||_2;  q_p = ||e_p||_2
s = dot(e_r/q_r, e_p/q_p)
score = s + min(0, beta*s - alpha) * min(q_r, q_p)
alpha = 0.077428;  beta = 0.125926
```

The published QMagFace-100 coefficients are used throughout, without refitting on LFW or by degradation condition. The adjustment is nonpositive and can make the score less than -1; the result is a score, not a probability. Below the weight cutoff, a higher minimum quality produces a larger subtraction at the same cosine similarity. Cosine and QMagFace therefore need separate decision thresholds.

The MagFace iResNet100 checkpoint was trained on MS1MV2 [3]. Input images are aligned to 112 x 112 pixels, converted to BGR and divided by 255. Inference uses one forward pass without flipped-image aggregation. Loading checks all 925 backbone entries with weights-only deserialization. The cache stores raw 512-dimensional features, their norms and normalized features separately. The adapter produced bitwise-identical output to the pinned author network on two test crops. A 16-image development pilot, with no image-path overlap with evaluation, also checked finite outputs and CPU/MPS agreement before the recognition experiments.

## 3. Author-feature benchmark reproduction

| Method | Paper (%) | Measured (%) | Gap (pp) | FM / FNM |
| --- | --- | --- | --- | --- |
| MagFace cosine | 99.73 | 99.7497 | +0.0197 | 3 / 12 |
| QMagFace | 99.83 | 99.8331 | +0.0031 | 2 / 8 |

The paper accuracies above come from Table 2 of the final WACV proceedings paper. The authors' archive contains 13,227 raw LFW feature rows. Six requested pairs lack features: four genuine and two impostor. Coverage is 99.9%, leaving 2,996 genuine and 2,998 impostor pairs. The official loader excludes these pairs and then applies contiguous, unshuffled ten-fold KFold to the surviving list, so its fold membership differs from the original 6,000-pair assignments. The saved exclusion file lists all six pairs.

The released evaluator tests 600 thresholds between the global score minimum and maximum. It accepts score > threshold and takes the first threshold when calibration accuracies tie. Reproduction retains these conventions, including the use of held-out score extrema to construct the grid. The controlled experiment below uses a different calibration protocol. QMagFace corrects five cosine errors and introduces none; both methods misclassify ten pairs. Its measured accuracy rounds to the published 99.83%, while the cosine baseline is slightly above the paper's 99.73%. The result is close to the published benchmark, subject to the six missing pairs and the evaluator conventions described here.

## 4. Controlled investigation protocol

The investigation uses the original LFW funneled images, ordered 6,000-pair list and ten predefined folds. A facenet-pytorch MTCNN selects the highest-confidence face containing the image center with probability at least 0.90. Five landmarks are fitted to a fixed ArcFace 112-pixel similarity template; Pillow bilinear affine sampling uses a black border. These crops may differ from those used for the authors' released features. Alignment runs once on each original image, before degradation, so detection failures caused by degraded input are outside this experiment.

| Family | Levels | Operation |
| --- | --- | --- |
| Original | 112 x 112 | No synthetic degradation |
| Resolution | 56, 28, 14 pixels | Bilinear downsample, then restore to 112 |
| Gaussian blur | sigma 0.7, 1.4, 2.1 | Separable sampled Gaussian; radius ceil(3*sigma); reflect border |
| Brightness | 0.75, 0.50, 0.25 | Multiply encoded RGB, without gamma correction or added noise |

Each transformation starts from the original aligned probe crop; reference images stay unchanged. Blur and brightness use float64 arithmetic, clipping and nearest-even uint8 quantization. The two scorers use the same embeddings and pairs. For each held-out fold, each method chooses its threshold by maximizing accuracy on successful original-quality pairs from the other nine folds. Candidates are score midpoints and finite accept-all/reject-all endpoints; ties select the largest threshold, and acceptance uses >=. That threshold is then fixed for every condition in the held-out fold. Test scores and labels do not select thresholds, model settings or coefficients. The condition grid was finalized after the development pilot, before test inference.

Alignment succeeds for 7,700 of 7,701 unique evaluation images. Each condition requests 6,000 pairs and scores 5,999: 2,999 genuine and 3,000 impostor. Failed attempts remain in the prediction exports, coverage counts and separate reject-on-failure summaries. The recognition rates below use only successfully preprocessed pairs.

## 5. Measured robustness results

| Condition | Cosine accuracy (%) | QMagFace accuracy (%) | Delta (pp) | Net corrected |
| --- | --- | --- | --- | --- |
| Original | 99.800 | 99.817 | +0.017 | +1 |
| Resolution 56 | 99.767 | 99.817 | +0.050 | +3 |
| Resolution 28 | 98.683 | 99.417 | +0.733 | +44 |
| Resolution 14 | 58.827 | 65.411 | +6.585 | +395 |
| Blur 0.7 | 99.800 | 99.817 | +0.017 | +1 |
| Blur 1.4 | 99.750 | 99.783 | +0.033 | +2 |
| Blur 2.1 | 99.500 | 99.733 | +0.233 | +14 |
| Brightness 0.75 | 99.817 | 99.817 | +0.000 | +0 |
| Brightness 0.50 | 99.800 | 99.817 | +0.017 | +1 |
| Brightness 0.25 | 99.783 | 99.800 | +0.017 | +1 |

Accuracy is the mean of the ten original folds, weighted equally; pp denotes percentage points. Net corrected counts cosine errors fixed by QMagFace minus new errors on the same pairs. The counts and fold means weight pairs slightly differently because one fold has an exclusion. Saved fold SD and SE describe variation across folds; they are not confidence intervals or significance tests.

![Figure 1. Accuracy under probe degradation, using thresholds calibrated on original images. Each family includes the same original control.](../../results/project2/report_figures/report_accuracy.png)

QMagFace reduces the total error count in 9 conditions and ties in 1. The largest improvement occurs at 14-pixel resolution. Total errors alone can hide cases where one method fixes some decisions but gets others wrong, so the paired-change exports record both corrected and introduced errors for every fold and condition.

| Condition | Cosine FM / FNM | QMagFace FM / FNM | Cosine FMR / FNMR (%) | QMagFace FMR / FNMR (%) |
| --- | --- | --- | --- | --- |
| Original | 1 / 11 | 1 / 10 | 0.033 / 0.367 | 0.033 / 0.333 |
| Resolution 56 | 1 / 13 | 0 / 11 | 0.033 / 0.433 | 0.000 / 0.367 |
| Resolution 28 | 0 / 79 | 2 / 33 | 0.000 / 2.634 | 0.067 / 1.100 |
| Resolution 14 | 0 / 2470 | 1 / 2074 | 0.000 / 82.361 | 0.033 / 69.156 |
| Blur 0.7 | 1 / 11 | 1 / 10 | 0.033 / 0.367 | 0.033 / 0.333 |
| Blur 1.4 | 1 / 14 | 0 / 13 | 0.033 / 0.467 | 0.000 / 0.433 |
| Blur 2.1 | 1 / 29 | 1 / 15 | 0.033 / 0.967 | 0.033 / 0.500 |
| Brightness 0.75 | 1 / 10 | 1 / 10 | 0.033 / 0.333 | 0.033 / 0.333 |
| Brightness 0.50 | 1 / 11 | 1 / 10 | 0.033 / 0.367 | 0.033 / 0.333 |
| Brightness 0.25 | 2 / 11 | 1 / 11 | 0.067 / 0.367 | 0.033 / 0.367 |

FM and FNM count false matches and false nonmatches across all scored pairs. FMR divides FM by the impostor count; FNMR divides FNM by the genuine count. All decisions use thresholds calibrated on original images. A reduction in false nonmatches can come with an increase in false matches.

At 14-pixel resolution, accuracy rises from 58.827% with cosine to 65.411% with QMagFace. QMagFace corrects 398 cosine errors but introduces 3 new errors. False nonmatches fall from 2,470 to 2,074, while false matches rise from 0 to 1. Both methods remain far below their original-image accuracy: cosine loses 40.973 percentage points and QMagFace loses 34.405.

The strongest blur has a smaller effect. At sigma 2.1, cosine accuracy is 99.500% and QMagFace accuracy is 99.733%. QMagFace fixes 14 cosine errors and adds 0. FNMR drops from 0.967% to 0.500%, with FMR unchanged at 0.033%. The losses from original-image accuracy are 0.300 and 0.083 percentage points, respectively.

Reducing brightness to 25% leaves accuracy close to the original condition: 99.783% for cosine and 99.800% for QMagFace. QMagFace corrects 2 errors and introduces 1, a net difference of one decision. The false-nonmatch count is unchanged between methods, while FMR falls from 0.067% to 0.033%. Each method loses 0.017 percentage points from its own original accuracy.

![Figure 2. Mean raw MagFace probe norms under degradation. Each scored pair contributes once, so repeated images receive more weight.](../../results/project2/report_figures/report_quality.png)

The pair-weighted mean raw probe norm is 28.658 on original images, 22.397 at 14 pixels, 23.983 at blur sigma 2.1 and 26.939 at brightness 0.25. The mean norm decreases as degradation becomes stronger within each tested family. This describes the average response; individual images need not follow the same pattern, and the result does not establish a quality ranking for all real-world degradations. Reference-quality distributions stay fixed because reference images are unchanged.

| Condition | Cosine AUC | QMagFace AUC | Cosine EER (%) | QMagFace EER (%) |
| --- | --- | --- | --- | --- |
| Original | 0.999528 | 0.999579 | 0.267 | 0.267 |
| Resolution 56 | 0.999439 | 0.999436 | 0.233 | 0.267 |
| Resolution 28 | 0.998936 | 0.998924 | 0.667 | 0.633 |
| Resolution 14 | 0.951571 | 0.947902 | 11.667 | 12.104 |
| Blur 0.7 | 0.999491 | 0.999525 | 0.233 | 0.267 |
| Blur 1.4 | 0.999391 | 0.999368 | 0.233 | 0.233 |
| Blur 2.1 | 0.999226 | 0.999127 | 0.433 | 0.367 |
| Brightness 0.75 | 0.999507 | 0.999554 | 0.267 | 0.267 |
| Brightness 0.50 | 0.999494 | 0.999535 | 0.233 | 0.267 |
| Brightness 0.25 | 0.999418 | 0.999399 | 0.267 | 0.267 |

Pooled ROC AUC and interpolated EER summarize performance across thresholds using all scored pairs. These threshold sweeps do not select the thresholds used for held-out predictions. Full ROC/DET curves, score and quality distributions, and paired changes from the original condition are saved with the results.

The 14-pixel result needs careful interpretation. QMagFace gains 6.585 accuracy points, yet AUC decreases from 0.951571 to 0.947902 and EER increases from 11.667% to 12.104%. The improvement is specific to the decision thresholds calibrated on original images. When the quality adjustment is active and m is the minimum feature norm, the score becomes (1 + beta*m)*s - alpha*m. A threshold T on this score is equivalent to a cosine cutoff of (T + alpha*m)/(1 + beta*m). At the negative thresholds measured here, that cutoff falls as quality decreases. More genuine pairs can therefore be accepted, at the cost of possible additional false matches. This explains the accuracy gain without implying that downsampling's lost information has been recovered.

## 6. Limitations and interpretation

The reproduction covers the scoring function and the LFW benchmark with released features. It does not repeat MagFace training, QMagFace parameter fitting or experiments on the paper's other datasets. Coverage and threshold selection differ between the author-feature benchmark and the local image experiment, so their absolute accuracies should be interpreted separately. The 14-pixel condition preserves the relative resolution scale of Project 1's 20-pixel condition, but the crops differ.

LFW repeats subjects and images, and the calibration sets overlap. Fold and condition results are therefore dependent; the paired changes reported here are descriptive. Differences of a few errors do not establish a general advantage. With about 3,000 impostor comparisons, one false match changes FMR by approximately 0.033 percentage points. This sample is too small to assess very low false-match rates reliably, and observing zero false matches does not establish zero risk.

Multiplying pixel values simulates dimming but omits sensor noise and exposure artifacts from real low-light capture. Fixed alignment also leaves out detection failures caused by degradation. Training/evaluation identity overlap and demographic fairness were not audited, so the study does not establish deployment suitability. Downloaded research artifacts are used locally; the submission excludes face images, model weights and large feature caches.

## 7. Conclusions and reproducibility

The independent scorer agrees numerically with the authors' implementation, and the LFW result closely matches the published QMagFace accuracy under the released evaluator's conventions. The controlled experiment compares scoring rules on the same features and pairs, keeping each method's original-image thresholds fixed across degradations. QMagFace improves fixed-threshold accuracy in 9 of ten conditions and ties in 1; its largest gain is 6.585 points at 14 pixels. However, even there it rejects 69.156% of genuine pairs, and its ranking metrics are worse than cosine. The blur benefit is smaller, and brightness changes produce only zero or one net corrected decision. The main finding is that QMagFace can reduce false rejections when thresholds set on original images are reused on degraded images. Severe resolution loss remains a substantial problem, and the results do not show an advantage at every decision threshold.

The reference environment is Python 3.13.7, NumPy 2.5.3, PyTorch 2.8.0, torchvision 0.23.0, facenet-pytorch 2.5.3 and Pillow 11.2.1 on macOS arm64. Inference used MPS with batches of 32 and two CPU threads for alignment. A separate benchmark environment adds scipy 1.16.2 and scikit-learn 1.7.2 while keeping Project 1 dependencies intact. The downloaded checkpoint's SHA-256 is cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58. This locally recorded hash identifies the file used; it is not a publisher-signed checksum.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/check_project2.py
.venv/bin/python scripts/project2_reanalyze.py
.venv/bin/python -m src.project2_experiment \
  --stage run --device mps --grid full --batch-size 32
```

Use --device cpu on systems without MPS. Setup and download instructions are in docs/project2/reproduce.md. The artifact audit, protocol, AI-assistance record and experiment log explain how the work was carried out. Saved scores include failed attempts, original folds, both qualities and both scores; predictions include the threshold used for each decision. Manifests record file hashes, software versions, checkpoint identity and timing. The checker recomputes metrics from saved pairs, with optional checks that reconstruct scores from raw features. The submission includes the full Git history.

## References

[1] P. Terhorst, M. Ihlefeld, M. Huber, N. Damer, F. Kirchbuchner, K. Raja and A. Kuijper. QMagFace: Simple and Accurate Quality-Aware Face Recognition. IEEE/CVF WACV, 2023, pp. 3484-3494. https://openaccess.thecvf.com/content/WACV2023/html/Terhorst_QMagFace_Simple_and_Accurate_Quality-Aware_Face_Recognition_WACV_2023_paper.html

[2] Authors' QMagFace code and artifact links. https://github.com/pterhoer/QMagFace Commit 395797145da1b1b263cbe7fb1564c6ea91f55b23 (accessed October 18, 2026).

[3] Q. Meng, S. Zhao, Z. Huang and F. Zhou. MagFace: A Universal Representation for Face Recognition and Quality Assessment. IEEE/CVF CVPR, 2021. https://github.com/IrvingMeng/MagFace Commit 99bae614ac2643b9694bf18e0c5645272ff6acfa.

[4] Labeled Faces in the Wild, funneled images and official pair files, distributed through the scikit-learn/figshare mirror. Source URLs and verified dataset hashes are recorded in src/lfw_dataset.py and docs/dataset_research.md.
