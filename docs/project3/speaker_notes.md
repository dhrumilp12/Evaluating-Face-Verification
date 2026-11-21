# Project 3 speaker notes and questions

The deck has 12 core slides and three backup slides. Aim for about 10–12 minutes, then questions. Use these notes as preparation rather than reading every sentence aloud. Paper and local-data references remain in the PowerPoint notes.

## Suggested timing

- Slides 1–3: problem and earlier projects, about 2 minutes.
- Slides 4–5: new method and evaluation, about 2 minutes.
- Slides 6–10: findings and investigations, about 5 minutes.
- Slides 11–12: limitations and conclusions, about 2 minutes.
- Slides 13–15: backup material for questions.

## Slide 1: Quality-aware calibration for face verification

This capstone continues the face-verification problem from Projects 1 and 2. I investigate whether a small learned decision rule that uses image quality helps beyond simply changing a threshold. The key finding is that exposure to degraded calibration examples explains a large gain. The more elaborate learned score does not consistently beat the simpler recalibrated QMagFace control. The study uses saved Project 2 scores, not a newly trained face network. This presentation has twelve core slides and three backup slides, suitable for roughly ten to twelve minutes plus questions.

## Slide 2: The same problem across three projects

The biometric task is one-to-one face verification on LFW: decide whether two face images belong to the same person. Project 1 established a FaceNet baseline and controlled probe degradations. Project 2 reproduced the QMagFace comparison function on fixed MagFace embeddings. Project 3 changes the comparison and decision layer and asks what calibration contributes. The historical Project 1 accuracy is context only. Its backbone, training data and 160-pixel box crops differ from the 112-pixel landmark-aligned MagFace pipeline. We cannot attribute cross-project differences solely to scoring. Sources: docs/final_report.md; docs/project2/final_report.md; docs/project3/protocol.md.

## Slide 3: Severe resolution loss exposed a decision problem

Project 2 calibrated thresholds on original-quality training pairs and froze them for degraded probes. At 14 pixels, cosine accuracy was 58.83% and QMagFace accuracy was 65.41%. Their false-nonmatch rates were 82.36% and 69.15%, respectively. That motivates testing calibration mismatch, but it does not prove that lost visual information can be recovered. Both scores still have imperfect ranking. This figure uses the Project 3 original_only controls, which exactly recover all twenty Project 2 thresholds. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts.

## Slide 4: A small quality-dependent decision score

The new model takes cosine similarity s and q, the minimum of the two raw MagFace embedding norms. Its three feature columns are s, q and their product. Each training split fits a StandardScaler and an L2-regularized logistic regression with C=1. The displayed equation is the equivalent raw-feature form after absorbing scaling into the coefficients. Accept when the decision score is at least zero. These logits are decision scores, not validated probabilities. The ablations remove the interaction and then remove quality. The two additional controls fit accuracy-maximizing thresholds to cosine and to the unchanged published QMagFace score using exactly the same calibration examples. Repository: docs/project3/protocol.md; src/project3_calibration.py; results/project3/calibration/models.json. Research context: https://arxiv.org/abs/2111.13475; https://arxiv.org/abs/2103.06627.

## Slide 5: Calibration regimes and held-out pairs

There are 6,000 requested pairs and 5,999 eligible pairs in each of ten conditions. For each original LFW fold, every variant of its pairs stays out of training. All fitted statistics use only the other nine folds. We test original-only calibration, mixed calibration on all ten conditions, and three runs withholding an entire degradation family. A mixed-training result evaluates new pair folds with known synthetic degradation types, whereas family exclusion tests transfer to the omitted type. Every underlying training pair has total weight one across its variants. LFW pair folds can share images or identities across different pairs. Repository: docs/project3/protocol.md; src/project3_calibration.py; results/project3/calibration/models.json.

## Slide 6: Recalibration explains a large accuracy gain

At 14 pixels, the unchanged QMagFace score rises from 65.41% with original-only calibration to 85.75% with mixed calibration, a 20.34 percentage-point increase. Mixed-calibration cosine reaches 81.31%. The full learned model reaches 84.80%, below recalibrated QMagFace. This is why the same-data threshold controls matter: the gain cannot all be attributed to adding learned quality features. All methods use the same held-out pairs. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts.

## Slide 7: Higher acceptance brings false-match costs

The methods trade false non-matches against false matches. The original-only QMagFace threshold makes only one false match but 2,074 false nonmatches at 14 pixels. With mixed calibration it makes 91 false matches and 764 false nonmatches. The learned score makes 121 and 791, so recalibrated QMagFace has lower rates of both errors at this setting. Relative to that control, the full model corrects 51 mistakes but introduces 108, giving 57 fewer correct decisions. The scalar controls maximize training accuracy and the logistic models minimize regularized log loss. No low-FMR target was optimized. One false match among 3,000 impostors changes pooled FMR by about 0.0333 percentage points. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts. Paired evidence: results/project3/calibration/paired_changes.csv.

## Slide 8: Quality helps, but the interaction adds little

The fair feature ablation compares three logistic models with the same training data, regularization and zero decision threshold. At 14 pixels, quality raises accuracy from 79.56% to 84.68%. The interaction gives only a further 0.117 percentage points. Across the ten equally weighted conditions, the quality-only model has 98.026% accuracy versus 98.001% for the full model. The extra term does not consistently help. These are descriptive results, not significance claims. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts.

## Slide 9: Transfer depends on the degradation family

Each row withholds all three levels of one degradation family from training. The other two families and original faces remain available. The learned score reaches 67.48% at 14 pixels when resolution is unseen, down from 84.80% with mixed training. Cosine performs best among these three on the strongest withheld blur. The learned score does better on withheld brightness, but the benefit is small. These patterns do not support general robustness to arbitrary degradations. The quality-support diagnostic finds only four of 5,999 held-out 14-pixel q values outside their fold's training range, so simple range membership is not enough to explain transfer. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts. Auxiliary source: results/project3/analysis/summary.json.

## Slide 10: Regularization changes the operating tradeoff

The protocol fixed C=1 as the primary setting before measurement. C=0.1 and C=10 are sensitivity checks, not a search for a winning outer-test setting. Weaker regularization increases severe-resolution accuracy but also raises mixed-training false matches: 2.53%, 4.03%, 4.43%. Original-quality mixed accuracy decreases slightly as C grows. The family-excluded gap remains large throughout. Keep all three settings visible and retain C=1 in the main conclusions. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts.

## Slide 11: What this evidence can establish

The experiment establishes controlled differences at the decision layer for these saved comparisons. It does not establish deployment performance. First, the benchmark is retrospective: earlier project outcomes motivated the question. Second, original LFW pair folds can share images or identities, and repeated conditions are correlated. Third, alignment happens before degradation, so detection failures under poor images are outside scope. Fourth, the synthetic brightness and blur operations are not a complete model of real sensors. Finally, a better accuracy operating point can coexist with worse ranking. At 14 pixels the full mixed model has mean-fold AUC 0.939861, below cosine 0.951617. I make no significance, demographic-fairness or low-FMR deployment claim. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts. Repository: docs/project3/protocol.md; src/project3_calibration.py; results/project3/calibration/models.json.

## Slide 12: Conclusions

My main contribution is a controlled calibration study, not a claim of a new state-of-the-art recognizer. Matching calibration to degraded inputs produced the largest operational improvement. Quality helped within the logistic family, but the additional interaction did not consistently help and did not beat the same-data QMagFace control at severe resolution. The held-out-family experiment showed limited transfer. A sensible next experiment would use an untouched dataset and a predeclared FMR requirement, with calibration and testing separated at the identity level where possible. These are future directions, not completed results. The code, fitted models, predictions and analysis allow the professor to inspect these findings. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts. Repository: docs/project3/protocol.md; src/project3_calibration.py; results/project3/calibration/models.json.

## Slide 13: Appendix: reproducing the study

The repository contains the saved Project 2 input comparisons, so reproducing the Project 3 calibration study needs no new face-image download or GPU. Install the small requirements-project3.txt environment, run the tests, then use the runner with a fresh output directory. The independent checker reconstructs every saved score from the serialized coefficients, verifies membership and feature scaling, and recomputes metrics using a separate implementation. Thirteen tests passed. The completed study has 290 fitted models and 1,128,000 prediction rows including failures and repeated methods/conditions. A fresh full run reproduced all five payloads byte for byte. Sources: docs/project3/README.md; validation.json; reproducibility.json; results/project3/calibration/manifest.json. AI assisted the development, testing and presentation; the repository contains the disclosure in docs/project3/ai_usage.md.

## Slide 14: Appendix: all ten mixed-calibration conditions

This table includes all ten conditions so the severe-resolution finding is not presented in isolation. All methods receive the same calibration conditions. The simple logistic quality model has the highest equal-condition mean among the five primary methods, but C=1 with the interaction remains the declared full method. No method wins every condition. The report includes all five primary methods and the source CSV includes every sensitivity and excluded-family result. Repository: results/project3/calibration/aggregates.csv and folds.csv. All stated rates are arithmetic means across the ten original LFW folds unless identified as pooled counts.

## Slide 15: Appendix: sources and attribution

Research sources: Terhorst et al., QMagFace: Simple and Accurate Quality-Aware Face Recognition, WACV 2023, https://arxiv.org/abs/2111.13475, official implementation https://github.com/pterhoer/QMagFace. Meng et al., MagFace: A Universal Representation for Face Recognition and Quality Assessment, CVPR 2021, https://arxiv.org/abs/2103.06627. Implementation reference: https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html and https://scikit-learn.org/1.8/common_pitfalls.html. Measured evidence is the project repository, particularly results/project3/calibration, docs/project3/protocol.md and docs/project3/results.md. Project 3 is a course modification of the comparison/decision stage, not a reproduction claim for a new paper. AI assisted study design, implementation, testing, analysis, documentation and slide preparation. The student remains responsible for understanding and presenting the work accurately.

## Questions to prepare for

### What did Project 3 add beyond Project 2?

Project 2 used a published quality-aware score and original-quality thresholds. Project 3 learns a supervised decision score from cosine and raw embedding quality, tests calibration on multiple conditions, removes features in ablations, and evaluates excluded degradation families. The face network remains fixed.

### Did the new method beat QMagFace?

At 14 pixels it beats the old original-calibrated QMagFace operating point, but it loses to QMagFace recalibrated on the same mixed training data: 84.798% versus 85.748%. It also has more false matches and false nonmatches there. This is why I report the simpler control and do not claim a universal improvement.

### Why is the negative result still useful?

It separates the effect of changing calibration data from the effect of adding model complexity. Quality helps relative to cosine-only logistic regression, but the interaction adds little consistently. The experiment identifies a useful simpler alternative and the limits of transfer.

### What prevents train/test leakage?

Each model and scaler use only the other nine official pair folds. All degraded versions of every held-out pair stay outside training. Excluded-family runs remove the entire family from training. Tests changed held-out labels, scores and qualities and verified unchanged fitted parameters. LFW pair folds can still share images or identities across different pairs, which is an explicit limitation.

### Why use C = 1?

It was fixed in the protocol before the new measurements. C = 0.1 and 10 are sensitivity checks. They are reported without choosing the best setting using outer-fold results. Training rows have weight 1 divided by the number of included conditions, so each underlying pair contributes total weight one.

### Does this prove the outputs are calibrated probabilities?

No. The logistic decision function is used as a score with a fixed zero threshold. This study evaluates verification decisions and ranking. It does not validate probabilities or forensic likelihood ratios.

### How can accuracy improve while AUC becomes worse?

Accuracy uses one fitted decision boundary. AUC summarizes ranking over a threshold sweep. Moving a boundary can accept more degraded genuine pairs without improving ranking. The report distinguishes those outcomes and computes AUC within each fold before averaging.

### Why does the method struggle with unseen resolution degradation?

The measurements show dependence on the calibration distribution. They do not identify one unique cause. Only four of 5,999 test quality values in that setting lie outside their respective training ranges, so a scalar range check does not explain the failure. Quality magnitude does not fully describe the joint relationship between identity, similarity and degradation.

### Are 1,128,000 predictions independent examples?

No. They repeat the same 6,000 requested pairs across methods, conditions and training regimes. One inherited genuine pair failed preprocessing, so recognition metrics use 5,999 eligible pairs per condition. I do not treat repeated rows or shared calibration folds as independent evidence.

### Why not train a new backbone?

The capstone permits modifications to the existing approach. Holding the embeddings fixed isolates the comparison and decision layer and makes the experiment reproducible without a new GPU training run. It does not claim to restore facial information lost in the image.

### What would you investigate next?

Use an untouched evaluation dataset, a calibration split separated by identity where possible, and a predeclared operational FMR target. Test realistic capture noise and degraded-image detection. These are future directions, not completed experiments.

### How was AI used?

AI supported the study design, implementation, testing, experiment execution, analysis, documentation and presentation. The repository includes an accurate disclosure. I am responsible for reviewing the work, understanding the method and reporting the measured results accurately.
