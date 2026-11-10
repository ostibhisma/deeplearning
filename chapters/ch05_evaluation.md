# Chapter 5 — Evaluation Design and Status of the Empirical Work

## 5.1 Introduction and status

This chapter specifies the empirical evaluation and reports its current status. The status is unusual and needs to be stated at the outset: no training run has been completed, because the FER2013 file is not available in the project environment. There are therefore no measured accuracies, no confusion matrices and no ablation results to report. What follows is the design of the evaluation, the run matrix that will produce the results, the baselines that can be computed without training, the reporting template into which measured values will be placed, and the decision rules that will be used to interpret them.

Writing the chapter this way is a deliberate choice. The alternative was to fill it with numbers taken from published papers or from intuition about how the model "should" behave, presented as findings. That would have produced a more conventional chapter and a dishonest thesis. The distinction between a result and an expectation is the difference between an evaluation and a performance.

The chapter is organised as follows. Section 5.2 defines the experimental configurations and the run matrix. Section 5.3 computes the trivial baselines from the published test distribution; these are arithmetic rather than measurements, and they are labelled as such. Section 5.4 sets out the tables and figures that will be produced, with their exact structure. Section 5.5 states the decision rules in advance. Section 5.6 discusses threats to validity. Section 5.7 summarises.

## 5.2 Experimental configurations and run matrix

### 5.2.1 Configurations

Four configurations are defined. Each changes exactly one component relative to the full model, so that a difference between two configurations can be attributed to that component rather than to a changed training budget or a different data split.

**Configuration A — Full model.** The architecture in Table 3.2, trained with the augmentation in Section 3.6.3, dropout as specified, batch normalisation in all three blocks, Adam at 10⁻³, batch size 64, maximum 100 epochs, early stopping on validation loss with patience 6, checkpoint on the best validation loss.

**Configuration B — No augmentation.** The same architecture and training settings, but the training generator applies rescaling only. Rotation, flip, zoom and shear are disabled. This isolates the contribution of geometric augmentation.

**Configuration C — No dropout.** The same architecture and training settings, with every dropout rate set to zero. This isolates the contribution of dropout. Batch normalisation remains, because removing both at once would confound the two effects.

**Configuration D — No batch normalisation.** The same architecture and training settings, with the five batch-normalisation layers removed. Dropout remains. Removing BN changes the parameter count slightly (by 1,664 trainable and 1,664 non-trainable parameters) and changes the optimisation dynamics substantially, which is the point of the comparison.

The four configurations share the data split, the seed set, the batch size, the optimiser and the stopping rule. They differ only in the component under test. That is what makes the ablation interpretable.

### 5.2.2 Seeds and repetitions

Each configuration is trained with three seeds: 42, 123 and 2026. The seeds control Python's `random`, NumPy, TensorFlow initialisation and the augmentation pipeline. Three seeds are enough to estimate run-to-run variability roughly and to detect a difference that is large relative to that variability; they are not enough for a precise variance estimate, and the report says so rather than pretending otherwise.

Twelve runs result. On a CPU, each run is expected to take hours rather than minutes, because the model performs about 3.05×10⁸ multiply–accumulates per image and an epoch contains 404 steps of 64 images. The exact duration is unknown until measured, and it is one of the quantities the first run will record. If time is short, the priority order is A, then B, then C, then D, because A establishes the baseline and B tests the component most likely to matter on a small dataset.

**Table 5.1. Run matrix. All entries are pending; no run has been executed.**

| Run | Configuration | Seed | Purpose | Status |
|---|---|---|---:|---|
| A-42 | Full | 42 | Baseline | not executed |
| A-123 | Full | 123 | Baseline variance | not executed |
| A-2026 | Full | 2026 | Baseline variance | not executed |
| B-42 | No augmentation | 42 | Augmentation effect | not executed |
| B-123 | No augmentation | 123 | Augmentation effect | not executed |
| B-2026 | No augmentation | 2026 | Augmentation effect | not executed |
| C-42 | No dropout | 42 | Dropout effect | not executed |
| C-123 | No dropout | 123 | Dropout effect | not executed |
| C-2026 | No dropout | 2026 | Dropout effect | not executed |
| D-42 | No batch norm | 42 | BN effect | not executed |
| D-123 | No batch norm | 123 | BN effect | not executed |
| D-2026 | No batch norm | 2026 | BN effect | not executed |

### 5.2.3 Environment

The runs require Python 3.11 or 3.12, TensorFlow 2.15 with the bundled Keras, NumPy, pandas, OpenCV, Matplotlib and tqdm, with versions pinned in `requirements.txt`. The exact versions will be recorded in each run manifest. The hardware is whatever is available; the manifest records the CPU or GPU model, the memory and the operating system. Because the environment does not yet exist on the machine used for this report, none of these details can be given now.

### 5.2.4 Artefacts per run

Each run writes: the checkpoint in `.keras` format, a CSV of per-epoch training and validation loss and accuracy, a JSON manifest with the configuration and seed, a JSON summary with the test metrics, a confusion-matrix figure, and the accuracy and loss curves. The evaluation script is given in Appendix D. The artefacts are named by configuration and seed so that a result can be traced to the run that produced it.

## 5.3 Baselines computable without training

Two baselines follow from the published test distribution and require no model. They are arithmetic, not measurements, and they are included because they set the floor against which any trained model must be judged.

The test partition contains 7,178 images, of which 1,774 are labelled happy. A classifier that always predicts happy therefore achieves

  1,774 / 7,178 = 0.2471,

or 24.7 % accuracy. This is the majority-class baseline. Its macro-F1 follows from the per-class scores. Happy has precision 0.2471 and recall 1.0, so F1 = 2 × 0.2471 × 1.0 / (0.2471 + 1.0) = 0.396. The other six classes have zero precision (the classifier never predicts them) and therefore zero F1. Macro-F1 is the mean over seven classes:

  0.396 / 7 = 0.0566.

The gap between 24.7 % accuracy and 0.057 macro-F1 is a compact illustration of why both metrics are needed. A model that always says "happy" looks a quarter right by accuracy and almost entirely wrong by macro-F1.

The uniform random baseline predicts each class with probability 1/7. Its expected accuracy is 1/7 = 0.1429, or 14.3 %. Under uniform random guessing the expected per-class precision and recall are both 1/7, so the expected per-class F1 is also 1/7, and the expected macro-F1 is 1/7 ≈ 0.143. Cohen's κ is zero for both baselines by construction, since neither uses information about the image.

**Table 5.2. Baselines computed from the published test distribution. These are arithmetic values, not measured model performance.**

| Baseline | Rule | Accuracy | Macro-F1 | κ |
|---|---|---:|---:|---:|
| Uniform random | Sample a class uniformly | 0.143 (expected) | 0.143 (expected) | 0.000 |
| Majority class | Always predict happy | 0.247 | 0.057 | 0.000 |

Two implications follow. First, any trained model must clear 24.7 % accuracy by a wide margin before it is worth discussing; a result in the high twenties would be barely distinguishable from the majority rule. Second, macro-F1 is the metric that shows whether the model has learned anything about the minority classes. A model with 50 % accuracy and 0.30 macro-F1 is more interesting than one with 55 % accuracy and 0.20 macro-F1, even though the second looks better in a headline.

A third baseline, a small two-layer CNN or a logistic regression on raw pixels, would separate the contribution of depth from the contribution of fitting any decision boundary at all. Neither is implemented. They are listed in Chapter 7 as the next experiments rather than reported here.

## 5.4 Reporting template

The following tables and figures are defined now so that the measured values can be inserted without redesigning the chapter. Every empty cell in the template will be filled from an actual run or left as "not run"; none will be filled from expectation.

### 5.4.1 Main results

Table 5.3 will report, for each configuration and seed, the test accuracy, the macro-F1, the 95 % bootstrap interval for both, the best validation loss, the epoch at which it occurred, and the number of epochs trained. The mean and standard deviation across seeds will be given in a summary row.

**Table 5.3. Main results template.**

| Run | Accuracy | 95 % CI | Macro-F1 | 95 % CI | Best val loss | Best epoch | Epochs |
|---|---:|---|---:|---|---:|---:|---:|
| A-42 | — | — | — | — | — | — | — |
| A-123 | — | — | — | — | — | — | — |
| A-2026 | — | — | — | — | — | — | — |
| Mean ± SD | — | — | — | — | — | — | — |
| B-42 … B-2026 | — | — | — | — | — | — | — |
| C-42 … C-2026 | — | — | — | — | — | — | — |
| D-42 … D-2026 | — | — | — | — | — | — | — |

The dashes are placeholders, not zeros. A dash means "no value exists"; a zero would be a measurement.

### 5.4.2 Per-class analysis

Table 5.4 will report precision, recall and F1 for each of the seven classes, for the best run of Configuration A. The per-class rows make the imbalance visible: if disgust recall is near zero while happiness recall is near one, the aggregate accuracy is hiding the failure.

**Table 5.4. Per-class metrics template for the best full-model run.**

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Angry | 958 | — | — | — |
| Disgust | 111 | — | — | — |
| Fear | 1,024 | — | — | — |
| Happy | 1,774 | — | — | — |
| Sad | 1,247 | — | — | — |
| Surprise | 831 | — | — | — |
| Neutral | 1,233 | — | — | — |

The support column is computed from the published distribution and is the only column that is currently filled. It is included because per-class metrics without support are hard to interpret: a recall of 0.5 on 111 images and a recall of 0.5 on 1,774 images carry very different uncertainty.

### 5.4.3 Figures

Three figures will be produced. Figure 5.1 shows training and validation accuracy against epoch for Configuration A, with the best epoch marked. Figure 5.2 shows the corresponding loss curves. Figure 5.3 is the confusion matrix for the best run, with counts and row-normalised values. If the ablation runs are completed, Figure 5.4 will show validation accuracy across configurations as a box plot over seeds.

The curves are not decoration. They show whether the model underfits (both curves low), overfits (training high, validation low), or converges (both high and close). The confusion matrix shows which pairs are confused. The box plot shows whether a difference between configurations is larger than the spread across seeds.

### 5.4.4 Paired comparisons

Table 5.5 will report McNemar tests for the three planned contrasts: A versus B, A versus C, and A versus D. For each, the table gives the discordant counts *b* and *c*, the χ² statistic with continuity correction, the *p*-value, and the paired bootstrap difference in accuracy with its interval.

**Table 5.5. Paired comparison template.**

| Contrast | b | c | χ² | p | Δ accuracy | 95 % CI |
|---|---:|---:|---:|---:|---:|---|
| A vs B (augmentation) | — | — | — | — | — | — |
| A vs C (dropout) | — | — | — | — | — | — |
| A vs D (batch norm) | — | — | — | — | — | — |

Three tests are planned. With a family-wise α of 0.05, a Bonferroni correction gives a per-test threshold of 0.0167, and the report will state both the uncorrected and corrected interpretations rather than choosing whichever is more convenient.

## 5.5 Decision rules

The interpretation is fixed in advance so that it does not depend on which way the numbers fall.

**A component is judged to contribute** if the paired difference in accuracy between the full model and the ablated model has a 95 % bootstrap interval that excludes zero, the McNemar test is significant at the corrected threshold, and the direction of the difference is consistent across all three seeds. If the interval includes zero, the component is reported as "no detectable contribution at this sample size", not as "no effect".

**Overfitting is judged present** if training accuracy exceeds validation accuracy by more than ten percentage points at the best epoch. This threshold is arbitrary but stated in advance; it is a convention for description, not a statistical test.

**The deployment gap is judged material** if the demo's predictions on detected faces differ systematically from its predictions on the same faces cropped tightly, or if the detector fails on a substantial fraction of frames. Because there is no labelled video, this is a qualitative judgement supported by screenshots and a log of failure modes, not a measured accuracy.

**No claim of state-of-the-art performance will be made**, whatever the numbers. The model is smaller than the published reference points and is trained with a fixed recipe. If it lands near them, that is interesting; if it lands below, that is expected.

These rules are the closest thing this project has to pre-registration. They do not prevent every form of bias, but they prevent the most common one, which is choosing the story after seeing the result.

## 5.6 Threats to validity

**Internal validity.** The main threat was test-set leakage, which the protocol in Section 3.3.4 removes. A second threat is early stopping on a validation set that is small for the minority classes: with about 44 disgust images in validation, the validation loss is dominated by the majority classes, so the selected checkpoint may not be the best for disgust. A third is seed variance: three seeds give a rough estimate, not a precise one, and a difference that survives three seeds could still be noise.

**External validity.** The model is trained and tested on FER2013, a dataset of 48×48 web images with posed or acted expressions. It does not follow that it generalises to CK+, RAF-DB, AffectNet, or to a webcam in a room with poor lighting. The demo is a demonstration, not a field trial. No demographic subgroup analysis is possible because the dataset has no demographic metadata.

**Construct validity.** The labels are annotations, not measurements of internal states. A model that predicts them well has learned the annotators' conventions. The report uses "predicted label" rather than "detected emotion" for this reason, and Chapter 6 discusses the gap between the two.

**Statistical conclusion validity.** The test set has 7,178 images, so a percentage point is about 72 images. Confidence intervals will be wide enough that differences of one or two points may not be distinguishable. Three paired tests with a Bonferroni correction reduce power further. The report will therefore emphasise effect sizes and intervals rather than *p*-values, and will avoid ranking configurations whose intervals overlap.

**Reliability.** The evaluation depends on the checkpoint, the split index and the prediction array being saved and reused. The protocol in Section 4.8 specifies this. If those artefacts are lost, the numbers cannot be reproduced exactly even with the same code and seed.

## 5.7 Summary

This chapter specified the empirical evaluation and reported its status: twelve runs across four configurations and three seeds, all pending because the dataset is unavailable. It computed the two trivial baselines from the published distribution — 24.7 % accuracy and 0.057 macro-F1 for the majority rule, 14.3 % for random guessing — and explained why the gap between accuracy and macro-F1 matters. It defined the tables, figures and statistical tests that will report the results, fixed the decision rules in advance, and set out the threats to validity.

The chapter contains no measured result, and it says so on every page where a result would normally appear. When the dataset becomes available, the template can be filled without rewriting the argument: the configurations, the metrics, the tests and the interpretation rules are already fixed. That is the most useful thing a results chapter can do before the results exist.
