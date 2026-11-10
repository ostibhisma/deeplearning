# Chapter 7 — Conclusion and Future Work

## 7.1 Summary of the work

This thesis set out to design, implement, evaluate and critically examine a convolutional neural network for seven-class facial expression recognition on FER2013, and to connect it to a live demonstration. The design and implementation are complete. The evaluation is specified in full but not executed, because the dataset was not available in the project environment.

The system that exists is compact and readable. A loader parses the FER2013 CSV, encodes labels as one-hot vectors and reshapes the images to 48×48×1. A VGG-style network with three convolutional blocks, batch normalisation, dropout and a two-layer classifier head is trained with Adam and categorical cross-entropy, with geometric augmentation and three callbacks that adapt the learning rate, stop early and keep the best checkpoint. A webcam demonstration couples the classifier to a Haar cascade detector and overlays the predicted label on the frame.

The analysis produced four things that were not in the original code or the original draft. First, an exact characterisation of the architecture: 2,818,375 parameters, of which 2,816,711 are trainable; 59.3 % of them in the classifier head; a receptive field of about 40×40; and a forward cost of about 3.05×10⁸ multiply–accumulates per image. Second, an audit of 21 defects, including test-set leakage, a missing evaluation step, misleading training plots and two bugs that made the demonstration classify the wrong image region. Third, a leakage-free evaluation protocol with pre-specified metrics, baselines, confidence intervals and significance tests. Fourth, a qualitative analysis of the gap between benchmark crops and detector crops, which is the project's answer to the deployment question.

## 7.2 Answers to the research questions

**RQ1. How does the implemented CNN perform on a leakage-free FER2013 test partition in terms of accuracy and macro-F1?**

This question cannot be answered numerically at present, because no training run has been completed. The protocol that will answer it is specified in Chapter 3 and the reporting template is in Chapter 5. What can be said now is what the answer will be compared against: a majority-class baseline of 24.7 % accuracy and 0.057 macro-F1, computed from the published test distribution, and a uniform random baseline of 14.3 %. Any trained model must clear those by a wide margin before its accuracy is interesting, and its macro-F1 will show whether it has learned anything about the minority classes.

It is worth being explicit about why the question was not answered by the code as shipped. The shipped pipeline uses the test partitions as validation, so its accuracy would have been selected on the test labels; and it never computes test accuracy at all, so there would have been no number to report even if the data had been present. The question is therefore unanswered for two independent reasons, and both had to be fixed before an answer would have meant anything.

**RQ2. What contribution do batch normalisation, dropout and geometric augmentation make to the generalisation gap?**

This question is designed but not executed. The ablation compares the full model with three single-component removals across three seeds, with the decision rules fixed in advance: a component counts as contributing if the paired accuracy difference has a bootstrap interval excluding zero, McNemar's test is significant after correction, and the direction is consistent across seeds. The expectation, stated as a hypothesis rather than a finding, is that augmentation reduces the training–validation gap, that removing dropout raises training accuracy and lowers test accuracy, and that removing batch normalisation produces the largest seed-to-seed variance. If the results contradict this, the contradiction is the finding.

**RQ3. How does classification behaviour change when the model moves from benchmark crops to Haar-detected faces in a live stream?**

This question is answered qualitatively, and the answer is that three shifts separate the two settings. Framing: the detector returns a loose rectangle that includes hair and background, so the face occupies a smaller fraction of the 48×48 input than it did in training. Illumination: webcam scenes may be backlit or dim, and the pipeline applies no equalisation. Geometry: the detector does not align the face, so rotation and profile views fall outside the augmented training distribution. A fourth, temporal, shift is that single-frame predictions can flicker, and no smoothing is applied.

None of these shifts is measured, because there is no labelled video. What can be said is that the demonstration, once the crop and resize bugs are fixed, exercises the pipeline end to end, and that its output should be read as a demonstration of the system rather than as evidence of field accuracy. That is a weaker answer than a measured one, and it is the honest one.

## 7.3 Contributions

The contributions are modest and mostly of the kind that helps the next person who works with the code.

1. **A complete, readable implementation** of a FER2013 pipeline, from CSV parsing to a live demo, with loading, modelling and inference separated into modules.
2. **A precise architectural characterisation**, computed from the code rather than estimated, which corrects the earlier draft's figure of about 1.2 million parameters and shows that most of the model sits in the classifier head.
3. **An implementation audit** identifying 21 defects, with their consequences and corrections, including the leakage problem that would have invalidated any reported accuracy and the two demo bugs that made its predictions meaningless.
4. **A leakage-free evaluation protocol** with pre-specified metrics, baselines, intervals, significance tests and decision rules.
5. **A deployment analysis** that identifies the framing, illumination, geometry and temporal shifts between benchmark and webcam conditions, and proposes concrete mitigations.

We do not claim a new architecture, a new loss, or state-of-the-art performance. The architecture is conventional by design, and the report says so throughout.

## 7.4 Future work

The first priority is obvious: obtain FER2013, create the pinned environment, apply the corrections in Chapter 4, and run the twelve configurations in Table 5.1. Until that is done, the empirical questions remain open, and the most valuable next step is not a new idea but a completed baseline.

Beyond that, five directions follow directly from the limitations in Chapter 6.

**Architecture.** Replace the flatten-and-dense head with global average pooling, or with a 1×1 convolution followed by pooling, and compare parameter counts, accuracy and calibration. Add a residual connection and deepen the stack to test whether the degradation problem becomes relevant at this scale. Add a lightweight attention module, such as a Squeeze-and-Excitation block (Hu et al., 2018) or CBAM (Woo et al., 2018), and check whether it improves the minority classes.

**Imbalance and label noise.** Add class-weighted cross-entropy and focal loss (Lin et al., 2017) as separate configurations, and compare macro-F1 rather than accuracy. Add label smoothing (Szegedy et al., 2016) and, if FER+ can be obtained, train on annotator distributions (Barsoum et al., 2016). These are the changes most likely to improve disgust and fear, which are the classes the current loss neglects.

**Transfer learning.** Fine-tune a pretrained network on FER2013 and compare it with the from-scratch baseline. This would almost certainly improve accuracy, and the size of the improvement would quantify how much of the problem is data-limited rather than architecture-limited. It would also break the "from scratch" constraint that the current design imposes, which is why it is a separate experiment rather than a replacement.

**Deployment.** Replace the Haar cascade with MTCNN (Zhang et al., 2016) or RetinaFace (Deng et al., 2019), add landmark-based alignment, apply histogram equalisation, and add temporal smoothing over a short window. Measure latency per stage, since the current analysis can only guess where the time goes. Apply temperature scaling so that displayed probabilities are calibrated (Guo et al., 2017).

**Interpretability.** Visualisation methods such as Grad-CAM (Selvaraju et al., 2017) and the deconvolution approach of Zeiler and Fergus (2014) would show whether the network attends to the eyes and mouth or to background artefacts. This serves both as an explanation and as a debugging tool, and it would make the error analysis in Chapter 5 considerably more informative.

**Evaluation.** Move from a single split to five-fold cross-validation, add a small-CNN and a logistic-regression baseline, and evaluate on CK+ or RAF-DB to measure the domain gap directly. If demographic metadata can be obtained from another dataset, add subgroup analysis; FER2013 cannot support it.

A sixth direction is less technical. The audit in this project found that most defects were in the glue rather than the model, and that they were silent. A short smoke test — run the pipeline on ten images, assert shapes and label order, assert that the output directory exists, and check that a known image receives a plausible label — would have caught several of them. Packaging that test with the code, and running it before every experiment, is a small change with a large return.

## 7.5 Closing remarks

The project did not produce the result we hoped for, in the sense that it produced no accuracy figure at all. It produced something else: a clear account of what the system is, what it costs, where it fails, and what would be needed to evaluate it properly. That account changed our view of the problem. The model was the easy part. The data handling, the evaluation and the deployment were where the real difficulty lay, and they are the parts that a report written from memory tends to skip.

For a project at this level, that is a reasonable outcome. The code is small enough to understand completely, the numbers that describe it are exact, and the path to a measured result is specified in enough detail that someone else could follow it. The next run will produce numbers; the work in this thesis is what makes those numbers mean something.
