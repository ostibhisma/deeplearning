# Chapter 1 — Introduction

## 1.1 Background and motivation

A face is one of the densest communication channels available to a person. Long before a sentence is finished, a raised eyebrow, a tightened mouth or a held gaze has already told the listener something about how the speaker feels and how the message should be taken. This is not a marginal part of interaction. It is how people coordinate turn-taking, signal disagreement without starting a conflict, and recognise that someone needs help. Any system that interacts with people in a natural way eventually runs into the question of whether it can read those signals, or at least whether it can avoid misreading them badly.

Facial expression recognition (FER) is the task of assigning a label to a face image automatically. The version of the task studied here is the most common one in the machine learning literature: a greyscale image of a face, cropped to a fixed size, is classified into one of seven categories — anger, disgust, fear, happiness, sadness, surprise and neutrality. The output is a probability over those categories, and the system is judged by how often its highest-probability choice matches a human annotator's label.

The motivation for working on this is practical as well as scientific. Affective interfaces, driver monitoring, clinical support tools, tutoring systems and games have all been proposed as applications, and several of them are plausible: a system that notices a driver's eyes closing or a student's confusion has an obvious use. The scientific motivation is different. FER sits at the intersection of computer vision, psychology and machine learning, and each of those fields asks something different of it. Computer vision asks whether a representation can be learned from pixels. Psychology asks whether the categories themselves are the right ones. Machine learning asks how to generalise from a small, noisy, imbalanced dataset to faces the model has never seen.

That last question is where the difficulty lies. Progress on laboratory datasets, where faces are posed, frontal and evenly lit, has been substantial for more than a decade. Progress on images taken in ordinary conditions has been slower. Faces in the wild are rotated, partly occluded, unevenly lit and often small. People differ in how much they express, and in what their expressions mean. The labels themselves come from annotators, who disagree with each other, especially on fear, disgust and neutrality. A model trained on such labels can reach a respectable accuracy while still being wrong in ways that matter.

This project is a small, deliberate entry into that problem. We built a complete pipeline: a loader for the FER2013 dataset, a convolutional network trained from scratch, an evaluation protocol, and a live webcam demonstration. The model is not novel. It is a VGG-style stack of the kind that has been standard since 2014, with batch normalisation and dropout. What we have tried to do is understand every part of it, measure what it actually costs, and be clear about what it can and cannot show. That last part turned out to be the most demanding, and it shapes the structure of this report.

## 1.2 Problem statement

Formally, let *X* ⊂ ℝ<sup>48×48</sup> be the space of greyscale face images and *Y* = {0, 1, …, 6} the set of emotion labels in the FER2013 encoding. Given a training set *D* = {(x<sub>i</sub>, y<sub>i</sub>)}, the task is to learn a function *f*: *X* → *Y* that minimises the expected misclassification on unseen images drawn from the same distribution. In practice *f* is parameterised by a convolutional network with weights θ, trained to minimise the categorical cross-entropy

  L(θ) = −(1/N) Σ<sub>i</sub> Σ<sub>c</sub> y<sub>ic</sub> log p<sub>θ</sub>(c | x<sub>i</sub>),

where y<sub>ic</sub> is the one-hot target and p<sub>θ</sub>(c | x<sub>i</sub>) is the softmax output for class *c*.

The difficulty is not the optimisation. It is everything around it. Six problems recur in the literature and in our own work.

**Class imbalance.** Happiness is 16.5 times more frequent in the training set than disgust. A model can achieve a respectable accuracy by learning the majority classes well and the minority classes hardly at all, and the loss will barely notice.

**Label noise.** FER2013 labels are human annotations, not measurements. Annotators disagree, and the dataset records only the majority decision. Some training images are therefore labelled in a way that most people would not endorse, which puts a ceiling on what any model can learn from them.

**Low resolution.** At 48×48 pixels, the eyes occupy a few pixels and the fine structure around the mouth is gone. Subtle expressions, which are exactly the ones that matter in spontaneous interaction, are the ones most damaged by downsampling.

**Pose and illumination.** The images are not aligned. The same expression appears at different scales, rotations and lighting conditions, and the network has to learn invariance from examples rather than from geometry.

**Distribution shift at deployment.** A model trained on tight benchmark crops is used in a demo that receives loose rectangles from a face detector. The face occupies a different fraction of the frame, the background is included, and the model's accuracy is not guaranteed to survive the change.

**Validity of the target.** The seven categories come from a particular theoretical tradition, and that tradition is contested. Barrett et al. (2019) argue that facial movements do not map one-to-one onto emotional states, and Russell (1994) and Jack et al. (2012) question the cross-cultural evidence for universal recognition. A classifier trained on FER2013 learns the labelling conventions of its annotators. Calling its output "emotion" is a shorthand that needs to be unpacked, and we try to unpack it in Chapter 6.

FER2013 is a reasonable place to study these problems because it is small enough to train on modest hardware, widely used enough to permit comparison, and hard enough that a good result is not trivial. It contains 35,887 images: 28,709 for training and 7,178 for testing, split into public and private test partitions by the original challenge (Goodfellow et al., 2013). The test set is small. One percentage point of accuracy is about 72 images, so differences of one or two points between models are within the range that sampling noise can produce. That fact constrains every comparison in this report.

## 1.3 Research questions

The project is organised around three questions. They are deliberately modest, and each is answerable with the system we have built.

**RQ1. How does the implemented VGG-style CNN, trained from scratch with the implemented augmentation and regularisation, perform on a leakage-free FER2013 test partition in terms of accuracy and macro-F1?**

This is the baseline question. It asks for a number, but it also asks for the right number: measured on data that played no part in model selection, and summarised in a way that exposes the minority classes rather than hiding them behind accuracy.

**RQ2. What contribution do batch normalisation, dropout and geometric augmentation make to the gap between training and validation performance?**

The architecture contains three regularising mechanisms, and the project's own documentation claims they are necessary. The claim is testable by removing each one in turn and observing what happens to the training–validation gap and to test accuracy. Without that comparison, the design is a set of conventions rather than a set of decisions.

**RQ3. How does classification behaviour change when the model moves from benchmark crops to Haar-detected faces in a live video stream?**

This question is about the deployment gap. It is qualitative as much as quantitative, because we have no labelled video data. But the difference between a tight 48×48 benchmark crop and a loose detector rectangle is real, and it is the kind of difference that determines whether a model that looks good on a test set is useful in a demonstration.

The questions are ordered by dependency. RQ3 presupposes a trained model, and RQ2 presupposes that the baseline in RQ1 has been measured cleanly.

## 1.4 Aims and objectives

The overall aim is to design, implement, evaluate and critically examine a convolutional network for seven-class facial expression recognition on FER2013, and to connect it to a working real-time demonstration.

The objectives are:

1. Implement a data pipeline that parses FER2013, encodes labels, and produces correctly shaped tensors for training and evaluation, with a split that does not leak test information.
2. Implement a convolutional architecture with batch normalisation, dropout and geometric augmentation, and characterise it precisely in terms of layer shapes, parameter counts, receptive field and computational cost.
3. Train the model with a documented optimisation procedure, saving the best checkpoint according to a validation signal that is independent of the test set.
4. Evaluate the model with accuracy, per-class precision and recall, macro-F1, a confusion matrix, confidence intervals and paired significance tests against trivial baselines.
5. Determine empirically, through controlled ablations, what each regularising component contributes.
6. Build a live demonstration that detects faces, classifies them, and displays the result, and analyse the differences between this setting and the benchmark setting.
7. Document the engineering defects found during the work and the corrections required, so that the pipeline can be reproduced and extended.

Objectives 1 to 3 and 6 are implemented. Objective 4 is specified in Chapter 3 and implemented as a script, but not yet executed, because the dataset is not available to us. Objective 5 is designed but not run. Objective 7 is complete. We state this distribution plainly because the report would otherwise imply a completeness it does not have.

## 1.5 Contributions

The contributions of the project are modest and mostly of the kind that is useful to the next person who picks up the code.

First, a **complete, readable implementation** of a FER2013 pipeline, from CSV parsing to a live demo, with the model, the training loop and the inference path in separate modules.

Second, a **precise characterisation of the architecture**. The network is often described loosely as "a small VGG-style CNN". We give the layer table, the exact parameter count (2,818,375, of which 2,816,711 are trainable), the distribution of those parameters (59.3 % in the classifier head), the receptive field (about 40×40) and the forward cost (about 3.05×10⁸ multiply–accumulates per image). These numbers are computed from the code, and they correct an earlier draft of this project that reported roughly 1.2 million parameters.

Third, an **audit of the implementation** that identifies 21 defects, including a test-set leakage problem, a missing evaluation step, misleading training plots, and two bugs in the demo that cause it to classify the wrong image region. The audit is not a boast. It is the part of the work that changed how we think about the system, and it is documented so that the same mistakes are easier to avoid.

Fourth, a **leakage-free evaluation protocol** with pre-specified metrics, baselines, confidence intervals and significance tests. The protocol is written before the experiments, which is unusual in a student project and, we think, correct.

We do not claim a new architecture, a new loss, or state-of-the-art accuracy. The architecture is conventional by design, and the report says so.

## 1.6 Scope and delimitations

The scope is deliberately narrow. The system classifies static, greyscale, 48×48 crops into seven categories. It does not model temporal dynamics, estimate continuous valence or arousal, detect action units, or identify individuals. It is trained on one dataset, FER2013, and evaluated on that dataset's test partition. No external training data and no pretrained weights are used, so the results reflect what can be learned from FER2013 alone.

Several things therefore lie outside the scope. Cross-dataset generalisation to CK+, RAF-DB or AffectNet is not evaluated, although it would be the natural next step. Demographic fairness is not evaluated, because FER2013 does not record demographic attributes. Real-world accuracy is not measured, because there is no labelled in-the-wild video. The live demo is assessed qualitatively, not as a benchmark.

The most important delimitation is temporal. At the time of writing, the FER2013 file is not available in the project environment, and no training run has been completed. Chapter 5 is therefore written as an evaluation protocol with the results tables left open, and it states explicitly which values are pending. We have chosen this over the alternative, which would be to fill the chapter with expected numbers borrowed from the literature and present them as findings. That alternative would have been easier to write and dishonest to submit.

## 1.7 Thesis structure

Chapter 2 reviews the literature: the psychology of facial expression, pre-deep-learning methods, convolutional networks, deep FER, datasets, regularisation, evaluation and deployment. Chapter 3 sets out the methodology, including the leakage-free protocol and the architecture derived from the code. Chapter 4 documents the implementation and the defects found. Chapter 5 specifies the evaluation and reports results where they exist. Chapter 6 discusses the design, compares it with published work, and states the limitations and ethical issues. Chapter 7 concludes and sets out future work. The appendices contain the configuration, code listings, the defect register and the AI-use log.

## 1.8 Summary

Facial expression recognition is a well-studied problem with a persistent gap between benchmark and real-world performance. This project builds a from-scratch convolutional classifier for seven-class FER2013, characterises it precisely, audits its implementation and specifies a rigorous evaluation. The model is conventional; the contribution lies in the clarity of the implementation, the honesty of the audit and the discipline of the evaluation protocol. The next chapter situates the work in the literature that produced it.
