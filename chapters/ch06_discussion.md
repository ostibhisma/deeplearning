# Chapter 6 — Discussion

## 6.1 Introduction

This chapter discusses what the project shows, what it does not show, and what the gap between the two means. It is written without measured results, so it cannot interpret an accuracy figure or explain a confusion pattern. What it can do is examine the design against the literature, explain why the evaluation problem turned out to be more demanding than the modelling problem, draw out what the audit reveals about the practice of building small deep learning systems, and state the limitations and ethical issues plainly.

The discussion is organised in three parts. Sections 6.2 and 6.3 deal with the technical design and the evaluation. Sections 6.4 and 6.5 deal with what the audit and the deployment analysis revealed. Sections 6.6 to 6.8 deal with limitations, ethics and reproducibility.

## 6.2 The architecture in context

The network is a small VGG-style stack: three blocks of two 3×3 convolutions, batch normalisation after each convolution, pooling and dropout after each block, then a flattened feature vector and two dense layers. It is conventional, and that was the intention. Comparing it with the literature clarifies both what it inherits and where it departs.

The VGG design principle is that small kernels stacked deeply give a large receptive field with few parameters (Simonyan & Zisserman, 2015). Our network follows this: three 3×3 convolutions in sequence have the receptive field of a 7×7 kernel with fewer weights and more non-linearities, and by the last block the receptive field is about 40×40, close to the full 48×48 input. The progressive widening from 64 to 128 to 256 channels is also VGG's pattern, and it keeps the activation volume roughly constant as spatial resolution falls.

Where the network departs from VGG is in scale and in the classifier head. VGG-16 has 138 million parameters; ours has 2.8 million. But the distribution is inverted: 59.3 % of our parameters sit in the three dense layers, because `Flatten` produces a 6,400-dimensional vector that feeds a 256-unit layer. VGG's convolutional stack dominates its parameter budget, with a comparatively small classifier. Our network is therefore not a miniature VGG so much as a convolutional feature extractor with a large classifier bolted on. That is a structural observation with practical consequences: the head is where overfitting is most likely, and global average pooling would remove most of it at the cost of discarding spatial position.

Whether spatial position should be discarded is an open question for this task. Expressions are configurations — the relationship between brow, eye and mouth — and a global average would collapse that relationship. But the dense head does not explicitly model relationships either; it learns them from position-dependent weights, which is a blunt instrument and expensive in parameters. An intermediate design, such as a 1×1 convolution followed by global pooling, would preserve some spatial structure at a fraction of the cost. We did not implement it, and it is the first architectural experiment we would run.

The regularisation is heavy: dropout at 0.5 after every block and 0.4–0.5 in the head. On a dataset of 28,709 images with noisy labels, heavy regularisation is defensible, and the literature on small-data training supports it. But it is also a blunt response. Label smoothing (Szegedy et al., 2016) or a soft-target loss built from annotator distributions (Barsoum et al., 2016) would address the noise directly rather than by suppressing capacity. We did not implement either, and the comparison is left for future work.

One detail deserves comment because it is the kind of thing that is invisible in a paper and visible in code: the first convolution uses valid padding while every other convolution uses same padding. The docstring describes the blocks symmetrically and does not mention it. The effect is small — the first feature map is 46×46 instead of 48×48, and the flattened vector is 6,400 instead of 9,216 — but it means the network described in the documentation is not quite the network that runs. Documentation drift of this kind is common, and it is one reason the audit in Chapter 4 recomputed every shape from the layer arguments rather than trusting the comments.

## 6.3 Evaluation as a design problem

The most surprising thing about this project was not the model. It was how much of the work went into making the evaluation trustworthy, and how many defects stood between the code and a number that meant what it appeared to mean.

The first was test-set leakage. Merging the public and private partitions into a single validation set is a natural mistake if one assumes FER2013 has two partitions, as the code's own docstring does. It is also fatal to the interpretation of any result: the checkpoint, the learning-rate schedule and the stopping point all depend on the test labels. The correction — a stratified validation split from the training partition — costs 10 % of the training data and buys an unbiased estimate. That trade is not close.

The second was the absence of any evaluation. The training function fits, checkpoints and plots, but never computes test accuracy, per-class metrics or a confusion matrix, and discards the history. A project can look complete — data loader, model, training loop, demo — and still be unable to answer its own research question. This is a useful corrective to the habit of treating "the model trains" as the finish line.

The third was subtler: the training plots drew only the training curve while the legend claimed two, and the output directory did not exist, so the figure failed to save inside a caught exception. A reader who saw the resulting figure would have inferred a validation curve that was never there. Misleading artefacts are worse than missing ones, because they are trusted.

These three problems share a structure. Each is a small piece of glue code, each is invisible in a high-level description of the system, and each changes what the numbers mean. The literature on reproducibility in machine learning makes the general point — Henderson et al. (2018) on seed variance, Recht et al. (2019) on benchmark reuse — but the specific lesson here is narrower: in a small project, the evaluation code deserves as much care as the model, and it usually gets less.

Recht et al.'s finding is also relevant to how we should read published FER2013 numbers. A benchmark that has been used for a decade is not a neutral measuring instrument. Models have been selected against it, the public partition invites tuning, and the test images are easier than fresh data collected the same way. Our protocol avoids selecting on the test set, but it cannot undo the benchmark's history. When Chapter 5's numbers eventually exist, they should be read as "performance on this benchmark under this protocol", not as an estimate of real-world accuracy.

## 6.4 What the audit reveals about practice

Twenty-one defects were found. Their distribution is informative. Almost none were in the model. The architecture, the loss and the optimiser were correct and conventional. The defects clustered in three places: the data pipeline (partition handling, one-hot encoding, silent exceptions), the training loop (missing evaluation, misleading plots, missing output directory) and the demo (transposed crop, wrong resize function).

This distribution is not unique to this project. It reflects where the difficulty actually lies in applied deep learning: the model is usually the part with the most documentation, the most examples and the most community scrutiny, while the glue around it is written once, by one person, under time pressure, and never reviewed. The glue is also where a mistake is most likely to be silent. A wrong architecture produces a shape error; a transposed crop produces a plausible-looking label; a missing directory produces a printed message that is easy to ignore.

Two practices would have caught most of the defects. The first is a smoke test that runs the whole pipeline on a handful of images and asserts the shapes, the label order and the existence of the output paths. The second is a written evaluation script that computes metrics from a saved prediction array, which forces the author to check that predictions correspond to the intended inputs. Neither is expensive. Both are absent from the original code, and both are specified in Chapter 4.

The audit also changed how we read our own earlier draft. That draft reported roughly 1.2 million parameters; the computed total is 2.8 million. It described an iterative search over architectures that left no trace in the repository. It presented expected accuracies as if they were findings. None of this was deliberate deception — it is what happens when a report is written from memory and ambition rather than from artefacts. The lesson is that claims should be generated from the code and the logs, not reconstructed afterwards.

## 6.5 The deployment gap

The live demonstration is where the project meets reality, and it is where the difference between benchmark and deployment is easiest to see.

The classifier is trained on tight, roughly centred 48×48 crops. The detector returns a rectangle around the face that typically includes hair, background and sometimes shoulders. Resizing that rectangle to 48×48 makes the face smaller relative to the frame than it was in training. The classifier has never seen this distribution, and its accuracy on it is unknown. This is a domain shift produced by the pipeline itself, not by a change of scene.

There is a second shift from illumination. FER2013 images vary in lighting, but the webcam scene may be backlit, dim or colour-cast. The conversion to greyscale removes colour but not the contrast problem. Histogram equalisation before detection and classification would reduce it and is a one-line change; it is not implemented.

There is a third from geometry. The detector does not align the face. A head tilted by 20° produces a crop that the classifier may still handle, because augmentation included rotations up to 20°, but a profile view is outside the training distribution entirely. Landmark-based alignment with MTCNN or RetinaFace would address this at the cost of a heavier dependency (Zhang et al., 2016; Deng et al., 2019).

Finally, there is the temporal dimension. The model classifies single frames, so its output can flicker between adjacent classes when the input is marginal. A majority vote over a short window would stabilise the display. This is a small engineering fix, but it changes the user's perception of reliability more than a one-point accuracy improvement would.

None of these issues is measured. The analysis is structural: it follows from comparing the training crop distribution with the detector's output distribution. That is enough to say that the demo is a demonstration of the pipeline, not evidence of field performance. Saying more would require labelled video, which the project does not have.

Calibration belongs here too. The demo displays the argmax label, not the probability, which avoids the worst misreading. If probabilities were shown, Guo et al.'s (2017) finding that modern networks are poorly calibrated would matter: a softmax value of 0.99 does not mean 99 % of such predictions are correct. Temperature scaling on a validation set is cheap and would make the displayed confidence meaningful.

## 6.6 Limitations

The limitations are substantial and should be stated without hedging.

**No empirical results.** This is the first and largest. The project has a model, a protocol and an evaluation script, but no completed run. Every claim about performance is therefore absent by design. The report documents a design and an audit, not an empirical finding.

**One dataset, one split.** FER2013 is the only dataset, and the evaluation uses a single stratified split with three seeds. This does not test cross-dataset generalisation, and it does not test robustness to a different sampling of the training data. A five-fold cross-validation would give a better estimate of variability, at five times the compute.

**A small, conventional architecture.** The model is deliberately small and has no attention, no residual connections and no pretrained weights. It is a baseline, not a contender. The published reference points — 75.2 % for an ensemble (Pramerdorfer & Kampel, 2016) and 73.28 % for a single tuned network (Khaireddin & Chen, 2021) — use larger models and more extensive tuning. Any comparison must account for both differences.

**Unaddressed imbalance and label noise.** The loss is unweighted, so disgust contributes 1.5 % of the gradient. The labels are hard, so annotator disagreement is discarded. Both are known problems with known remedies — class weighting, focal loss, label smoothing, distribution training — and none is implemented. This limits what the model can learn about the minority classes, and it is the most obvious place to improve.

**No demographic analysis.** FER2013 has no demographic metadata, so subgroup performance cannot be measured. Given the evidence that commercial classifiers can fail unevenly across groups (Buolamwini & Gebru, 2018), this is a real gap, not a technicality.

**No temporal modelling.** The model classifies frames independently. Expressions unfold over time, and a video model could exploit that. The project does not.

**No latency measurements.** The demo's frame rate is unmeasured. The analysis identifies the detector and the Python loop as the likely bottlenecks, but that is an inference, not a measurement.

**A single training recipe.** The optimiser settings, augmentation policy and dropout rates are fixed by convention rather than search. The one schedule analysis in Chapter 3 suggests that the learning-rate schedule is nearly inert, which means the model is effectively trained at a constant rate. A proper schedule could change the results substantially.

## 6.7 Ethical considerations

The ethical issues fall into four groups: validity, consent, privacy and bias.

**Validity.** A system that predicts "anger" from a face is making a claim about an internal state. Barrett et al. (2019) argue that the evidence does not support such inferences from facial movement alone. The model actually predicts a label assigned by annotators, which is a different thing. Presenting the output as emotion detection would overstate it, and the report avoids that language. This is not pedantry: a system that says "this person is angry" can be used to justify actions that a system saying "this face resembles the angry category in FER2013" would not.

**Consent.** The people in FER2013 did not consent to being in a benchmark or to having their expressions classified. The dataset is licensed for research, and we use it within that licence, but the absence of consent is a real limitation of the data and it propagates to any model trained on it. A deployment that classifies the faces of people who have not agreed to it inherits the problem in a sharper form.

**Privacy.** The demonstration processes frames locally and stores nothing. No images are written to disk, transmitted, or linked to an identity. If the system were deployed more widely, the relevant obligations would include a lawful basis for processing biometric data, data minimisation, retention limits and a documented intended use. The project meets none of these, and it is not intended to be deployed.

**Bias.** FER2013 does not record demographic attributes, so the model's performance across groups is unknown. The literature suggests that this is not a neutral omission: datasets and models can encode the demographics of their collection process (Torralba & Efros, 2011), and commercial systems have shown large subgroup disparities (Buolamwini & Gebru, 2018). A model that works well on average may work poorly for a group that is under-represented in the data, and without metadata the failure is invisible. Model cards (Mitchell et al., 2019) exist precisely to make such gaps visible; this project cannot fill in the subgroup section, and says so.

**Misuse.** Facial expression recognition can be used for surveillance, for emotion policing in workplaces or classrooms, or for manipulative interfaces. The technology is dual-use, and a student project cannot control how it is used. What it can do is avoid overstating what the system knows, document its limits, and refuse to present a benchmark result as a claim about people.

The report itself is written under Roskilde University's guidelines on generative AI (Roskilde University, 2024). The guidelines permit AI assistance in project reports if the use is declared, quotations of AI-generated text are cited, and the authors remain responsible for the content. The declaration in the front matter records how the tool was used, and the verification log in Appendix E records what was checked. The group is responsible for the final text, and the declaration says so.

## 6.8 Reproducibility

The original repository is not reproducible. It has no seeds, no pinned dependencies, no logs, an empty `requirements.txt`, and a broad exception handler that hides failures. Two runs of the same script would produce different models, and a reader could not reconstruct the environment in which the original was produced.

The experimental build changes this in six ways: seeds for Python, NumPy and TensorFlow; a pinned environment; a per-run manifest with the configuration, seed and git commit; per-epoch CSV history; a saved split index; and a single prediction array from which all metrics are computed. These are not novel practices, but they are the difference between a result and an anecdote.

Residual non-determinism remains. GPU operations such as atomic additions are not associative, cuDNN may choose different algorithms depending on available memory, and parallel reductions can vary in order. Bitwise reproducibility is therefore not guaranteed even with seeds. The realistic target is statistical reproducibility: the same configuration should produce results within the seed-to-seed spread, and the spread should be reported. Three seeds give a rough estimate of that spread, not a precise one.

## 6.9 Summary

The architecture is a conventional VGG-style baseline whose parameters are concentrated in a large classifier head, and its most interesting structural feature is how much of it sits in the dense layers. The evaluation turned out to be the harder design problem: leakage, a missing evaluation step and misleading plots each changed what a number would have meant. The audit showed that defects cluster in the glue rather than the model, which is where they are hardest to see. The deployment analysis identified three shifts — crop framing, illumination and geometry — that separate the demo from the benchmark, none of which is measured.

The limitations are real: no results, one dataset, a small model, unaddressed imbalance and no demographic analysis. The ethical issues are equally real: the labels are annotations rather than emotions, the images were collected without consent, and the model's behaviour across groups is unknown. These are not reasons to abandon the work. They are the reasons the work is described as it is, and they set the agenda for Chapter 7.
