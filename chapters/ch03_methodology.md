# Chapter 3 — Methodology

## 3.1 Introduction

This chapter explains how the emotion recognition system was designed, how it is trained, and how it is evaluated. It is written before any experiment has been run, and that order is intentional. Fixing the protocol in advance removes the temptation to choose the analysis after seeing the numbers, which is a persistent problem in student projects and, more quietly, in published work. Where the code that exists in the repository departs from sound experimental practice, the departure is named here and the protocol adopted for the experiments is described separately. The implementation itself, including the defects found and the fixes applied, is documented in Chapter 4.

The chapter is organised around the pipeline. Section 3.2 gives the overall research design and the system architecture. Section 3.3 describes the dataset and the evaluation splits, including a leakage problem in the shipped code that had to be resolved before any result could be trusted. Section 3.4 covers preprocessing. Section 3.5 derives the network layer by layer, with tensor shapes, parameter counts and computational cost computed from the code rather than estimated. Section 3.6 describes the training procedure and analyses what the callback configuration actually does. Section 3.7 sets out the evaluation metrics and the statistical treatment. Section 3.8 describes the live inference pipeline, and Section 3.9 discusses ethics and data protection. Section 3.10 summarises.

## 3.2 Research design and system overview

The work follows a design-science approach common in applied computer science. We build an artefact, a complete pipeline from raw data to a working demonstration, and evaluate it quantitatively against a held-out benchmark and against trivial baselines. The unit of analysis is the trained classifier: its accuracy, its per-class behaviour, and the gap between its performance on benchmark crops and on faces detected in a live video stream. There are no human participants and no user study, so the design is a controlled computational experiment rather than an empirical study of people.

The pipeline has six stages, shown in Figure 3.1. Raw data is read from a single CSV file and split according to the partition labels it carries. Each row is parsed into a 48×48 greyscale image and a one-hot label. Images are rescaled to the unit interval and passed through an augmentation pipeline during training only. The classifier is a six-layer convolutional network with three pooling stages and a classifier head of two hidden dense layers plus a seven-way output. Training is controlled by three callbacks that adapt the learning rate, stop early, and keep the best checkpoint. Evaluation reports accuracy, per-class precision and recall, macro-averaged F1, and a confusion matrix. The final stage is a webcam demonstration that couples the trained classifier to a Haar cascade face detector.

```
fer2013.csv
   │  read + split by Usage
   ▼
┌─────────────────┐    ┌──────────────────┐
│ training split  │    │ validation split │
│ (stratified)    │    │ (stratified)     │
└────────┬────────┘    └────────┬─────────┘
         │ augmentation          │ rescale only
         ▼                       ▼
   ┌───────────────────────────────────┐
   │  CNN: 3 conv blocks + 2 dense     │
   └───────────────┬───────────────────┘
                   │ softmax (7)
                   ▼
        checkpoint (best val_loss)
                   │
                   ▼
   evaluation ──► metrics + confusion matrix
                   │
                   ▼
   webcam: Haar detect → crop → 48×48 → predict → overlay
```

*Figure 3.1. End-to-end system architecture. The vertical flow from the saved checkpoint to evaluation marks the point at which the official test partition is opened exactly once.*

The design separates concerns in a way that suits a project of this size. `data_loading/data.py` owns parsing and encoding, `training/model.py` owns architecture and fitting, `main.py` wires the two together, and `app.py` owns inference. The separation is not elaborate, but it means the model can be rebuilt without re-parsing the CSV and the demo can be changed without touching training. The main constraint throughout is compactness: 48×48 inputs, seven outputs, and a model small enough to train on a laptop CPU. That constraint is a deliberate response to the available hardware, not a claim that small models are preferable.

Three design principles guided the choices. First, comparability: the model is trained from scratch on FER2013 alone, with no external data and no pretrained weights, so that its performance can be attributed to the architecture and the training recipe rather than to a large upstream corpus. Second, transparency: every architectural number in this chapter is computed from the layer definitions and can be reproduced by reading the source. Third, honesty about scope: the system targets seven-way classification of aligned, low-resolution greyscale crops. It does not attempt continuous affect estimation, temporal modelling, or person identification, and it should not be presented as if it did.

## 3.3 Data

### 3.3.1 Composition of FER2013

FER2013 was released for the ICML 2013 Challenges in Representation Learning and distributed through Kaggle (Goodfellow et al., 2013). It contains 35,887 greyscale images of faces at 48×48 pixels, each labelled with one of seven expressions: anger, disgust, fear, happiness, sadness, surprise and neutrality. The images were gathered from web image search and annotated by human raters, which makes the dataset a record of how people label photographs rather than a record of measured internal states. That distinction matters for interpretation and is taken up again in Chapters 6 and 7.

The file has three columns. `emotion` holds an integer from 0 to 6 in the order just listed. `pixels` holds 2,304 space-separated integers in row-major order, representing the 48×48 image. `Usage` holds one of `Training`, `PublicTest` or `PrivateTest`. Table 3.1 gives the class distribution.

**Table 3.1. FER2013 partitions and class distribution.**

| Emotion | Index | Training | Test (Public + Private) | Share of training |
|---|---:|---:|---:|---:|
| Angry | 0 | 3,995 | 958 | 13.9 % |
| Disgust | 1 | 436 | 111 | 1.5 % |
| Fear | 2 | 4,097 | 1,024 | 14.3 % |
| Happy | 3 | 7,215 | 1,774 | 25.1 % |
| Sad | 4 | 4,830 | 1,247 | 16.8 % |
| Surprise | 5 | 3,171 | 831 | 11.0 % |
| Neutral | 6 | 4,965 | 1,233 | 17.3 % |
| **Total** | | **28,709** | **7,178** | **100 %** |

The two test partitions are 3,589 images each, giving 7,178 in total. These counts are the published composition of the dataset; they are not measurements we made, because the dataset is not currently available to us. The percentages are arithmetic on those counts.

### 3.3.2 Properties that shape the modelling problem

Four properties of FER2013 influence every decision that follows.

**Class imbalance.** Happiness is 16.5 times more frequent in training than disgust. Disgust accounts for 1.5 % of the training set, so a model that never predicts disgust at all loses very little training loss. Accuracy is therefore a misleading summary on its own, and macro-averaged F1 is needed to show what happens to the rare classes (Sokolova & Lapalme, 2009). Nothing in the current training code compensates for the imbalance: there is no class weighting and no resampling. The expected consequence is a strong prior toward the majority classes, which the evaluation must measure rather than assume.

**Label noise.** The labels come from human annotation, and annotators disagree, particularly on fear, disgust and neutrality. Barsoum et al. (2016) made this explicit when they relabelled a subset of FER2013 with ten annotators per image and showed that training on the full label distribution outperforms training on majority-vote labels. FER2013 as distributed provides only the majority vote, so some fraction of the training labels is wrong in the sense that most annotators would not agree with them. We cannot estimate that fraction from the data available to us, and we do not quote a figure for it. The practical implication is a ceiling on achievable accuracy and a reason to prefer a regularised model over one that fits the training set closely.

**Low resolution and pose variation.** At 48×48 pixels, the eyes occupy a handful of pixels and fine wrinkles are absent. Images are not aligned, so the same expression appears at different scales, rotations and translations. This is why geometric augmentation is appropriate, and also why a face detector that returns a loose crop creates a domain shift at inference time: the classifier is trained on crops that are roughly centred on the face, while the detector returns a rectangle that may include hair, background and shoulders.

**No demographic metadata.** FER2013 does not record age, gender or ethnicity, so subgroup performance cannot be measured. This is a limitation of the benchmark rather than of our code, but it constrains what can be claimed about fairness. Model reporting frameworks such as Mitchell et al. (2019) treat subgroup evaluation as a basic requirement, and we can only note its absence.

### 3.3.3 The partition problem in the shipped code

The `DataLoader` routes rows by comparing `Usage` to the string `Training` (`data.py:34`). Rows that match go to the training arrays; every other row goes to the test arrays. Because `PublicTest` and `PrivateTest` both fail that comparison, the two test partitions are merged into one 7,178-image set. That set is then passed to `model.fit` as `validation_data` (`model.py:104`), and all three callbacks monitor validation loss on it.

The consequence is leakage. `ModelCheckpoint` keeps the epoch with the lowest loss on the official test images, `ReduceLROnPlateau` adjusts the learning rate in response to them, and `EarlyStopping` decides when to stop by looking at them. The final weights have therefore been selected using the labels of the data that later provide the reported accuracy. The resulting number would be optimistic, and the size of the optimism would be unknown. This is the single most important methodological defect in the project, and no result produced by the code as shipped should be reported as a test result.

### 3.3.4 The leakage-free protocol adopted here

The protocol used for all experiments reported in Chapter 5 is as follows.

1. Only rows with `Usage == 'Training'` are used for fitting and model selection.
2. Those 28,709 rows are split into a training set (90 %) and a validation set (10 %), stratified by class. That gives approximately 25,838 training and 2,871 validation images. The split is made once with a fixed seed and reused across configurations so that comparisons are paired.
3. The validation set controls checkpoint selection, learning-rate reduction and early stopping. It is never used to report final performance.
4. The official test partition (PublicTest and PrivateTest together, 7,178 images) is evaluated once per trained model, after training has finished. Its labels play no part in any decision.
5. Each configuration is trained with three seeds (42, 123 and 2026), and results are reported as mean ± standard deviation across seeds.
6. If a comparison with a published number requires the public test partition specifically, that number is reported in a separate column and labelled accordingly.

Two trade-offs are worth stating. First, holding out 10 % of the training partition removes about 2,871 images that the shipped pipeline would have trained on, so our model sees slightly less data than the original code intended. That is the price of an unbiased validation signal, and it is the correct price to pay. Second, a stratified 10 % split leaves only about 44 disgust images in validation, so validation loss for that class is noisy and early stopping is effectively driven by the majority classes. We accept this because the alternative, using test data for validation, is worse; the per-class test results remain the place where disgust is properly assessed.

## 3.4 Preprocessing

### 3.4.1 Parsing and encoding

Each row's `pixels` string is split on the space character, producing 2,304 tokens, which are converted to a `float32` array. The values are the raw greyscale intensities in the range 0–255. Labels are converted to one-hot vectors of length seven with `to_categorical`, and the image arrays are reshaped from `(N, 2304)` to `(N, 48, 48, 1)`.

The shipped loader builds Python lists row by row and converts them to arrays at the end (`data.py:32–47`). This is correct but inefficient: `DataFrame.iterrows` creates a Series per row, and the list of 28,709 arrays is copied again when `np.array` is called. The result is a parsing stage that takes minutes and a peak memory footprint of roughly 265 MB for the training tensor alone, before the copy. A vectorised implementation would split the column once and stack the result, and caching the arrays to `.npy` would make repeated ablation runs much cheaper. We describe the shipped approach here because it is what the project does; the optimised version is listed as future work.

Two details in the encoding are easy to get wrong and are worth recording. First, `to_categorical` is called without an explicit `num_classes`, so a split that happens to contain no examples of a class would silently produce six columns instead of seven. With the full dataset this cannot happen, and with a stratified 10 % validation split every class is represented because the smallest class has 436 members, so it does not arise here. Adding `num_classes=7` is still the correct defensive choice. Second, the label order must match the order used at inference. The `emotions` list in `app.py:7` is `['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']`, which matches the FER2013 integer encoding. If the two orders had diverged, every prediction in the demo would have been displayed under the wrong name, and the error would be invisible without ground-truth labels. We verified the correspondence explicitly.

### 3.4.2 Intensity scaling

The loader leaves pixel values in 0–255. Scaling happens inside the data generators, which apply `rescale=1./255` to both the training and the validation streams (`model.py:36–37`). This gives inputs in [0, 1], which is the range the network is trained on and the range used at inference (`app.py:20`).

We did not apply dataset-level mean and standard deviation normalisation. For a greyscale face dataset the difference is usually small, and batch normalisation after the first convolution absorbs much of it, but per-dataset standardisation is the more conventional choice and its absence is a reasonable simplification to note rather than defend. The important property is consistency: the same scaling must be applied at training, validation and inference. Because the generators hold the scaling and the demo hard-codes the same divisor, a future change to one without the other would silently degrade accuracy. That coupling is a design weakness, and Chapter 4 records it.

### 3.4.3 Shape and channel conventions

The model declares `input_shape=(48, 48, 1)` with `data_format='channels_last'` (`model.py:41`), and the tensors are shaped `(N, 48, 48, 1)`. This is the TensorFlow default and requires no transposition. The single channel is explicit rather than squeezed, because the convolution expects a channel dimension; feeding `(N, 48, 48)` would raise a rank error. At inference the demo expands a single image to `(1, 48, 48, 1)` before prediction (`app.py:21`).

### 3.4.4 Batching and shuffling

Training batches are drawn with `ImageDataGenerator.flow`, which shuffles by default and yields batches of 64. With 25,838 training images under the protocol, an epoch is 404 steps; under the shipped split of 28,709 it is 449 steps. The validation stream is also a generator, and its batch size defaults to 32 because none is specified. Batch size affects the validation loss only through the averaging of per-batch losses, so the inconsistency is cosmetic, but it is the kind of detail that makes exact reproduction harder than it needs to be.

## 3.5 Architecture

### 3.5.1 Design rationale

The classifier is a VGG-style convolutional network: small 3×3 kernels stacked in pairs, each pair followed by pooling, with the channel count doubling at each stage (Simonyan & Zisserman, 2015). Three such blocks are used, followed by a flatten operation and two fully connected layers. The choice is deliberately conservative. FER2013 is small, noisy and low-resolution, and the published evidence suggests that deeper and wider architectures do not automatically help at this scale (Pramerdorfer & Kampel, 2016). A compact stack is easier to train on limited hardware, easier to reason about, and a fair from-scratch baseline.

Several alternatives were considered and rejected for this iteration. Residual connections (He et al., 2016) would ease optimisation of a deeper stack, but at six convolutional layers the degradation problem they address is not yet severe. Inception-style multi-branch modules (Szegedy et al., 2015) increase capacity per layer at the cost of implementation complexity that a project of this size cannot justify. Transfer learning from a large face or object corpus would almost certainly raise accuracy, but it would confound the question we are asking, which is what a from-scratch model can do on this dataset. Mobile-style depthwise separable convolutions (Howard et al., 2017) would reduce cost, but the model is already small enough for CPU inference. These are recorded as future directions in Chapter 7 rather than as omissions.

Two architectural choices deserve explicit justification because they are the most consequential. The network uses `Flatten` followed by `Dense(256)` rather than global average pooling. Flattening preserves spatial position, which plausibly matters for expressions, since a mouth corner in one place means something different from the same texture elsewhere. The cost is large: the flattened vector has 6,400 elements, and the first dense layer alone holds 1.64 million parameters. A global-average-pooling variant would reduce the head to about 2,000 parameters and would be a strong regularisation baseline; we did not implement it, and it is the first ablation we would add if compute allowed. The second choice is dropout at 0.5 after every pooling layer. This is aggressive. It is defensible given the label noise and the size of the training set, but it may also suppress learning, and without an ablation we cannot say which effect dominates.

### 3.5.2 Layer specification

Table 3.2 lists every layer with its output shape and parameter count. The counts are computed from the layer arguments using the standard formulas for convolution and batch normalisation, not read from a model summary.

**Table 3.2. Layer-by-layer specification of the implemented network.**

| # | Layer | Configuration | Output shape | Parameters |
|---|---|---|---:|---:|
| 1 | Conv2D | 64 filters, 3×3, stride 1, valid padding, ReLU | 46×46×64 | 640 |
| 2 | Conv2D | 64 filters, 3×3, stride 1, same padding, ReLU | 46×46×64 | 36,928 |
| 3 | BatchNormalization | default ε, momentum | 46×46×64 | 128 |
| 4 | MaxPooling2D | 2×2, stride 2 | 23×23×64 | 0 |
| 5 | Dropout | rate 0.5 | 23×23×64 | 0 |
| 6 | Conv2D | 128 filters, 3×3, same, ReLU | 23×23×128 | 73,856 |
| 7 | BatchNormalization | default | 23×23×128 | 256 |
| 8 | Conv2D | 128 filters, 3×3, same, ReLU | 23×23×128 | 147,584 |
| 9 | BatchNormalization | default | 23×23×128 | 256 |
| 10 | MaxPooling2D | 2×2, stride 2 | 11×11×128 | 0 |
| 11 | Dropout | rate 0.5 | 11×11×128 | 0 |
| 12 | Conv2D | 256 filters, 3×3, same, ReLU | 11×11×256 | 295,168 |
| 13 | BatchNormalization | default | 11×11×256 | 512 |
| 14 | Conv2D | 256 filters, 3×3, same, ReLU | 11×11×256 | 590,080 |
| 15 | BatchNormalization | default | 11×11×256 | 512 |
| 16 | MaxPooling2D | 2×2, stride 2 | 5×5×256 | 0 |
| 17 | Dropout | rate 0.5 | 5×5×256 | 0 |
| 18 | Flatten | — | 6,400 | 0 |
| 19 | Dense | 256 units, ReLU | 256 | 1,638,656 |
| 20 | Dropout | rate 0.4 | 256 | 0 |
| 21 | Dense | 128 units, ReLU | 128 | 32,896 |
| 22 | Dropout | rate 0.5 | 128 | 0 |
| 23 | Dense | 7 units, softmax | 7 | 903 |
| | **Total** | | | **2,818,375** |

Of the total, 2,816,711 parameters are trainable and 1,664 are the moving mean and variance buffers of the five batch-normalisation layers.

### 3.5.3 Shape arithmetic

For a convolution with input size *I*, kernel size *K*, padding *P* and stride *S*, the output size is

  O = ⌊(I − K + 2P) / S⌋ + 1.

The first layer applies a 3×3 kernel to a 48×48 input with no padding, giving (48 − 3)/1 + 1 = 46. Every later convolution uses `same` padding with stride 1, so the spatial size is unchanged and only pooling reduces it. The sequence is 48 → 46 → 23 → 11 → 5, with each pooling step halving and rounding down (23/2 → 11, 11/2 → 5).

The asymmetry at layer 1 is not documented in the code's docstring, which describes the blocks symmetrically. It is a small thing, but it changes the flattened dimension: had the first convolution used `same` padding, the final feature map would have been 6×6×256 = 9,216 rather than 6,400, and the first dense layer would have been 44 % larger. We report the network as it is, not as the docstring implies.

### 3.5.4 Parameter budget

The parameter count follows from the standard formulas. A convolution with *C*<sub>in</sub> input channels, *C*<sub>out</sub> output channels and a *k*×*k* kernel has (*k*²·*C*<sub>in</sub> + 1)·*C*<sub>out</sub> parameters, the +1 being the bias. A batch-normalisation layer has 2·*C* trainable parameters (scale and offset) and 2·*C* non-trainable buffers. A dense layer has (*n*<sub>in</sub> + 1)·*n*<sub>out</sub>.

The convolutional stack accounts for 1,144,256 parameters and the normalisation layers for 1,664, leaving 1,672,455 in the classifier head. That is 59.3 % of the model in three dense layers, with the single connection between the flattened features and the first hidden layer responsible for 1,638,656 of them. This distribution is the clearest structural fact about the network: it is not really a convolutional model with a small classifier, but a convolutional feature extractor feeding a large fully connected classifier. Whether that is the right allocation for a 48×48 input is an empirical question, and the global-average-pooling comparison is the obvious way to answer it.

### 3.5.5 Receptive field and computational cost

The receptive field grows as layers are added. Let *r* be the receptive field and *j* the cumulative stride, starting at 1. A 3×3 convolution with stride 1 adds (3 − 1)·*j* = 2 to *r*. A 2×2 pooling layer with stride 2 doubles *j* and adds 1·*j*. Following the stack, the receptive field reaches approximately 40×40 pixels by the last convolution, against a 48×48 input. The deepest features can therefore integrate information from almost the whole face, which is appropriate for expressions that depend on the configuration of brows, eyes and mouth rather than on a single local texture.

Cost is estimated by counting multiply–accumulate operations. For a convolution, MACs = *H*<sub>out</sub>·*W*<sub>out</sub>·*C*<sub>out</sub>·*k*²·*C*<sub>in</sub>; for a dense layer, MACs = *n*<sub>in</sub>·*n*<sub>out</sub>. Summing over the network gives 304,945,536 MACs per image, or about 3.05×10⁸, which corresponds to roughly 0.61 GFLOPs for a forward pass. The figure excludes batch normalisation, activation functions and the softmax, so it is a lower bound. It is small enough that CPU inference is practical and CPU training is feasible, if slow.

### 3.5.6 Normalisation and regularisation

Batch normalisation (Ioffe & Szegedy, 2015) standardises each channel across the batch and then rescales:

  μ<sub>B</sub> = (1/m) Σ x<sub>i</sub>,  σ²<sub>B</sub> = (1/m) Σ (x<sub>i</sub> − μ<sub>B</sub>)²,
  x̂<sub>i</sub> = (x<sub>i</sub> − μ<sub>B</sub>) / √(σ²<sub>B</sub> + ε),
  y<sub>i</sub> = γ x̂<sub>i</sub> + β.

During training, μ<sub>B</sub> and σ²<sub>B</sub> are computed from the current mini-batch; during inference, the moving averages accumulated during training are used. Keras defaults are ε = 10⁻³ and momentum 0.99. The practical effect here is to stabilise the scale of activations after each convolution, which matters because the loss surface of a six-layer network with ReLU units and a large dense head is otherwise sensitive to initialisation. It also introduces a dependence on batch composition: with batches of 64 the statistics are reasonably stable, but the last batch of an epoch, which may be smaller, produces a slightly different normalisation than the rest.

Dropout (Srivastava et al., 2014) randomly sets a fraction *p* of activations to zero during training. Keras uses inverted dropout: surviving activations are scaled by 1/(1 − *p*) during training, so no rescaling is needed at inference, where dropout is the identity. Dropout at 0.5 after pooling means that half of the feature maps reaching the next block are masked on each forward pass. Combined with the 0.4 and 0.5 rates in the head, the network is heavily regularised. This is a deliberate response to label noise and a small dataset, but it also slows convergence, and the training curves in Chapter 5 will show whether it is too strong.

### 3.5.7 Output layer and objective

The final layer has seven units with a softmax activation, so the output is a probability distribution over the seven classes:

  p<sub>c</sub> = exp(z<sub>c</sub>) / Σ<sub>j</sub> exp(z<sub>j</sub>).

The loss is categorical cross-entropy between the one-hot target *y* and the predicted distribution:

  L = − Σ<sub>c=1</sub><sup>7</sup> y<sub>c</sub> log p<sub>c</sub>.

With a one-hot target this reduces to −log p<sub>ŷ</sub>, the negative log-likelihood of the correct class. The gradient with respect to the logits is p<sub>c</sub> − y<sub>c</sub>, which is clean and well behaved.

The loss is unweighted. Each training example contributes equally, so the expected gradient contribution of a class is proportional to its frequency. Disgust, at 1.5 % of the training set, contributes about one sixty-sixth as much as happiness. A class-weighted loss or focal loss (Lin et al., 2017) would change that, and both are discussed in Chapter 6, but the experiments reported here use the unweighted objective because that is what the code implements and because changing it would confound the comparison with the original pipeline.

## 3.6 Training procedure

### 3.6.1 Optimiser

Training uses Adam (Kingma & Ba, 2015) with Keras defaults: learning rate 10⁻³, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁷. Adam maintains first and second moment estimates,

  m<sub>t</sub> = β₁ m<sub>t−1</sub> + (1 − β₁) g<sub>t</sub>,
  v<sub>t</sub> = β₂ v<sub>t−1</sub> + (1 − β₂) g<sub>t</sub>²,

corrects them for initialisation bias,

  m̂<sub>t</sub> = m<sub>t</sub> / (1 − β₁<sup>t</sup>),  v̂<sub>t</sub> = v<sub>t</sub> / (1 − β₂<sup>t</sup>),

and updates the parameters as

  θ<sub>t</sub> = θ<sub>t−1</sub> − α m̂<sub>t</sub> / (√v̂<sub>t</sub> + ε).

Adam was chosen because it is robust to the scale of gradients and requires little tuning, which matters when the available time for hyperparameter search is small. No weight decay is applied, and there is no learning-rate warm-up. The default ε is used.

### 3.6.2 Batch size and epoch budget

The batch size is 64. Under the leakage-free protocol this gives 404 optimisation steps per epoch; the shipped code, training on all 28,709 images, gives 449. The maximum is 100 epochs, but training almost never reaches it, for reasons analysed below. Batch size 64 is a compromise between gradient noise and memory: smaller batches give noisier but more frequent updates, and 64 fits comfortably in memory on both CPU and a modest GPU.

### 3.6.3 Augmentation

Training images pass through an `ImageDataGenerator` configured with rotation up to 20°, horizontal flipping, zoom up to 20 %, shear of 0.2, and rescaling by 1/255 (`model.py:36`). Validation images are only rescaled.

One detail deserves care because it is widely misreported: in Keras, `rotation_range` is in degrees but `shear_range` is in radians. A shear of 0.2 is therefore about 11.5°, not 0.2°. It is a moderate geometric distortion, not a negligible one.

The augmentation is chosen to reflect the transformations that preserve an expression label. A face rotated by 20° is still happy; a horizontally mirrored face is still happy; a slightly zoomed face is still happy. Shear is more questionable, because it changes the geometry of the mouth and eyes in a way that no real camera does, and with the default `fill_mode='nearest'` the borders are replicated, producing streaks that the network could learn to associate with particular classes. There is no photometric augmentation: brightness, contrast and gamma are untouched, even though illumination variation is one of the documented difficulties of in-the-wild FER. This is a gap in the design, and adding mild brightness and contrast jitter is a natural extension.

The augmentation is applied only during training. Validation and test images are rescaled but not transformed, so the evaluation measures performance on undistorted inputs, which is the correct choice for a benchmark comparison.

### 3.6.4 Callbacks and what they actually do

Three callbacks are registered (`model.py:94–100`):

- `ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3)`, which multiplies the learning rate by 0.8 after three epochs without improvement;
- `EarlyStopping(monitor='val_loss', min_delta=0, patience=6, mode='auto')`, which stops training after six epochs without improvement;
- `ModelCheckpoint('training/emotion_model.h5', monitor='val_loss', save_best_only='True')`, which writes the best model seen so far; the argument is a string in the shipped code and is discussed below.

The interaction between the first two is worth working through, because it determines the effective training budget. Suppose the best validation loss occurs at epoch *k*. The learning-rate reducer waits three epochs, so it acts at *k* + 3. Early stopping waits six, so it acts at *k* + 6. Between those points there are three more epochs, enough for a second reduction only if the first did not help; but at *k* + 6 training stops regardless. In the common case, therefore, the model trains at 10⁻³ for most of its life, drops once to 8×10⁻⁴, and stops. The schedule is close to inert. A factor of 0.8 is also unusually gentle; 0.1 to 0.5 is the usual range. If the intention was to let the model settle into a minimum, the early-stopping patience needed to be larger than the reducer's patience plus a few epochs. This is a design flaw, not a bug, and it is recorded as D6 in Chapter 4.

`ModelCheckpoint` receives `save_best_only='True'`, a string rather than a boolean. It is truthy, so the behaviour is correct, but it is a type error and is fixed in the revised code. Because the checkpoint stores the best weights, the fact that `EarlyStopping` does not set `restore_best_weights=True` does not affect the saved model; it would matter only if the in-memory model were used after training, which the shipped script does not do.

### 3.6.5 Reproducibility controls

The shipped code sets no random seed. Python's `random`, NumPy and TensorFlow all draw from sources that vary between runs, and the augmentation pipeline is stochastic, so two runs of the same script produce different weights. For the experiments in this thesis, every run is seeded at the start (Python, NumPy, TensorFlow and the generator), the seed is recorded in a run manifest alongside the configuration, and results are reported across three seeds. Exact determinism is not guaranteed even then: some GPU operations are non-deterministic by design, and cuDNN may select different algorithms depending on available memory. We therefore aim for statistical reproducibility across seeds rather than bitwise reproducibility, and we report variance rather than a single number.

## 3.7 Evaluation protocol

### 3.7.1 Metrics

Accuracy is the proportion of correctly classified images. It is reported because it is the standard summary and because published FER2013 results are usually given in accuracy, which makes comparison possible. It is not sufficient on its own.

For each class *c*, precision is TP<sub>c</sub> / (TP<sub>c</sub> + FP<sub>c</sub>) and recall is TP<sub>c</sub> / (TP<sub>c</sub> + FN<sub>c</sub>), with F1<sub>c</sub> their harmonic mean. Macro-averaged F1 is the unweighted mean of the seven per-class F1 scores:

  F1<sub>macro</sub> = (1/7) Σ<sub>c</sub> F1<sub>c</sub>.

Because every class counts equally, macro-F1 punishes a model that ignores disgust even if its accuracy is high. The gap between accuracy and macro-F1 is itself informative: a large gap indicates that performance is concentrated in the majority classes.

The confusion matrix *C*, where *C*<sub>ij</sub> counts images of true class *i* predicted as class *j*, is reported in full. It shows not only which classes are weak but which pairs are confused, and the pattern is interpretable: fear and surprise share widened eyes, sadness and neutral share a still face, and disgust is often absorbed into anger. Those patterns are discussed in Chapter 5 when the matrix exists; we do not assume them in advance.

### 3.7.2 Uncertainty and significance

A single accuracy figure on 7,178 images carries sampling error. We compute 95 % confidence intervals by bootstrapping (Efron, 1979): resample the test set with replacement 1,000 times, compute accuracy on each resample, and take the 2.5th and 97.5th percentiles. The same procedure is applied to macro-F1.

For paired comparisons between two models evaluated on the same test images, we use McNemar's test (McNemar, 1947) on the 2×2 table of agreements and disagreements. With *b* images the first model gets right and the second gets wrong, and *c* the reverse, the statistic is

  χ² = (|b − c| − 1)² / (b + c),

with one degree of freedom and a continuity correction. This is the appropriate test when the same test set is used for both models, which is exactly our situation (Dietterich, 1998). A paired bootstrap of the accuracy difference is reported alongside it as a check, since the two can disagree when the number of discordant pairs is small.

One caution applies throughout. The test set has 7,178 images, so one percentage point is about 72 images. Differences of one or two points between models are within the range that sampling noise can produce, and a difference that is statistically significant is not necessarily practically meaningful. We report effect sizes with intervals rather than treating a *p*-value as a verdict.

### 3.7.3 Baselines

Three baselines bound the interpretation. A uniform random classifier achieves 1/7 ≈ 14.3 % accuracy. A majority-class classifier that always predicts happiness achieves 1,774/7,178 ≈ 24.7 %, computed from the published test counts. A trivial model therefore sets a floor that any trained network must clear by a wide margin to be worth discussing. We also report Cohen's κ as a chance-corrected agreement measure, because accuracy alone flatters models on imbalanced data.

A stronger baseline, a small two-layer CNN or a logistic regression on raw pixels, would help separate the contribution of depth from the contribution of simply fitting a decision boundary. Neither is implemented in the current code. They are listed in Chapter 7 as the next experiments rather than reported as if they had been run.

### 3.7.4 Ablation design

RQ2 asks what BatchNorm, dropout and augmentation contribute. The design is a set of controlled removals from the full model, each changing one component and leaving everything else, including the seed set, identical:

- **Full model**: as specified in Table 3.2.
- **No augmentation**: generators with rescaling only.
- **No dropout**: all dropout rates set to zero.
- **No batch normalisation**: BN layers removed.

Four configurations times three seeds is twelve runs. On a CPU this is a substantial amount of computation, and if time is short the priority order is the full model first, then no augmentation (the component most likely to matter on a small dataset), then no dropout, then no batch norm. Each configuration is evaluated once on the test set, and the paired comparisons use McNemar's test as described above.

The hypotheses are stated in advance so that they can be falsified. We expect augmentation to reduce the training–validation gap and to improve test accuracy modestly. We expect removing dropout to increase training accuracy and decrease test accuracy. We expect removing batch normalisation to slow convergence and produce the largest seed-to-seed variance. If the results contradict these expectations, that is a finding, not a failure, and it will be reported as such.

### 3.7.5 What will not be claimed

The evaluation cannot support claims about state-of-the-art performance, because it uses a single architecture, a single dataset and no external training data. It cannot support claims about demographic fairness, because FER2013 has no demographic metadata. It cannot support claims about real-world accuracy, because the test set consists of cropped benchmark images and the demo operates on detector crops that differ in framing and scale. It cannot support claims about temporal behaviour, because the model classifies single frames. These limits are stated here so that Chapter 5 does not have to walk them back.

## 3.8 Inference pipeline design

The demonstration captures frames from the default camera, converts each to greyscale, and runs a Haar cascade classifier (`app.py:5, 15–16`). Haar cascades use boosted rectangular features and a cascade of stages to reject background windows quickly (Viola & Jones, 2001). The detector is configured with `scaleFactor=1.3` and `minNeighbors=3`, with no minimum or maximum face size.

For each detected rectangle `(x, y, w, h)`, the design calls for cropping the face, resizing it to 48×48, scaling to [0, 1], expanding to a batch of one, and taking the argmax of the softmax output. The rectangle is drawn on the frame and the predicted label is written above it.

Two implementation details are part of the design rather than incidental. The crop must index rows by *y* and columns by *x*, that is `image[y:y+h, x:x+w]`; getting this backwards silently classifies a different part of the frame while drawing the box in the correct place. The resize must be a spatial resampling operation, `cv2.resize` with area interpolation for downscaling, not `np.resize`, which repeats or truncates the flattened array. Both are corrected in the revised code and discussed in Chapter 4.

Three further design decisions concern the live setting. First, the detector returns a rectangle that is usually looser than the tight crops in FER2013, so the classifier sees more background and hair than it did in training. Face alignment using landmarks, as in MTCNN or RetinaFace (Zhang et al., 2016; Deng et al., 2019), would reduce this mismatch but adds a dependency and latency. Second, single-frame predictions flicker. A majority vote over a short window of recent frames, or an exponential moving average of the class probabilities, would stabilise the display at the cost of a few frames of lag. Third, inference latency is dominated by the detector rather than the classifier, since the network costs about 0.61 GFLOPs per face. No latency measurements have been taken, so we make no frames-per-second claim.

## 3.9 Ethics and data protection

Three issues need to be stated plainly.

The first is consent and provenance. FER2013 consists of photographs of people collected from the web, and the people in them did not consent to being in a benchmark or to having their expressions classified. The dataset is widely used for research, and its licence permits this use, but the absence of consent is a real limitation and not a technicality. Any deployment that classifies faces of people who have not agreed to it inherits that problem in a sharper form.

The second is validity. The seven categories come from a particular tradition in psychology, and that tradition is contested. Russell (1994) questioned the cross-cultural evidence for universal recognition, Jack et al. (2012) showed that expressive signals vary by culture, and Barrett et al. (2019) argued that facial movements do not map one-to-one onto emotional states. A classifier trained on FER2013 therefore learns to reproduce the labelling conventions of its annotators. It does not read internal emotion, and presenting its output as an emotion reading would overstate what the model does. The report uses "predicted label" rather than "detected emotion" wherever precision matters.

The third is privacy in the demonstration. The webcam pipeline processes frames locally, displays them, and stores nothing. No frames are written to disk, sent to a server, or used to identify anyone. If the system were deployed more widely, the relevant obligations would include a lawful basis for processing biometric data, data minimisation, retention limits, and a documented intended use. Model reporting frameworks (Mitchell et al., 2019) and the broader AI ethics literature (Jobin et al., 2019) provide structures for that documentation; the project does not meet them, and Chapter 6 discusses the gap.

The report itself is subject to Roskilde University's guidelines on generative AI in written exams (Roskilde University, 2024), which permit AI assistance in project reports provided its use is declared, quotations of AI-generated text are cited, and the authors remain responsible for verifying content. The declaration in the front matter records how the tool was used.

## 3.10 Summary

This chapter set out the method. FER2013 provides 28,709 training and 7,178 test images across seven classes, with a 16.5:1 imbalance between happiness and disgust and labels that are noisy by construction. The shipped code merges the two test partitions and uses them as validation, which leaks test information into model selection; the protocol adopted here instead holds out a stratified 10 % of the training partition for validation and touches the test set once. Preprocessing parses the pixel strings, one-hot encodes the labels, reshapes to 48×48×1, and scales to [0, 1] inside the generators. The classifier is a three-block VGG-style network with 2,818,375 parameters, 59.3 % of them in the classifier head, a receptive field of about 40×40, and a forward cost of about 3.05×10⁸ MACs. It is trained with Adam, unweighted categorical cross-entropy, batch size 64, geometric augmentation, and three callbacks whose interaction limits the effective schedule to roughly one learning-rate reduction. Evaluation reports accuracy, per-class precision and recall, macro-F1, the confusion matrix, bootstrap intervals, McNemar tests against paired alternatives, and trivial baselines. The live pipeline couples a Haar detector to the classifier, with known differences between detector crops and benchmark crops that the discussion returns to.

Chapter 4 turns from design to implementation: how each of these stages is realised in code, which parts of the shipped implementation differ from the design described here, and what had to be fixed before the evaluation could be trusted.
