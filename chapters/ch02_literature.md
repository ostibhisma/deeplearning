# Chapter 2 — Literature Review

## 2.1 Introduction and review method

This chapter places the project in the literature that produced it. It covers four bodies of work: the psychology of facial expression, the classical computer vision methods that preceded deep learning, the convolutional architectures that made the current generation of recognisers possible, and the specific work on FER2013 and related benchmarks. It closes with the literatures on regularisation, evaluation and deployment that the methodology draws on.

The review is narrative rather than systematic. Sources were located through arXiv and OpenAlex searches on terms including "facial expression recognition", "FER2013", "convolutional neural network", "label noise", "batch normalisation" and "dataset bias", and then followed backwards and forwards through the papers those searches returned. Inclusion required a verifiable identifier — a DOI, an arXiv number or a publisher page — and a contribution that the chapter actually uses. Sources that could not be verified were excluded rather than approximated, and every quantitative claim is tied to the paper and evaluation split that produced it. This matters more than it might seem: FER2013 results are routinely reported on different partitions, and a table that mixes them is worse than no table.

The review has a bias of its own. It concentrates on work that is relevant to a small, from-scratch model evaluated on a single benchmark, and it says relatively little about the large-scale multimodal and transformer-based systems that dominate current research. That is a limitation of scope, not a judgement about importance, and Chapter 7 returns to it.

## 2.2 Foundations: what is being recognised

### 2.2.1 The basic emotion tradition

The modern study of facial expression is usually traced to Ekman and Friesen's cross-cultural work, which argued that a small set of expressions — happiness, sadness, fear, disgust, anger and surprise — are recognised across cultures and therefore reflect something shared rather than something learned (Ekman & Friesen, 1971). Ekman later developed the theoretical case for basic emotions as discrete, biologically grounded states with characteristic expressions (Ekman, 1992). This framework is the reason FER datasets have the labels they have. FER2013's seven categories are the six basic emotions plus neutral, and the annotation instructions given to raters presuppose that those categories are distinguishable and mutually exclusive.

The framework also supplied a descriptive vocabulary. The Facial Action Coding System decomposes facial movement into action units rather than emotion words, and much of the computer vision literature uses action units as an intermediate representation. We do not model action units, but the distinction is useful: a system can detect a movement without inferring a state, and conflating the two is a category error that the critique below makes explicit.

### 2.2.2 The critique

The universality claim has been contested on three fronts.

Russell (1994) reviewed the cross-cultural evidence and argued that the standard experimental paradigm — showing posed photographs and asking subjects to choose from a fixed list — builds the answer into the method. Forced choice inflates agreement, the photographs are often produced by the same researchers who propose the categories, and recognition within a culture is not the same as recognition across cultures.

Jack et al. (2012) provided direct evidence for cultural variation in how expressions are produced and read. Using computational modelling of dynamic faces, they showed that Western and East Asian observers rely on different facial signals, which undermines the idea of a single universal code.

Barrett et al. (2019) went further, reviewing the psychological and neuroscientific evidence and concluding that facial movements do not reliably diagnose emotional states. People vary in how they express the same emotion, the same movement can occur in different states, and context changes interpretation. Their conclusion is not that faces carry no information but that the mapping from movement to emotion is many-to-many, and that inferring a specific state from a still image is not something the evidence supports.

This critique matters for the project in a direct way. A classifier trained on FER2013 learns to reproduce the labelling behaviour of its annotators. It does not learn to read internal states. The output is best described as a predicted label, and the report uses that language wherever precision matters. Chapter 6 returns to the ethical consequences.

### 2.2.3 Posed and spontaneous expressions

A second distinction cuts across the first. Laboratory datasets such as CK+ contain posed expressions, often recorded in controlled conditions with the subject deliberately producing a facial configuration (Lucey et al., 2010). FER2013 is different in provenance — the images were collected from the web — but many appear to be posed or acted, because people often post photographs in which they perform an expression. Spontaneous expressions in natural interaction are different: they are briefer, often asymmetrical, and frequently blended.

A model trained mostly on posed faces can therefore learn the stereotyped version of an expression and miss the subtle version. This is one explanation for the gap between laboratory and in-the-wild performance, and it is a reason to be cautious about generalising from FER2013 to a webcam conversation.

### 2.2.4 Consequences for supervised learning

Three consequences follow for the modelling in this thesis.

First, the labels are noisy by construction. Annotators disagree, particularly on fear, disgust and neutrality, and the dataset records only the majority label. Barsoum et al. (2016) made this concrete by collecting ten annotations per image in FER+ and showing that training on the full distribution outperforms training on majority-vote labels. FER2013 as distributed does not provide the distribution, so some training targets are wrong in the sense that most annotators would not agree with them.

Second, the class boundaries are fuzzy. Disgust and anger share a lowered brow and a tightened mouth; fear and surprise share widened eyes; sadness and neutral share a still face. A model that confuses these pairs is not necessarily broken — it may be reproducing a genuine ambiguity in the labels. This is why the confusion matrix, rather than accuracy alone, is the informative summary, and why the discussion of errors in Chapter 6 treats certain confusions as expected rather than as failures.

Third, an accuracy ceiling exists and is below 100 %. Human annotators do not agree perfectly with each other, so no model can agree perfectly with a single annotator's labels. We do not quote a human ceiling because we have not measured one on this dataset, and repeating a figure from secondary sources would violate the rule set out in Section 2.1. The qualitative point is enough: perfect accuracy is not the target, and a model that fits the training labels very closely is probably fitting noise.

## 2.3 Before deep learning

### 2.3.1 Face detection

Every FER system needs to find the face before it can classify it. The classical solution is the Viola–Jones detector, which combines Haar-like rectangular features, an integral image for fast evaluation, and a cascade of boosted classifiers that rejects background windows early (Viola & Jones, 2001). The detector is fast enough for real-time use on ordinary hardware, which is why it remains in OpenCV and why it appears in our demonstration.

Its limitations are equally well known. It assumes a roughly frontal, unoccluded face; it is sensitive to lighting and pose; and its rectangular features give a coarse localisation that includes hair and background. Later detectors based on convolutional networks, such as MTCNN (Zhang et al., 2016) and RetinaFace (Deng et al., 2019), improve accuracy and add landmarks that allow alignment. The trade-off is complexity and latency. For a student demonstration on a laptop, the Haar cascade is a reasonable default, and Chapter 4 discusses what it costs.

### 2.3.2 Handcrafted descriptors

Before learned representations, FER depended on descriptors designed by hand. Local Binary Patterns (LBP) encode the relationship between each pixel and its neighbours as a binary pattern, producing a texture descriptor that is invariant to monotonic illumination changes (Ojala et al., 2002). Shan et al. (2009) studied LBP for facial expression recognition systematically and showed that it is competitive on laboratory datasets when combined with a suitable classifier.

Histograms of Oriented Gradients (HOG) describe the distribution of gradient directions in local cells and were developed for pedestrian detection (Dalal & Triggs, 2005). They capture shape and contour information that is useful for expressions, particularly around the mouth and eyes.

Patch-based methods take a different route. Happy and Routray (2015) argued that discriminative information is concentrated in a small number of facial regions — primarily around the eyes and mouth — and built a descriptor from salient patches rather than the whole face. The idea anticipates the attention mechanisms that later became standard, though the implementation is hand-designed rather than learned.

### 2.3.3 Classifiers and pipelines

These descriptors were typically fed to a shallow classifier: a support vector machine, a random forest, or a boosted ensemble. The pipeline had three stages — detect, describe, classify — and each stage was tuned separately. This modularity was a strength for interpretation and a weakness for performance, because errors in early stages could not be corrected by later ones and the descriptor was fixed before the classifier saw any data.

The results on laboratory datasets were good enough to sustain the field for a decade. The results on unconstrained images were not, and the reason is structural: a descriptor designed for frontal, well-lit faces does not capture the variation present in the wild, and no amount of classifier tuning can recover information the descriptor discarded.

### 2.3.4 Why the transition happened

Convolutional networks replaced this pipeline for three reasons. They learn the representation rather than receiving it, so the features can be optimised for the task. They compose layers into a hierarchy, so simple edge detectors give way to parts and then to configurations of parts. And they share weights across space, which makes them efficient in parameters and naturally tolerant of translation. The next section traces that development.

## 2.4 Convolutional networks

### 2.4.1 LeNet and the convolutional idea

LeCun et al. (1998) demonstrated that a network with convolutional layers, pooling and shared weights could read handwritten digits reliably. The architecture introduced the ingredients still used today: local receptive fields, weight sharing, spatial subsampling, and a fully connected classifier on top. The motivation was partly biological and partly practical — shared weights reduce the parameter count dramatically compared with a dense network on the same image — and the paper established that gradient-based training of such a network was feasible.

For FER, the relevance is direct. A face is a spatially structured object, and the same local pattern (an eye corner, a mouth line) can appear anywhere in the frame. Convolution exploits that structure instead of ignoring it.

### 2.4.2 AlexNet and the deep learning shift

The 2012 ImageNet result is usually treated as the point at which deep convolutional networks became the default for image classification. Krizhevsky et al. (2012) trained a large network on two GPUs, used ReLU activations instead of saturating units, applied dropout to the fully connected layers, and augmented the training data with crops and flips. The combination produced a substantial improvement over the hand-engineered features that had dominated the benchmark.

Two lessons transferred to other vision tasks. First, capacity plus data plus regularisation beats feature engineering when the data is large enough. Second, the training tricks matter as much as the architecture: ReLU made deep networks trainable, dropout made them generalise, and augmentation stretched a finite dataset.

### 2.4.3 VGG and the value of small kernels

Simonyan and Zisserman (2015) asked what depth alone could do. Their answer was a stack of 3×3 convolutions, 16 to 19 weight layers deep, with channel counts doubling after each pooling stage. Two 3×3 convolutions have the same receptive field as one 5×5 convolution but fewer parameters and an extra non-linearity, and three 3×3 convolutions substitute for a 7×7. The design is simple, regular and easy to reason about.

Our classifier is a small version of this pattern: three blocks, each with two 3×3 convolutions, doubling channels, pooling after each block. The lineage is direct, and the reasons for choosing it are the same ones VGG gave — regularity, parameter efficiency and a receptive field that grows with depth.

### 2.4.4 Inception and multi-scale processing

Szegedy et al. (2015) took a different approach. The Inception module applies several filter sizes in parallel and concatenates the results, so the network can use multi-scale information at the same layer. The GoogLeNet instantiation used in the 2014 ImageNet challenge had 22 layers and was considerably cheaper than a comparable VGG network, because 1×1 convolutions reduced dimensionality before expensive 3×3 and 5×5 operations.

The idea is attractive for faces, where a smile involves both a local mouth shape and a global configuration of cheeks and eyes. We did not implement it, for the reasons given in Section 3.5.1: the implementation complexity is not justified at this scale, and a single-path network is easier to analyse.

### 2.4.5 ResNet and the degradation problem

He et al. (2016) observed that simply stacking more layers made training worse, not better, once a certain depth was reached. Their solution was to learn a residual function: each block computes F(x) and adds the input, so the block only has to learn the difference from the identity. With this change, networks of 152 layers trained successfully and won the 2015 ImageNet challenge.

At six convolutional layers, the degradation problem is not the binding constraint, and a residual version of our network would add parameters and complexity for little expected gain. It is, however, the obvious next architectural step if the model were deepened, and Chapter 7 lists it as future work.

### 2.4.6 Training enablers

Three techniques made deep networks practical and are used in this project.

Batch normalisation (Ioffe & Szegedy, 2015) normalises the activations of each channel across a mini-batch and then applies a learned scale and offset. The original motivation was to reduce internal covariate shift, and the empirical result was that networks could be trained with much higher learning rates and far fewer steps — the paper reports the same accuracy with fourteen times fewer training steps on ImageNet. Whether the mechanism is really covariate shift or a smoothing of the loss landscape is still debated, but the practical effect is not: BN makes deep stacks easier to train, and it acts as a mild regulariser.

Dropout (Srivastava et al., 2014) randomly deactivates a fraction of units during training, which prevents units from co-adapting and approximates an ensemble of thinned networks. At inference the full network is used, with the scaling folded into the weights. Dropout was essential in the fully connected era and remains useful in classifier heads, though in convolutional stacks it is often replaced by batch normalisation or used more sparingly.

Adam (Kingma & Ba, 2015) adapts the learning rate per parameter using estimates of the first and second moments of the gradient. It is robust to gradient scale, requires little tuning, and became a default for problems where the practitioner cannot afford an extensive learning-rate search. Our training uses Adam with its default hyperparameters for exactly that reason.

A fourth enabler, careful initialisation, is relevant even when BN is present. He et al. (2015) derived an initialisation that accounts for the variance of ReLU activations, which allows very deep rectified networks to train from scratch. We rely on Keras defaults, which implement Glorot uniform initialisation; with BN after every block the sensitivity to this choice is reduced, but it is not zero.

### 2.4.7 Learning-rate schedules

The learning rate is the hyperparameter that most often decides whether a run is useful. A constant rate that is large enough to escape a poor region early is usually too large to settle into a good minimum late, and a rate small enough to settle is slow to start. Schedules address this by reducing the rate over training.

The simplest schedules are step decay, which multiplies the rate by a factor at fixed epochs, and exponential decay, which does so continuously. Cosine annealing instead follows a cosine curve from an initial to a minimum rate, and SGDR adds periodic warm restarts that raise the rate again to escape local minima (Loshchilov & Hutter, 2017). Adaptive schedules take a different approach: they monitor a validation metric and reduce the rate when it stops improving. Keras implements this as `ReduceLROnPlateau`, which is what our training uses.

The adaptive approach has the advantage of not requiring the total number of epochs to be known in advance, which suits early stopping. It has two disadvantages. The first is that it adds another validation-dependent decision, which interacts with early stopping and checkpointing; Chapter 3 shows how the patience values in our configuration combine to make the schedule nearly inert. The second is that a gentle factor, such as 0.8, produces a slow decay that may never reach the low rates where fine-tuning happens.

Weight decay is a related but distinct lever. In stochastic gradient descent, L2 regularisation and weight decay are equivalent, but in adaptive methods they are not: the L2 term is scaled by the adaptive learning rate, so the effective decay depends on the gradient history. Loshchilov and Hutter (2019) proposed decoupling the two, giving the AdamW variant, which generally generalises better than Adam with L2. Our optimiser applies neither. That is a deliberate simplification for a baseline, but it means the model relies entirely on dropout, batch normalisation and early stopping for regularisation, and the discussion in Chapter 6 treats that as a limitation rather than a neutral choice.

### 2.4.8 Why convolution suits faces

Three properties explain why convolutional networks dominate this task.

Locality: facial features are local patterns. An eye, a nostril or a mouth corner occupies a small region, and a convolution with a small kernel is the natural detector for such patterns.

Weight sharing: the same feature can occur at different positions. A network that learned a separate detector for every location would need far more parameters and far more data. Sharing makes the model translation-equivariant by construction, and pooling adds a degree of invariance.

Hierarchy: expressions are configurations of parts. Early layers detect edges and textures, middle layers detect parts such as eyes and mouth corners, and deeper layers combine them into configurations. This is the same hierarchy that the action-unit tradition describes by hand, learned instead from data.

### 2.4.9 The small-data caveat

The same capacity that makes convolutional networks powerful makes them prone to overfitting when data is scarce. FER2013 has 28,709 training images, which is small for a network with millions of parameters. The literature's response is a combination of regularisation (dropout, weight decay, batch normalisation), augmentation, early stopping, and architecture choices that keep capacity in check.

Our network sits at the conservative end: 2.8 million parameters, heavy dropout, geometric augmentation and early stopping. Whether that is too conservative, too loose, or about right is an empirical question, and it is the question behind RQ2.

## 2.5 Deep learning for facial expression recognition

### 2.5.1 Early convolutional entries

FER2013 was released for the ICML 2013 challenge, and the entries that followed were among the first convolutional networks applied to the problem at scale (Goodfellow et al., 2013). Tang (2013) showed that replacing the softmax classifier with a linear support vector machine gave a small but consistent improvement on several benchmarks, including the challenge's expression task. The finding is a useful reminder that the loss and the classifier head are part of the model, not an afterthought.

The early results established the pattern that still holds: convolutional features outperform handcrafted descriptors on this dataset, but the absolute accuracy is limited by label noise, resolution and class imbalance.

### 2.5.2 Reviewing the state of the art

Pramerdorfer and Kampel (2016) reviewed CNN-based FER and found that reported architectures differed so much that the effect of individual design choices was unclear from published results alone. They identified the relatively basic architectures used in the field as a bottleneck and showed that an ensemble of modern deep networks reached 75.2 % test accuracy on FER2013 without auxiliary training data or face registration.

Two things are worth taking from this. First, architecture matters: moving from shallow, custom networks to standard deep architectures produced a substantial gain. Second, the gain came from an ensemble. A single network of the same family would be lower, and the comparison in Chapter 6 has to keep that distinction.

### 2.5.3 A strong single network

Khaireddin and Chen (2021) took the opposite approach: rather than combining models, they fine-tuned a single VGGNet and tuned its hyperparameters carefully, reporting 73.28 % single-network accuracy on FER2013 without additional training data. The result is close to the ensemble figure above, which suggests that a large part of the ensemble's advantage can be recovered by getting one model right.

The paper is also a useful example of how much depends on training details. Architecture families are public; the difference between a mediocre and a strong result often lies in augmentation policy, learning-rate schedule, batch size and the length of training. Our project fixes most of those details by default rather than by search, which is a limitation we return to in Chapter 6.

### 2.5.4 Label noise and label distributions

Barsoum et al. (2016) addressed the label problem directly. They collected ten annotations per image for a subset of FER2013, released the result as FER+, and compared four ways of using multiple labels: majority voting, multi-label learning, probabilistic label drawing, and cross-entropy against the full distribution. The last two outperformed majority voting, which is intuitive: a soft target that says "six annotators said happy, three said neutral, one said sad" carries more information than a hard label that discards the disagreement.

FER2013 as distributed provides only the hard label. A model trained on it therefore cannot distinguish a clear example from a contested one. This is one reason to expect a gap between training and test performance that no amount of capacity can close, and it is a reason to treat label noise as a property of the data rather than a failure of the model.

### 2.5.5 In-the-wild benchmarks

The limitations of FER2013 motivated newer datasets. Li and Deng (2017) introduced RAF-DB, built with a reliable crowdsourcing procedure and paired with a deep locality-preserving learning method designed to keep samples of the same class close in feature space. RAF-DB contains real-world images with compound expressions, which makes it harder and more representative than FER2013's posed web images.

AffectNet (Mollahosseini et al., 2017) scaled the problem up, annotating a very large number of images for both discrete expressions and continuous valence–arousal. CK+ (Lucey et al., 2010) remains the standard laboratory benchmark, with posed sequences and action-unit annotations.

These datasets are not interchangeable. A model trained on one and tested on another faces a domain shift that is often larger than the difference between two architectures on the same dataset. Our work stays within FER2013, and Chapter 6 is explicit that cross-dataset generalisation is untested.

### 2.5.6 Attention and other directions

Attention mechanisms developed for general vision offer a way to focus computation on informative regions. The Squeeze-and-Excitation block recalibrates channel responses (Hu et al., 2018), and CBAM adds a spatial attention branch (Woo et al., 2018). For faces, the motivation is obvious: the eyes and mouth carry most of the signal, and a module that learns to weight them could improve both accuracy and interpretability.

We did not implement attention. The reason is scope rather than scepticism: adding a module changes the architecture, complicates the ablation design, and requires a separate set of runs. It is a natural extension and is listed in Chapter 7.

### 2.5.7 Where the present model sits

The verified reference points for FER2013 are a 75.2 % ensemble (Pramerdorfer & Kampel, 2016) and a 73.28 % single network (Khaireddin & Chen, 2021). Both use architectures larger than ours and both tune training details extensively. Our model is smaller, trained with default optimiser settings and a fixed augmentation policy, and its purpose is to serve as a transparent from-scratch baseline rather than to compete at the frontier.

Any comparison in Chapter 6 will therefore be framed carefully: same dataset, different splits, different training budgets and different model sizes. The point of the comparison is to locate the model in the range of published results, not to claim parity.

## 2.6 Datasets

### 2.6.1 FER2013

FER2013 is the dataset used here. It has 35,887 greyscale 48×48 images labelled with seven expressions and split into 28,709 training and 7,178 test images, the latter divided into public and private partitions by the original challenge (Goodfellow et al., 2013). The images were collected from web search and annotated by human raters.

Its advantages are size, standardisation and ubiquity. Almost every FER paper reports on it, so comparisons are possible. Its disadvantages are equally well documented: low resolution, label noise, class imbalance (16.5:1 between happiness and disgust), no demographic metadata, and a test set small enough that a percentage point is about 72 images. It is a benchmark of convenience as much as of difficulty.

### 2.6.2 FER+

FER+ is a relabelling of FER2013 with ten annotations per image and a revised set of categories (Barsoum et al., 2016). It is not a different image collection; it is a different set of labels for (mostly) the same images. Its value is that it makes annotator disagreement visible and supports training on distributions rather than hard labels. We did not use it, because the project is built around the original FER2013 file, but it is the most direct way to address the label-noise limitation.

### 2.6.3 Laboratory datasets

CK+ extends the Cohn–Kanade collection with posed sequences and action-unit annotations (Lucey et al., 2010). Because the expressions are deliberate and the conditions controlled, accuracy on CK+ is much higher than on FER2013, and models trained on it do not transfer well to unconstrained images. It is useful for studying expression dynamics and action units, less useful as a test of real-world robustness.

### 2.6.4 In-the-wild datasets

RAF-DB (Li & Deng, 2017) and AffectNet (Mollahosseini et al., 2017) were built to address the gap. RAF-DB provides real-world images with reliable crowd annotations and compound expressions; AffectNet provides a very large collection annotated for both discrete categories and continuous affect. Both are harder than FER2013 and both are more representative of deployment conditions.

The existence of these datasets is a reminder that FER2013 is not the final word. A model that performs well on it has demonstrated something about learning from noisy 48×48 crops, not about understanding faces in general.

### 2.6.5 Provenance, consent and bias

Most of these datasets consist of photographs collected from the web, and the people depicted did not consent to being in a research benchmark or to having their expressions classified. Laboratory datasets such as CK+ are an exception in one respect, since participants took part in recorded sessions, but the categories and instructions still reflect the choices of the researchers who built them. This is a structural feature of the field, not a criticism of any single dataset, and it has consequences for what the models can be used for.

Dataset bias compounds the problem. Torralba and Efros (2011) showed that datasets carry a signature of their collection process, and that classifiers can exploit that signature rather than learning the intended concept. For FER, the collection process selects for posed, well-lit, front-facing faces, which means that a model trained on such data may be learning the conventions of the dataset as much as the structure of expressions. Buolamwini and Gebru (2018) demonstrated the practical consequence for commercial gender classification: accuracy differed substantially across intersectional subgroups, and the aggregate figure hid the disparity. FER2013 has no demographic metadata, so the equivalent analysis cannot be performed on it. That absence is itself a finding, and Chapter 6 discusses it.


Table 2.1 summarises the datasets discussed in this section. The comparison is qualitative because we have verified the composition only of FER2013; sizes and label counts for the others are deliberately omitted rather than quoted from memory.

**Table 2.1. Qualitative comparison of facial expression datasets.**

| Dataset | Setting | Label type | Main strength | Main limitation |
|---|---|---|---|---|
| FER2013 | Web images, 48×48 greyscale | Seven discrete classes | Standard benchmark; large enough to train from scratch | Label noise, low resolution, no demographics |
| FER+ | Relabelled FER2013 images | Distribution over categories from ten annotators | Exposes annotator disagreement | Same images and resolution as FER2013 |
| CK+ | Laboratory, posed sequences | Discrete emotions and action units | Controlled; useful for dynamics and action units | Does not represent unconstrained conditions |
| RAF-DB | Real-world images | Discrete and compound expressions | Realistic; reliable crowd annotation | Smaller and harder; different label space |
| AffectNet | Very large in-the-wild collection | Discrete emotions and continuous affect | Scale; continuous annotations | Annotation cost and coverage vary by class |

The table makes the central point of this section concrete: the datasets differ in setting, label type and purpose, and a result on one is not a result on another.

## 2.7 Learning from small, noisy data

### 2.7.1 Augmentation

Data augmentation expands the effective training set by applying label-preserving transformations. For faces, the natural transformations are geometric: small rotations, translations, flips and zooms. Photometric transformations — brightness, contrast, gamma — address illumination variation. Both are standard, and both carry assumptions: a horizontal flip preserves most expressions but not all (a wink becomes the other eye), and aggressive shear or zoom can create artefacts that the model learns instead of the expression.

More recent methods mix images or erase regions. Cutout randomly masks a square region during training, forcing the model to use information from the rest of the face and improving robustness (DeVries & Taylor, 2017). Mixup trains on convex combinations of images and labels (Zhang et al., 2018), and CutMix pastes a patch from one image into another while mixing the labels proportionally (Yun et al., 2019). These methods are effective on large datasets; on a small, noisy dataset their benefit is less certain, because they assume that interpolating between labels is meaningful, which is questionable when the labels themselves are noisy.

### 2.7.2 Label smoothing and loss shaping

Label smoothing replaces the one-hot target with a softened distribution, typically (1 − ε) on the true class and ε/(K − 1) on the others (Szegedy et al., 2016). It discourages overconfidence, improves calibration, and can improve accuracy when labels are noisy — all relevant here. We did not implement it, and it is a cheap experiment for future work.

Focal loss reshapes cross-entropy so that well-classified examples contribute less, focusing training on hard examples (Lin et al., 2017). It was designed for extreme foreground–background imbalance in object detection, but the same mechanism applies to class imbalance: a rare class that is consistently misclassified retains a large loss while an easy majority class is down-weighted. Class-weighted cross-entropy is a simpler alternative that scales each class's contribution by its inverse frequency.

Neither is implemented. The network is trained with unweighted cross-entropy, which means the imbalance is unaddressed and the evaluation must reveal its cost.

### 2.7.3 Transfer learning

The most reliable way to improve performance on a small dataset is to start from a model trained on a large one. Khaireddin and Chen (2021) fine-tuned a VGGNet and reported a strong single-network result, and the general practice in the field is to initialise from ImageNet or from a large face corpus.

We deliberately did not. The research question is what can be learned from FER2013 alone, and pretraining would confound that question. The cost is accuracy; the benefit is interpretability of the result. Chapter 6 discusses the trade-off, and Chapter 7 recommends a transfer-learning comparison as the most valuable next experiment.

### 2.7.4 Class imbalance

Imbalance can be addressed at three points in the pipeline. At the data level, the training set can be resampled: oversample the minority classes, undersample the majority, or draw batches with class-balanced probabilities. At the loss level, each class can be weighted by the inverse of its frequency, or the loss can be reshaped so that easy examples contribute less (Lin et al., 2017). At the decision level, the predicted probabilities can be adjusted with class-specific thresholds so that the operating point favours the minority classes.

Each has costs. Oversampling duplicates images and encourages memorisation unless it is combined with augmentation. Undersampling discards data that is already scarce. Class weighting increases the gradient contribution of rare classes but also increases the variance of the updates, and with 436 disgust images the effective sample size remains small however the loss is weighted. Threshold adjustment changes the precision–recall trade-off without changing the model, which is useful when the application has asymmetric costs but misleading when the goal is a single benchmark number.

Our training uses none of these. The consequence is predictable: the model will favour the majority classes, accuracy will exceed macro-F1 by a margin that reflects that favour, and disgust will have low recall. The evaluation is designed to show this rather than to hide it, and Chapter 6 discusses whether weighting would be worth the variance it introduces.

## 2.8 Evaluation and reproducibility

### 2.8.1 Metrics

Accuracy is the proportion of correct predictions and is the default in the FER literature. It is also easy to mislead with when classes are imbalanced. Sokolova and Lapalme (2009) give a systematic account of precision, recall, F-measure and their macro and micro variants, and the distinction matters here: micro-averaging over classes is equivalent to accuracy in single-label classification, while macro-averaging weights each class equally and therefore exposes poor performance on disgust.

We report both, and we treat the gap between them as a result rather than a nuisance.

### 2.8.2 Confidence and significance

A single accuracy on 7,178 images has sampling error. Efron's bootstrap (Efron, 1979) provides a distribution-free way to estimate it: resample the test set with replacement, recompute the metric, and take percentiles. The method assumes that the test images are exchangeable, which is reasonable here, and it is more informative than a standard error computed under a binomial model when the metric is F1 rather than accuracy.

For comparing two models on the same test set, McNemar's test (McNemar, 1947) uses the discordant pairs and is the standard choice; Dietterich (1998) explains why it is preferable to comparing independent proportions in this setting. We use it with a continuity correction and report paired bootstrap intervals alongside.

### 2.8.3 Variance and benchmark reuse

Henderson et al. (2018) showed, in the context of deep reinforcement learning, that seed variance can be large enough to make single-run comparisons meaningless, and argued for reporting across seeds with significance measures. The argument transfers to supervised training: initialisation, data order and augmentation randomness all vary, and a one-point difference between two runs may be noise. Our protocol uses three seeds and reports standard deviations, which is a minimum rather than a thorough treatment.

Recht et al. (2019) showed that classifiers evaluated on a reused test set can lose several points when a fresh test set is collected in the same way. The mechanism is partly adaptivity — many models have been tuned against the same benchmark — and partly that the original test set is easier than new data. FER2013's test set has been used for a decade, and the existence of a public partition invites model selection on it. Our leakage-free protocol avoids selection on the test set, but it cannot undo the benchmark's history.

### 2.8.4 Calibration and reporting

Guo et al. (2017) showed that modern networks are often poorly calibrated: their confidence does not match their accuracy, and deeper and wider models are worse. This matters for a live demonstration, where a 0.99 probability is displayed as if it meant something. Temperature scaling is a cheap post-hoc fix. We did not apply it, but the discussion of the demo in Chapter 6 notes the issue.

Mitchell et al. (2019) proposed model cards as a structured way to report intended use, performance, and subgroup results. The format is a useful checklist even for a student project, and our limitations section maps onto several of its categories.

## 2.9 Deployment considerations

### 2.9.1 Detection and alignment

A classifier trained on tight crops needs a detector that returns comparable crops. Haar cascades are fast but loose and frontal-only (Viola & Jones, 2001). MTCNN returns a tighter box plus five landmarks, which allows similarity-transform alignment before classification (Zhang et al., 2016). RetinaFace improves localisation further and can run in real time on a CPU with a lightweight backbone (Deng et al., 2019). Alignment is not cosmetic: it removes rotation and scale variation that the classifier was trained to tolerate but not to prefer.

Our pipeline uses Haar and no alignment. That is a deliberate simplification for a laptop demo, and it is the main reason the deployment setting differs from the benchmark setting.

### 2.9.2 Model compression

If a model has to run on a phone or an embedded device, three families of techniques apply. Pruning removes weights or connections that contribute little, and Han et al. (2015) showed an order-of-magnitude reduction in storage without accuracy loss on ImageNet-era networks. Quantisation reduces the precision of weights and activations, and Jacob et al. (2018) described a scheme for integer-only inference that is now standard on mobile hardware. Distillation trains a small model to mimic a large one, compressing an ensemble into a single network (Hinton et al., 2015).

Efficient architectures take the idea further. MobileNets use depthwise separable convolutions to reduce computation by a large factor (Howard et al., 2017), and MobileNetV2 adds inverted residuals and linear bottlenecks (Sandler et al., 2018). Our model is already small — 2.8 million parameters and about 0.61 GFLOPs per image — so compression is not the binding constraint. The demonstration's latency is dominated by the detector and the Python loop, not by the network.

### 2.9.3 Latency and the demo

A live demo has to process frames faster than a human notices lag. At 30 frames per second the budget is 33 milliseconds per frame, including capture, detection, classification and drawing. We have not measured our pipeline's latency, so we make no frames-per-second claim. The design discussion in Section 3.8 identifies the detector as the likely bottleneck and suggests batching and temporal smoothing as fixes. Those are hypotheses to be tested, not results.

## 2.10 Synthesis: the gap this project addresses

The literature is rich on architectures and benchmarks and increasingly rich on label noise and dataset bias. Three things are less well covered, and they define the space this project occupies.

First, **honest small-scale baselines**. Published results are dominated by large models, ensembles and extensive tuning. A reader who wants to know what a compact, from-scratch network achieves under a fixed, documented recipe has to assemble that answer from scattered papers. A clear baseline with a full architectural characterisation is useful even when it is not competitive.

Second, **implementation audits**. Papers describe what was intended; code reveals what was done. The defects documented in Chapter 4 — test-set leakage, a missing evaluation step, a transposed crop — are easy to introduce and hard to see, and they change what the numbers mean. Making the audit part of the report, rather than an embarrassment to be hidden, is a small contribution to how this work is taught.

Third, **the deployment gap in small systems**. The distance between a benchmark crop and a detector rectangle is well known in industry and under-discussed in coursework. Our RQ3 addresses it qualitatively, and the analysis of detector crops versus training crops is transferable even though our measurements are limited.

None of these is a research breakthrough. Together they describe a project that is honest about its scale and useful to the next student who inherits the code.

## 2.11 Summary

The chapter traced the field from the basic-emotion tradition and its critiques, through handcrafted descriptors and classical detectors, to convolutional networks and their application to FER. It identified the two verified FER2013 reference points — a 75.2 % ensemble and a 73.28 % single network — and the reasons why neither is directly comparable to a small from-scratch model. It reviewed the datasets that situate FER2013, the regularisation methods that make small-data training possible, the evaluation practices that keep comparisons honest, and the deployment techniques that connect a classifier to a camera. It closed by identifying the gap the project addresses: a transparent baseline, an implementation audit, and an analysis of the gap between benchmark and deployment conditions. Chapter 3 turns to the method.
