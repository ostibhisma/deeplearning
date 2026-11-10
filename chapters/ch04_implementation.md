# Chapter 4 — Implementation

## 4.1 Environment and dependencies

The project is written in Python and depends on TensorFlow/Keras (Abadi et al., 2016) for the model, OpenCV for the demonstration, and the usual scientific stack for data handling and plotting. The README advertises Python 3.7 and instructs the reader to install dependencies from `requirements.txt`, but that file is empty. In practice the environment is whatever the author had installed when the code was written, and nothing in the repository records it. This is the first reproducibility problem, and it is worth being concrete about why it matters rather than treating it as housekeeping.

The code uses `tensorflow.keras`, the Keras API bundled with TensorFlow 2.x. Three APIs used here have changed or been deprecated since the code was written. `ImageDataGenerator` still works in TensorFlow 2.15 but is legacy in Keras 3, where the preprocessing layer approach (`tf.keras.layers.RandomRotation`, `RandomFlip`, `RandomZoom`) is preferred. Loading an `.h5` file through `load_model` remains supported but is discouraged in favour of the native `.keras` format. `ModelCheckpoint` still accepts a string for `save_best_only`, but the documented type is boolean. None of these breaks the code on TensorFlow 2.x, and all of them mean that "it runs" depends on which version is installed. A pinned environment is therefore not a nicety.

At the time of writing, the machine used for this report has Python 3.14 as its system interpreter, no TensorFlow, no OpenCV, no NumPy and no pandas. TensorFlow does not publish wheels for that Python version, so the pipeline cannot be executed here at all. The experimental build described in Chapter 3 therefore targets Python 3.11 or 3.12 with a pinned TensorFlow 2.15 installation, plus NumPy, pandas, OpenCV, Matplotlib and tqdm. That environment has not yet been created, because the dataset is not available. This is a statement about the state of the work, not a limitation of the design, and it is the reason Chapter 5 is written as a protocol rather than a results chapter.

The full dependency set implied by the imports is small: `tensorflow`, `numpy`, `pandas`, `tqdm`, `matplotlib` and `opencv-python`. There is no experiment-tracking library, no configuration system and no test framework. For a project of this size that is a reasonable economy, but it means that every run has to be described by hand, which is exactly the kind of manual record-keeping that tends to go missing.

## 4.2 Repository structure and module responsibilities

The repository has four Python modules and one shell script:

```
deeplearning/
├── main.py                 training entry point
├── app.py                  webcam demonstration
├── run.sh                  interactive launcher
├── data_loading/
│   └── data.py             DataLoader
├── training/
│   └── model.py            Model: architecture and fitting
├── README.md
└── requirements.txt        empty
```

`data_loading/` and `training/` have no `__init__.py`. Python 3 treats them as namespace packages, so `from data_loading.data import DataLoader` works as long as the interpreter is started from the repository root. It would fail if the script were run from inside a subdirectory, which is a small fragility worth knowing about.

`main.py` is a dozen lines of orchestration. It constructs a `DataLoader` pointed at `./fer2013/fer2013.csv`, calls `get_data()`, passes the four arrays to `Model`, builds the network with `cnn_model()`, and calls `training(model)`. There is no argument parsing, no configuration file, and no error handling. The hard-coded path is the reason the README insists that the CSV live in a folder called `fer2013`, which also happens to be where the Haar cascade XML is expected.

`run.sh` is a three-branch menu. It reads a line from standard input and runs `main.py` if the answer is `training`, `app.py` if it is `no`, and prints an instruction otherwise. The script assumes it is executed from the repository root and that `python` resolves to an interpreter with the dependencies installed. Neither assumption is checked.

`app.py` is a top-level script rather than a module. Importing it starts a camera loop, which is convenient for a demonstration and inconvenient for testing. There is no `if __name__ == '__main__'` guard, so the loop cannot be imported without side effects.

The separation between loading, modelling and inference is the strongest structural feature of the code. Each concern is in one place, the classes have narrow interfaces, and a reader can understand the whole system in twenty minutes. The weaknesses are equally clear: configuration is scattered through the source as literals, the two entry points duplicate path strings, and nothing records what happened during a run.

## 4.3 The data loader

### 4.3.1 Reading and routing

`DataLoader.__init__` stores the path and initialises four empty lists. `get_data()` reads the CSV with pandas and iterates over rows:

```python
df = pd.read_csv(self.path_to_data)
for index, rows in tqdm(df.iterrows()):
    x = rows['pixels'].split(" ")
    if rows['Usage'] == 'Training':
        self.X_train.append(np.array(x, 'float32'))
        self.y_train.append(rows['emotion'])
    else:
        self.X_test.append(np.array(x, "float32"))
        self.y_test.append(rows['emotion'])
```

The `tqdm` wrapper gives a progress bar, which is useful because the loop takes minutes. `np.array(x, 'float32')` converts the list of 2,304 strings to a float array; the second positional argument is the dtype, which is valid NumPy usage but reads oddly. Labels are appended as Python integers and converted later.

The routing logic is the source of the leakage discussed in Section 3.3.3. `Usage` takes three values, and the `else` branch catches both test partitions. The result is a single `X_test` of 7,178 images, which `training()` then uses as validation. It is worth noting that the code's own docstring describes the split as "Training or Testing", so the author appears to have believed there were two partitions. FER2013 has three, and the distinction between public and private test matters because the challenge used it to prevent leaderboard overfitting.

### 4.3.2 Encoding and reshaping

After the loop, the lists are converted to arrays, labels are one-hot encoded with `to_categorical`, and the images are reshaped:

```python
self.y_train = to_categorical(self.y_train)
self.y_test = to_categorical(self.y_test)
self.X_train = self.X_train.reshape(self.X_train.shape[0], 48, 48, 1)
self.X_test = self.X_test.reshape(self.X_test.shape[0], 48, 48, 1)
```

The reshape relies on the pixel string being in row-major order, which it is. The single channel dimension is explicit, matching the model's `input_shape`. The one-hot call omits `num_classes`, as discussed in Section 3.4.1; with the full dataset every class is present in both partitions, so the output is 7 columns.

The method returns the four arrays in the order `X_train, X_test, y_train, y_test`. That order is unusual, since most APIs return inputs and targets together, and `main.py` unpacks it positionally. A reader who assumes the conventional order would pass labels where images are expected and get a shape error, so the order is documented in the docstring and repeated in the caller. It works, but it is a trap for anyone extending the code.

### 4.3.3 Cost and failure modes

The row-wise loop is the dominant cost. `iterrows` constructs a pandas Series for every row, and splitting the pixel string in Python is slow relative to a vectorised operation. For 35,887 rows the parsing stage takes minutes and allocates a list of arrays that is then copied by `np.array`. A vectorised alternative splits the column once and stacks the result, and caching the arrays to `.npy` once would make every subsequent run start in seconds. Neither optimisation is present.

The method wraps everything in `try/except` and prints `"Exception occurs"` followed by the exception, returning `None` on failure (`data.py:50–51`). If the CSV is missing, `main.py` receives `None` and fails with `AttributeError: 'NoneType' object has no attribute ...`, which points at the caller rather than the cause. The original exception is printed, so a careful reader can find it, but the traceback is swallowed. For a teaching project this is a common pattern; for anything used repeatedly it is a time sink.

Memory is the other constraint. The training images occupy 28,709 × 2,304 × 4 bytes ≈ 265 MB as `float32`, and during construction the list of individual arrays and the final array coexist, so peak use is roughly double that. On a laptop this is manageable; on a small cloud instance it is not negligible, and it is one reason to cache to disk.

## 4.4 Building the model

### 4.4.1 Construction

`cnn_model()` builds a `Sequential` model imperatively, adding layers one at a time. The first convolution declares `input_shape=(48, 48, 1)` and `data_format='channels_last'`; subsequent layers infer their shapes. The block structure is:

- Block 1: Conv(64, 3×3, valid) → Conv(64, 3×3, same) → BatchNorm → MaxPool(2×2) → Dropout(0.5)
- Block 2: Conv(128, 3×3, same) → BatchNorm → Conv(128, 3×3, same) → BatchNorm → MaxPool(2×2) → Dropout(0.5)
- Block 3: Conv(256, 3×3, same) → BatchNorm → Conv(256, 3×3, same) → BatchNorm → MaxPool(2×2) → Dropout(0.5)
- Head: Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(128, ReLU) → Dropout(0.5) → Dense(7, softmax)

This is exactly the structure analysed in Section 3.5. Two details are only visible in the code. The first convolution has no `padding='same'`, unlike every other convolution in the network, which is why the spatial size drops to 46 immediately. The docstring above the method lists the blocks as "2 × CONV — MAXP — DROPOUT" and does not mention batch normalisation at all, although BN appears after every convolution except the first. The docstring is therefore an approximate description of the code rather than a specification of it.

Initialisation is left at the Keras defaults: `glorot_uniform` for convolution and dense kernels, zeros for biases, and gamma/beta initialised to one and zero in the BN layers. Given BN after each block, the choice of initialiser matters less than it would in a plain stack, but it is still an unexamined default. The same is true of the ReLU slope and of the absence of any regularisation on the kernels themselves: there is no L2 weight decay.

### 4.4.2 What the model does not do

The model has no explicit input normalisation layer, relying on the generator's rescaling. It has no global average pooling, no residual connections, no attention, and no auxiliary loss. Each of these is a defensible omission for a baseline; together they define the model as a straightforward VGG-style classifier, and Chapter 6 discusses what that costs.

The network is compiled in `training()` rather than in `cnn_model()`. This keeps architecture and optimisation separate, which is tidy, but it also means that a model can be built and then compiled twice with different settings, and nothing prevents it. The compile call uses `categorical_crossentropy`, `adam` and `accuracy`, with no additional metrics.

## 4.5 The training loop

### 4.5.1 Data generators

Two generators are created in `cnn_model()` and stored on the instance: a training generator with rescaling and geometric augmentation, and a validation generator with rescaling only. `training()` then calls `flow` on each:

```python
history = model.fit(self.train_aug.flow(self.X_train, self.y_train, batch_size=BATCH_SIZE),
                    callbacks=[lr_reducer, early_stopper, checkpoint], epochs=EPOCHS,
                    validation_data=self.test_aug.flow(self.X_test, self.y_test))
```

The training stream shuffles by default. The validation stream also shuffles by default, which does not affect the aggregate loss but does mean the order of validation batches changes between epochs; with a fixed batch size the aggregate is the same, so this is harmless.

The callbacks are constructed before the call and passed as a list. Keras invokes them in list order at the end of each epoch: the learning-rate reducer first, then early stopping, then the checkpoint. That order matters for the edge case where a reduction and a stop occur in the same epoch; the reduction is applied and then training stops, so the reduced rate is used only if training continues.

### 4.5.2 Callbacks in practice

`ReduceLROnPlateau` monitors `val_loss` with `factor=0.8` and `patience=3`. It does not set `min_lr`, so the rate can in principle decay toward zero over a very long run, though early stopping prevents that here. `EarlyStopping` monitors `val_loss` with `min_delta=0` and `patience=6`, and does not restore the best weights. `ModelCheckpoint` monitors `val_loss`, saves only improvements, and writes to a path relative to the working directory.

The combined effect was analysed in Section 3.6.4: the model normally receives one learning-rate reduction before training stops, so the schedule is nearly constant. This is not visible from reading any single callback configuration; it emerges from their interaction, and it is the kind of thing that only becomes obvious when the timeline is written out.

### 4.5.3 Plotting and its failure

After fitting, the function plots the history:

```python
plt.plot(history.history["accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Train", "Validation"], loc='upper_left')
plt.savefig("training_images/accuracy.png")
```

Two problems are visible immediately. Only the training curve is passed to `plot`; `val_accuracy` is never plotted, yet the legend declares two series. A reader of the resulting figure would see one line and a legend that promises a second. The same applies to the loss figure.

The third problem is less visible. `plt.savefig("training_images/accuracy.png")` writes into a directory that does not exist in the repository. On a clean checkout, `savefig` raises `FileNotFoundError`. Because the whole training routine is wrapped in `try/except` that prints and returns, the exception is caught after training has finished, printed as `"Exception occurs"`, and the plots are never written. The model checkpoint survives because it was written during training, so the run appears to have succeeded apart from a cryptic message. This is a good example of why broad exception handling is dangerous: it converted a missing directory into a silent loss of the only diagnostic output the project produces.

`plt.close()` is not called, so the figure objects remain open. With a single call the leak is trivial; with repeated runs in one process it accumulates.

### 4.5.4 What the training routine does not return

`history` is a local variable. It is not returned, saved to CSV, or passed to a logger. The `Model` instance does not keep a reference to the fitted model either, so after `training()` returns, the only artefacts are the checkpoint file and, in principle, the two PNGs. There is no record of the loss values, the epoch at which the best model was found, the learning rate at each epoch, or the final training accuracy.

This is the single biggest gap between the code and the evaluation described in Chapter 3. Even with the data present, the shipped pipeline would produce a model and no numbers. Any results chapter requires an evaluation script that loads the checkpoint, runs predictions on the test set, and computes the metrics described in Section 3.7. That script does not exist in the repository and is specified in Appendix D.

## 4.6 The demonstration

### 4.6.1 Startup

`app.py` executes at import time. It loads the Haar cascade from `fer2013/haarcascade_frontalface_default.xml`, defines the emotion list, loads the model from `training/emotion_model.h5`, and opens the default camera with `cv2.VideoCapture(0)`. If any of these fails, the script raises before the loop starts, with an error that names the missing file. There is no fallback and no check that the camera opened.

Loading the model at import has a practical cost of a second or two, which is acceptable for a demonstration. It also means the module cannot be imported by a test without loading TensorFlow and a model file, which is why the code is effectively untestable in its current form.

### 4.6.2 The frame loop

Each iteration reads a frame, converts it to greyscale, and runs the detector:

```python
_, frame = video.read()
image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces_region = faces.detectMultiScale(image_gray, 1.3, 3)
```

The first line discards the success flag. If the camera is unavailable, `frame` is `None` and `cvtColor` raises a `cv2.error` that says little about the cause. `detectMultiScale` with `scaleFactor=1.3` and `minNeighbors=3` is a coarse configuration: a 1.3 step is faster but less thorough than the more common 1.1, and three neighbours is permissive, so the detector will occasionally return false positives. No `minSize` is set, so tiny regions are also considered.

For each rectangle, the shipped code crops and resizes:

```python
roi = image_gray[x:x+w, y:y+h]
image_resize = np.resize(roi, (48, 48))
```

Both lines are wrong, and together they make the demonstration's predictions meaningless even when the detector works.

The first indexes a NumPy array as `[x:x+w, y:y+h]`. NumPy indexes images as `[row, column]`, which corresponds to `[y, x]`, so the correct crop is `image_gray[y:y+h, x:x+w]`. As written, the code takes rows from `x` to `x+w` and columns from `y` to `y+h`. When the face is centred and roughly square, and when `x` and `y` are close, the crop may still contain part of the face by accident; for a face in the right half of the frame it contains mostly background. The rectangle drawn on the display uses the original coordinates and is therefore in the right place, which makes the bug hard to see: the box tracks the face while the classifier looks somewhere else.

The second line uses `np.resize`, which does not resample an image. It flattens the input, repeats or truncates the values to fill 48×48, and returns an array of the right shape but the wrong content. A 200×200 crop becomes the first 2,304 values of that crop, tiled, rather than a downscaled face. The correct call is `cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)`, where area interpolation is the appropriate choice for downscaling because it averages over the source region rather than sampling single pixels.

The rest of the loop is conventional. The crop is divided by 255, reshaped to `(1, 48, 48, 1)`, passed to `model.predict`, and the argmax is mapped to a label. A green rectangle of thickness 5 is drawn, the label is written in red above it, and the frame is resized to 1000×700 before display. The resize ignores aspect ratio, so the display is stretched; predictions are unaffected because they are made before it. The loop exits when `q` is pressed, and the camera is released afterwards.

### 4.6.3 Consequences

Because of the two bugs, the demonstration as shipped cannot be used as evidence that the classifier works. It may appear to work, since it draws boxes and labels, but the label is computed from a distorted, incorrectly located crop. Any screenshot or recording made with the original code would misrepresent the system. The correction is two lines, and it must be applied before the qualitative examples in Chapter 5 are captured.

There is a second, subtler issue that survives the fix. The detector returns a loose rectangle around the face, while FER2013 crops are tight and roughly centred. The classifier has never seen the extra hair, background and shoulder region that the detector includes. Resizing the loose crop to 48×48 shrinks the face relative to the training distribution. This is a domain shift, and it is a property of the pipeline design rather than a bug. Face alignment or a tighter landmark-based crop would reduce it, as discussed in Section 3.8.

## 4.7 Defects found during the audit

The audit identified 21 issues. They are listed in full in Appendix C; this section discusses the ones that affect what the project can claim.

**Test-set leakage (D1).** The two test partitions are merged and used as validation, so checkpoint selection and early stopping see the test labels. This is the most consequential defect because it invalidates any test accuracy produced by the shipped pipeline. The correction is the protocol in Section 3.3.4: a stratified validation split from the training partition, and a single evaluation on the test set.

**No evaluation (D10).** The training routine never calls `evaluate`, never computes a confusion matrix, and discards the history. Even with data, the pipeline produces no accuracy figure. The correction is an evaluation script that loads the checkpoint and reports the metrics in Section 3.7. This is the defect that most directly blocks Chapter 5.

**Misleading plots (D11).** The validation curves are never drawn, the legend claims them, and the output directory is missing, so the figures fail to save inside a swallowed exception. The correction is to plot both series, create the directory, label the axes and legend correctly, and close the figures.

**Transposed crop (D14) and `np.resize` (D15).** Both affect the demonstration and make its predictions meaningless. The corrections are the two-line changes described above.

**No seeds (D8).** Two runs of the same script differ. The correction is to seed Python, NumPy and TensorFlow at the start of every run and to record the seed. Bitwise determinism is not guaranteed on GPU, but statistical reproducibility across seeds is achievable and is what the evaluation reports.

**No class weighting (D9).** The loss is unweighted, so disgust contributes 1.5 % of the gradient mass. This is a design limitation rather than a bug, and the correction is an experiment rather than a patch: class-weighted cross-entropy or focal loss (Lin et al., 2017), reported as a comparison.

**`save_best_only='True'` (D7).** A string where a boolean is expected. It works because non-empty strings are truthy, but it is a type error and is corrected to `True`.

**Blanket exception handling (D4, D12).** Both the loader and the trainer catch every exception, print a short message, and return `None`. This hides missing files, missing directories and shape errors behind a generic line. The correction is to let exceptions propagate with context, or to catch specific ones.

**Empty requirements (D20).** The README's install step does nothing, and no versions are recorded. The correction is a pinned `requirements.txt` matching the environment used for the reported runs.

**Docstring drift (D21, and the architecture docstring).** The data docstring describes two partitions where there are three, and the model docstring omits batch normalisation and the asymmetric padding. Documentation that does not match the code is worse than none, because it is trusted. The correction is to regenerate both docstrings from the implementation.

Two further issues are worth naming because they are easy to miss. `to_categorical` is called without `num_classes`, which is safe for this dataset but fragile for any subset (D3). And the validation generator uses the default batch size of 32 while training uses 64, a cosmetic inconsistency that complicates exact reproduction (D13).

None of these corrections changes the architecture. They change the protocol (D1), the availability of evidence (D10, D11), the correctness of the demonstration (D14, D15) and the reproducibility of the environment (D8, D20). That distribution is itself informative: the model was the part that worked, and the experimental scaffolding around it was the part that did not.

## 4.8 Reproducibility measures for the experimental build

The evaluation in Chapter 3 depends on a small set of practices that the shipped code does not implement. They are specified here so that the run manifests in Chapter 5 can be checked against them.

Every run begins by seeding Python's `random`, NumPy and TensorFlow, and by setting the environment variable that requests deterministic cuDNN operations where available. The seed is written into a manifest file together with the configuration: batch size, epochs, augmentation settings, dropout rates, optimiser, learning rate, and the git commit of the code that produced the run. The manifest is stored alongside the checkpoint, so a result can be traced to the exact code and settings that generated it.

Training history is written to CSV at the end of each epoch, not only at the end of training, so that a crash does not lose the curve. The best epoch and the corresponding validation loss are recorded explicitly. Checkpoints use the `.keras` format in the experimental build, with the `.h5` file kept only for compatibility with the demonstration.

The evaluation script loads a checkpoint, computes predictions once, and derives every metric from the same prediction array. This guarantees that accuracy, per-class scores, macro-F1 and the confusion matrix are mutually consistent; computing them from separate passes is a common source of small discrepancies. The script writes a JSON summary and a figure per run, and prints the confusion matrix as a table.

Finally, the data split is saved to disk as an index array. Recomputing the split from a seed is possible, but saving it removes any dependence on library version changes in the splitting function. The same index array is reused for every configuration, which is what makes the paired comparisons in Section 3.7.2 valid.

## 4.9 Summary

The implementation is compact and readable. Data loading, model construction, training and inference are separated into four modules with clear responsibilities, and the architecture is built explicitly enough that every shape and parameter count can be derived from the source. The weaknesses are concentrated in the scaffolding rather than the model: the test partitions are merged and used for validation, no evaluation is performed, the training curves are plotted incorrectly and fail to save, the demonstration crops and resizes incorrectly, and the environment is unpinned and unseeded.

Chapter 3 described the method the project needs. This chapter described the code the project has. The gap between them is the work that remains before results can be reported, and it is a gap of engineering rather than of design: none of the corrections requires changing the network, the loss or the optimiser. Chapter 5 sets out the evaluation that will be run once the dataset and environment are in place, with the results tables left open until the numbers exist.
