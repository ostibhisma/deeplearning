# Appendices

## Appendix A — Configuration as Implemented

| Parameter | Value | Location |
|---|---|---|
| Input shape | 48 × 48 × 1, channels last | `model.py:41` |
| Convolution kernel | 3 × 3, stride 1 | `model.py:41–58` |
| Filters per block | 64, 128, 256 | `model.py:41,47,54` |
| Padding | valid on first conv, same thereafter | `model.py:41–42` |
| Activation | ReLU | all conv and dense layers |
| Batch normalisation | after conv 2, 3, 4, 5, 6 | `model.py:43,48,50,55,57` |
| Pooling | 2 × 2, stride 2, three times | `model.py:44,51,58` |
| Dropout | 0.5 after each block; 0.4 then 0.5 in head | `model.py:45,52,59,64,67` |
| Dense units | 256, 128, 7 | `model.py:63,66,69` |
| Output activation | softmax | `model.py:69` |
| Loss | categorical cross-entropy | `model.py:92` |
| Optimiser | Adam, lr 10⁻³, β₁ 0.9, β₂ 0.999, ε 10⁻⁷ | `model.py:92` |
| Batch size | 64 | `model.py:89` |
| Maximum epochs | 100 | `model.py:90` |
| Training augmentation | rotation 20°, horizontal flip, zoom 0.2, shear 0.2 rad, rescale 1/255 | `model.py:36` |
| Validation preprocessing | rescale 1/255 | `model.py:37` |
| LR reduction | factor 0.8, patience 3, monitor val_loss | `model.py:94` |
| Early stopping | patience 6, min_delta 0, monitor val_loss | `model.py:96–97` |
| Checkpoint | best val_loss, `training/emotion_model.h5` | `model.py:99–100` |
| Seed | none in shipped code | — |

## Appendix B — Source Code Listing

The three modules below are reproduced from commit `208236e`, with comments preserved and line wrapping adjusted to the page width. The corrections discussed in Chapter 4 are specified there and in Appendix C; they are not applied in this listing.

### B.1 `data_loading/data.py`

```python
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

class DataLoader:
    """
    This class shall obtain a data and preprocess the data for training
    """
    def __init__(self,path_to_data):
        self.path_to_data = path_to_data
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

    def get_data(self):
        """
        Method Name : get_data
        Description : This method read the data using pandas dataframe and
                      according to the Training or Testion given in the
                      'Usage' column in df it load the training and testing
                      data and convert it to the numpy array and reshape the
                      data in (48,48,1) format and returns the data
        """
        try:
            df = pd.read_csv(self.path_to_data)
            for index,rows in tqdm(df.iterrows()):
                x = rows['pixels'].split(" ")
                if rows['Usage'] =='Training':
                    self.X_train.append(np.array(x,'float32'))
                    self.y_train.append(rows['emotion'])
                else:
                    self.X_test.append(np.array(x,"float32"))
                    self.y_test.append(rows['emotion'])
            self.X_train = np.array(self.X_train)
            self.X_test = np.array(self.X_test)
            self.y_train = np.array(self.y_train)
            self.y_test = np.array(self.y_test)
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
            self.X_train = self.X_train.reshape(self.X_train.shape[0],48,48,1)
            self.X_test = self.X_test.reshape(self.X_test.shape[0],48,48,1)
            return self.X_train,self.X_test,self.y_train,self.y_test
        except Exception as e:
            print("Exception occurs",str(e))
```

### B.2 `training/model.py`

```python
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Flatten, Dense, Conv2D, MaxPooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        EarlyStopping)
import matplotlib.pyplot as plt

class Model:
    """
    This Class Train the model and save the best model according to
    validation accuracy in the form of .h5 file
    """
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def cnn_model(self):
        """
        [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
        [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
        [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
        Dense (256) — DROPOUT (0.4)
        Dense (128) — DROPOUT (0.5)
        """
        try:
            num_features = 64
            num_labels = 7
            width, height = 48, 48
            self.train_aug = ImageDataGenerator(
                rescale=1./255, rotation_range=20, horizontal_flip=True,
                zoom_range=0.2, shear_range=0.2)
            self.test_aug = ImageDataGenerator(rescale=1./255)

            model = Sequential()
            model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu',
                             input_shape=(width, height, 1),
                             data_format='channels_last'))
            model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu',
                             padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(2*num_features, kernel_size=(3, 3),
                             activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(2*num_features, kernel_size=(3, 3),
                             activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(2*2*num_features, kernel_size=(3, 3),
                             activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(2*2*num_features, kernel_size=(3, 3),
                             activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(2*2*num_features, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(2*num_features, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_labels, activation='softmax'))
            return model
        except Exception as e:
            print("Exceptions occurs",str(e))

    def training(self,model):
        try:
            BATCH_SIZE = 64
            EPOCHS = 100
            model.compile(loss='categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                           patience=3, verbose=1)
            early_stopper = EarlyStopping(monitor='val_loss', min_delta=0,
                                          patience=6, verbose=1, mode='auto')
            checkpoint = ModelCheckpoint('training/emotion_model.h5',
                                         monitor='val_loss', verbose=0,
                                         save_best_only='True', mode='auto')
            history = model.fit(
                self.train_aug.flow(self.X_train, self.y_train,
                                    batch_size=BATCH_SIZE),
                callbacks=[lr_reducer, early_stopper, checkpoint],
                epochs=EPOCHS,
                validation_data=self.test_aug.flow(self.X_test, self.y_test))
            plt.plot(history.history["accuracy"])
            plt.title("Model Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epochs")
            plt.legend(["Train","Validation"], loc='upper_left')
            plt.savefig("training_images/accuracy.png")
            plt.plot(history.history["loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.legend(["Train","Validation"], loc='upper_left')
            plt.savefig("training_images/loss.png")
        except Exception as e:
            print("Exception occurs",str(e))
```

### B.3 `app.py`

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

faces = cv2.CascadeClassifier("fer2013/haarcascade_frontalface_default.xml")
emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']
model = load_model("training/emotion_model.h5")
video = cv2.VideoCapture(0)

while True:
    _,frame = video.read()
    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces_region = faces.detectMultiScale(image_gray,1.3,3)
    for (x,y,w,h) in faces_region:
        roi = image_gray[x:x+w,y:y+h]
        image_resize = np.resize(roi,(48,48))
        resized = image_resize/255
        image = np.reshape(resized,(1,48,48,1))
        result = model.predict(image)
        label = np.argmax(result,axis=1)[0]
        emotion = emotions[label]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        cv2.putText(frame, emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
```

## Appendix C — Defect Register

| ID | Location | Defect | Consequence | Correction |
|---|---|---|---|---|
| D1 | `data.py:34` | PublicTest and PrivateTest merged and used as validation | Test-set leakage in checkpoint, LR and early stopping | Stratified validation split from Training |
| D2 | `data.py:31–48` | No stratified split or seed | Split not reproducible | `train_test_split(..., stratify=y, random_state=42)` |
| D3 | `data.py:44–45` | `to_categorical` without `num_classes` | Silent width change on subsets | `num_classes=7` |
| D4 | `data.py:50–51` | Broad `except` returns `None` | Opaque downstream failure | Raise with context |
| D5 | `data.py:32` | Row-wise `iterrows` loop | Slow parsing, high memory | Vectorise or cache `.npy` |
| D6 | `model.py:94–97` | LR patience 3 vs ES patience 6 | Schedule nearly inert | Increase ES patience or use cosine schedule |
| D7 | `model.py:100` | `save_best_only='True'` string | Type error that works by accident | `True` |
| D8 | all | No random seeds | Runs not reproducible | Seed Python, NumPy, TensorFlow |
| D9 | `model.py:92` | No class weights | Majority-class bias, low macro-F1 | Class weighting or focal loss |
| D10 | `model.py:79–119` | No evaluation step | No accuracy, F1 or confusion matrix produced | Add evaluation script (Appendix D) |
| D11 | `model.py:107–119` | Validation curve not plotted; legend claims it; output directory missing | Misleading figure; save fails silently | Plot both series, create directory, close figures |
| D12 | `model.py:88–123` | Blanket `try/except` | Bugs hidden after training | Narrow exception handling |
| D13 | `model.py:104` | Validation generator batch 32 | Minor inconsistency | Set `batch_size=64` |
| D14 | `app.py:18` | Crop indexed as `[x:x+w, y:y+h]` | Classifies the wrong region | `[y:y+h, x:x+w]` |
| D15 | `app.py:19` | `np.resize` used as image resize | Tiled/truncated input | `cv2.resize(..., INTER_AREA)` |
| D16 | `app.py:14` | Frame read result discarded | Crash when camera unavailable | Guard and exit with message |
| D17 | `app.py:16` | Untuned Haar parameters | Jitter and false positives | Tune, add `minSize` |
| D18 | `app.py:28` | Display resize ignores aspect ratio | Stretched image | Preserve aspect ratio |
| D19 | `app.py:22` | Per-face `predict` call | Latency overhead | Batch ROIs or call model directly |
| D20 | `requirements.txt` | Empty, no pins | Environment not reproducible | Pin versions |
| D21 | `README.md` | Python 3.7 badge, paths under `fer2013/` | Misleading setup | Update documentation |

## Appendix D — Evaluation Script

The script below implements the protocol in Chapter 3. It is written for the corrected pipeline (stratified validation split, `.keras` checkpoint) and has not been executed, because FER2013 is not available in the environment. It is included so that the evaluation can be reproduced and audited.

```python
"""Evaluate a trained FER2013 model: metrics, intervals, McNemar, figures."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             cohen_kappa_score)

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_images(frame):
    """Parse the pixel column into an (N, 48, 48, 1) float array in [0, 1]."""
    values = frame["pixels"].str.split()
    array = np.stack([np.asarray(v, dtype="float32") for v in values])
    return array.reshape(-1, 48, 48, 1) / 255.0


def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=42):
    """Percentile bootstrap interval for a metric of two label arrays."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(metric(y_true[idx], y_pred[idx]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def mcnemar(y_true, pred_a, pred_b):
    """McNemar test with continuity correction for two paired classifiers."""
    from scipy.stats import chi2 as chi2_dist
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - chi2_dist.cdf(statistic, df=1))
    return {"b": b, "c": c, "chi2": float(statistic), "p": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="fer2013/fer2013.csv")
    parser.add_argument("--model", default="training/emotion_model.keras")
    parser.add_argument("--out", default="results")
    parser.add_argument("--run-id", default="A-42")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frame = pd.read_csv(args.csv)
    test = frame[frame["Usage"] != "Training"].reset_index(drop=True)
    x_test = load_images(test)
    y_true = test["emotion"].to_numpy()

    model = tf.keras.models.load_model(args.model)
    probabilities = model.predict(x_test, batch_size=64, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(7), zero_division=0)
    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = float(np.mean(f1))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    acc_ci = bootstrap_ci(y_true, y_pred, lambda t, p: float(np.mean(t == p)))
    f1_ci = bootstrap_ci(
        y_true, y_pred,
        lambda t, p: float(np.mean(precision_recall_fscore_support(
            t, p, labels=range(7), zero_division=0)[2])))

    matrix = confusion_matrix(y_true, y_pred, labels=range(7))
    summary = {
        "run_id": args.run_id,
        "n_test": int(len(y_true)),
        "accuracy": accuracy,
        "accuracy_ci95": acc_ci,
        "macro_f1": macro_f1,
        "macro_f1_ci95": f1_ci,
        "kappa": kappa,
        "per_class": {
            EMOTIONS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            } for i in range(7)
        },
        "confusion_matrix": matrix.tolist(),
    }
    with open(os.path.join(args.out, f"{args.run_id}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.save(os.path.join(args.out, f"{args.run_id}_pred.npy"), y_pred)
    print(json.dumps({k: summary[k] for k in
                      ("run_id", "accuracy", "macro_f1", "kappa")}, indent=2))


if __name__ == "__main__":
    main()
```

## Appendix E — Verification and AI-Use Log

The table records the checks carried out on the report and the use of the AI assistant. Dates refer to the drafting sessions in October 2026.

| Date | Item | Check performed | Result |
|---|---|---|---|
| 28 Oct | Parameter count | Recomputed every layer from `model.py` | 2,818,375 total; 2,816,711 trainable |
| 28 Oct | Receptive field | Computed from kernel sizes and strides | ≈ 40 × 40 |
| 28 Oct | MAC count | Summed per-layer products | 304,945,536 per image |
| 28 Oct | Class distribution | Summed published per-class counts | 28,709 train, 7,178 test |
| 28 Oct | Majority baseline | 1,774 / 7,178 | 0.247 accuracy; 0.057 macro-F1 |
| 28 Oct | Dataset identifiers | arXiv API lookup | Goodfellow et al. (2013) confirmed |
| 28 Oct | FER2013 reference points | arXiv abstracts read | 75.2 % ensemble; 73.28 % single network |
| 28 Oct | Bibliography DOIs | OpenAlex DOI lookup | 58 entries verified; four wrong identifiers corrected before use |
| 28 Oct | RUC AI guidelines | PDF downloaded and read in full | Declaration requirements applied |
| 28 Oct | Draft consistency | Numbers and claims cross-checked | Ch. 3: 43→44 %, head description, `save_best_only` type; Ch. 2: consent paragraph, posed claim, public-partition claim |
| 28 Oct | Claims audit | Every "we measured" statement checked | None present; no fabricated results |

**AI assistance.** The assistant (DeepSeek Harness) was used to analyse the code, compute the architectural quantities, assist with source verification, draft chapter text and check consistency. All code-derived numbers were verified independently; all citations were checked against external records; no data or results were generated. The authors reviewed and edited the text and accept responsibility for it, as stated in the front matter.
