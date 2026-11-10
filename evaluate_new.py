
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ====================================
# PROJECT CONFIGURATION
# ====================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "training" / "emotion_model.h5"
DATA_PATH = BASE_DIR / "fer2013" / "fer2013.csv"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# ====================================
# LOAD TRAINED MODEL
# ====================================

print("\nLoading trained CNN model...")

model = load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model loaded successfully!")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# ====================================
# LOAD FER2013 DATASET
# ====================================

print("\nLoading FER2013 dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset columns:", df.columns.tolist())
print("Total dataset images:", len(df))

required = {"emotion", "pixels", "Usage"}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing CSV columns: {required - set(df.columns)}"
    )

# Use the standard FER2013 held-out test split
test_df = df[
    df["Usage"] == "PrivateTest"
].copy()

if test_df.empty:
    raise RuntimeError(
        "No PrivateTest images found."
    )

print("Test images:", len(test_df))

# ====================================
# PREPROCESS TEST IMAGES
# ====================================

print("\nPreprocessing test images...")

X_test = []

for pixels in test_df["pixels"]:

    image = np.fromstring(
        pixels,
        dtype=np.uint8,
        sep=" "
    )

    if image.size != 48 * 48:
        raise ValueError(
            "Invalid image size in FER2013 CSV."
        )

    image = image.reshape(48, 48)
    X_test.append(image)

X_test = np.array(
    X_test,
    dtype=np.float32
) / 255.0

X_test = X_test.reshape(
    -1, 48, 48, 1
)

y_test = test_df["emotion"].to_numpy(
    dtype=np.int64
)

print("Input shape:", X_test.shape)

# ====================================
# PREDICT EMOTIONS
# ====================================

print("\nRunning model predictions...")

predictions = model.predict(
    X_test,
    batch_size=64,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ====================================
# CALCULATE PERFORMANCE METRICS
# ====================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test, y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 45)
print("       CNN MODEL EVALUATION RESULTS")
print("=" * 45)

print(f"Accuracy:      {accuracy * 100:.2f}%")
print(f"Precision:     {precision * 100:.2f}%")
print(f"Recall:        {recall * 100:.2f}%")
print(f"Macro F1:      {macro_f1 * 100:.2f}%")
print(f"Weighted F1:   {weighted_f1 * 100:.2f}%")

# ====================================
# CLASSIFICATION REPORT
# ====================================

report = classification_report(
    y_test,
    y_pred,
    labels=list(range(7)),
    target_names=EMOTIONS,
    zero_division=0
)

print("\nCLASSIFICATION REPORT\n")
print(report)

# Save classification report
report_path = BASE_DIR / "classification_report.txt"

with open(report_path, "w") as file:
    file.write("CNN Emotion Recognition Evaluation\n")
    file.write("=" * 40 + "\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n")
    file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    file.write(report)

# ====================================
# CONFUSION MATRIX
# ====================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(7))
)

fig, ax = plt.subplots(
    figsize=(11, 9)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=EMOTIONS
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "FER2013 CNN Emotion Recognition"
)

plt.xticks(rotation=45)
plt.tight_layout()

matrix_path = BASE_DIR / "confusion_matrix.png"

plt.savefig(
    matrix_path,
    dpi=300
)

plt.close(fig)

print("\nEvaluation completed!")
print("Saved classification_report.txt")
print("Saved confusion_matrix.png")


