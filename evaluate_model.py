import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from utils.preprocess import load_dataset


# ============================================
# SETTINGS
# ============================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

MODEL_PATH = (
    "model/best_emotion_model_v3.keras"
)


# ============================================
# LOAD MODEL
# ============================================

print("Loading CNN V3...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully!"
)


# ============================================
# LOAD TEST DATASET
# ============================================

print("\nLoading test dataset...")

X_test, y_test = load_dataset(
    "test"
)

print(
    "Test images shape:",
    X_test.shape
)

print(
    "Test labels shape:",
    y_test.shape
)


# ============================================
# PREPROCESS TEST IMAGES
# ============================================

print("\nPreprocessing test images...")

X_test = (
    X_test.astype("float32")
    / 255.0
)

X_test = np.expand_dims(
    X_test,
    axis=-1
)

print(
    "Final test shape:",
    X_test.shape
)


# ============================================
# PREDICTIONS
# ============================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================
# OVERALL ACCURACY
# ============================================

accuracy = np.mean(
    predicted_labels == y_test
)

print("\n========================================")
print("CNN V3 TEST RESULTS")
print("========================================")

print(
    f"\nTest accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    y_test,
    predicted_labels,
    target_names=EMOTIONS,
    digits=4,
    zero_division=0
)

print(report)


# ============================================
# CONFUSION MATRIX
# ============================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    y_test,
    predicted_labels
)

print(cm)


# ============================================
# PER-EMOTION ACCURACY
# ============================================

print("\n========================================")
print("PER-EMOTION ACCURACY")
print("========================================")

for index, emotion in enumerate(EMOTIONS):

    total_samples = np.sum(
        y_test == index
    )

    correct_samples = np.sum(
        (y_test == index)
        &
        (predicted_labels == index)
    )

    if total_samples > 0:

        emotion_accuracy = (
            correct_samples
            / total_samples
        )

    else:

        emotion_accuracy = 0.0

    print(
        f"{emotion:10} : "
        f"{emotion_accuracy * 100:.2f}%"
    )


# ============================================
# SAVE CONFUSION MATRIX
# ============================================

print("\nCreating confusion matrix image...")

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=EMOTIONS
)

fig, ax = plt.subplots(
    figsize=(9, 9)
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    "CNN V3 Confusion Matrix"
)

plt.tight_layout()

os.makedirs(
    "model",
    exist_ok=True
)

confusion_matrix_path = (
    "model/confusion_matrix_v3.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=150
)

plt.show()

print(
    "\nConfusion matrix saved at:"
)

print(
    confusion_matrix_path
)