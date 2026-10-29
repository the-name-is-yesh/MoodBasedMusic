import os

import numpy as np
import pandas as pd


# ========================================
# SETTINGS
# ========================================

CSV_PATH = "data/ckextended.csv"

OUTPUT_DIR = "data/ckplus_processed"

IMAGE_SIZE = 48


# CK+ label mapping
CKPLUS_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
    7: "contempt"
}


# We only want these emotions.
# Neutral and contempt are intentionally excluded
# from CK+ for this experiment.
EMOTIONS_TO_KEEP = {
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise"
}


# ========================================
# LOAD DATASET
# ========================================

print("========================================")
print("       PREPARING CK+ DATASET")
print("========================================")

if not os.path.exists(CSV_PATH):

    print("\nERROR:")
    print("CK+ CSV file not found:")
    print(CSV_PATH)

    raise SystemExit


print("\nLoading CK+ CSV...")

data = pd.read_csv(CSV_PATH)

print("Loaded successfully!")

print("Total rows:", len(data))


# ========================================
# CONVERT PIXELS
# ========================================

images = []
labels = []


print("\nConverting pixel data...")


for index, row in data.iterrows():

    emotion_number = int(row["emotion"])

    emotion_name = CKPLUS_LABELS.get(
        emotion_number
    )

    # Skip unknown labels
    if emotion_name is None:
        continue

    # Skip neutral and contempt
    if emotion_name not in EMOTIONS_TO_KEEP:
        continue

    # Pixel values are stored as a string
    pixel_string = row["pixels"]

    pixel_values = np.array(
        pixel_string.split(),
        dtype=np.uint8
    )

    # Make sure each image has exactly
    # 48 x 48 = 2304 pixels.
    if len(pixel_values) != IMAGE_SIZE * IMAGE_SIZE:

        print(
            f"\nWARNING: Row {index} has "
            f"{len(pixel_values)} pixels."
        )

        continue

    image = pixel_values.reshape(
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    images.append(image)
    labels.append(emotion_name)


# ========================================
# CONVERT TO NUMPY ARRAYS
# ========================================

images = np.array(images)

labels = np.array(labels)


print("\n========================================")
print("CONVERSION COMPLETE")
print("========================================")

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# ========================================
# SHOW EMOTION COUNTS
# ========================================

print("\n========================================")
print("EMOTION COUNTS")
print("========================================")


unique_emotions, emotion_counts = np.unique(
    labels,
    return_counts=True
)


for emotion, count in zip(
    unique_emotions,
    emotion_counts
):

    print(
        f"{emotion:<10} {count}"
    )


# ========================================
# NORMALIZE
# ========================================

print("\nNormalizing pixel values...")

images_normalized = (
    images.astype("float32") / 255.0
)


print(
    "Pixel range:",
    images_normalized.min(),
    "to",
    images_normalized.max()
)


# ========================================
# ADD CHANNEL DIMENSION
# ========================================

images_normalized = np.expand_dims(
    images_normalized,
    axis=-1
)


print(
    "Final image shape:",
    images_normalized.shape
)


# ========================================
# CREATE OUTPUT DIRECTORY
# ========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ========================================
# SAVE DATA
# ========================================

images_path = os.path.join(
    OUTPUT_DIR,
    "images.npy"
)

labels_path = os.path.join(
    OUTPUT_DIR,
    "labels.npy"
)


np.save(
    images_path,
    images_normalized
)

np.save(
    labels_path,
    labels
)


# ========================================
# FINAL INFORMATION
# ========================================

print("\n========================================")
print("FILES SAVED")
print("========================================")

print(
    "Images:",
    images_path
)

print(
    "Labels:",
    labels_path
)


print("\n========================================")
print("CK+ PREPARATION FINISHED")
print("========================================")