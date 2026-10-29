import os

import numpy as np
import tensorflow as tf

from PIL import Image


# ============================================================
# EMOTION CLASSES
# ============================================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 48

MODEL_PATH = "model/best_emotion_model.keras"

IMAGE_PATH = "test_image1.jpg"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading emotion model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print("\nError: Image not found.")

    print(
        "Please place an image named "
        f"'{IMAGE_PATH}' in the project folder."
    )

    exit()


# ============================================================
# LOAD IMAGE
# ============================================================

print("\nLoading image...")

image = Image.open(
    IMAGE_PATH
)


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

image = image.convert("L")


# ============================================================
# RESIZE IMAGE
# ============================================================

image = image.resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ============================================================
# CONVERT TO NUMPY ARRAY
# ============================================================

image_array = np.array(
    image
)


# ============================================================
# NORMALIZE PIXEL VALUES
# ============================================================

image_array = image_array.astype(
    "float32"
) / 255.0


# ============================================================
# ADD CHANNEL DIMENSION
# ============================================================

image_array = np.expand_dims(
    image_array,
    axis=-1
)


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_array = np.expand_dims(
    image_array,
    axis=0
)


print(
    "\nFinal image shape:",
    image_array.shape
)


# ============================================================
# PREDICT EMOTION
# ============================================================

print("\nPredicting emotion...")

predictions = model.predict(
    image_array,
    verbose=0
)


# ============================================================
# FIND HIGHEST PROBABILITY
# ============================================================

predicted_index = np.argmax(
    predictions[0]
)

predicted_emotion = EMOTIONS[
    predicted_index
]

confidence = predictions[0][
    predicted_index
]


# ============================================================
# DISPLAY ALL PROBABILITIES
# ============================================================

print("\n========================================")
print("EMOTION PROBABILITIES")
print("========================================")

for index, emotion in enumerate(EMOTIONS):

    probability = predictions[0][index]

    print(
        f"{emotion:10} : "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

print("\n========================================")
print("PREDICTION RESULT")
print("========================================")

print(
    f"Predicted Emotion : "
    f"{predicted_emotion}"
)

print(
    f"Confidence        : "
    f"{confidence * 100:.2f}%"
)

print("========================================")