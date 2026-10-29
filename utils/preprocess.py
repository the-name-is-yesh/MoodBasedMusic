import os

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Emotion classes
# The order here determines the numerical label.
# ---------------------------------------------------------

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ---------------------------------------------------------
# Image configuration
# ---------------------------------------------------------

IMAGE_SIZE = 48


# ---------------------------------------------------------
# Find the project root
# ---------------------------------------------------------
# __file__ points to:
# MoodBasedMusic/utils/preprocess.py
#
# dirname(__file__) gives:
# MoodBasedMusic/utils
#
# dirname(dirname(__file__)) gives:
# MoodBasedMusic
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Dataset location:
# MoodBasedMusic/dataset

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset"
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset(dataset_type):
    """
    Load images and labels from the dataset.

    dataset_type should be:
        "train"
        or
        "test"

    Returns:
        images -> NumPy array containing image pixels
        labels -> NumPy array containing emotion labels
    """

    images = []
    labels = []

    # Create the path to either:
    # dataset/train
    # or
    # dataset/test

    dataset_path = os.path.join(
        DATASET_PATH,
        dataset_type
    )

    # Go through each emotion folder

    for label, emotion in enumerate(EMOTIONS):

        emotion_folder = os.path.join(
            dataset_path,
            emotion
        )

        # Check whether the emotion folder exists

        if not os.path.exists(emotion_folder):
            print(
                f"Warning: Folder not found: {emotion_folder}"
            )
            continue

        # Read every file inside the emotion folder

        for filename in os.listdir(emotion_folder):

            # Only process image files

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            # Complete image path

            image_path = os.path.join(
                emotion_folder,
                filename
            )

            # Open image

            image = Image.open(image_path)

            # Convert image to grayscale

            image = image.convert("L")

            # Make sure image is 48 x 48

            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Convert image into NumPy array

            image_array = np.array(image)

            # Store image and its label

            images.append(image_array)
            labels.append(label)

    # Convert lists into NumPy arrays

    return np.array(images), np.array(labels)


# ---------------------------------------------------------
# Normalize images
# ---------------------------------------------------------

def normalize_images(images):
    """
    Convert pixel values from:

        0 - 255

    to:

        0 - 1

    This makes the data easier for the neural network
    to process.
    """

    images = images.astype("float32")

    images = images / 255.0

    return images


# ---------------------------------------------------------
# Add channel dimension
# ---------------------------------------------------------

def add_channel_dimension(images):
    """
    Convert image shape from:

        (number_of_images, 48, 48)

    to:

        (number_of_images, 48, 48, 1)

    The final 1 represents the grayscale channel.
    """

    return np.expand_dims(
        images,
        axis=-1
    )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 50)
    print("MOOD BASED MUSIC - DATA PREPROCESSING")
    print("=" * 50)

    print("\nDataset location:")
    print(DATASET_PATH)

    # -----------------------------------------------------
    # Load training dataset
    # -----------------------------------------------------

    print("\nLoading training dataset...")

    X_train, y_train = load_dataset("train")

    print(
        "Training images shape:",
        X_train.shape
    )

    print(
        "Training labels shape:",
        y_train.shape
    )

    # -----------------------------------------------------
    # Load test dataset
    # -----------------------------------------------------

    print("\nLoading test dataset...")

    X_test, y_test = load_dataset("test")

    print(
        "Test images shape:",
        X_test.shape
    )

    print(
        "Test labels shape:",
        y_test.shape
    )

    # -----------------------------------------------------
    # Normalize images
    # -----------------------------------------------------

    print("\nNormalizing images...")

    X_train = normalize_images(X_train)

    X_test = normalize_images(X_test)

    print(
        "Training pixel range:",
        X_train.min(),
        "to",
        X_train.max()
    )

    print(
        "Test pixel range:",
        X_test.min(),
        "to",
        X_test.max()
    )

    # -----------------------------------------------------
    # Add channel dimension
    # -----------------------------------------------------

    print("\nAdding channel dimension...")

    X_train = add_channel_dimension(X_train)

    X_test = add_channel_dimension(X_test)

    # -----------------------------------------------------
    # Final information
    # -----------------------------------------------------

    print(
        "\nFinal training shape:",
        X_train.shape
    )

    print(
        "Final test shape:",
        X_test.shape
    )

    print("\nEmotion labels:")

    for number, emotion in enumerate(EMOTIONS):

        print(
            f"{number} -> {emotion}"
        )

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 50)