import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils.preprocess import (
    load_dataset,
    normalize_images,
    add_channel_dimension,
    EMOTIONS
)


# ========================================
# SETTINGS
# ========================================

IMAGE_SIZE = 48

NUM_CLASSES = 7

BATCH_SIZE = 64

EPOCHS = 30

RANDOM_STATE = 42

CKPLUS_IMAGES_PATH = (
    "data/ckplus_processed/images.npy"
)

CKPLUS_LABELS_PATH = (
    "data/ckplus_processed/labels.npy"
)

BEST_MODEL_PATH = (
    "model/best_emotion_model_v4.keras"
)

FINAL_MODEL_PATH = (
    "model/emotion_model_v4.keras"
)


# ========================================
# DISPLAY INFORMATION
# ========================================

print("========================================")
print("       CNN V4 TRAINING")
print("========================================")

print("\nTraining data:")
print("Original dataset + CK+")

print("\nValidation data:")
print("Original dataset only")

print("\nTest data:")
print("Untouched original test dataset")


# ========================================
# LOAD ORIGINAL DATASET
# ========================================

print("\n========================================")
print("LOADING ORIGINAL DATASET")
print("========================================")

print("\nLoading training data...")

X_train_full, y_train_full = load_dataset(
    "train"
)

print(
    "Original training images:",
    X_train_full.shape
)

print(
    "Original training labels:",
    y_train_full.shape
)


print("\nLoading test data...")

X_test, y_test = load_dataset(
    "test"
)

print(
    "Test images:",
    X_test.shape
)

print(
    "Test labels:",
    y_test.shape
)


# ========================================
# NORMALIZE ORIGINAL DATA
# ========================================

print("\nNormalizing original images...")

X_train_full = normalize_images(
    X_train_full
)

X_test = normalize_images(
    X_test
)


# ========================================
# CREATE VALIDATION SET
# ========================================

print("\n========================================")
print("CREATING VALIDATION SET")
print("========================================")

X_train_original, X_val, y_train_original, y_val = (
    train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )
)


print(
    "Original training portion:",
    X_train_original.shape
)

print(
    "Validation portion:",
    X_val.shape
)


# ========================================
# LOAD CK+ DATA
# ========================================

print("\n========================================")
print("LOADING CK+ DATA")
print("========================================")


if not os.path.exists(
    CKPLUS_IMAGES_PATH
):

    print(
        "\nERROR: CK+ images file not found:"
    )

    print(
        CKPLUS_IMAGES_PATH
    )

    raise SystemExit


if not os.path.exists(
    CKPLUS_LABELS_PATH
):

    print(
        "\nERROR: CK+ labels file not found:"
    )

    print(
        CKPLUS_LABELS_PATH
    )

    raise SystemExit


X_ckplus = np.load(
    CKPLUS_IMAGES_PATH
)

y_ckplus_names = np.load(
    CKPLUS_LABELS_PATH,
    allow_pickle=True
)

if X_ckplus.ndim == 4:
    X_ckplus = np.squeeze(
        X_ckplus,
        axis=-1
    )

    
print(

    "CK+ images:",
    X_ckplus.shape
)

print(
    "CK+ labels:",
    y_ckplus_names.shape
)


# ========================================
# CONVERT CK+ STRING LABELS TO NUMBERS
# ========================================

print("\nConverting CK+ labels...")


emotion_to_index = {
    emotion: index
    for index, emotion in enumerate(
        EMOTIONS
    )
}


y_ckplus = np.array(
    [
        emotion_to_index[
            str(label)
        ]
        for label in y_ckplus_names
    ],
    dtype=np.int64
)


print(
    "CK+ numeric labels:",
    y_ckplus.shape
)


# ========================================
# COMBINE TRAINING DATA
# ========================================

print("\n========================================")
print("COMBINING TRAINING DATA")
print("========================================")


X_train = np.concatenate(
    [
        X_train_original,
        X_ckplus
    ],
    axis=0
)


y_train = np.concatenate(
    [
        y_train_original,
        y_ckplus
    ],
    axis=0
)


print(
    "Combined training images:",
    X_train.shape
)

print(
    "Combined training labels:",
    y_train.shape
)


# ========================================
# ADD CHANNEL DIMENSION
# ========================================

print("\nAdding grayscale channel dimension...")


X_train = add_channel_dimension(
    X_train
)

X_val = add_channel_dimension(
    X_val
)

X_test = add_channel_dimension(
    X_test
)


print(
    "Final training shape:",
    X_train.shape
)

print(
    "Final validation shape:",
    X_val.shape
)

print(
    "Final test shape:",
    X_test.shape
)


# ========================================
# SHOW TRAINING CLASS COUNTS
# ========================================

print("\n========================================")
print("COMBINED TRAINING CLASS COUNTS")
print("========================================")


for index, emotion in enumerate(
    EMOTIONS
):

    count = np.sum(
        y_train == index
    )

    print(
        f"{emotion:<10} {count}"
    )


# ========================================
# CLASS WEIGHTS
# ========================================

print("\n========================================")
print("CALCULATING CLASS WEIGHTS")
print("========================================")


classes = np.unique(
    y_train
)


class_weights_array = (
    compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
)


class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        classes,
        class_weights_array
    )
}


for class_index, weight in class_weights.items():

    print(
        f"{EMOTIONS[class_index]:<10} "
        f"{weight:.4f}"
    )


# ========================================
# DATA AUGMENTATION
# ========================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        )
    ],
    name="data_augmentation"
)


# ========================================
# BUILD CNN V4
# ========================================

print("\n========================================")
print("BUILDING CNN V4")
print("========================================")


model = tf.keras.Sequential(
    [

        tf.keras.layers.Input(
            shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                1
            )
        ),

        # --------------------------------
        # Data augmentation
        # --------------------------------

        data_augmentation,


        # --------------------------------
        # Block 1
        # --------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Dropout(
            0.20
        ),


        # --------------------------------
        # Block 2
        # --------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Dropout(
            0.25
        ),


        # --------------------------------
        # Block 3
        # --------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Dropout(
            0.30
        ),


        # --------------------------------
        # Feature extraction
        # --------------------------------

        tf.keras.layers.GlobalAveragePooling2D(),


        # --------------------------------
        # Dense layer
        # --------------------------------

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.40
        ),


        # --------------------------------
        # Output
        # --------------------------------

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ]
)


# ========================================
# COMPILE MODEL
# ========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


print("\nModel summary:")

model.summary()


# ========================================
# CALLBACKS
# ========================================

early_stopping = (
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
)


model_checkpoint = (
    tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
)


reduce_learning_rate = (
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
)


# ========================================
# TRAIN
# ========================================

print("\n========================================")
print("STARTING V4 TRAINING")
print("========================================")


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_learning_rate
    ],

    verbose=1
)


# ========================================
# SAVE FINAL MODEL
# ========================================

print("\n========================================")
print("SAVING FINAL MODEL")
print("========================================")


model.save(
    FINAL_MODEL_PATH
)


print(
    "\nFinal model saved to:"
)

print(
    FINAL_MODEL_PATH
)


print(
    "\nBest model saved to:"
)

print(
    BEST_MODEL_PATH
)


# ========================================
# TRAINING SUMMARY
# ========================================

print("\n========================================")
print("V4 TRAINING COMPLETE")
print("========================================")


best_val_accuracy = max(
    history.history[
        "val_accuracy"
    ]
)


best_val_loss = min(
    history.history[
        "val_loss"
    ]
)


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)


print(
    "\nIMPORTANT:"
)

print(
    "The original test set has NOT "
    "been used during training."
)

print(
    "Next step will be evaluating "
    "V4 on the untouched test set."
)