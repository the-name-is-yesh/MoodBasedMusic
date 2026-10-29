import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model.emotion_model import create_emotion_model
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

EPOCHS = 30

BATCH_SIZE = 64

VALIDATION_SIZE = 0.15

RANDOM_SEED = 42


# ============================================
# V3 MODEL FILES
# ============================================

BEST_MODEL_PATH = (
    "model/best_emotion_model_v3.keras"
)

FINAL_MODEL_PATH = (
    "model/emotion_model_v3.keras"
)


# ============================================
# REPRODUCIBILITY
# ============================================

tf.keras.utils.set_random_seed(
    RANDOM_SEED
)


# ============================================
# LOAD TRAINING DATA
# ============================================

print("Loading training dataset...")

X_train_full, y_train_full = load_dataset(
    "train"
)

print(
    "Full training images shape:",
    X_train_full.shape
)

print(
    "Full training labels shape:",
    y_train_full.shape
)


# ============================================
# LOAD TEST DATA
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
# NORMALIZE IMAGES
# ============================================

print("\nNormalizing images...")

X_train_full = (
    X_train_full.astype("float32")
    / 255.0
)

X_test = (
    X_test.astype("float32")
    / 255.0
)


# ============================================
# ADD CHANNEL DIMENSION
# ============================================

print("\nAdding channel dimension...")

X_train_full = np.expand_dims(
    X_train_full,
    axis=-1
)

X_test = np.expand_dims(
    X_test,
    axis=-1
)

print(
    "Full training data:",
    X_train_full.shape
)

print(
    "Test data:",
    X_test.shape
)


# ============================================
# TRAIN / VALIDATION SPLIT
# ============================================

print("\nCreating training and validation split...")

X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_train_full
    )
)


print(
    "Training split:",
    X_train.shape
)

print(
    "Validation split:",
    X_validation.shape
)

print(
    "Test set:",
    X_test.shape
)


# ============================================
# CLASS WEIGHTS
# ============================================

print("\nCalculating class weights...")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    enumerate(class_weights_array)
)


print("\nClass weights:")

for index, emotion in enumerate(EMOTIONS):

    print(
        f"{emotion:10} : "
        f"{class_weights[index]:.2f}"
    )


# ============================================
# CREATE CNN V3
# ============================================

print("\nCreating CNN V3...")

model = create_emotion_model()


# ============================================
# COMPILE
# ============================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================
# MODEL SUMMARY
# ============================================

print("\nCNN V3 model summary:")

model.summary()


# ============================================
# CREATE MODEL DIRECTORY
# ============================================

os.makedirs(
    "model",
    exist_ok=True
)


# ============================================
# CALLBACKS
# ============================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# ============================================
# START TRAINING
# ============================================

print("\n========================================")
print("STARTING CNN V3 TRAINING")
print("========================================")

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ],

    verbose=1
)


# ============================================
# SAVE FINAL MODEL
# ============================================

model.save(
    FINAL_MODEL_PATH
)


# ============================================
# TRAINING RESULTS
# ============================================

print("\n========================================")
print("CNN V3 TRAINING COMPLETED")
print("========================================")

print(
    "\nBest model saved at:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nFinal model saved at:"
)

print(
    FINAL_MODEL_PATH
)

print(
    "\nNumber of epochs completed:"
)

print(
    len(history.history["loss"])
)


# ============================================
# BEST VALIDATION ACCURACY
# ============================================

best_validation_accuracy = max(
    history.history["val_accuracy"]
)

print(
    "\nBest validation accuracy:"
)

print(
    f"{best_validation_accuracy * 100:.2f}%"
)