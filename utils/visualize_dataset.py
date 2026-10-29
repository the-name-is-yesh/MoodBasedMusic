import matplotlib.pyplot as plt

from preprocess import load_dataset


# ---------------------------------------------------------
# Emotion labels
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
# Load training dataset
# ---------------------------------------------------------

print("Loading training dataset...")

X_train, y_train = load_dataset("train")

print("Dataset loaded.")

print(
    "Training dataset shape:",
    X_train.shape
)

print(
    "Training labels shape:",
    y_train.shape
)


# ---------------------------------------------------------
# Count images for each emotion
# ---------------------------------------------------------

print("\n" + "=" * 40)
print("TRAINING DATASET DISTRIBUTION")
print("=" * 40)

emotion_counts = {}

for label, emotion in enumerate(EMOTIONS):

    count = (y_train == label).sum()

    emotion_counts[emotion] = count

    print(
        f"{emotion:10} : {count}"
    )


# ---------------------------------------------------------
# Display sample images
# ---------------------------------------------------------

plt.figure(figsize=(12, 8))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.imshow(
        X_train[i],
        cmap="gray"
    )

    emotion = EMOTIONS[y_train[i]]

    plt.title(emotion)

    plt.axis("off")


plt.tight_layout()

plt.show()


# ---------------------------------------------------------
# Create class distribution graph
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    emotion_counts.keys(),
    emotion_counts.values()
)

plt.xlabel("Emotion")

plt.ylabel("Number of Images")

plt.title("Training Dataset - Emotion Distribution")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()