import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# EMOTION LABELS
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
# IMAGE SIZE
# ============================================================

IMAGE_SIZE = 48


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = "model/best_emotion_model.keras"


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

print("Loading emotion model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Emotion model loaded successfully!")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# OPEN WEBCAM
# ============================================================

print("\nStarting camera...")

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Error: Could not open camera.")

    exit()


print("Camera started successfully!")
print("Press 'q' to quit.")


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------------

    success, frame = camera.read()


    if not success:

        print("Error: Could not read camera frame.")

        break


    # --------------------------------------------------------
    # CONVERT FRAME TO GRAYSCALE
    # --------------------------------------------------------

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # DETECT FACES
    # --------------------------------------------------------

    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )


    # --------------------------------------------------------
    # PROCESS EACH DETECTED FACE
    # --------------------------------------------------------

    for (x, y, width, height) in faces:

        # ====================================================
        # CROP FACE
        # ====================================================

        face = gray_frame[
            y:y + height,
            x:x + width
        ]


        # ====================================================
        # RESIZE FACE TO 48 × 48
        # ====================================================

        face = cv2.resize(
            face,
            (IMAGE_SIZE, IMAGE_SIZE)
        )


        # ====================================================
        # NORMALIZE PIXEL VALUES
        # ====================================================

        face = face.astype(
            "float32"
        ) / 255.0


        # ====================================================
        # ADD CHANNEL DIMENSION
        # Shape:
        # (48, 48)
        #       ↓
        # (48, 48, 1)
        # ====================================================

        face = np.expand_dims(
            face,
            axis=-1
        )


        # ====================================================
        # ADD BATCH DIMENSION
        # Shape:
        # (48, 48, 1)
        #       ↓
        # (1, 48, 48, 1)
        # ====================================================

        face = np.expand_dims(
            face,
            axis=0
        )


        # ====================================================
        # PREDICT EMOTION
        # ====================================================

        predictions = model.predict(
            face,
            verbose=0
        )


        # ====================================================
        # FIND HIGHEST PROBABILITY
        # ====================================================

        predicted_index = np.argmax(
            predictions[0]
        )


        # ====================================================
        # GET EMOTION NAME
        # ====================================================

        predicted_emotion = EMOTIONS[
            predicted_index
        ]


        # ====================================================
        # GET CONFIDENCE
        # ====================================================

        confidence = predictions[0][
            predicted_index
        ]


        # ====================================================
        # DRAW FACE RECTANGLE
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )


        # ====================================================
        # CREATE DISPLAY TEXT
        # ====================================================

        emotion_text = (
            f"{predicted_emotion} "
            f"{confidence * 100:.1f}%"
        )


        # ====================================================
        # DISPLAY EMOTION ABOVE FACE
        # ====================================================

        cv2.putText(
            frame,
            emotion_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Live Emotion Detection",
        frame
    )


    # ========================================================
    # CHECK FOR 'q'
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# RELEASE CAMERA
# ============================================================

camera.release()


cv2.destroyAllWindows()


print("\nCamera closed.")
print("Emotion detection stopped.")