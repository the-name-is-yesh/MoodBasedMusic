
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from services.music_recommender import (
    load_songs,
    recommend_songs
)


# ==========================================
# CONFIGURATION
# ==========================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMAGE_SIZE = 48
MODEL_PATH = "model/best_emotion_model_v3.keras"
CONFIDENCE_THRESHOLD = 0.50


# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Mood-Based Music",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Mood-Based Music Recommendation")

st.write(
    "Use your camera to detect your facial emotion "
    "and receive music recommendations based on your mood."
)

st.divider()


# ==========================================
# LOAD EMOTION MODEL
# ==========================================

@st.cache_resource
def load_emotion_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_emotion_model()


# ==========================================
# LOAD MUSIC SERVICE
# ==========================================

@st.cache_data
def load_song_data():
    songs = load_songs()
    return songs


songs = load_song_data()


# ==========================================
# FACE DETECTOR
# ==========================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# ==========================================
# EMOTION DETECTION
# ==========================================

st.header("😊 Emotion Detection")

st.write(
    "Take a picture using your camera "
    "to identify your facial emotion."
)

camera_image = st.camera_input(
    "Take a picture"
)


# Default values
detected_emotion = "Waiting..."
detected_confidence = "--"

recommendations = None
low_confidence = False


# ==========================================
# PROCESS CAMERA IMAGE
# ==========================================

if camera_image is not None:

    image_bytes = camera_image.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "Could not process the captured image."
        )

    else:

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:

            st.warning(
                "No face detected. Please face the camera "
                "directly and make sure your face is well lit."
            )

        else:

            # Select the largest detected face
            faces = sorted(
                faces,
                key=lambda face: face[2] * face[3],
                reverse=True
            )

            x, y, width, height = faces[0]

            face = gray_frame[
                y:y + height,
                x:x + width
            ]

            # Preprocess face for CNN
            face = cv2.resize(
                face,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            face = face.astype("float32") / 255.0

            face = np.expand_dims(
                face,
                axis=-1
            )

            face = np.expand_dims(
                face,
                axis=0
            )

            # Predict emotion
            predictions = model.predict(
                face,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            detected_emotion = EMOTIONS[
                predicted_index
            ]

            confidence = predictions[0][
                predicted_index
            ]

            detected_confidence = (
                f"{confidence * 100:.2f}%"
            )

            # Confidence check
            if confidence < CONFIDENCE_THRESHOLD:

                low_confidence = True
                recommendations = None

            else:

                low_confidence = False

                # Fetch live songs from online API
                recommendations = recommend_songs(
                    songs,
                    detected_emotion,
                    3
                )

            # Draw rectangle around face
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                frame_rgb,
                caption="Face Detection Result"
            )

            if low_confidence:

                st.warning(
                    "The emotion prediction has low confidence. "
                    "Try taking another picture with better "
                    "lighting and a clearer view of your face."
                )

            else:

                st.success(
                    "Face detected and emotion predicted!"
                )


# ==========================================
# DISPLAY EMOTION & CONFIDENCE
# ==========================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Detected Emotion",
        value=detected_emotion
    )

with col2:

    st.metric(
        label="Confidence",
        value=detected_confidence
    )


# ==========================================
# LIVE MUSIC RECOMMENDATIONS
# ==========================================

st.divider()

st.header("🎵 Recommended Music")


if low_confidence:

    st.info(
        "🎵 Music recommendations are hidden because "
        "the emotion prediction is not confident enough."
    )

elif recommendations is None:

    st.info(
        "📷 Take a picture to receive music recommendations."
    )

elif recommendations.empty:

    st.warning(
        f"No music recommendations found for "
        f"{detected_emotion}. Please try again."
    )

else:

    st.write(
        f"Your detected mood is **{detected_emotion}**."
    )

    st.write(
        "Here are some songs found online for your mood:"
    )

    # Display recommended songs
    for _, song in recommendations.iterrows():

        with st.container(border=True):

            # Album artwork
            album_art = song.get("album_art", "")

            if isinstance(album_art, str) and album_art:

                st.image(
                    album_art,
                    width=150
                )

            # Song information
            st.subheader(
                f"🎵 {song['title']}"
            )

            st.write(
                f"👤 **Artist:** {song['artist']}"
            )

            st.write(
                f"🎸 **Genre:** {song['genre']}"
            )

            # Audio preview
            preview_url = song.get(
                "preview_url",
                ""
            )

            if isinstance(preview_url, str) and preview_url:

                st.audio(
                    preview_url
                )

            else:

                st.caption(
                    "Preview not available for this song."
                )

            # Direct song link
            track_url = song.get(
                "track_url",
                ""
            )

            if isinstance(track_url, str) and track_url:

                st.link_button(
                    "🎧 Open Song",
                    track_url
                )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MBMS | Mood-Based Music Recommendation System"
)

st.caption(
    "Emotion detection is an AI prediction and "
    "may not accurately represent a person's actual mood."
)
