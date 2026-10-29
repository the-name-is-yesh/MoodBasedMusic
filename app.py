import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from urllib.parse import quote_plus

from services.music_recommender import (
    load_songs,
    recommend_songs
)


# ============================================================
# EMOTION SETTINGS
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

IMAGE_SIZE = 48

MODEL_PATH = "model/best_emotion_model_v3.keras"

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Mood-Based Music",
    page_icon="🎵",
    layout="centered"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🎵 Mood-Based Music Recommendation"
)

st.write(
    "Use your camera to detect your facial emotion "
    "and receive music recommendations based on your mood."
)

st.divider()


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_emotion_model()


# ============================================================
# LOAD SONG DATASET
# ============================================================

@st.cache_data
def load_song_data():

    songs = load_songs()

    return songs


songs = load_song_data()


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# EMOTION DETECTION
# ============================================================

st.header(
    "😊 Emotion Detection"
)

st.write(
    "Take a picture using your camera."
)

camera_image = st.camera_input(
    "Take a picture"
)


# ============================================================
# DEFAULT VALUES
# ============================================================

detected_emotion = "Waiting..."

detected_confidence = "--"

recommendations = None

low_confidence = False


# ============================================================
# PROCESS CAMERA IMAGE
# ============================================================

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

        # ----------------------------------------------------
        # Convert image to grayscale
        # ----------------------------------------------------

        gray_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        # ----------------------------------------------------
        # No face detected
        # ----------------------------------------------------

        if len(faces) == 0:

            st.warning(
                "No face detected. "
                "Please face the camera directly "
                "and make sure your face is well lit."
            )


        # ----------------------------------------------------
        # Face detected
        # ----------------------------------------------------

        else:

            # Select the largest detected face.

            faces = sorted(
                faces,
                key=lambda face: face[2] * face[3],
                reverse=True
            )

            x, y, width, height = faces[0]


            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            face = gray_frame[
                y:y + height,
                x:x + width
            ]


            # ------------------------------------------------
            # Resize face to 48 x 48
            # ------------------------------------------------

            face = cv2.resize(
                face,
                (IMAGE_SIZE, IMAGE_SIZE)
            )


            # ------------------------------------------------
            # Normalize pixel values
            # ------------------------------------------------

            face = face.astype(
                "float32"
            ) / 255.0


            # ------------------------------------------------
            # Add channel dimension
            # ------------------------------------------------

            face = np.expand_dims(
                face,
                axis=-1
            )


            # ------------------------------------------------
            # Add batch dimension
            # ------------------------------------------------

            face = np.expand_dims(
                face,
                axis=0
            )


            # ------------------------------------------------
            # Predict emotion
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Get confidence
            # ------------------------------------------------

            confidence = predictions[0][
                predicted_index
            ]


            detected_confidence = (
                f"{confidence * 100:.2f}%"
            )


            # ------------------------------------------------
            # Check confidence level
            # ------------------------------------------------

            if confidence < CONFIDENCE_THRESHOLD:

                low_confidence = True

                recommendations = None

            else:

                low_confidence = False

                recommendations = recommend_songs(
                    songs,
                    detected_emotion,
                    3
                )


            # ------------------------------------------------
            # Draw face rectangle
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # Convert BGR to RGB
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # Display image
            # ------------------------------------------------

            st.image(
                frame_rgb,
                caption="Face Detection Result"
            )


            # ------------------------------------------------
            # Show detection status
            # ------------------------------------------------

            if low_confidence:

                st.warning(
                    "The emotion prediction has low "
                    "confidence. Try taking another "
                    "picture with better lighting and "
                    "a clearer view of your face."
                )

            else:

                st.success(
                    "Face detected and emotion predicted!"
                )


# ============================================================
# EMOTION RESULT
# ============================================================

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


# ============================================================
# MUSIC RECOMMENDATIONS
# ============================================================

st.divider()

st.header(
    "🎵 Recommended Music"
)


# ============================================================
# LOW CONFIDENCE
# ============================================================

if low_confidence:

    st.info(
        "🎵 Music recommendations are hidden "
        "because the emotion prediction is "
        "not confident enough."
    )


# ============================================================
# NO PHOTO YET
# ============================================================

elif recommendations is None:

    st.info(
        "📷 Take a picture to receive "
        "music recommendations."
    )


# ============================================================
# NO SONGS AVAILABLE
# ============================================================

elif recommendations.empty:

    st.warning(
        f"No music recommendations are "
        f"available for {detected_emotion}."
    )


# ============================================================
# SHOW RECOMMENDATIONS
# ============================================================

else:

    # --------------------------------------------------------
    # Disgust fallback
    # --------------------------------------------------------

    if detected_emotion == "disgust":

        st.write(
            "Your detected emotion is "
            "**disgust**."
        )

        st.info(
            "There are currently no disgust "
            "songs in our dataset, so we are "
            "using angry/intense songs as a "
            "fallback."
        )

    else:

        st.write(
            f"Your detected mood is "
            f"**{detected_emotion}**."
        )

        st.write(
            "Here are some songs selected "
            "for your mood:"
        )


    # --------------------------------------------------------
    # Display song cards
    # --------------------------------------------------------

    for _, song in recommendations.iterrows():

        with st.container(
            border=True
        ):

            st.subheader(
                f"🎵 {song['title']}"
            )

            st.write(
                f"👤 **Artist:** {song['artist']}"
            )

            st.write(
                f"🎸 **Genre:** {song['genre']}"
            )


            # ------------------------------------------------
            # Create YouTube search link
            # ------------------------------------------------

            search_text = (
                f"{song['title']} "
                f"{song['artist']}"
            )

            encoded_search = quote_plus(
                search_text
            )

            youtube_url = (
                "https://www.youtube.com/results"
                f"?search_query={encoded_search}"
            )


            # ------------------------------------------------
            # Listen button
            # ------------------------------------------------

            st.link_button(
                "▶️ Listen / Search on YouTube",
                youtube_url
            )