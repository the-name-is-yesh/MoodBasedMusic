import os

import pandas as pd


SONG_DATASET_PATH = "data/songs.csv"


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


FALLBACK_EMOTIONS = {
    "disgust": "angry"
}


def load_songs():
    """
    Load the song dataset from CSV.
    """

    if not os.path.exists(SONG_DATASET_PATH):

        raise FileNotFoundError(
            f"Song dataset not found: {SONG_DATASET_PATH}"
        )

    songs = pd.read_csv(
        SONG_DATASET_PATH
    )

    return songs


def get_songs_by_emotion(
    songs,
    emotion
):
    """
    Return songs that match the detected emotion.
    """

    filtered_songs = songs[
        songs["emotion"].str.lower()
        == emotion.lower()
    ]

    return filtered_songs


def recommend_songs(
    songs,
    emotion,
    number_of_songs=3
):
    """
    Recommend random songs based on emotion.

    If no songs exist for an emotion,
    a fallback emotion is used.
    """

    matching_songs = get_songs_by_emotion(
        songs,
        emotion
    )

    # Check whether a fallback emotion is needed.
    if matching_songs.empty:

        fallback_emotion = FALLBACK_EMOTIONS.get(
            emotion.lower()
        )

        if fallback_emotion is not None:

            matching_songs = get_songs_by_emotion(
                songs,
                fallback_emotion
            )

    # If there are still no songs,
    # return an empty DataFrame.
    if matching_songs.empty:

        return matching_songs

    # Randomly select songs.
    number_to_select = min(
        number_of_songs,
        len(matching_songs)
    )

    recommended_songs = matching_songs.sample(
        n=number_to_select
    )

    return recommended_songs.reset_index(
        drop=True
    )


if __name__ == "__main__":

    print("Loading song dataset...")

    songs = load_songs()

    print(
        "\nSong dataset loaded successfully!"
    )

    print(
        "\nTotal songs:",
        len(songs)
    )

    print(
        "\n========================================"
    )

    print(
        "TESTING RANDOM RECOMMENDATIONS"
    )

    print(
        "========================================"
    )

    for emotion in EMOTIONS:

        print(
            f"\nEmotion: {emotion.upper()}"
        )

        recommendations = recommend_songs(
            songs,
            emotion,
            3
        )

        if recommendations.empty:

            print(
                "No songs found."
            )

            continue

        for _, song in recommendations.iterrows():

            print(
                f"  {song['title']} - "
                f"{song['artist']}"
            )