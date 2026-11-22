
import json
import urllib.parse
import urllib.request

import pandas as pd


EMOTION_SEARCH_TERMS = {
    "happy": "feel good upbeat pop",
    "sad": "sad emotional mellow songs",
    "angry": "calm relaxing music",
    "fear": "peaceful soothing music",
    "neutral": "chill soft pop music",
    "surprise": "energetic popular music",
    "disgust": "calm uplifting music",
}

FALLBACK_EMOTIONS = {
    "disgust": "angry"
}


def load_songs():
    """
    Return an empty DataFrame.
    Song data will be fetched online instead of loaded from CSV.
    """
    return pd.DataFrame()


def recommend_songs(songs, emotion, number_of_songs=3):
    """
    Fetch songs dynamically from the iTunes Search API
    based on the detected emotion.
    """

    emotion = emotion.lower()

    search_term = EMOTION_SEARCH_TERMS.get(
        emotion,
        "popular music"
    )

    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode({
        "term": search_term,
        "media": "music",
        "entity": "song",
        "limit": number_of_songs
    })

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "MBMS-Music-App"}
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = data.get("results", [])

        songs_list = []

        for track in results:
            songs_list.append({
                "title": track.get("trackName", "Unknown"),
                "artist": track.get("artistName", "Unknown"),
                "genre": track.get("primaryGenreName", "Unknown"),
                "album_art": track.get("artworkUrl100", ""),
                "preview_url": track.get("previewUrl", ""),
                "track_url": track.get("trackViewUrl", "")
            })

        return pd.DataFrame(songs_list)

    except Exception as e:
        print(f"Music API error: {e}")
        return pd.DataFrame()
