# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm  # Make sure lightgbm is imported

# --- Page Configuration (Set this at the top) ---
st.set_page_config(
    page_title="Spotify Genre Predictor",
    page_icon="🎵",
    layout="wide"
)

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("data/dataset.csv")
        if 'playlist_genre' not in df.columns or 'track_artist' not in df.columns or 'track_name' not in df.columns:
            st.warning("The dataset must have 'playlist_genre', 'track_artist', and 'track_name' columns.")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error("The dataset file 'data/dataset.csv' was not found.")
        return pd.DataFrame()


# --- Load The Models and Objects ---
@st.cache_resource
def load_resources():
    try:
        with open('spotify_genre_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('label_encoder.pkl', 'rb') as encoder_file:
            label_encoder = pickle.load(encoder_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, label_encoder, scaler
    except FileNotFoundError:
        st.error("Model or supporting files not found. Please ensure spotify_genre_model.pkl, label_encoder.pkl, and scaler.pkl are in the same folder as app.py.")
        return None, None, None

model, label_encoder, scaler = load_resources()

# --- Genre Profile Data ---
GENRE_PROFILES = {
    'edm': [0.655, 0.802, 5, -5.42, 1, 0.086, 0.081, 0.218, 0.211, 0.400, 125.7, 222540],
    'latin': [0.713, 0.708, 5, -6.26, 1, 0.102, 0.210, 0.044, 0.180, 0.605, 118.6, 216863],
    'pop': [0.639, 0.701, 5, -6.31, 1, 0.073, 0.170, 0.059, 0.176, 0.503, 120.7, 217768],
    'r&b': [0.670, 0.590, 5, -7.86, 1, 0.116, 0.259, 0.028, 0.175, 0.531, 114.2, 237599],
    'rap': [0.718, 0.650, 5, -7.04, 1, 0.197, 0.192, 0.075, 0.191, 0.505, 120.6, 214163],
    'rock': [0.520, 0.732, 5, -7.58, 1, 0.057, 0.145, 0.062, 0.203, 0.537, 124.9, 248576]
}
FEATURE_ORDER = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
]

# --- Initialize Session State ---
# This dictionary will hold the current values of all sliders
if 'slider_values' not in st.session_state:
    st.session_state.slider_values = dict(zip(FEATURE_ORDER, GENRE_PROFILES['pop']))

# This will track if we are in demo mode
if 'loaded_profile' not in st.session_state:
    st.session_state.loaded_profile = None

# --- Callback Functions ---
def set_profile_features():
    """Called when the dropdown selection changes to enter Demo Mode."""
    profile_name = st.session_state.genre_profile_key
    if profile_name != "---Select a Genre Profile---":
        st.session_state.loaded_profile = profile_name
        profile_values = GENRE_PROFILES[profile_name]
        # Update the slider values in session state
        for i, feature in enumerate(FEATURE_ORDER):
            st.session_state.slider_values[feature] = profile_values[i]
    else:
        st.session_state.loaded_profile = None

def slider_change_callback():
    """Called when any slider is manually changed. This exits Demo Mode."""
    st.session_state.loaded_profile = None
    # No need to reset the dropdown key, as the user might want to re-select
    # The important part is setting loaded_profile to None

# --- UI Layout ---
st.title("🎵 Spotify Song Genre Predictor")
st.write(
    "This app uses a Machine Learning model to predict the genre of a song. "
    "Use the dropdown to load a genre profile for a demo, or move the sliders for a live prediction."
)

st.selectbox(
    "Load a Genre Profile (Demo Mode):",
    options=["---Select a Genre Profile---"] + list(GENRE_PROFILES.keys()),
    key="genre_profile_key",
    on_change=set_profile_features
)

st.header("Audio Features")
col1, col2 = st.columns(2)

# --- THE FIX IS HERE: We use a separate key for each slider and pass its current value ---
with col1:
    st.session_state.slider_values['danceability'] = st.slider("Danceability", 0.0, 1.0, st.session_state.slider_values['danceability'], on_change=slider_change_callback)
    st.session_state.slider_values['energy'] = st.slider("Energy", 0.0, 1.0, st.session_state.slider_values['energy'], on_change=slider_change_callback)
    st.session_state.slider_values['loudness'] = st.slider("Loudness (dB)", -60.0, 0.0, st.session_state.slider_values['loudness'], on_change=slider_change_callback)
    st.session_state.slider_values['speechiness'] = st.slider("Speechiness", 0.0, 1.0, st.session_state.slider_values['speechiness'], on_change=slider_change_callback)
    st.session_state.slider_values['acousticness'] = st.slider("Acousticness", 0.0, 1.0, st.session_state.slider_values['acousticness'], on_change=slider_change_callback)
    st.session_state.slider_values['instrumentalness'] = st.slider("Instrumentalness", 0.0, 1.0, st.session_state.slider_values['instrumentalness'], on_change=slider_change_callback)

with col2:
    st.session_state.slider_values['liveness'] = st.slider("Liveness", 0.0, 1.0, st.session_state.slider_values['liveness'], on_change=slider_change_callback)
    st.session_state.slider_values['valence'] = st.slider("Valence", 0.0, 1.0, st.session_state.slider_values['valence'], on_change=slider_change_callback)
    st.session_state.slider_values['tempo'] = st.slider("Tempo (BPM)", 0.0, 250.0, st.session_state.slider_values['tempo'], on_change=slider_change_callback)
    st.session_state.slider_values['duration_ms'] = st.slider("Duration (ms)", 0, 600000, st.session_state.slider_values['duration_ms'], on_change=slider_change_callback)
    st.session_state.slider_values['key'] = st.slider("Key", 0, 11, st.session_state.slider_values['key'], on_change=slider_change_callback)
    st.session_state.slider_values['mode'] = st.slider("Mode (Major=1, Minor=0)", 0, 1, st.session_state.slider_values['mode'], on_change=slider_change_callback)

# --- Prediction Logic ---
if st.button("Predict Genre", type="primary"):
    song_df = load_dataset()

    if model:
        st.subheader("Prediction Result")
        
        # If a profile is loaded (and sliders haven't been touched), we're in Demo Mode
        if st.session_state.loaded_profile:
            prediction_genre = st.session_state.loaded_profile
            st.success(f"The selected genre profile is: **{prediction_genre.upper()}**")
            st.balloons()
        else:
            # If no profile is loaded, it means a slider was moved, so do a real prediction
            st.info("Live prediction based on current slider values...")
            features_df = pd.DataFrame([st.session_state.slider_values])
            features_df = features_df[FEATURE_ORDER]
            
            scaled_features = scaler.transform(features_df)
            prediction_encoded = model.predict(scaled_features)
            prediction_genre = label_encoder.inverse_transform(prediction_encoded)[0]
            st.success(f"The model predicts the genre is: **{prediction_genre.upper()}**")

            # --- Show Related Songs ---
        if not song_df.empty:
            matching_songs = song_df[song_df['playlist_genre'].str.lower() == prediction_genre.lower()]
            if not matching_songs.empty:
                st.subheader(f"🎧 Songs in the {prediction_genre.upper()} Genre")
                st.dataframe(matching_songs[['track_name', 'track_artist']].sample(n=min(5, len(matching_songs))).reset_index(drop=True))
            else:
                st.info(f"No songs found in the dataset for genre: {prediction_genre}")

    else:
        st.error("Model is not loaded. Cannot make a prediction.")