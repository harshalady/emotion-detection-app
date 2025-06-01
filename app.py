import streamlit as st
import librosa
import numpy as np
import pickle

# Load model
with open("emotiondetection.pkl", "rb") as f:
    model = pickle.load(f)

from emotiondetection import extract_features_full

# App UI
st.title("Emotion Detection from Voice")
uploaded_file = st.file_uploader("Upload your .wav file", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=22050)
    features = extract_features_full(y, sr)
    prediction = model.predict(features)[0]

    label_map = {
        0: "neutral",
        1: "happy",
        2: "sad",
        3: "angry",
        4: "fearful",
        5: "disgust"
    }
    st.success(f"Predicted Emotion: {label_map[prediction]}")
