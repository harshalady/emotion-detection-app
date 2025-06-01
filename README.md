# 🎙️ Voice Emotion Detection App

This Streamlit app predicts the emotion conveyed in a `.wav` audio file using machine learning. It uses MFCC features and a trained Random Forest Classifier to classify audio into six emotions: **neutral, happy, sad, angry, fearful, disgust**.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://emotionsdetectionapp.streamlit.app/)  
---

## 🧠 Model Overview

- **Feature Extraction**: MFCC (Mel Frequency Cepstral Coefficients)
- **Classifier**: Random Forest (sklearn)
- **Accuracy**: ~99.75% on validation set
- **Training Dataset**: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

---

## 📁 Upload Format

- Only `.wav` files are supported
- Keep audio clips short (1–4 seconds) and focused on a single emotion

---

## 🛠️ How It Works

1. Upload your `.wav` file
2. App extracts MFCC features
3. Model predicts the emotion
4. Output is displayed instantly

---

## 🧪 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/voice-emotion-detection.git
cd voice-emotion-detection
