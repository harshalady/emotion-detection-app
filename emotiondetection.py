
import numpy as np
import librosa
import pickle

def extract_features_full(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    expected_length = 216
    if mfcc.shape[1] < expected_length:
        pad_width = expected_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width = ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :expected_length]
    mfcc_flat = mfcc.flatten()
    return mfcc_flat.reshape(1, -1)

# Assuming 'model' is your trained RandomForestClassifier model
# You would need to load your model here if running this script independently
# with open('emotiondetection.pkl', 'rb') as f:
#     model = pickle.load(f)

# Example usage (you would replace this with how you plan to use the functions)
# if __name__ == "__main__":
#     # Load an audio file
#     audio_path = 'your_audio_file.wav'
#     y, sr = librosa.load(audio_path, sr=22050)
#
#     # Extract features
#     features = extract_features_full(y, sr)
#
#     # Make a prediction (assuming model is loaded)
#     # predicted_label = model.predict(features)[0]
#
#     # label_map = {
#     #     0: "neutral",
#     #     1: "happy",
#     #     2: "sad",
#     #     3: "angry",
#     #     4: "fearful",
#     #     5: "disgust"
#     # }
#     # print("Predicted Emotion:", label_map[predicted_label])
