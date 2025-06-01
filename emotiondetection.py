
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


