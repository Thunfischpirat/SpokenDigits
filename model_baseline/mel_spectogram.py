import math
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

parent_dir = Path(__file__).parent.parent


def extract_melspectrogram(signal, sr, num_mels):
    """
    Given a time series speech signal (.wav), sampling rate (sr),
    and the number of mel coefficients, return a mel-scaled
    representation of the signal as numpy array.
    """

    mel_features = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=200,  # with sampling rate = 8000, this corresponds to 25 ms
        hop_length=80,  # with sampling rate = 8000, this corresponds to 10 ms
        n_mels=num_mels,  # number of frequency bins, use either 13 or 39
        fmin=50,  # min frequency threshold
        fmax=sr / 2,  # max frequency threshold, set to SAMPLING_RATE/2
    )

    return mel_features


def downsample_spectrogram(spectrogram, num_frames):
    """
    Given a mel-scaled representation of a signal, return a fixed-size
    representation of the signal as numpy array of size (1, num_frames)
    by taking num_frames equal sized chunks of the signal and averaging
    them over the frequency axis.
    """

    signal_length = spectrogram.shape[1]
    window_size = int(math.ceil(spectrogram.shape[1] / num_frames))
    padding = num_frames * window_size - signal_length

    spectrogram_downsampled = np.zeros((spectrogram.shape[0], num_frames))

    # pad signal with zeros
    spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), "constant")

    for section in range(num_frames):
        spectrogram_downsampled[:, section] = np.mean(
            spectrogram[:, section * window_size : (section + 1) * window_size], axis=1
        )

    spectrogram_downsampled = np.reshape(spectrogram_downsampled, (1, -1))
    return spectrogram_downsampled


def create_features(split, num_mels=13, num_frames=10):
    """
    Given a split of the dataset (train, val, test), return a numpy array
    of mel-scaled representations of the signals in the split.
    """

    # Load from lazy loading if possible.
    if Path(f"data/{split.lower()}_features.npy").exists():
        features = np.load(f"data/{split.lower()}_features.npy")
        labels = np.load(f"data/{split.lower()}_labels.npy")
        return features, labels

    sdr_df = pd.read_csv(
        parent_dir / "SDR_metadata.tsv", sep="\t", header=0, index_col="Unnamed: 0"
    )
    filenames = sdr_df[sdr_df["split"] == split].file.values
    audio_samples = [librosa.load(parent_dir / sample) for sample in filenames]

    features = None
    for i, (audio, sr) in enumerate(audio_samples):
        mel_features = extract_melspectrogram(audio, sr, num_mels)
        mel_features = downsample_spectrogram(mel_features, num_frames)
        if i == 0:
            features = mel_features
        else:
            features = np.vstack((features, mel_features))

    labels = sdr_df[sdr_df["split"] == split].label.values

    # Save for lazy loading
    os.makedirs("data", exist_ok=True)
    np.save(f"data/{split.lower()}_features.npy", features)
    np.save(f"data/{split.lower()}_labels.npy", labels)

    return features, labels
