from pathlib import Path
import librosa
import numpy as np
from sklearn import preprocessing
import math

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

    # for numerical stability added this line
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    # 20 * log10 to convert to log scale
    log_mel_features = 20 * np.log10(mel_features)

    # feature scaling
    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)

    return scaled_log_mel_features


def downsample_spectrogram(spectrogram, num_frames):
    """
    Given a mel-scaled representation of a signal, return a fixed-size
    representation of the signal as numpy array of size (1, num_frames)
    by taking num_frames equal sized chunks of the signal and averaging
    them over the frequency axis.
    """

    signal_length = spectrogram.shape[1]
    window_size = int(math.ceil(spectrogram.shape[1] / num_frames))
    padding = window_size - (signal_length % window_size)

    spectrogram_downsampled = np.zeros((spectrogram.shape[0], num_frames))

    # pad signal with zeros
    spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), 'constant')


    for section in range(num_frames):
        spectrogram_downsampled[:, section] = np.mean(spectrogram[:, section*window_size:(section+1)*window_size], axis=1)

    spectrogram_downsampled = np.reshape(spectrogram_downsampled, (1, -1))
    return spectrogram_downsampled

if __name__ == "__main__":

        # load audio file
        audio_path = parent_dir / "speech_data/2_nicolas_16.wav"
        signal, sr = librosa.load(audio_path)

        # extract mel-scaled representation
        scaled_log_mel_features = extract_melspectrogram(signal, sr, num_mels=13)

        # extract 1d representation
        mel_features_1d = downsample_spectrogram(scaled_log_mel_features, num_frames=10)

        print(mel_features_1d.shape)
