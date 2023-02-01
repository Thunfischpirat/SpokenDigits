import os
import librosa
import pandas as pd
from torch.utils.data import Dataset

class MNISTAudio(Dataset):
    def __init__(self, annotations_dir, audio_dir, split="TRAIN"):
        metadata = pd.read_csv(annotations_dir, sep="\t", header=0, index_col="Unnamed: 0")
        self.audio_labels = metadata[metadata["split"] == split].label.values
        self.audio_names = metadata[metadata["split"] == split].file.values
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_names[idx])
        audio, _ = librosa.load(audio_path)
        label = self.audio_labels[idx]
        return audio, label