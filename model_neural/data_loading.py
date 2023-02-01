"""
    Data loading utilities for the MNIST audio dataset.
    See also:
    https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
"""

import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class MNISTAudio(Dataset):
    def __init__(self, annotations_dir, audio_dir, split="TRAIN"):
        metadata = pd.read_csv(annotations_dir, sep="\t", header=0, index_col="Unnamed: 0")
        audio_labels = torch.tensor(metadata[metadata["split"] == split].label.values)
        self.audio_labels = one_hot(audio_labels, 10)
        self.audio_names = metadata[metadata["split"] == split].file.values
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_names[idx])
        audio, _ = torchaudio.load(audio_path)
        label = self.audio_labels[idx]
        return audio, label


def collate_audio(batch):
    """Collate a batch of audio samples and labels into a batch of tensors."""
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0.0)
    targets = torch.stack(targets)

    return tensors, targets
