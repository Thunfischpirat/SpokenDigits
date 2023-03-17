"""
    Data loading utilities for the MNIST audio dataset.
    See also:
    https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
"""

import os
from pathlib import Path
from typing import List, Tuple, Union, Callable

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

base_dir = Path(__file__).parent.parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"


class MNISTAudio(Dataset):
    def __init__(
        self,
        annotations_dir: str,
        audio_dir: str,
        split: Union[str, List[str]] = "TRAIN",
        to_mel: bool = False,
        audio_transforms: List[Tuple[Callable, float]] = None,
        spec_transforms: List[nn.Module] = None
    ):
        """
        Wrapper for MNIST audio dataset.

        Args:
            annotations_dir: Path to the annotations file.
            audio_dir: Path to the audio directory.
            split: Which split of the data to use. One of "TRAIN", "DEV", "TEST"
                   or a set of speaker names such as ["george", "lucas"].
            to_mel: Whether to convert the raw audio to a mel spectrogram first.
            audio_transforms: A list of tuples of functions from torchaudio.functional to perform raw audio transforms
                                with execution probability. Example: [(torchaudio.functional.contrast, 0.5)]
            spec_transforms: A list of spectrogram transforms from torchaudio.transforms.
                             Example: [torchaudio.transforms.FrequencyMasking(freq_mask_param=15)]
        """
        metadata = pd.read_csv(annotations_dir, sep="\t", header=0, index_col="Unnamed: 0")
        if isinstance(split, str):
            audio_labels = torch.tensor(metadata[metadata["split"] == split].label.values)
            self.audio_labels = audio_labels
            self.audio_names = metadata[metadata["split"] == split].file.values
        else:
            audio_labels = torch.tensor(metadata[metadata["speaker"].isin(split)].label.values)
            self.audio_labels = audio_labels
            self.audio_names = metadata[metadata["speaker"].isin(split)].file.values
        self.audio_dir = audio_dir
        self.to_mel = to_mel
        self.audio_transforms = audio_transforms
        self.spec_transforms = spec_transforms

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_names[idx])
        audio, sample_rate = torchaudio.load(audio_path)
        if self.audio_transforms is not None:
            for transform, exec_prob in self.audio_transforms:
                if torch.rand(1) < exec_prob:
                    audio = transform(audio)
        if self.to_mel:
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate, n_fft=200, hop_length=80, n_mels=39
            )(audio)
            mel_log = 20 * torch.log(mel_spectrogram + 1e-9)
            mel_normalized = mel_log - mel_log.mean(1, keepdim=True) / (
                mel_log.std(1, keepdim=True) + 1e-10
            )
            audio = mel_normalized
            # Apply random transforms to the spectrogram
            if self.spec_transforms is not None:
                for transform in self.spec_transforms:
                    audio = transform(audio)
            audio = audio.squeeze(0)
        else:
            audio = audio - audio.mean() / (audio.std() + 1e-10)
        label = self.audio_labels[idx]
        return audio, label


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_audio(batch):
    """Collate a batch of audio samples and labels into a batch of tensors."""
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def create_loaders(loader_names: Union[List[str], List[List[str]]], to_mel: bool):
    """
    Create a dictionary of PyTorch DataLoader objects for the MNIST audio dataset for each split of the data.

    Args:
        loader_names: A list of the splits or list of a list of speaker names to use.
        to_mel: Whether to convert the raw audio to a mel spectrogram first.
    """
    loaders = dict(
        [
            (
                # If speaker names are given: use concatenation of the first two letters of the speaker name as key.
                split if isinstance(split, str) else "_".join([name[:2] for name in split]),
                DataLoader(
                    MNISTAudio(
                        annotations_dir=annotations_dir,
                        audio_dir=base_dir,
                        split=split,
                        to_mel=to_mel,
                    ),
                    batch_size=32,
                    collate_fn=collate_audio,
                    shuffle=True,
                ),
            )
            for split in loader_names
        ]
    )
    return loaders


if __name__ == "__main__":
    min_length = float("inf")
    for audio, label in MNISTAudio(
        annotations_dir,
        base_dir,
        split=["jackson", "lucas", "nicolas", "yweweler", "theo", "george"],
    ):
        if audio.shape[1] < min_length:
            min_length = audio.shape[1]
    print(min_length)
