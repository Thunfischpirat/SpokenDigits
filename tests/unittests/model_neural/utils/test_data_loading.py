from pathlib import Path
from unittest import TestCase

import torch
import torchaudio
from torch import nn

from model_neural.utils.data_loading import MNISTAudio, collate_audio, ContrastiveTransformations

base_dir = Path(__file__).parent.parent.parent.parent.parent


class TestMNISTAudio(TestCase):
    """Test the MNISTAudio class."""

    def setUp(self):
        """Prepare variables to be used in tests."""
        self.annotations_dir = base_dir / "SDR_metadata.tsv"
        self.audio_dir = base_dir

    def test_mnist_audio(self):
        """Test that the MNISTAudio class returns audio and labels."""
        dataset = MNISTAudio(
            annotations_dir=self.annotations_dir, audio_dir=self.audio_dir, split="TRAIN"
        )
        audio, label = dataset[0]
        # train dataset has 2000 samples. See DataExploration.ipynb.
        self.assertTrue(len(dataset) == 2000)
        self.assertTrue(isinstance(audio, torch.Tensor))
        self.assertTrue(audio.shape[0] == 1)
        self.assertTrue(isinstance(label, torch.Tensor))

    def test_mnist_audio_mel(self):
        dataset = MNISTAudio(
            annotations_dir=self.annotations_dir,
            audio_dir=self.audio_dir,
            split="TRAIN",
            to_mel=True,
        )
        audio, label = dataset[0]
        self.assertTrue(audio.shape[0] == 39)

    def test_mnist_audio_speaker(self):
        dataset = MNISTAudio(
            annotations_dir=self.annotations_dir,
            audio_dir=self.audio_dir,
            split=["george", "lucas"],
        )
        audio, label = dataset[0]
        self.assertTrue(audio.shape[0] == 1)

    def test_spec_transforms(self):
        dataset = MNISTAudio(
            annotations_dir=self.annotations_dir,
            audio_dir=self.audio_dir,
            split="TRAIN",
            to_mel=True,
            spec_transforms=nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35),
            ),
        )
        audio, label = dataset[0]
        self.assertTrue(audio.shape[0] == 39)

    def test_contrastive_transforms(self):
        transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )
        spec_transforms = ContrastiveTransformations(transforms, n_views=2)
        dataset = MNISTAudio(
            annotations_dir=self.annotations_dir,
            audio_dir=self.audio_dir,
            split="TRAIN",
            to_mel=True,
            spec_transforms=spec_transforms,
        )
        audio, label = dataset[0]
        self.assertTrue(audio.shape[0] == 2)
        self.assertTrue(audio.shape[1] == 39)


class TestCollation(TestCase):
    def setUp(self):
        self.tensor_1 = torch.tensor([[1, 2, 3]])
        self.label_1 = torch.tensor([1, 0, 0, 0])
        self.tensor_2 = torch.tensor([[4, 5, 6, 7]])
        self.label_2 = torch.tensor([0, 1, 0, 0])
        self.tensor_3 = torch.tensor([[8, 9]])
        self.label_3 = torch.tensor([0, 0, 1, 0])

    def test_collate_audio(self):
        """Test that the pad_sequence function returns the correct shape."""
        batch = [
            (self.tensor_1, self.label_1),
            (self.tensor_2, self.label_2),
            (self.tensor_3, self.label_3),
        ]
        padded_batch, targets = collate_audio(batch)
        self.assertTrue(padded_batch.shape == (3, 1, 4))
        self.assertTrue(targets.shape == (3, 4))
