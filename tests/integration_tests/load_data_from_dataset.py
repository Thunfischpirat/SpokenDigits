from pathlib import Path

import torchaudio
from torch import nn

from model_neural.utils.data_loading import MNISTAudio, collate_audio, ContrastiveTransformations, collate_contrastive
from torch.utils.data import DataLoader

from unittest import TestCase

base_dir = Path(__file__).parent.parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"

class VanillaAudio(TestCase):
    """Test the vanilla MNISTAudio class."""
    def test_vanilla_audio(self):
        dataset = MNISTAudio(annotations_dir=annotations_dir, audio_dir=base_dir, split="TRAIN")

        train_dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_audio, shuffle=True)

        train_features, train_labels = next(iter(train_dataloader))

        self.assertTrue(train_features.shape[0] == 64)


class ContrastiveAudio(TestCase):
    """Test the MNISTAudio class when configured for contrastive loss based training."""
    def test_contrastive_audio(self):
        transforms = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )
        spec_transforms = ContrastiveTransformations(transforms, n_views=2)
        dataset = MNISTAudio(
            annotations_dir=annotations_dir,
            audio_dir=base_dir,
            split="TRAIN",
            spec_transforms=spec_transforms,
            to_mel=True,
        )

        train_dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_contrastive, shuffle=True)
        train_features, train_labels = next(iter(train_dataloader))
        self.assertTrue(train_features.shape[0] == 128)
        self.assertTrue(train_features.shape[1] == 39)

