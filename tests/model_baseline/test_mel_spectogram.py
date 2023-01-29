from unittest import TestCase

import numpy as np
from model_baseline.mel_spectogram import (create_features,
                                           downsample_spectrogram)


class TestDownsampleSpectrum(TestCase):
    """Test the downsample_spectrogram function."""

    def setUp(self) -> None:
        """Prepare variables to be used in tests."""
        self.spectrogram = 2 * np.array(
            [
                [4.0, 3.0, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [4.0, 0.0, 1.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0],
            ]
        )
        self.expected_result = np.array([[7.0, 4.0, 1.0, 2.0, 1.0, 4.0, 1.0, 3.0, 3.0, 0.0]])

    def test_downsample_spectrogram(self):
        """Test that the downsample_spectrogram function returns the correct shape and value."""
        representation_1d = downsample_spectrogram(self.spectrogram, num_frames=5)
        self.assertEqual(representation_1d.shape, (1, 10))
        self.assertTrue(np.array_equal(representation_1d, self.expected_result))


class TestCreateFeatures(TestCase):
    """Test the create_features function."""

    def setUp(self) -> None:
        self.num_mels = 13
        self.num_frames = 10
        self.split = "DEV"

    def test_create_features(self):
        """Test that the create_features function returns the correct shape and no entry is nan."""
        features, labels = create_features(self.split, self.num_mels, self.num_frames)
        self.assertEqual(features.shape[1], 130)
        self.assertFalse(np.isnan(features).any())
