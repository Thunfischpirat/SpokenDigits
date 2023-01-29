from unittest import TestCase
import numpy as np

from model_baseline.mel_spectogram import downsample_spectrogram


class TestDownsampleSpectrum(TestCase):
    """Test the mel_1d_representation function."""

    def setUp(self) -> None:
        """Prepare variables to be used in tests."""
        self.spectrogram = 2 * np.array(
            [[4., 3., 4., 0., 0., 1., 1., 1., 1.], [4., 0., 1., 0., 0., 3., 1., 2., 0.]]
        )
        self.expected_result = np.array([[7., 4., 1., 2., 1., 4., 1., 3., 3., 0.]])

    def test_downsample_spectrogram(self):
        """Test that the mel_1d_representation function returns the correct shape."""
        representation_1d = downsample_spectrogram(self.spectrogram, num_frames=5)
        self.assertEqual(representation_1d.shape, (1, 10))
        self.assertTrue(np.array_equal(representation_1d, self.expected_result))
