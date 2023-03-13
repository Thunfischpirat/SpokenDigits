from unittest import TestCase

import torch
from model_neural.conv1d_model import Conv1dModel, Conv1dMelModel


class TestConv1dModel(TestCase):
    def test_conv1d_model(self):
        """Test that the conv1d_model returns the correct shape."""
        model = Conv1dModel()
        x = torch.randn(64, 1, 4830)
        y = model(x)
        self.assertTrue(y.shape == (64, 1, 10))

    def test_conv1d_mel_model(self):
        """Test that the conv1d_model returns the correct shape."""
        model = Conv1dMelModel()
        x = torch.randn(64, 39, 80)
        y = model(x)
        self.assertTrue(y.shape == (64, 1, 10))
