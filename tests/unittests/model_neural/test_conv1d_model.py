import torch
from model_neural.conv1d_model import conv1d_model
from unittest import TestCase

class TestConv1dModel(TestCase):
    def test_conv1d_model(self):
        """Test that the conv1d_model returns the correct shape."""
        model = conv1d_model()
        x = torch.randn(64, 1, 4830)
        y = model(x)
        self.assertTrue(y.shape == (64, 1, 10))