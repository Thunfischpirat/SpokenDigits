from pathlib import Path
from unittest import TestCase

import numpy as np

from model_neural.data_loading import MNISTAudio

base_dir = Path(__file__).parent.parent.parent
class TestMNISTAudio(TestCase):
    """ Test the MNISTAudio class. """

    def setUp(self):
        """ Prepare variables to be used in tests. """
        self.annotations_dir = base_dir / "SDR_metadata.tsv"
        self.audio_dir = base_dir

    def test_mnist_audio(self):
        """ Test that the MNISTAudio class returns audio and labels. """
        dataset = MNISTAudio(annotations_dir=self.annotations_dir, audio_dir=self.audio_dir, split="TRAIN")
        audio, label = dataset[0]
        # train dataset has 2000 samples. See DataExploration.ipynb.
        self.assertTrue(len(dataset) == 2000)
        self.assertTrue(isinstance(audio, np.ndarray))
        self.assertTrue(audio.ndim == 1)
        self.assertTrue(isinstance(label, np.int64))
