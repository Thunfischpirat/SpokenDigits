from typing import Union

import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class conv1d_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pool_size=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class Conv1dModel(nn.Module):
    """Convolutional neural network for 1D data."""

    def __init__(
        self,
        n_input=1,
        n_channel=32,
        n_output=10,
        initial_kernel_size=60,
        initial_stride=8,
    ):
        super().__init__()
        self.conv_block1 = conv1d_block(n_input, n_channel, initial_kernel_size, initial_stride)
        self.conv_block2 = conv1d_block(n_channel, n_channel)
        self.conv_block3 = conv1d_block(n_channel, 2 * n_channel)
        self.conv_block4 = conv1d_block(2 * n_channel, 2 * n_channel)
        if n_output is None:
            self.fc1 = nn.Identity()
        else:
            self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


class Conv1dMelModel(nn.Module):
    """Convolutional neural network for 1D data."""

    def __init__(
        self,
        n_input: int = 39,
        n_channel: int = 32,
        n_output: Union[int, None] = 10,
        initial_kernel_size: int = 16,
        initial_stride: int = 3,
    ):
        """
        Args:
            n_input: Number of input channels.
            n_channel: Number of channels in the first convolutional layer.
                       Subsequent layers will have 2 times the number of channels.
            n_output: Number of output classes. Can be set to None to use the model for contrastive learning.
            initial_kernel_size: Kernel size of the first convolutional layer.
            initial_stride: Stride of the first convolutional layer.
        """
        super().__init__()
        self.conv_block1 = conv1d_block(
            n_input, n_channel, initial_kernel_size, initial_stride, pool_size=2
        )
        self.conv_block2 = conv1d_block(n_channel, n_channel, pool_size=1)
        self.conv_block3 = conv1d_block(n_channel, 2 * n_channel, pool_size=1)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    # Based on https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
    from utils.helpers import count_parameters, optimize_hyperparams, train_model

    to_mel = True
    optimize_hp = False
    train_set = "TRAIN"
    val_set = "DEV"
    train_set = ["george"]
    val_set = ["jackson", "lucas", "nicolas", "yweweler", "theo"]

    if to_mel:
        model = Conv1dMelModel()
        audio_transforms = [
            (torchaudio.functional.contrast, 0.5),
            # (torchaudio.functional.dither, 0.5),
        ]
        audio_transforms = None
        spec_transforms = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        spec_transforms = None
    else:
        # Expect error when training with speaker based datasets due to small input size for some samples.
        model = Conv1dModel()

    print(f"Number of parameters: {count_parameters(model)}")

    if not optimize_hp:
        trained_model, _ = train_model(
            model,
            train_set,
            val_set,
            to_mel,
            audio_transforms=audio_transforms,
            spec_transforms=spec_transforms,
            lr=0.0008,
            weight_decay=0.1,
            step_size=20,
            gamma=0.2,
        )
    else:
        trained_model, _ = optimize_hyperparams(
            model,
            train_set,
            val_set,
            to_mel,
            spec_transforms=spec_transforms,
            learning_rates=[0.008, 0.004, 0.002, 0.001],
            weight_decays=[0.01, 0.002, 0.001],
            step_sizes=[10, 15, 20],
            gammas=[0.2, 0.1],
        )
