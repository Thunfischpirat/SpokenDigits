import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self, n_input=1, n_channel=32, n_output=10, initial_kernel_size=60, initial_stride=8
    ):
        super().__init__()
        self.conv_block1 = conv1d_block(n_input, n_channel, initial_kernel_size, initial_stride)
        self.conv_block2 = conv1d_block(n_channel, n_channel)
        self.conv_block3 = conv1d_block(n_channel, 2 * n_channel)
        self.conv_block4 = conv1d_block(2 * n_channel, 2 * n_channel)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class Conv1dMelModel(nn.Module):
    """Convolutional neural network for 1D data."""

    def __init__(
        self, n_input=39, n_channel=32, n_output=10, initial_kernel_size=16, initial_stride=3
    ):
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
        return F.log_softmax(x, dim=2)


if __name__ == "__main__":
    # Based on https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

    torch.manual_seed(32)
    from utils.helpers import count_parameters, train_model, optimize_hyperparams

    to_mel = False

    if to_mel:
        model_name = "Conv1dMelModel"
        model = Conv1dMelModel()
    else:
        model_name = "Conv1dModel"
        model = Conv1dModel()

    print(f"Number of parameters: {count_parameters(model)}")

    trained_model = optimize_hyperparams(
        model,
        learning_rates=[0.002, 0.001],
        weight_decays=[0.002, 0.001],
        step_sizes=[10, 15],
        gammas=[0.2, 0.1],
        to_mel=to_mel
    )

