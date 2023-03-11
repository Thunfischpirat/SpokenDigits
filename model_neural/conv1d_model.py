import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import train_model, count_parameters



class conv1d_model(nn.Module):
    def __init__(
        self, n_input=1, n_channel=32, n_output=10, initial_kernel_size=60, initial_stride=8
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_input, n_channel, kernel_size=initial_kernel_size, stride=initial_stride
        )
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


if __name__ == "__main__":
    # Based on https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

    to_mel = False

    if to_mel:
        model = conv1d_model(n_input=39, initial_kernel_size=6, initial_stride=1)
    else:
        model = conv1d_model()

    model = train_model(model, lr=0.01, to_mel=False)

    print(f"Number of parameters: {count_parameters(model)}")

    torch.save(model.state_dict(), "models/conv1d_model.pt")



