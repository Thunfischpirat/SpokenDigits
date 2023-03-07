import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim



class M5(nn.Module):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    # Based on https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
    from pathlib import Path
    from torch.utils.data import DataLoader
    from model_neural.data_loading import MNISTAudio, collate_audio

    torch.manual_seed(32)
    base_dir = Path(__file__).parent.parent
    annotations_dir = base_dir / "SDR_metadata.tsv"

    to_mel = False

    trainset = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split="TRAIN", to_mel=to_mel
    )
    train_loader = DataLoader(trainset, batch_size=64, collate_fn=collate_audio, shuffle=True)

    valset = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split="DEV", to_mel=to_mel
    )
    validation_loader = DataLoader(valset, batch_size=64, collate_fn=collate_audio, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for training.")

    if to_mel:
        model = M5(n_input=39, initial_kernel_size=6, initial_stride=1)
    else:
        model = M5()

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    log_interval = 5
    n_epoch = 100

    loss_func = nn.NLLLoss()

    for epoch in range(n_epoch):
        # -------------------- TRAINING --------------------
        model.train()
        losses_train = []
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss = loss_func(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_train.append(loss.item())

        # -------------------- VALIDATION --------------------
        model.eval()
        losses_val = []
        correct = 0
        with torch.no_grad():
            for data, target in validation_loader:

                data = data.to(device)
                target = target.to(device)

                output = model(data)

                loss = loss_func(output.squeeze(), target)

                pred = output.argmax(dim=2, keepdim=True).squeeze()
                correct += pred.eq(target).sum().item()

                losses_val.append(loss.item())

        # Update the learning rate of the optimizer
        scheduler.step()


        if epoch % log_interval == 0:
            mean_loss_train = sum(losses_train) / len(losses_train)
            mean_loss_val = sum(losses_val) / len(losses_val)
            accuracy = 100.0 * correct / len(validation_loader.dataset)
            print(
                f"Epoch: {epoch} Train-Loss: {mean_loss_train:.6f} "
                f"Val-Loss: {mean_loss_val:.6f} "
                f"Val-Accuracy: {accuracy:.2f} "
            )

    torch.save(model.state_dict(), "models/conv1d_model.pt")



