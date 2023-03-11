from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model_neural.utils.data_loading import MNISTAudio, collate_audio

torch.manual_seed(32)
base_dir = Path(__file__).parent.parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, lr=0.01, n_epoch=100, log_interval=5, batch_size=64, to_mel=False):
    trainset = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split="TRAIN", to_mel=to_mel
    )
    train_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate_audio, shuffle=True)

    valset = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split="DEV", to_mel=to_mel
    )
    validation_loader = DataLoader(valset, batch_size=batch_size, collate_fn=collate_audio, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for training.")

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

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

    return model
