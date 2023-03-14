import copy
import itertools
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics as tm
from model_neural.utils.data_loading import MNISTAudio, collate_audio
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler

base_dir = Path(__file__).parent.parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"


def count_parameters(model: nn.Module):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_loaders(batch_size: int = 64, to_mel: bool = False):
    """Get the data loaders for the MNIST audio dataset."""
    train_set = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split="TRAIN", to_mel=to_mel
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_audio, shuffle=True
    )

    validation_set = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split="DEV", to_mel=to_mel
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, collate_fn=collate_audio, shuffle=True
    )

    return train_loader, validation_loader


def train_model(
        model: nn.Module,
        lr: float = 0.01,
        weight_decay: float = 0.01,
        step_size: int = 20,
        gamma: float = 0.1,
        n_epoch: int = 100,
        batch_size: int = 32,
        early_stopping: bool = True,
        to_mel: bool = False,
):
    """Train a model on the MNIST audio dataset."""

    train_loader, validation_loader = get_data_loaders(batch_size=batch_size, to_mel=to_mel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for training.")

    model.to(device)
    model_name = model.__class__.__name__

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_func = nn.NLLLoss()
    loss_val = None

    # Used for early stopping, if enabled.
    best_loss_val = float("inf")
    counter = 15

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
                output_sm = F.log_softmax(output, dim=2)

                loss = loss_func(output_sm.squeeze(), target)

                pred = output.argmax(dim=2, keepdim=True).squeeze()
                correct += pred.eq(target).sum().item()

                losses_val.append(loss.item())

        # Update the learning rate of the optimizer
        scheduler.step()

        if epoch % 5 == 0:
            loss_train = sum(losses_train) / len(losses_train)
            loss_val = sum(losses_val) / len(losses_val)
            accuracy = 100.0 * correct / len(validation_loader.dataset)
            print(
                f"Epoch: {epoch} Train-Loss: {loss_train:.6f} "
                f"Val-Loss: {loss_val:.6f} "
                f"Val-Accuracy: {accuracy:.2f} "
            )

        if early_stopping and epoch >= 10:
            loss_val = sum(losses_val) / len(losses_val)
            if loss_val < best_loss_val:
                accuracy = 100.0 * correct / len(validation_loader.dataset)
                print(
                    f"Improved validation loss to {loss_val:.6f} in epoch {epoch}."
                    f" Current accuracy: {accuracy:.2f}. Saving model to disk!"
                )
                best_loss_val = loss_val
                torch.save(
                    model.state_dict(),
                    f"models/{model_name}.pt",
                )
                counter = 15
            elif counter == 0:
                print(f"Validation loss didnt improve further. Stopping training in epoch {epoch}!")
                return model, best_loss_val

            counter -= 1

    return model, loss_val


def optimize_hyperparams(
        model: nn.Module,
        learning_rates: List[float],
        weight_decays: List[float],
        step_sizes: List[int],
        gammas: List[float],
        to_mel: bool = False,
):
    """Optimize the hyperparameters of a model."""
    grid_space = itertools.product(learning_rates, weight_decays, step_sizes, gammas)
    size_grid_space = len(learning_rates) * len(weight_decays) * len(step_sizes) * len(gammas)

    original_model = model
    model_name = model.__class__.__name__

    best_loss = float("inf")
    best_model = None
    best_params = None

    # Removing old results file.
    filename = f"logs/hyp_opt_{model_name}.txt"
    try:
        os.remove(filename)
    except OSError:
        pass

    i = 0
    for lr, weight_decay, step_size, gamma in grid_space:
        i += 1
        print(
            f"\n-------------Training model with parameter combination {i}/{size_grid_space}. -------------"
        )
        # Copying the original model to avoid using weights from previous training runs.
        model = copy.deepcopy(original_model)
        trained_model, loss = train_model(
            model, lr=lr, weight_decay=weight_decay, step_size=step_size, gamma=gamma, to_mel=to_mel
        )

        # Logging experiment results.
        with open(f"{filename}", "a") as file:
            file.write(
                f"Iteration: {i}/{size_grid_space},"
                f" Loss: {loss:.4f} for parameters: {lr}, {weight_decay}, {step_size}, {gamma}.\n"
            )

        if loss < best_loss:
            best_loss = loss
            best_model = trained_model
            best_params = {
                "lr": lr,
                "weight_decay": weight_decay,
                "step_size": step_size,
                "gamma": gamma,
            }
            print(
                f"\nNew best model found at iteration {i}/{size_grid_space} with loss: {best_loss:.6f} and params: {best_params}."
            )

    # Save best model to disk and include parameters in filename.
    str_lr = str(best_params["lr"]).replace(".", "")
    str_weight_decay = str(best_params["weight_decay"]).replace(".", "")
    str_step_size = str(best_params["step_size"]).replace(".", "")
    str_gamma = str(best_params["gamma"]).replace(".", "")

    torch.save(
        best_model.state_dict(),
        f"models/{model_name}_{str_lr}_{str_weight_decay}_{str_step_size}_{str_gamma}.pt",
    )
    with open(f"{filename}", "a") as file:
        file.write(f"Best parameters: {best_params}. Best loss: {best_loss}.\n")

    return best_model, best_params


def test_statistical_significance(model: nn.Module, baseline: nn.Module, device: torch.device):
    """Test for statistical significance between models w.r.t accuracy
    :return: p-value"""
    # Adapted from https://aclanthology.org/D12-1091.pdf
    # Sample with replacement for val (DEV) set
    ds = MNISTAudio(annotations_dir=annotations_dir, audio_dir=base_dir, split="DEV", to_mel=True)
    n = ds.__len__()
    sampler = RandomSampler(ds, replacement=True, num_samples=n)
    dl = DataLoader(ds, sampler=sampler, batch_size=32)
    s = 0
    b = 10 ** 6
    model_accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)
    model_accuracy_metric.to(device)
    baseline_accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)
    baseline_accuracy_metric.to(device)
    for _ in range(b):
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dl):
                data = data.to(device)
                target = target.to(device)
                model_output = model(data)
                baseline_output = baseline(data)
                model_pred = model_output.argmax(dim=2, keepdim=True).squeeze()
                baseline_pred = baseline_output.argmax(dim=2, keepdim=True).squeeze()
                model_accuracy_metric(model_pred, target)
                baseline_accuracy_metric(baseline_pred, target)
            model_accuracy = model_accuracy_metric.compute()
            baseline_accuracy = baseline_accuracy_metric.compute()
            model_accuracy_metric.reset()
            baseline_accuracy_metric.reset()
            if model_accuracy > baseline_accuracy:
                s+=1
    return s/b
