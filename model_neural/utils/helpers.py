import copy
import itertools
import os
from pathlib import Path
from typing import List, Union

import torch
import torch.nn.functional as F

from model_neural.utils.data_loading import MNISTAudio, collate_audio
from torch import nn, optim
from torch.utils.data import DataLoader

base_dir = Path(__file__).parent.parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"


def count_parameters(model: nn.Module):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_loaders(
    train_set: Union[str, List[str]],
    val_set: Union[str, List[str]],
    to_mel: bool = False,
    batch_size: int = 64,
):
    """
    Get the data loaders for the MNIST audio dataset.

    Args:
        train_set: Name of the train set to use. Can be either a split or list of speakers.
        val_set: Name of the validation set to use.
        to_mel: Whether to convert the raw audio to a mel spectrogram first.
        batch_size: The batch size to use.
    """
    train_set = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split=train_set, to_mel=to_mel
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_audio, shuffle=True
    )

    validation_set = MNISTAudio(
        annotations_dir=annotations_dir, audio_dir=base_dir, split=val_set, to_mel=to_mel
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, collate_fn=collate_audio, shuffle=True
    )

    return train_loader, validation_loader


def train_model(
    model: nn.Module,
    train_set: Union[str, List[str]] = "TRAIN",
    val_set: Union[str, List[str]] = "DEV",
    to_mel: bool = False,
    lr: float = 0.01,
    weight_decay: float = 0.01,
    step_size: int = 20,
    gamma: float = 0.1,
    batch_size: int = 32,
):
    """
    Train a model on a split of subset of speakers of the MNIST audio dataset.

    Args:
        model: The model to train.
        train_set: Name of the train set to use. Can be either a split or list of speakers.
        val_set: Name of the validation set to use.
        to_mel: Whether to convert the raw audio to a mel spectrogram first.
        lr: The learning rate to use.
        weight_decay: The weight decay to use.
        step_size: The step size to use for the learning rate scheduler.
        gamma: The gamma to use for the learning rate scheduler.
        batch_size: The batch size to use.
    """

    train_loader, validation_loader = get_data_loaders(
        train_set=train_set, val_set=val_set, to_mel=to_mel, batch_size=batch_size
    )

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

    n_epochs = 100
    for epoch in range(n_epochs):
        # -------------------- TRAINING --------------------
        model.train()
        losses_train = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            output_sm = F.log_softmax(output, dim=2)

            loss = loss_func(output_sm.squeeze(), target)

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

        if epoch >= 5:
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
    train_set: Union[str, List[str]],
    val_set: Union[str, List[str]],
    to_mel: bool,
    learning_rates: List[float],
    weight_decays: List[float],
    step_sizes: List[int],
    gammas: List[float],
):
    """
    Optimize the hyperparameters of a model.

    Args:
        model: The model to optimize.
        train_set: Name of the train set to use. Can be either a split or list of speakers.
        val_set: Name of the validation set to use.
        to_mel: Whether to convert the raw audio to a mel spectrogram first.
        learning_rates: List of learning rates to try.
        weight_decays: List of weight decays to try.
        step_sizes: List of step sizes to try.
        gammas: List of gammas to try.
    """
    grid_space = itertools.product(learning_rates, weight_decays, step_sizes, gammas)
    size_grid_space = len(learning_rates) * len(weight_decays) * len(step_sizes) * len(gammas)

    original_model = model
    model_name = model.__class__.__name__

    best_loss = float("inf")
    best_model = None
    best_params = None

    # Removing old results file.
    if not os.path.exists("./logs"):
        os.mkdir("./logs")

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
            model, train_set, val_set, to_mel, lr, weight_decay, step_size, gamma
        )

        # Logging experiment results.
        with open(f"{filename}", 'a') as file:
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
