import os
from pathlib import Path

import torch
import torchmetrics as tm
from typing import Dict
import matplotlib.pyplot as plt

from torch import nn

from model_neural.conv1d_model import Conv1dModel, Conv1dMelModel
from model_neural.transformer_model import TransformerModel
from torch.utils.data import DataLoader

base_dir = Path(__file__).parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"


def classification_report(accuracy, *args):
    """Creates a classification report for a model in form of a formatted string."""
    metrics = torch.vstack([*args]).t()
    report = "   ".join(f"{col}" for col in ["class", "precision", "recall", "f1-score"]) + "\n"
    for i, row in enumerate(metrics):
        line = f"     {i}      "
        for value in row:
            line += f"{value.item():.2f}       "
        report += f"{line}\n"
    report += f"\naccuracy: {accuracy.item():.2f}\n"
    return report


def make_heatmap(cm, title, save_path):
    """Creates a heatmap of the confusion matrix."""
    fig, ax = plt.subplots()
    cm = cm.cpu()
    img = ax.imshow(cm, cmap="YlOrRd")

    ax.figure.colorbar(img, ax=ax)

    classes = list(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(classes)
    ax.set_yticks(classes)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(int(cm[i, j].item())), ha="center", va="center", color="black")

    ax.set_title(title)

    plt.savefig(f"{save_path}.png")
    plt.close()

def eval_models(model: nn.Module, loaders: Dict[str, DataLoader]):
    """Evaluate a model on various splits of the MNIST audio dataset."""
    confusion_matrix = tm.classification.MulticlassConfusionMatrix(num_classes=10)
    f1_score = tm.classification.MulticlassF1Score(num_classes=10, average="none")
    precision_metric = tm.classification.MulticlassPrecision(num_classes=10, average="none")
    recall_metric = tm.classification.MulticlassRecall(num_classes=10, average="none")
    accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)

    confusion_matrix.to(device)
    f1_score.to(device)
    precision_metric.to(device)
    recall_metric.to(device)
    accuracy_metric.to(device)

    filename = f"../logs/{model.__class__.__name__}_report.txt"
    try:
        os.remove(filename)
    except OSError:
        pass

    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
    with torch.no_grad():
        for loader_name in loaders:
            for batch_idx, (data, target) in enumerate(loaders[loader_name]):
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                pred = output.argmax(dim=2, keepdim=True).squeeze()

                confusion_matrix(pred, target)
                f1_score(pred, target)
                precision_metric(pred, target)
                recall_metric(pred, target)
                accuracy_metric(pred, target)

            cm = confusion_matrix.compute()
            f1 = f1_score.compute()
            precision = precision_metric.compute()
            recall = recall_metric.compute()
            accuracy = accuracy_metric.compute()

            confusion_matrix.reset()
            f1_score.reset()
            precision_metric.reset()
            recall_metric.reset()
            accuracy_metric.reset()

            make_heatmap(
                cm,
                f"Confusion matrix {loader_name.lower()}-set",
                f"../logs/{model.__class__.__name__}_cm_{loader_name.lower()}",
            )

            with open(filename, "a") as file:
                cm_string = f"\nConfusion matrix of {loader_name.lower()}-set:\n {cm}"
                print(cm_string)
                file.write(cm_string)

                report = classification_report(accuracy, precision, recall, f1)
                report_string = f"\nClassification report: {loader_name.lower()}-set:\n {report}"
                print(report_string)
                file.write(report_string)

    return report


if __name__ == "__main__":
    from model_neural.utils.data_loading import MNISTAudio, collate_audio

    to_mel = True
    # This code creates a dictionary of PyTorch DataLoader objects for the MNIST audio dataset for each split of the data.
    loaders = dict(
        [
            (
                split,
                DataLoader(
                    MNISTAudio(
                        annotations_dir=annotations_dir,
                        audio_dir=base_dir,
                        split=split,
                        to_mel=to_mel,
                    ),
                    batch_size=64,
                    collate_fn=collate_audio,
                    shuffle=True,
                ),
            )
            for split in ["TRAIN", "DEV", "TEST"]
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for report.")

    model = TransformerModel()
    model.load_state_dict(torch.load("../models/TransformerModel.pt"))
    model.to(device)
    model.eval()

    report = eval_models(model, loaders)
