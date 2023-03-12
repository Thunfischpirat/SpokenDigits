from pathlib import Path

import torch
import torchmetrics as tm
from conv1d_model import M5
from model_neural.utils.data_loading import MNISTAudio, collate_audio
from torch.utils.data import DataLoader

base_dir = Path(__file__).parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: '{device}' as device for report.")

model = M5()
model.load_state_dict(torch.load("../models/conv1d_model.pt"))
model.to(device)
model.eval()

# This code creates a dictionary of PyTorch DataLoader objects for the MNIST audio dataset for each split of the data.
loaders = dict(
    [
        (
            split,
            DataLoader(
                MNISTAudio(annotations_dir=annotations_dir, audio_dir=base_dir, split=split),
                batch_size=64,
                collate_fn=collate_audio,
                shuffle=True,
            ),
        )
        for split in ["TRAIN", "DEV", "TEST"]
    ]
)


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
        print(f"Confusion matrix of {loader_name.lower()}-set:\n {cm}\n")
        report = classification_report(accuracy, precision, recall, f1)
        print(f"Classification report: {loader_name.lower()}-set:\n {report}")
