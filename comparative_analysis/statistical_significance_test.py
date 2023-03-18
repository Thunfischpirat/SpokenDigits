import torch
from torch import nn
import torchmetrics as tm
from torch.utils.data import RandomSampler, DataLoader

from model_neural.utils.data_loading import MNISTAudio
from model_neural.utils.helpers import annotations_dir, base_dir


def test_statistical_significance(model, baseline, device: torch.device):
    """Test for statistical significance between models w.r.t accuracy
    :return: p-value"""
    # Adapted from https://aclanthology.org/D12-1091.pdf
    # Sample with replacement for val (DEV) set
    ds = MNISTAudio(annotations_dir=annotations_dir, audio_dir=base_dir, split="DEV", to_mel=True)
    n = ds.__len__()
    s = 0
    b = 10 ** 6

    if issubclass(model, nn.Module):
        model_accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)
        model_accuracy_metric.to(device)
    if issubclass(baseline, nn.Module):
        baseline_accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)
        baseline_accuracy_metric.to(device)

    for _ in range(b):
        sampler = RandomSampler(ds, replacement=True, num_samples=n)
        dl = DataLoader(ds, sampler=sampler, batch_size=32)
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dl):
                data = data.to(device)
                target = target.to(device)

                if issubclass(model, nn.Module):
                    model_output = model(data)
                    model_pred = model_output.argmax(dim=2, keepdim=True).squeeze()
                else:
                    model_pred = model.predict(data)
                model_accuracy_metric(model_pred, target)

                if issubclass(baseline, nn.Module):
                    baseline_output = baseline(data)
                    baseline_pred = baseline_output.argmax(dim=2, keepdim=True).squeeze()
                else:
                    baseline_pred = baseline.predict(data)
                baseline_accuracy_metric(baseline_pred, target)

        model_accuracy = model_accuracy_metric.compute()
        model_accuracy_metric.reset()
        baseline_accuracy = baseline_accuracy_metric.compute()
        baseline_accuracy_metric.reset()

        if model_accuracy > baseline_accuracy:
            s += 1

    return s / b