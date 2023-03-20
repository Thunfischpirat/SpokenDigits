import joblib
import numpy
import numpy as np
import torch
from torch import nn
import torchmetrics as tm
from torch.utils.data import RandomSampler, DataLoader

from model_baseline.data_loading import downsample_spectrogram
from model_neural.utils.data_loading import MNISTAudio, create_loaders, collate_audio
from model_neural.utils.helpers import annotations_dir, base_dir


def test_statistical_significance(model, baseline, device: torch.device):
    """Test for statistical significance between models w.r.t accuracy
    :return: p-value"""
    # Adapted from https://aclanthology.org/D12-1091.pdf
    # Sample with replacement for val (DEV) set

    ds39 = MNISTAudio(annotations_dir, base_dir, "DEV", True)
    ds13 = MNISTAudio(annotations_dir, base_dir, "DEV", True, n_mels=13)

    s = 0
    b = 100

    model_accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)
    model_accuracy_metric.to(device)
    baseline_accuracy_metric = tm.classification.MulticlassAccuracy(num_classes=10)
    baseline_accuracy_metric.to(device)

    for i in range(b):
        sampler = RandomSampler(ds39, replacement=True, num_samples=ds39.__len__())
        dl39 = DataLoader(ds39, batch_size=32, collate_fn=collate_audio, sampler=sampler)
        dl13 = DataLoader(ds13, batch_size=32, collate_fn=collate_audio, sampler=sampler)
        with torch.no_grad():
            for ((batch_idx, (data, target)), (idx2, (data2, target2))) in zip(enumerate(dl39), enumerate(dl13)):
                data = data.to(device)
                target = target.to(device)

                if model.__class__.__name__ in ["TransformerModel", "Conv1dMelModel"]:
                    model_output = model(data)
                    model_pred = model_output.argmax(dim=2, keepdim=True).squeeze()
                else:
                    model_pred = model.predict([downsample_spectrogram(i, 10) for i in data2])
                if not torch.is_tensor(model_pred):
                    model_pred = torch.from_numpy(model_pred)
                model_accuracy_metric(model_pred, target)

                if baseline.__class__.__name__ in ["TransformerModel", "Conv1dMelModel"]:
                    baseline_output = baseline(data)
                    baseline_pred = baseline_output.argmax(dim=2, keepdim=True).squeeze()
                else:
                    baseline_pred = baseline.predict([downsample_spectrogram(i, 10)[0] for i in data2])
                if not torch.is_tensor(baseline_pred):
                    baseline_pred = torch.from_numpy(baseline_pred)
                baseline_accuracy_metric(baseline_pred, target)

        model_accuracy = model_accuracy_metric.compute()
        model_accuracy_metric.reset()
        baseline_accuracy = baseline_accuracy_metric.compute()
        baseline_accuracy_metric.reset()

        if model_accuracy > baseline_accuracy:
            s += 1

    return s / b


if __name__ == "__main__":
    from model_neural.conv1d_model import Conv1dMelModel
    from model_neural.transformer_model import TransformerModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = joblib.load(
        "../model_baseline/linear_model.joblib")

    conv1d = Conv1dMelModel()
    conv1d.load_state_dict(torch.load(
        "../model_neural/models/Conv1dMelModel_0008_0001_20_02.pt", map_location=device))
    conv1d.to(device)
    conv1d.eval()

    transformer = TransformerModel()
    transformer.load_state_dict(torch.load(
        "../model_neural/models/TransformerModel_00001_00001_15_001.pt", map_location=device))
    transformer.to(device)
    transformer.eval()

    models = [conv1d, baseline, transformer]
    names = ["Conv1d", "Baseline", "Transformer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bc_pval = test_statistical_significance(conv1d, baseline, device)
    bt_pval = test_statistical_significance(transformer, baseline, device)
    ct_pval = test_statistical_significance(conv1d, transformer, device)

    print(f"p-value of {names[0]} x {names[1]}: {bc_pval}")
    print(f"p-value of {names[2]} x {names[1]}: {bt_pval}")
    print(f"p-value of {names[0]} x {names[2]}: {ct_pval}")
    print("When p < 0.5, 2nd model outperforms 1st with [p]")
    print("When p > 0.5, 1st model outperforms 2nd with [1 - p]")
