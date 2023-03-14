import matplotlib.pyplot as plt
import numpy as np
import torch

from model_baseline.data_loading import create_features
from model_neural.utils.data_loading import create_loaders
from sklearn.manifold import TSNE
from torch import nn


def tsne_model(model: nn.Module, device: torch.device, to_mel: bool = False, split: str = "TRAIN"):
    """Create tsne embedding of output of model applied to given data-split."""

    loader = create_loaders([split], to_mel=to_mel)[split]

    targets = torch.empty(0).to(device)
    # Output shape is (batch_size, 1, 10).
    outputs = torch.empty(0, 1, 10).to(device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            targets = torch.cat((targets, target), dim=0)
            outputs = torch.cat((outputs, output), dim=0)

    # Apply tsne to output of model.
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_embedding = tsne.fit_transform(outputs.view(-1, 10).cpu().numpy())
    labels = targets.view(-1).cpu().numpy()
    return tsne_embedding, labels


def tsne_linear(model, num_mels, num_frames, to_mel: bool = False, split: str = "TRAIN"):
    """Create tsne embedding of output of linear model applied to given data-split."""
    features, labels = create_features(split, num_mels, num_frames)
    preds = model.predict(features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_embedding = tsne.fit_transform(preds)
    return tsne_embedding, labels


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, filename: str = None):
    """Plot tsne embedding."""
    unique_labels = np.unique(labels)

    # Create a color map that maps each label to a unique color.
    color_map = plt.cm.get_cmap("hsv", len(unique_labels))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each point with its corresponding label color.
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings[:, 0][mask],
            embeddings[:, 1][mask],
            color=color_map(i),
            label=label,
            alpha=0.5,
        )

    ax.legend()
    model_name = model.__class__.__name__
    ax.set_title(f"t-SNE Embeddings {model_name}")
    if filename is not None:
        plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    from model_neural.conv1d_model import Conv1dModel
    from model_neural.transformer_model import TransformerModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for report.")

    model = Conv1dModel()
    model.load_state_dict(torch.load("../model_neural/models/Conv1dModel_0002_0002_10_01.pt", map_location=device))
    model.to(device)
    model.eval()

    if model.__class__.__name__ in ["TransformerModel", "Conv1dMelModel"]:
        to_mel = True
    else:
        to_mel = False

    tsne_embedding, labels = tsne_model(model, device, to_mel, split="DEV")
    plot_tsne(tsne_embedding, labels)
