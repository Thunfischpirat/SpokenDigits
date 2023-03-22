import os
from datetime import datetime
import torchmetrics as tm

import torch
import torchaudio
from torch import nn, optim

from model_neural.classification_report import eval_models, make_heatmap, classification_report
from model_neural.utils.data_loading import create_loaders
from model_neural.utils.helpers import train_model, get_data_loaders


class LogisticRegression(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    from model_neural.conv1d_model import Conv1dMelModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for report.")

    embedder = Conv1dMelModel(n_output=512)
    embedder.load_state_dict(torch.load("./models/Conv1dMelModel_contrastive.pt", map_location=device))

    embedder.to(device)
    embedder.eval()

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.0)

    train_set = ["george"]
    val_set = ['lucas', 'jackson', 'nicolas', 'yweweler', 'theo']

    if not os.path.isfile(f"models/{(LogisticRegression.__name__ + '_contrastive')}.pt"):
        torch.manual_seed(32)
        train_loader, validation_loader = get_data_loaders(
            train_set=train_set,
            val_set=val_set,
            batch_size=32,
            spec_transforms=nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35),
            ),
        )

        model = LogisticRegression(512, 10)

        model.to(device)
        model_name = model.__class__.__name__ + "_contrastive"

        n_epochs = 200

        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[int(n_epochs * 0.6), int(n_epochs * 0.8)], gamma=0.2)

        loss_val = None

        # Used for early stopping, if enabled.
        best_loss_val = float("inf")
        for epoch in range(n_epochs):
            # -------------------- TRAINING --------------------
            model.train()
            losses_train = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)

                output = model(embedder(data))

                loss = loss_func(output.squeeze(), target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_train.append(loss.item())

            # -------------------- VALIDATION --------------------
            model.eval()
            losses_val = []
            with torch.no_grad():
                for data, target in validation_loader:
                    data = data.to(device)
                    try:
                        output = model(embedder(data))

                        loss = loss_func(output.squeeze(), target)
                        losses_val.append(loss.item())
                    except RuntimeError:
                        print("Skipping batch due to small sample length.")

            # Update the learning rate of the optimizer
            scheduler.step()

            if epoch % 5 == 0:
                loss_train = sum(losses_train) / len(losses_train)
                loss_val = sum(losses_val) / len(losses_val)

                print(
                    f"Epoch: {epoch} Train-Loss: {loss_train:.4f} "
                    f"Val-Loss: {loss_val:.4f} "
                )

            if epoch >= 5:
                loss_val = sum(losses_val) / len(losses_val)
                if loss_val < best_loss_val:
                    print(
                        f"Improved validation loss to {loss_val:.4f} in epoch {epoch}."
                        f" Saving model to disk!"
                    )
                    best_loss_val = loss_val
                    torch.save(
                        model.state_dict(),
                        f"models/{model_name}.pt",
                    )
                    patience = 20
                elif patience == 0:
                    print(f"Validation loss didnt improve further. Stopping training in epoch {epoch}!")

                patience -= 1

    else:
        model = LogisticRegression(512, 10)
        model.load_state_dict(torch.load(f"models/LogisticRegression_contrastive.pt", map_location=device))

        model.to(device)
        model.eval()

    # Conv1dModel doesn't use mel-spectrogram, so we need to specify that.
    if embedder.__class__.__name__ in ["TransformerModel", "Conv1dMelModel"]:
        to_mel = True
    else:
        to_mel = False

    loaders = create_loaders(["TRAIN", "DEV", "TEST"], to_mel)

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

    if not os.path.exists("../logs"):
        os.mkdir("../logs")

    filename = f"../logs/{model.__class__.__name__}_report.txt"
    with open(filename, "a") as file:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M")
        file.write(f"\n------Writing report for evaluation run. Timestamp: {current_time}------\n")

    # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
    with torch.no_grad():
        for loader_name in loaders:
            for batch_idx, (data, target) in enumerate(loaders[loader_name]):
                data = data.to(device)
                target = target.to(device)

                output = model(embedder(data))
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

            if not os.path.exists("../logs"):
                os.mkdir("../logs")

            make_heatmap(
                cm,
                f"Confusion matrix",
                f"logs/{model.__class__.__name__}_cm_{loader_name.lower()}",
            )

            with open(filename, "a") as file:
                cm_string = f"\nConfusion matrix of {loader_name.lower()}-set:\n {cm}"
                print(cm_string)
                file.write(cm_string)

                report = classification_report(accuracy, precision, recall, f1)
                report_string = f"\nClassification report: {loader_name.lower()}-set:\n {report}"
                print(report_string)
                file.write(report_string)
