import torch
from sklearn.metrics import confusion_matrix, classification_report

from model_neural.utils.data_loading import create_loaders

if __name__ == "__main__":
    from model_neural.conv1d_model import Conv1dMelModel

    models = [Conv1dMelModel, ]
    states = ['Conv1dMelModel_contrastive.pt', ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for report.")

    for m, s in zip(models, states):
        model = m(n_output=512)
        model.load_state_dict(torch.load("./models/" + s, map_location=device))

        model.to(device)
        model.eval()

        # Conv1dModel doesn't use mel-spectrogram, so we need to specify that.
        if model.__class__.__name__ in ["TransformerModel", "Conv1dMelModel"]:
            to_mel = True
        else:
            to_mel = False

        print(f"------------------- {model.__class__.__name__} report -------------------")
