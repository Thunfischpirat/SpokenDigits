import torch

from model_neural.classification_report import eval_models

if __name__ == "__main__":
    from model_neural.conv1d_model import Conv1dMelModel, Conv1dModel
    from model_neural.transformer_model import TransformerModel
    from model_neural.utils.data_loading import MNISTAudio, collate_audio

    models = [Conv1dMelModel, TransformerModel]
    states = ['Conv1dMelModel_single_speaker.pt', 'TransformerModel_single_speaker.pt']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: '{device}' as device for report.")

    for m, s in zip(models, states):
        model = m()
        model.load_state_dict(torch.load("./models/" + s, map_location=device))

        model.to(device)
        model.eval()

        # Conv1dModel doesn't use mel-spectrogram, so we need to specify that.
        if model.__class__.__name__ in ["TransformerModel", "Conv1dMelModel"]:
            to_mel = True
        else:
            to_mel = False

        print(f"------------------- {model.__class__.__name__} report -------------------")
        eval_models(model, ["TRAIN", "DEV", "TEST"], device=device, to_mel=to_mel)