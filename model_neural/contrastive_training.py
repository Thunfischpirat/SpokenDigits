import torch
import torchaudio
from torch import nn

import torch.nn.functional as F

from model_neural.conv1d_model import Conv1dMelModel
from model_neural.utils.data_loading import ContrastiveTransformations
from model_neural.utils.helpers import contrastive_train

model = Conv1dMelModel(n_output=512)

train_set = ["george"]
val_set = ['lucas', 'jackson', 'nicolas', 'yweweler', 'theo']

transforms = nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)
spec_transforms = ContrastiveTransformations(transforms, n_views=2)

_, _ = contrastive_train(model, train_set, val_set, spec_transforms=spec_transforms)

trained_embedding_model = torch.load("model_neural/trained_models/Conv1dMelModel_contrastive_model.pt")