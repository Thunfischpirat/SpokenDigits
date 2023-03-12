from pathlib import Path
from torch.utils.data import DataLoader
from model_neural.utils.data_loading import MNISTAudio, collate_audio

base_dir = Path(__file__).parent.parent.parent
annotations_dir = base_dir / "SDR_metadata.tsv"

dataset = MNISTAudio(annotations_dir=annotations_dir, audio_dir=base_dir, split="TRAIN")

train_dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_audio, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")