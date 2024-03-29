import torch
import torch.nn as nn
from model_neural.utils.helpers import (count_parameters, optimize_hyperparams,
                                        train_model)


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=150):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        mask = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        # 0::2 means even indices, 1::2 means odd indices.
        self.P[:, :, 0::2] = torch.sin(mask)
        self.P[:, :, 1::2] = torch.cos(mask)

    def forward(self, x):
        x = x + self.P[:, : x.shape[1], :].to(x.device)
        return self.dropout(x)


class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hidden, mlp_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hidden)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))


class ViTBlock(nn.Module):
    def __init__(self, num_hidden, norm_shape, mlp_num_hidden, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = nn.MultiheadAttention(num_hidden, num_heads, 0.1, batch_first=True)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hidden, num_hidden)

    def forward(self, x, valid_lens=None):
        x = x + self.attention(*([self.ln1(x)] * 3))[0]
        return x + self.mlp(self.ln2(x))


class TransformerModel(nn.Module):
    def __init__(
        self,
        patch_size=16,
        stride=10,
        num_hidden=128,
        mlp_num_hidden=128,
        num_heads=4,
        num_blocks=2,
        num_classes=10,
    ):
        super().__init__()
        self.patch_embedding = nn.LazyConv2d(num_hidden, kernel_size=patch_size, stride=stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hidden))
        self.pos_embedding = PositionalEncoding(num_hidden, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                f"{i}", ViTBlock(num_hidden, num_hidden, mlp_num_hidden, num_heads)
            )
        self.head = nn.Sequential(nn.LayerNorm(num_hidden), nn.Linear(num_hidden, num_classes))

    def forward(self, x):
        # Add channel dimension.
        x = x.unsqueeze(1)
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        # Using .expand adds cls token for each sample in the batch.
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)[:, 0, :].unsqueeze(1)
        return x


if __name__ == "__main__":

    model_name = "TransformerModel"
    model = TransformerModel()

    # train_set = ["george"]
    # val_set = ["jackson", "lucas", "nicolas", "yweweler", "theo"]
    train_set = "TRAIN"
    val_set = "DEV"

    to_mel = True
    optimize_hp = False

    if not optimize_hp:
        trained_model, _ = train_model(
            model,
            train_set,
            val_set,
            to_mel,
            lr=0.0001,
            weight_decay=0.0001,
            step_size=15,
            gamma=0.01,
        )
    else:
        trained_model, _ = optimize_hyperparams(
            model,
            train_set,
            val_set,
            audio_transforms=None,
            spec_transforms=None,
            learning_rates=[0.001, 0.0001],
            weight_decays=[0.001, 0.0001],
            step_sizes=[15, 20],
            gammas=[0.01, 0.1],
            to_mel=True,
        )

    # Since we are using LazyLinear, we need to call the model once to initialize the weights.
    num_params = count_parameters(trained_model)
    print("Number of parameters: %s" % num_params)
