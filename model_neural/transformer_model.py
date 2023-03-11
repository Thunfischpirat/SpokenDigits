import torch
import torch.nn as nn
import torch.nn.functional as F

from model_neural.utils.helpers import count_parameters, train_model


class PatchEmbedding(nn.Module):
    def __init__(self, spec_dim, patch_size, stride, num_hidden):
        super().__init__()
        # Apply formula 14.12 from pml book to calculate num_patches.
        self.num_patches = ((spec_dim[0] - patch_size + stride) // stride) * (
            (spec_dim[1] - patch_size + stride) // stride
        )
        self.conv = nn.LazyConv2d(num_hidden, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)


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
        x = x + self.attention(*([self.ln1(x)] * 3), need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class transformer_model(nn.Module):
    def __init__(
        self,
        sample_size=(39,88),
        patch_size=8,
        stride=5,
        num_hidden=128,
        mlp_num_hidden=64,
        num_heads=4,
        num_blocks=1,
        num_classes=10,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(sample_size, patch_size, stride, num_hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hidden))
        num_steps = self.patch_embedding.num_patches + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hidden))
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                f"{i}", ViTBlock(num_hidden, num_hidden, mlp_num_hidden, num_heads)
            )
        self.head = nn.Sequential(nn.LayerNorm(num_hidden), nn.Linear(num_hidden, num_classes))

    def forward(self, x):
        x = x.unsqueeze(1)
        if x.shape[3] > 88:
            x = F.interpolate(x, (39,88))
        elif x.shape[3] < 88:
            # Pad last dimension of x with zeros.
            x = F.pad(x, (0, 88 - x.shape[3]))

        x = self.patch_embedding(x)
        # Add cls token for each sample in the batch.
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)[:, 0, :].unsqueeze(1)
        return F.log_softmax(x, dim=2)


if __name__ == "__main__":

    model = transformer_model()

    model = train_model(model, n_epoch=20, lr=0.0001, to_mel=True)

    num_params = count_parameters(model)
    print("Number of parameters: %s" % num_params)

    torch.save(model.state_dict(), "models/transformer_model.pt")


