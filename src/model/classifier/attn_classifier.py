import math
import torch
import torch.nn as nn

from ..base import BaseModel

def get_sinusoidal_embeddings(num_positions: int, dim: int) -> torch.Tensor:
    """Generates fixed sinusoidal positional embeddings."""
    position = torch.arange(num_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def mixup_data(batch, alpha=0.2):
    batch_size = batch['labels'].size(0)
    index = torch.randperm(batch_size).to(batch['labels'].device)

    mixed_batch = {}
    for key, x in batch.items():
        mixed_batch[key] = alpha * x + (1 - alpha) * x[index, :]

    return mixed_batch


class AttnClassifier(BaseModel):
    def __init__(
        self,
        num_channels: int = 9,       # C
        unishape_dim: int = 128,     # UniShape D
        mantis_dim: int = 256,       # Mantis D
        dim: int = 128,              # Unified Transformer D
        num_classes: int = 8,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        models: list[str] = ['unishape', 'mantis'],
        mixup_alpha: float = 0.2
    ):
        super().__init__()
        self.capture_config()

        self.num_channels = num_channels
        self.dim = dim
        self.models = models
        self.mixup_alpha = mixup_alpha

        # 1. Independent projections to unify the dimensions
        self.unishape_proj = nn.Linear(unishape_dim, dim) if unishape_dim != dim else nn.Identity()
        self.mantis_proj = nn.Linear(mantis_dim, dim) if mantis_dim != dim else nn.Identity()

        # The CLS token remains learnable as it acts as the pooling mechanism
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # 2. Embeddings
        # Model embedding differentiates which foundation model the token came from (Fixed)
        # self.register_buffer(
        #     "model_embed",
        #     get_sinusoidal_embeddings(3, dim)
        # )

        # Channel embedding differentiates the 9 time-series channels (Learnable)
        # Note: Size is now just `num_channels`, because we apply it BEFORE adding the CLS token
        self.channel_embed = nn.Parameter(
            torch.zeros(1, num_channels, dim)
        )
        self.model_embed = nn.Parameter(
            torch.zeros(3, dim)
        )

        # 3. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 4. Lightweight Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize the CLS token and the embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        nn.init.trunc_normal_(self.model_embed, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Mixup is applied only during training
        if self.mixup_alpha > 0 and self.training:
            batch = mixup_data(batch, alpha=self.mixup_alpha)

        features = []

        # Process each model independently, applying projections and embeddings
        if 'unishape' in self.models:
            uni_x = batch['unishape_features']
            uni_x = self.unishape_proj(uni_x)
            uni_x = uni_x + self.channel_embed
            uni_x = uni_x + self.model_embed[0].view(1, 1, -1)
            features.append(uni_x)

        if 'mantis' in self.models:
            man_x = batch['mantis_features']
            man_x = self.mantis_proj(man_x)
            man_x = man_x + self.channel_embed
            # man_x = man_x + self.model_embed[1].view(1, 1, -1)
            features.append(man_x)

        # 4. Concatenate sequences along the sequence dimension (dim=1)
        # If 1 model: (B, 9, dim)
        # If 2 models: (B, 18, dim)
        x = torch.cat(features, dim=1)
        B, seq_len, _ = x.shape

        # 5. Prepend CLS token
        # Shape becomes (B, seq_len + 1, dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 6. Transform
        x = self.transformer(x)

        # 7. Extract ONLY the CLS token output
        cls_out = x[:, 0]  # (B, dim)

        # 8. Classify
        logits = self.head(cls_out)

        return {
            'logits': logits,
            'labels': batch.get('labels'),
            'cls': cls_out
        }