import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base import BaseModel

class ChannelAttentionPooler(nn.Module):
    """Dynamically weights channels before pooling."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class LightweightCrossAttention(nn.Module):
    """Cross-attention where the sequence length of Keys/Values can vary."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, query_seq: torch.Tensor, kv_seq: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=query_seq,
            key=kv_seq,
            value=kv_seq,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False
        )
        x = self.norm(query_seq + self.dropout(attn_out))
        out = self.proj(x)
        return self.norm_out(x + self.dropout(out))


class Ensemble(BaseModel):
    def __init__(
        self,
        active_modalities: List[str] = ['unishape', 'mantis', 'nutime', 'trex', 'tspulse', 'fft'],
        num_channels: int = 9,
        dim: int = 128,
        num_classes: int = 8,
        num_heads: int = 4,
        dropout: float = 0.3,
        token_mask_prob: float = 0.15
    ):
        super().__init__()
        self.capture_config()
        self.token_mask_prob = token_mask_prob
        self.active_modalities = active_modalities

        # Define native input dimensions for all possible modalities
        self.seq_dims = {
            'unishape': 128,
            'mantis': 256,
            'nutime': 128,
            'trex': 512
        }
        self.global_dims = {
            'tspulse': 512,
            'fft': 2832
        }

        # Filter active modalities by type
        self.active_seq = [m for m in active_modalities if m in self.seq_dims]
        self.active_global = [m for m in active_modalities if m in self.global_dims]

        if not self.active_seq and not self.active_global:
            raise ValueError("At least one valid modality must be active.")

        # --- 1. Dynamic Initialization for Sequence Modalities ---
        self.seq_projs = nn.ModuleDict()
        self.seq_embeds = nn.ParameterDict()
        self.cross_attns = nn.ModuleDict()
        self.poolers = nn.ModuleDict()

        for mod in self.active_seq:
            # Projections
            self.seq_projs[mod] = nn.Sequential(
                nn.Linear(self.seq_dims[mod], dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout)
            )
            # Spatial/Channel Embeddings
            self.seq_embeds[mod] = nn.Parameter(torch.zeros(1, num_channels, dim))
            # Cross Attention Blocks
            self.cross_attns[mod] = LightweightCrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
            # Poolers
            self.poolers[mod] = ChannelAttentionPooler(dim)

        # --- 2. Dynamic Initialization for Global (Bypass) Modalities ---
        self.global_projs = nn.ModuleDict()

        for mod in self.active_global:
            if mod == 'fft':
                # Two-layer bottleneck for large 'fft' dimension
                self.global_projs[mod] = nn.Sequential(
                    nn.Linear(self.global_dims[mod], 512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(dropout),
                    nn.Linear(512, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout)
                )
            else:
                # Standard projection for medium dimensions like tspulse
                self.global_projs[mod] = nn.Sequential(
                    nn.Linear(self.global_dims[mod], dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout)
                )

        # --- 3. Dynamic Classification Head ---
        # Each active modality contributes `dim` (128) to the final concatenated vector
        total_active = len(self.active_seq) + len(self.active_global)
        head_in_dim = dim * total_active

        self.head = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for mod in self.active_seq:
            nn.init.trunc_normal_(self.seq_embeds[mod], std=0.02)

    def _generate_token_mask(self, batch_size: int, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.training or self.token_mask_prob <= 0.0:
            return None
        rand_tensor = torch.rand(batch_size, seq_len, device=device)
        return rand_tensor < self.token_mask_prob

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = next(iter(batch.values())).size(0)
        fused_features = []

        # --- Phase 1: Process Sequence Modalities ---
        if self.active_seq:
            # 1a. Project and embed all active sequence features
            seq_x = {}
            for mod in self.active_seq:
                raw_feat = batch[f'{mod}_features']
                seq_x[mod] = self.seq_projs[mod](raw_feat) + self.seq_embeds[mod]

            # 1b. Cross-attention and Pooling
            for mod in self.active_seq:
                # Gather all OTHER active sequence modalities for the KV sequence
                other_xs = [seq_x[other] for other in self.active_seq if other != mod]

                # If there are other sequence modalities, concatenate them.
                # If this is the ONLY sequence modality, it attends to itself.
                if other_xs:
                    kv_seq = torch.cat(other_xs, dim=1)
                else:
                    kv_seq = seq_x[mod]

                seq_len = kv_seq.size(1)
                mask = self._generate_token_mask(batch_size, seq_len, kv_seq.device)

                # Cross Attention
                fused = self.cross_attns[mod](seq_x[mod], kv_seq, key_padding_mask=mask)

                # Pooling
                pooled = self.poolers[mod](fused)
                fused_features.append(pooled)

        # --- Phase 2: Process Global Modalities ---
        for mod in self.active_global:
            raw_feat = batch[f'{mod}_features']
            proj = self.global_projs[mod](raw_feat)
            fused_features.append(proj)

        # --- Phase 3: Final Fusion and Classification ---
        # Concatenate all processed modalities
        fused_backbone = torch.cat(fused_features, dim=-1)

        logits = self.head(fused_backbone)

        return {
            'logits': logits,
            'labels': batch.get('labels')
        }