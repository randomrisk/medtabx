"""
Attention-based fusion for multi-modal data.
"""

import torch
from torch import Tensor
from torch.nn import Module, Linear, Parameter
import torch.nn.functional as F

from .fusion_base import MultiModalFusion


class AttentionPoolingFusion(MultiModalFusion):
    """Attention-based pooling within each modality, then concatenate.

    For each modality:
    1. Compute attention weights over tokens
    2. Weighted sum to get [B, C]
    3. Concatenate all modalities

    Args:
        channels (int): Channel dimension
        num_heads (int): Number of attention heads (default: 1)
    """

    def __init__(
        self,
        channels: int,
        modality_names: list[str] | None = None,
        num_heads: int = 1,
    ):
        super().__init__(channels, modality_names)
        self.num_heads = num_heads

        # Attention query for each modality
        self.attention_query = Parameter(torch.randn(channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.attention_query, std=0.01)

    def forward(self, modality_dict: dict[str, Tensor]) -> Tensor:
        pooled = []

        for name, feat in modality_dict.items():
            if feat.ndim == 3:  # [B, N, C]
                # Compute attention scores
                scores = torch.einsum('bnc,c->bn', feat, self.attention_query)
                weights = F.softmax(scores, dim=1)  # [B, N]

                # Weighted sum
                feat = torch.einsum('bn,bnc->bc', weights, feat)  # [B, C]

            pooled.append(feat)

        # Concatenate all modalities
        fused = torch.cat(pooled, dim=1)  # [B, num_modalities * C]
        return fused
