"""
Cross-modal attention fusion for multi-modal data.

This module allows different modalities to attend to each other,
enabling cross-modal information exchange.
"""

import torch
from torch import Tensor
from torch.nn import Module, Linear, MultiheadAttention, LayerNorm
import torch.nn.functional as F

from .fusion_base import MultiModalFusion


class CrossModalAttentionFusion(MultiModalFusion):
    """Cross-modal attention fusion.

    Each modality attends to all other modalities using multi-head attention.
    This allows modalities to exchange information before final fusion.

    Args:
        channels (int): Channel dimension
        num_heads (int): Number of attention heads (default: 4)
        dropout (float): Dropout rate (default: 0.1)

    Example:
        >>> fusion = CrossModalAttentionFusion(channels=64, num_heads=4)
        >>> modalities = {
        ...     'tabular': torch.randn(8, 10, 64),
        ...     'imaging': torch.randn(8, 100, 64),
        ... }
        >>> out = fusion(modalities)  # [8, 64]
    """

    def __init__(
        self,
        channels: int,
        modality_names: list[str] | None = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(channels, modality_names)
        self.num_heads = num_heads

        # Multi-head attention for cross-modal interaction
        self.cross_attention = MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm = LayerNorm(channels)

        # Final projection
        self.output_proj = Linear(channels, channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.output_proj.reset_parameters()

    def forward(self, modality_dict: dict[str, Tensor]) -> Tensor:
        """Apply cross-modal attention and fuse.

        Args:
            modality_dict: Dictionary of modality tensors

        Returns:
            Fused tensor [B, C]
        """
        # Step 1: Pool each modality to get representative tokens
        modality_tokens = []
        for name, feat in modality_dict.items():
            if feat.ndim == 3:  # [B, N, C]
                # Use mean pooling to get one token per modality
                token = feat.mean(dim=1, keepdim=True)  # [B, 1, C]
            else:  # [B, C]
                token = feat.unsqueeze(1)  # [B, 1, C]
            modality_tokens.append(token)

        # Step 2: Stack all modality tokens
        # [B, num_modalities, C]
        all_tokens = torch.cat(modality_tokens, dim=1)

        # Step 3: Apply cross-modal attention
        # Each modality attends to all other modalities
        attn_out, _ = self.cross_attention(
            query=all_tokens,
            key=all_tokens,
            value=all_tokens,
        )  # [B, num_modalities, C]

        # Step 4: Residual connection + normalization
        all_tokens = self.norm(all_tokens + attn_out)

        # Step 5: Aggregate all modalities (mean pooling)
        fused = all_tokens.mean(dim=1)  # [B, C]

        # Step 6: Final projection
        fused = self.output_proj(fused)  # [B, C]

        return fused
