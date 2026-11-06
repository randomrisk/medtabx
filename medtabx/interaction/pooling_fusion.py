"""
Pooling-based fusion strategies for multi-modal data.
"""

import torch
from torch import Tensor
from torch.nn import Module, Linear, Parameter
import torch.nn.functional as F

from .fusion_base import MultiModalFusion


class MeanPoolingFusion(MultiModalFusion):
    """Simple mean pooling fusion.

    Pools each modality to [B, C], then averages all modalities.

    Example:
        >>> fusion = MeanPoolingFusion(channels=64)
        >>> modalities = {
        ...     'tabular': torch.randn(8, 10, 64),
        ...     'imaging': torch.randn(8, 100, 64),
        ... }
        >>> out = fusion(modalities)  # [8, 64]
    """

    def forward(self, modality_dict: dict[str, Tensor]) -> Tensor:
        pooled = []

        for name, feat in modality_dict.items():
            if feat.ndim == 3:  # [B, N, C]
                feat = feat.mean(dim=1)  # Pool to [B, C]
            pooled.append(feat)

        # Average all modalities
        fused = torch.stack(pooled, dim=0).mean(dim=0)  # [B, C]
        return fused


class ConcatPoolingFusion(MultiModalFusion):
    """Concatenation-based fusion.

    Pools each modality to [B, C], then concatenates.
    Output: [B, num_modalities * C]

    Example:
        >>> fusion = ConcatPoolingFusion(channels=64)
        >>> modalities = {
        ...     'tabular': torch.randn(8, 10, 64),
        ...     'imaging': torch.randn(8, 100, 64),
        ... }
        >>> out = fusion(modalities)  # [8, 128]
    """

    def forward(self, modality_dict: dict[str, Tensor]) -> Tensor:
        pooled = []

        for name, feat in modality_dict.items():
            if feat.ndim == 3:  # [B, N, C]
                feat = feat.mean(dim=1)  # Pool to [B, C]
            pooled.append(feat)

        # Concatenate all modalities
        fused = torch.cat(pooled, dim=1)  # [B, num_modalities * C]
        return fused
