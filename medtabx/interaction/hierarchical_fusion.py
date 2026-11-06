"""
Hierarchical fusion for multi-modal data.

Combines multiple fusion strategies in a hierarchical manner.
"""

import torch
from torch import Tensor
from torch.nn import Module, Linear, ModuleDict

from .fusion_base import MultiModalFusion
from .pooling_fusion import MeanPoolingFusion
from .attention_fusion import AttentionPoolingFusion


class HierarchicalFusion(MultiModalFusion):
    """Hierarchical fusion combining multiple strategies.

    First applies intra-modality fusion (within each modality),
    then inter-modality fusion (across modalities).

    Args:
        channels (int): Channel dimension
        fusion_type (str): Type of inter-modality fusion
            Options: 'mean', 'attention', 'concat'
    """

    def __init__(
        self,
        channels: int,
        modality_names: list[str] | None = None,
        fusion_type: str = 'attention',
    ):
        super().__init__(channels, modality_names)
        self.fusion_type = fusion_type

        # Intra-modality attention (within each modality)
        self.intra_attention = ModuleDict()

        # Inter-modality fusion
        if fusion_type == 'attention':
            self.inter_fusion = AttentionPoolingFusion(channels)
        elif fusion_type == 'mean':
            self.inter_fusion = MeanPoolingFusion(channels)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, modality_dict: dict[str, Tensor]) -> Tensor:
        """Apply hierarchical fusion.

        Args:
            modality_dict: Dictionary of modality tensors

        Returns:
            Fused tensor [B, C]
        """
        # Step 1: Intra-modality fusion (pool each modality)
        pooled_dict = {}
        for name, feat in modality_dict.items():
            if feat.ndim == 3:  # [B, N, C]
                # Mean pooling within modality
                feat = feat.mean(dim=1)  # [B, C]
            pooled_dict[name] = feat

        # Step 2: Inter-modality fusion
        fused = self.inter_fusion(pooled_dict)

        return fused
