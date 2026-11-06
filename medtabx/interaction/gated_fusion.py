"""
Gated fusion for multi-modal data.

This module learns importance weights for each modality dynamically.
"""

import torch
from torch import Tensor
from torch.nn import Module, Linear, Sequential, ReLU, Sigmoid
import torch.nn.functional as F

from .fusion_base import MultiModalFusion


class GatedFusion(MultiModalFusion):
    """Gated fusion with learnable modality importance weights.

    Learns to weight each modality based on the input, allowing the model
    to dynamically adjust which modalities are most important.

    Args:
        channels (int): Channel dimension
        hidden_dim (int): Hidden dimension for gating network (default: channels//2)

    Example:
        >>> fusion = GatedFusion(channels=64)
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
        hidden_dim: int | None = None,
    ):
        super().__init__(channels, modality_names)
        self.hidden_dim = hidden_dim or (channels // 2)

        # Gating network: learns importance weight for each modality
        # Input: concatenated modality features [B, num_modalities * C]
        # Output: importance weights [B, num_modalities]
        self.gate_network = Sequential(
            Linear(channels, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, 1),  # Will be applied per modality
        )

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.gate_network:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, modality_dict: dict[str, Tensor]) -> Tensor:
        """Apply gated fusion with learned importance weights.

        Args:
            modality_dict: Dictionary of modality tensors

        Returns:
            Fused tensor [B, C]
        """
        # Step 1: Pool each modality to [B, C]
        pooled_modalities = []
        for name, feat in modality_dict.items():
            if feat.ndim == 3:  # [B, N, C]
                feat = feat.mean(dim=1)  # [B, C]
            pooled_modalities.append(feat)

        # Step 2: Compute gate weights for each modality
        gate_weights = []
        for feat in pooled_modalities:
            weight = self.gate_network(feat)  # [B, 1]
            gate_weights.append(weight)

        # Stack and normalize with softmax
        gate_weights = torch.cat(gate_weights, dim=1)  # [B, num_modalities]
        gate_weights = F.softmax(gate_weights, dim=1)  # [B, num_modalities]

        # Step 3: Weighted combination
        stacked = torch.stack(pooled_modalities, dim=1)  # [B, num_modalities, C]
        fused = torch.einsum('bm,bmc->bc', gate_weights, stacked)  # [B, C]

        return fused
