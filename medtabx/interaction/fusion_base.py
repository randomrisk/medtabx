"""
Multi-modal fusion base class for MedTabX.

This module handles fusion of features from different medical modalities
(tabular, imaging, time series, text, etc.) after encoding.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module


class MultiModalFusion(Module, ABC):
    """Base class for multi-modal fusion modules.

    Unlike TableConv (which processes single modality [B, num_cols, C]),
    MultiModalFusion combines multiple modalities into a unified representation.

    Args:
        channels (int): Channel dimension for all modalities
        modality_names (list[str]): Names of modalities to fuse
    """

    def __init__(
        self,
        channels: int,
        modality_names: list[str] | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.modality_names = modality_names or []

    @abstractmethod
    def forward(
        self,
        modality_dict: dict[str, Tensor],
    ) -> Tensor:
        """Fuse multiple modality embeddings.

        Args:
            modality_dict: Dictionary mapping modality names to tensors.
                Each tensor can be:
                - [B, num_tokens, C] (token-level features)
                - [B, C] (already pooled features)

        Returns:
            Fused tensor of shape [B, C] or [B, out_channels]
        """
        raise NotImplementedError
