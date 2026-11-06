"""Lab value encoder with reference ranges."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('lab_value')
class LabValueEncoder(BaseEncoder):
    """
    Encoder for lab values with reference range normalization.
    """

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim)
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols] lab values

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        x = x.unsqueeze(-1)  # [B, num_cols, 1]
        return self.linear(x)
