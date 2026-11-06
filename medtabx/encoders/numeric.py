"""Numeric column encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('numeric')
class NumericEncoder(BaseEncoder):
    """
    Encoder for numerical columns.

    Projects numerical values to embedding space via linear layer.
    """

    def __init__(self, embedding_dim: int, num_cols: int = 1):
        super().__init__(embedding_dim)
        self.num_cols = num_cols
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols] numerical values

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        # Reshape to [B, num_cols, 1]
        x = x.unsqueeze(-1)
        # Project to [B, num_cols, embedding_dim]
        return self.linear(x)
