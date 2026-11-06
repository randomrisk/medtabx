"""Text column encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('text_embedded')
class TextEmbeddedEncoder(BaseEncoder):
    """
    Encoder for pre-embedded text columns.

    Assumes text has already been embedded (e.g., via sentence transformers).
    """

    def __init__(self, embedding_dim: int, input_dim: int = 768):
        super().__init__(embedding_dim)
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, input_dim] pre-embedded text

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        return self.projection(x)
