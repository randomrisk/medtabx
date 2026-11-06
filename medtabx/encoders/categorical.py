"""Categorical column encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('categorical')
class CategoricalEncoder(BaseEncoder):
    """
    Encoder for categorical columns using embedding lookup.
    """

    def __init__(self, embedding_dim: int, num_categories: int):
        super().__init__(embedding_dim)
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols] categorical indices

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        return self.embedding(x.long())
