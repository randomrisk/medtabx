"""Base encoder classes for MedTabX."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseEncoder(nn.Module, ABC):
    """
    Base class for all feature encoders.

    All encoders must output column-wise embeddings of shape
    ``[batch_size, num_cols, embedding_dim]``.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to embedding.

        Args:
            x: Input tensor (shape varies by encoder type)

        Returns:
            Embedding tensor of shape [batch_size, num_cols, embedding_dim]
        """
        pass
