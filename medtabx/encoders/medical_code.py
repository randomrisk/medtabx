"""Medical code encoder (ICD-10, CPT)."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('medical_code')
class MedicalCodeEncoder(BaseEncoder):
    """
    Encoder for medical codes with hierarchical structure.

    Handles ICD-10, CPT codes with embedding lookup.
    """

    def __init__(self, embedding_dim: int, num_codes: int = 10000):
        super().__init__(embedding_dim)
        self.embedding = nn.Embedding(num_codes, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols] code indices

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        return self.embedding(x.long())
