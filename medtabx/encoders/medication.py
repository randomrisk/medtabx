"""Medication encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('medication')
class MedicationEncoder(BaseEncoder):
    """
    Encoder for medication data (drug + dosage + frequency).
    """

    def __init__(self, embedding_dim: int, num_drugs: int = 5000):
        super().__init__(embedding_dim)
        self.drug_embedding = nn.Embedding(num_drugs, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols] drug indices

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        return self.drug_embedding(x.long())
