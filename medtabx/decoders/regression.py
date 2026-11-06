"""Regression decoder."""
import torch
import torch.nn as nn
from ..core.registry import decoder_registry


@decoder_registry.register('regression')
class RegressionDecoder(nn.Module):
    """
    Decoder for regression tasks.

    Aggregates column embeddings and predicts continuous values.
    """

    def __init__(self, embedding_dim: int, num_targets: int = 1, pooling: str = 'mean'):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(embedding_dim, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, embedding_dim]

        Returns:
            [batch_size, num_targets] predictions
        """
        if self.pooling == 'mean':
            pooled = x.mean(dim=1)
        elif self.pooling == 'max':
            pooled = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.fc(pooled)
