"""Multilabel classification decoder."""

import torch
import torch.nn as nn

from ..core.registry import decoder_registry


@decoder_registry.register("multilabel")
class MultilabelClassificationDecoder(nn.Module):
    def __init__(self, embedding_dim: int, num_labels: int, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(embedding_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            pooled = x.mean(dim=1)
        elif self.pooling == "max":
            pooled = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.fc(pooled)
