"""ECG time series encoder."""

import torch
import torch.nn as nn

from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register("ecg")
class ECGEncoder(BaseEncoder):
    def __init__(self, embedding_dim: int, num_channels: int = 12, seq_len: int = 1000):
        super().__init__(embedding_dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_cols = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        features = self.conv_layers(x).squeeze(-1)
        embeddings = self.fc(features)
        return embeddings.view(batch_size, num_cols, self.embedding_dim)
