"""Vital sign time series encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('vital_sign')
class VitalSignEncoder(BaseEncoder):
    """
    Encoder for vital sign time series (heart rate, blood pressure, etc.).

    Handles missing values and temporal patterns.
    """

    def __init__(self, embedding_dim: int, seq_len: int = 100):
        super().__init__(embedding_dim)

        # 1D Conv for temporal patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, seq_len] vital sign sequences

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        B, num_cols, seq_len = x.shape

        # Reshape for 1D conv: [B*num_cols, 1, seq_len]
        x = x.view(B * num_cols, 1, seq_len)

        # Extract features
        features = self.conv_layers(x).squeeze(-1)  # [B*num_cols, 128]
        embeddings = self.fc(features)  # [B*num_cols, embedding_dim]

        return embeddings.view(B, num_cols, self.embedding_dim)
