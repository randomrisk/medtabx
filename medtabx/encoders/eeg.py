"""EEG time series encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('eeg')
class EEGEncoder(BaseEncoder):
    """
    Encoder for EEG time series data.

    Uses 1D convolutions to extract temporal features.
    """

    def __init__(self, embedding_dim: int, num_channels: int = 64, seq_len: int = 1000):
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
        """
        Args:
            x: [batch_size, num_cols, num_channels, seq_len]

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        B, num_cols = x.shape[:2]

        # Flatten batch and column dimensions
        x = x.view(-1, *x.shape[2:])  # [B*num_cols, num_channels, seq_len]

        # Extract features
        features = self.conv_layers(x)  # [B*num_cols, 256, 1]
        features = features.squeeze(-1)  # [B*num_cols, 256]

        # Project to embedding space
        embeddings = self.fc(features)  # [B*num_cols, embedding_dim]

        return embeddings.view(B, num_cols, self.embedding_dim)
