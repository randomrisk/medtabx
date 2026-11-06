"""fMRI time series encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('fmri')
class FMRIEncoder(BaseEncoder):
    """
    Encoder for fMRI time series (ROI-based or connectivity matrices).
    """

    def __init__(self, embedding_dim: int, num_rois: int = 100, seq_len: int = 200):
        super().__init__(embedding_dim)

        # Temporal convolution for ROI time series
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(num_rois, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, num_rois, seq_len]

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        B, num_cols = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # [B*num_cols, num_rois, seq_len]

        features = self.temporal_conv(x).squeeze(-1)  # [B*num_cols, 256]
        embeddings = self.fc(features)

        return embeddings.view(B, num_cols, self.embedding_dim)
