"""3D medical imaging encoder (CT/MRI)."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('imaging3d')
class Imaging3DEncoder(BaseEncoder):
    """
    Encoder for 3D medical images (CT/MRI).

    Uses 3D CNN to extract features from volumetric data.
    """

    def __init__(self, embedding_dim: int, in_channels: int = 1):
        super().__init__(embedding_dim)

        # Simple 3D CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, C, D, H, W] 3D volumes

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        B, num_cols = x.shape[:2]

        # Flatten batch and column dimensions
        x = x.view(-1, *x.shape[2:])  # [B*num_cols, C, D, H, W]

        # Extract features
        features = self.conv_layers(x)  # [B*num_cols, 128, 1, 1, 1]
        features = features.view(features.size(0), -1)  # [B*num_cols, 128]

        # Project to embedding space
        embeddings = self.fc(features)  # [B*num_cols, embedding_dim]

        # Reshape back
        return embeddings.view(B, num_cols, self.embedding_dim)
