"""3D segmentation mask encoder."""
import torch
import torch.nn as nn
from .base import BaseEncoder
from ..core.registry import encoder_registry


@encoder_registry.register('mask3d')
class Mask3DEncoder(BaseEncoder):
    """
    Encoder for 3D segmentation masks.

    Extracts features from binary or multi-class segmentation masks.
    """

    def __init__(self, embedding_dim: int, num_classes: int = 1):
        super().__init__(embedding_dim)
        self.num_classes = num_classes

        # 3D CNN for mask features
        self.conv_layers = nn.Sequential(
            nn.Conv3d(num_classes, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )

        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, num_classes, D, H, W] mask volumes

        Returns:
            [batch_size, num_cols, embedding_dim]
        """
        B, num_cols = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # [B*num_cols, num_classes, D, H, W]

        features = self.conv_layers(x)  # [B*num_cols, 64, 1, 1, 1]
        features = features.view(features.size(0), -1)  # [B*num_cols, 64]

        embeddings = self.fc(features)  # [B*num_cols, embedding_dim]
        return embeddings.view(B, num_cols, self.embedding_dim)
