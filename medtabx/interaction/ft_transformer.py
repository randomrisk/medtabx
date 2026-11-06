"""FT-Transformer interaction layer."""
import torch
import torch.nn as nn
from .base import TableConv
from ..core.registry import interaction_registry


@interaction_registry.register('ft_transformer')
class FTTransformer(TableConv):
    """
    Feature Tokenizer Transformer for tabular data.

    Uses standard transformer architecture for column interactions.
    """

    def __init__(self, channels: int, num_layers: int = 3, num_heads: int = 8):
        super().__init__(channels, num_layers)
        self.num_heads = num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, channels]

        Returns:
            [batch_size, num_cols, channels]
        """
        return self.transformer(x)
