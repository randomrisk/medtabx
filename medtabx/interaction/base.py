"""Base class for interaction/TableConv modules."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class TableConv(nn.Module, ABC):
    """
    Base class for table interaction layers.

    All TableConv modules take [B, num_cols, C] and output [B, num_cols, C].
    """

    def __init__(self, channels: int, num_layers: int = 1):
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_cols, channels]

        Returns:
            [batch_size, num_cols, channels]
        """
        pass
