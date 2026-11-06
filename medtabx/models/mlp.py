from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LayerNorm,
    Linear,
    Module,
    ReLU,
    Sequential,
)

import medtabx
from medtabx.core import stype
from medtabx.data import TensorFrame
from medtabx.data.stats import StatType
from medtabx.encoders import (
    StypeEncoder,
    StypeWiseFeatureEncoder,
    default_tabular_stype_encoder_dict,
)


class MLP(Module):
    r"""The light-weight MLP model that mean-pools column embeddings and
    applies MLP over it.

    Args:
        channels (int): The number of channels in the backbone layers.
        out_channels (int): The number of output channels in the decoder.
        num_layers (int): The number of layers in the backbone.
        col_stats(dict[str,Dict[:class:`medtabx.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`medtabx.core.stype`, List[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`medtabx.core.stype`,
            :class:`medtabx.encoders.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call :obj:`EmbeddingEncoder()`
            for categorical feature and :obj:`LinearEncoder()` for
            numerical feature)
        normalization (str, optional): The type of normalization to use.
            :obj:`batch_norm`, :obj:`layer_norm`, or :obj:`None`.
            (default: :obj:`layer_norm`)
        dropout_prob (float): The dropout probability (default: `0.2`).
    """
    def __init__(
        self,
        channels: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[medtabx.core.stype, list[str]],
        stype_encoder_dict: dict[medtabx.core.stype, StypeEncoder]
        | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        if stype_encoder_dict is None:
            stype_encoder_dict = default_tabular_stype_encoder_dict(
                stypes=list(col_names_dict))

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

        self.mlp = Sequential()

        for _ in range(num_layers - 1):
            self.mlp.append(Linear(channels, channels))
            if normalization == "layer_norm":
                self.mlp.append(LayerNorm(channels))
            elif normalization == "batch_norm":
                self.mlp.append(BatchNorm1d(channels))
            self.mlp.append(ReLU())
            self.mlp.append(Dropout(p=dropout_prob))
        self.mlp.append(Linear(channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        for param in self.mlp:
            if hasattr(param, 'reset_parameters'):
                param.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame): Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)

        x = torch.mean(x, dim=1)

        out = self.mlp(x)
        return out
