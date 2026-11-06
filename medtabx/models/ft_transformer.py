from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential

import medtabx
from medtabx.core import stype
from medtabx.data import TensorFrame
from medtabx.data.stats import StatType
from medtabx.encoders.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from medtabx.encoders import default_tabular_stype_encoder_dict
from medtabx.encoders.stypewise_encoder import StypeWiseFeatureEncoder
from medtabx.interaction.ft_transformer_convs import FTTransformerConvs


class FTTransformer(Module):
    r"""The FT-Transformer model introduced in the
    `"Revisiting Deep Learning Models for Tabular Data"
    <https://arxiv.org/abs/2106.11959>`_ paper.

    .. note::

        For an example of using FTTransformer, see `examples/revisiting.py
        <https://github.com/pyg-team/pytorch-frame/blob/master/examples/
        revisiting.py>`_.

    Args:
        channels (int): Hidden channel dimensionality
        out_channels (int): Output channels dimensionality
        num_layers (int): Number of layers.  (default: :obj:`3`)
        col_stats(dict[str,dict[:class:`medtabx.data.stats.StatType`,Any]]):
             A dictionary that maps column name into stats.
             Available as :obj:`dataset.col_stats`.
        col_names_dict (dict[:obj:`medtabx.core.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`. Available as
            :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`medtabx.core.stype`,
            :class:`medtabx.encoders.StypeEncoder`], optional):
            A dictionary mapping stypes into their stype encoders.
            (default: :obj:`None`, will call
            :class:`medtabx.encoders.EmbeddingEncoder()` for categorical
            feature and :class:`medtabx.encoders.LinearEncoder()`
            for numerical feature)
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
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer (got {num_layers})")

        if stype_encoder_dict is None:
            stype_encoder_dict = default_tabular_stype_encoder_dict(
                stypes=list(col_names_dict))

        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.backbone = FTTransformerConvs(channels=channels,
                                           num_layers=num_layers)
        self.decoder = Sequential(
            LayerNorm(channels),
            ReLU(),
            Linear(channels, out_channels),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.backbone.reset_parameters()
        for m in self.decoder:
            if not isinstance(m, ReLU):
                m.reset_parameters()

    def forward(self, tf: TensorFrame) -> Tensor:
        r"""Transforming :class:`TensorFrame` object into output prediction.

        Args:
            tf (TensorFrame):
                Input :class:`TensorFrame` object.

        Returns:
            torch.Tensor: Output of shape [batch_size, out_channels].
        """
        x, _ = self.encoder(tf)
        x, x_cls = self.backbone(x)
        out = self.decoder(x_cls)
        return out
