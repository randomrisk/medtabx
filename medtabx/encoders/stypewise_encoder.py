from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleDict

import medtabx
from medtabx.data import TensorFrame
from medtabx.data.stats import StatType
from medtabx.encoders.encoder import FeatureEncoder
from medtabx.encoders.stype_encoder import StypeEncoder


class StypeWiseFeatureEncoder(FeatureEncoder):
    r"""Feature encoder that transforms each stype tensor into embeddings and
    performs the final concatenation.

    Args:
        out_channels (int): Output dimensionality.
        col_stats
            (dict[str, dict[:class:`medtabx.data.stats.StatType`, Any]]):
            A dictionary that maps column name into stats. Available as
            :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`medtabx.core.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`medtabx.core.stype`,
            :class:`medtabx.encoders.StypeEncoder`]):
            A dictionary that maps :class:`medtabx.core.stype` into
            :class:`medtabx.encoders.StypeEncoder` class. Only
            parent :class:`stypes <medtabx.core.stype>` are supported
            as keys.
    """
    def __init__(
        self,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[medtabx.core.stype, list[str]],
        stype_encoder_dict: dict[medtabx.core.stype, StypeEncoder],
    ) -> None:
        super().__init__()

        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.encoder_dict = ModuleDict()
        for stype, stype_encoder in stype_encoder_dict.items():
            if stype != stype.parent:
                if stype.parent in stype_encoder_dict:
                    msg = (
                        f"You can delete this {stype} directly since encoder "
                        f"for parent stype {stype.parent} is already declared."
                    )
                else:
                    msg = (f"To resolve the issue, you can change the key from"
                           f" {stype} to {stype.parent}.")
                raise ValueError(f"{stype} is an invalid stype to use in the "
                                 f"stype_encoder_dcit. {msg}")
            if stype not in stype_encoder.supported_stypes:
                raise ValueError(
                    f"{stype_encoder} does not support encoding {stype}.")

            if stype in col_names_dict:
                stats_list = [
                    self.col_stats[col_name]
                    for col_name in self.col_names_dict[stype]
                ]
                # Set lazy attributes
                stype_encoder.stype = stype
                stype_encoder.out_channels = out_channels
                stype_encoder.stats_list = stats_list
                self.encoder_dict[stype.value] = stype_encoder

        # Catch an unencodable column here rather than at the first forward
        # pass, where it surfaces as a bare KeyError on the stype name.
        missing = [
            col_stype for col_stype in col_names_dict
            if col_stype.value not in self.encoder_dict
        ]
        if missing:
            raise ValueError(
                f"No encoder was given for "
                f"{sorted(s.value for s in missing)}, but the frame has "
                f"columns of those stypes "
                f"({ {s.value: col_names_dict[s] for s in missing} }). "
                f"Pass an encoder for them in stype_encoder_dict, or use "
                f"medtabx.encoders.default_tabular_stype_encoder_dict("
                f"stypes=list(col_names_dict)) to cover every stype in the "
                f"frame.")

    def forward(self, tf: TensorFrame) -> tuple[Tensor, list[str]]:
        all_col_names = []
        xs = []
        for stype in tf.stypes:
            feat = tf.feat_dict[stype]
            col_names = self.col_names_dict[stype]
            x = self.encoder_dict[stype.value](feat, col_names)
            xs.append(x)
            all_col_names.extend(col_names)
        x = torch.cat(xs, dim=1)
        return x, all_col_names
