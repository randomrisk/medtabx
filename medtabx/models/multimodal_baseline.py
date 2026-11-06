from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

import medtabx
from medtabx.data import TensorFrame
from medtabx.interaction.pooling_fusion import ConcatPoolingFusion, MeanPoolingFusion


class MultimodalBaseline(nn.Module):
    """Canonical multimodal medical baseline model.

    The model expects a materialized :class:`TensorFrame` and a mapping from
    semantic types to encoder modules. Each encoder should return
    ``[batch_size, num_cols, embedding_dim]``.
    """

    def __init__(
        self,
        stype_encoder_dict: Mapping[medtabx.core.stype, nn.Module],
        decoder: nn.Module,
        fusion: str = "mean",
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.stype_encoder_dict = dict(stype_encoder_dict)
        self.encoders = nn.ModuleDict({
            stype.value: module for stype, module in self.stype_encoder_dict.items()
        })
        self.decoder = decoder
        self.fusion_name = fusion
        if fusion == "mean":
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified for mean fusion.")
            self.fusion = MeanPoolingFusion(channels=embedding_dim)
            self.output_dim = embedding_dim
        elif fusion == "concat":
            if embedding_dim is None:
                raise ValueError("embedding_dim must be specified for concat fusion.")
            self.fusion = ConcatPoolingFusion(channels=embedding_dim)
            self.output_dim = embedding_dim * len(self.encoders)
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion}")

    def encode_modalities(self, tf: TensorFrame) -> dict[str, torch.Tensor]:
        modality_embeddings: dict[str, torch.Tensor] = {}
        for stype_name, feat in tf.feat_dict.items():
            key = stype_name.value
            if key not in self.encoders:
                continue
            modality_embeddings[key] = self.encoders[key](feat)
        if not modality_embeddings:
            raise ValueError("No compatible modality encoders found for the given TensorFrame.")
        return modality_embeddings

    def forward(
        self,
        tf: TensorFrame,
        return_modalities: bool = False,
    ):
        modality_embeddings = self.encode_modalities(tf)
        fused = self.fusion(modality_embeddings)
        decoder_input = fused.unsqueeze(1)
        output = self.decoder(decoder_input)
        if return_modalities:
            return output, modality_embeddings, fused
        return output
