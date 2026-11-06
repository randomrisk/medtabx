"""Tests for the canonical multimodal baseline model and new heads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

import medtabx
from medtabx.data import Dataset
from medtabx.decoders import (
    ClassificationDecoder,
    MultilabelClassificationDecoder,
    OrdinalDecoder,
    RegressionDecoder,
    SurvivalDecoder,
)
from medtabx.encoders import (
    FMRIEncoder,
    Imaging3DEncoder,
    Mask3DEncoder,
    NumericEncoder,
    TextEmbeddedEncoder,
    VitalSignEncoder,
)
from medtabx.models.multimodal_baseline import MultimodalBaseline


def _build_dataset() -> Dataset:
    fixture_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    vitals_path = Path(__file__).parent.parent / "examples" / "data" / "vitals" / "vitals_timeseries.csv"
    df = pd.DataFrame(
        {
            "dwi_path": [str(fixture_root / "imaging" / "sub-02e8eb42_DWI.nii.gz")],
            "stroke_mask_path": [str(fixture_root / "masks" / "sub-02e8eb42_stroke_mask.nii.gz")],
            "pwi_path": [str(fixture_root / "imaging" / "sub-02e8eb42_PWI.nii.gz")],
            "report_text": [(fixture_root / "text" / "sub-02e8eb42_dwi_report.txt").read_text()],
            "derived_volume": [123.4],
            "vitals_path": [str(vitals_path)],
        }
    )
    dataset = Dataset(
        df=df,
        col_to_stype={
            "dwi_path": medtabx.image_3d,
            "stroke_mask_path": medtabx.mask_3d,
            "pwi_path": medtabx.fmri,
            "report_text": medtabx.report_text,
            "derived_volume": medtabx.tabular_derived,
            "vitals_path": medtabx.vital_sign,
        },
    )
    dataset.materialize()
    return dataset


def test_multimodal_baseline_and_decoder_heads():
    dataset = _build_dataset()
    tf = dataset.tensor_frame
    emb_dim = 32

    stype_encoder_dict = {
        medtabx.image_3d: Imaging3DEncoder(embedding_dim=emb_dim, in_channels=1),
        medtabx.mask_3d: Mask3DEncoder(embedding_dim=emb_dim, num_classes=1),
        medtabx.fmri: FMRIEncoder(
            embedding_dim=emb_dim,
            num_rois=tf.feat_dict[medtabx.fmri].shape[2],
            seq_len=tf.feat_dict[medtabx.fmri].shape[-1],
        ),
        medtabx.vital_sign: VitalSignEncoder(
            embedding_dim=emb_dim,
            seq_len=tf.feat_dict[medtabx.vital_sign].shape[-1],
        ),
        medtabx.embedding: TextEmbeddedEncoder(
            embedding_dim=emb_dim,
            input_dim=tf.feat_dict[medtabx.embedding].shape[-1],
        ),
        medtabx.numerical: NumericEncoder(
            embedding_dim=emb_dim,
            num_cols=tf.feat_dict[medtabx.numerical].shape[1],
        ),
    }

    classifiers = [
        ClassificationDecoder(embedding_dim=emb_dim, num_classes=2),
        RegressionDecoder(embedding_dim=emb_dim, num_targets=1),
        MultilabelClassificationDecoder(embedding_dim=emb_dim, num_labels=3),
        OrdinalDecoder(embedding_dim=emb_dim, num_levels=4),
        SurvivalDecoder(embedding_dim=emb_dim),
    ]

    expected_shapes = [(1, 2), (1, 1), (1, 3), (1, 3), (1, 1)]
    for decoder, expected_shape in zip(classifiers, expected_shapes):
        model = MultimodalBaseline(
            stype_encoder_dict=stype_encoder_dict,
            decoder=decoder,
            fusion="mean",
            embedding_dim=emb_dim,
        )
        out, modality_embeddings, fused = model(tf, return_modalities=True)
        assert out.shape == expected_shape
        assert fused.shape == (1, emb_dim)
        assert len(modality_embeddings) >= 4

    logits = classifiers[0](torch.randn(1, 1, emb_dim))
    assert logits.shape == (1, 2)
