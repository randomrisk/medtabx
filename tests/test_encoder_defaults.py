"""Tests for default encoder selection and auto-configuration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import medtabx
from medtabx import stype
from medtabx.data import Dataset
from medtabx.decoders import ClassificationDecoder
from medtabx.encoders import (
    DEFAULT_MEDICAL_ENCODER_CLS,
    DEFAULT_TABULAR_ENCODER_CLS,
    LinearEncoder,
    StypeWiseFeatureEncoder,
    build_stype_encoder_dict,
    default_tabular_stype_encoder_dict,
    infer_encoder_kwargs,
)
from medtabx.models import MLP
from medtabx.models.multimodal_baseline import MultimodalBaseline

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402


@pytest.fixture
def multimodal_dataset(tmp_path):
    """A small frame covering image, mask, 4D series, text, and tabular."""
    rng = np.random.default_rng(0)
    num_rows = 4

    for i in range(num_rows):
        nib.save(nib.Nifti1Image(rng.random((8, 8, 4)).astype(np.float32),
                                 np.eye(4)), tmp_path / f"v{i}.nii.gz")
        nib.save(nib.Nifti1Image((rng.random((8, 8, 4)) > 0.5).astype(
            np.float32), np.eye(4)), tmp_path / f"m{i}.nii.gz")
        nib.save(nib.Nifti1Image(rng.random((8, 8, 4, 10)).astype(np.float32),
                                 np.eye(4)), tmp_path / f"p{i}.nii.gz")

    df = pd.DataFrame({
        "t1": [str(tmp_path / f"v{i}.nii.gz") for i in range(num_rows)],
        "mask": [str(tmp_path / f"m{i}.nii.gz") for i in range(num_rows)],
        "pwi": [str(tmp_path / f"p{i}.nii.gz") for i in range(num_rows)],
        "age": rng.normal(65, 10, num_rows),
        "nihss": rng.normal(8, 3, num_rows),
        "sex": pd.Categorical(rng.integers(0, 2, num_rows)),
        "report": ["left MCA stroke"] * num_rows,
        "y": pd.Categorical(rng.integers(0, 2, num_rows)),
    })
    return Dataset(df, col_to_stype={
        "t1": stype.image_3d,
        "mask": stype.mask_3d,
        "pwi": stype.timeseries_4d,
        "age": stype.numerical,
        "nihss": stype.numerical,
        "sex": stype.categorical,
        "report": stype.report_text,
        "y": stype.categorical,
    }, target_col="y").materialize()


# Tabular path ###############################################################


def test_tabular_defaults_cover_every_supported_stype():
    encoders = default_tabular_stype_encoder_dict()
    assert set(encoders) == set(DEFAULT_TABULAR_ENCODER_CLS)
    for st, encoder in encoders.items():
        assert st in encoder.supported_stypes


def test_tabular_defaults_return_fresh_instances():
    # StypeWiseFeatureEncoder mutates the encoders it is handed, so two calls
    # must not hand back the same objects.
    first = default_tabular_stype_encoder_dict()
    second = default_tabular_stype_encoder_dict()
    for st in first:
        assert first[st] is not second[st]


def test_tabular_defaults_restricted_to_frame_stypes():
    encoders = default_tabular_stype_encoder_dict(
        stypes=[stype.numerical, stype.categorical])
    assert set(encoders) == {stype.numerical, stype.categorical}


def test_tabular_defaults_map_child_stypes_to_their_parent():
    # report_text is stored under the embedding parent after materialization.
    encoders = default_tabular_stype_encoder_dict(stypes=[stype.report_text])
    assert set(encoders) == {stype.embedding}


def test_tabular_defaults_reject_a_raw_medical_stype():
    with pytest.raises(ValueError, match="image_3d"):
        default_tabular_stype_encoder_dict(stypes=[stype.image_3d])


def test_tabular_defaults_honour_overrides():
    override = LinearEncoder()
    encoders = default_tabular_stype_encoder_dict(
        stypes=[stype.numerical], overrides={stype.numerical: override})
    assert encoders[stype.numerical] is override


def test_report_text_column_reaches_a_tabular_model():
    # Regression: the default dict used to omit `embedding`, so a frame with
    # report text raised KeyError('embedding') on the first forward pass.
    rng = np.random.default_rng(0)
    num_rows = 12
    df = pd.DataFrame({
        "age": rng.normal(65, 10, num_rows),
        "sex": pd.Categorical(rng.integers(0, 2, num_rows)),
        "report": ["left MCA stroke"] * num_rows,
        "y": pd.Categorical(rng.integers(0, 2, num_rows)),
    })
    dataset = Dataset(df, col_to_stype={
        "age": stype.numerical,
        "sex": stype.categorical,
        "report": stype.report_text,
        "y": stype.categorical,
    }, target_col="y").materialize()
    frame = dataset.tensor_frame

    model = MLP(channels=16, out_channels=2, num_layers=2,
                col_stats=dataset.col_stats,
                col_names_dict=frame.col_names_dict)
    _, encoded_cols = model.encoder(frame)

    assert set(encoded_cols) == {"age", "sex", "report"}
    assert model(frame).shape == (num_rows, 2)


def test_stypewise_encoder_names_the_missing_stype_at_construction():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.normal(size=8),
        "sex": pd.Categorical(rng.integers(0, 2, 8)),
        "y": pd.Categorical(rng.integers(0, 2, 8)),
    })
    dataset = Dataset(df, col_to_stype={
        "age": stype.numerical,
        "sex": stype.categorical,
        "y": stype.categorical,
    }, target_col="y").materialize()

    with pytest.raises(ValueError, match="categorical"):
        StypeWiseFeatureEncoder(
            out_channels=8, col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
            stype_encoder_dict={stype.numerical: LinearEncoder()})


# Multimodal path ############################################################


def test_inferred_kwargs_match_the_materialized_shapes(multimodal_dataset):
    frame = multimodal_dataset.tensor_frame
    stats = multimodal_dataset.col_stats

    def kwargs_for(st):
        return infer_encoder_kwargs(st, frame.feat_dict[st], stats,
                                    frame.col_names_dict.get(st))

    assert kwargs_for(stype.image_3d) == {"in_channels": 1}
    assert kwargs_for(stype.mask_3d) == {"num_classes": 1}
    assert kwargs_for(stype.timeseries_4d) == {"num_rois": 2, "seq_len": 10}
    assert kwargs_for(stype.numerical) == {"num_cols": 2}
    assert kwargs_for(stype.embedding) == {"input_dim": 16}
    assert kwargs_for(stype.categorical)["num_categories"] >= 1


def test_build_stype_encoder_dict_covers_the_whole_frame(multimodal_dataset):
    frame = multimodal_dataset.tensor_frame
    encoders = build_stype_encoder_dict(
        frame, embedding_dim=8, col_stats=multimodal_dataset.col_stats)

    assert set(encoders) == set(frame.feat_dict)
    for st, encoder in encoders.items():
        assert isinstance(encoder, DEFAULT_MEDICAL_ENCODER_CLS[st])
        assert encoder(frame.feat_dict[st]).shape == (
            len(frame), len(frame.col_names_dict[st]), 8)


def test_auto_configured_multimodal_model_runs(multimodal_dataset):
    frame = multimodal_dataset.tensor_frame
    model = MultimodalBaseline(
        stype_encoder_dict=build_stype_encoder_dict(
            frame, embedding_dim=8, col_stats=multimodal_dataset.col_stats),
        decoder=ClassificationDecoder(embedding_dim=8, num_classes=2),
        embedding_dim=8,
    )
    assert model(frame).shape == (len(frame), 2)


def test_include_selects_a_single_modality_arm(multimodal_dataset):
    frame = multimodal_dataset.tensor_frame
    encoders = build_stype_encoder_dict(
        frame, embedding_dim=8, col_stats=multimodal_dataset.col_stats,
        include=[stype.image_3d])
    assert set(encoders) == {stype.image_3d}


def test_exclude_drops_a_modality(multimodal_dataset):
    frame = multimodal_dataset.tensor_frame
    encoders = build_stype_encoder_dict(
        frame, embedding_dim=8, col_stats=multimodal_dataset.col_stats,
        exclude=[stype.image_3d, stype.mask_3d])
    assert stype.image_3d not in encoders
    assert stype.mask_3d not in encoders
    assert stype.numerical in encoders


def test_include_rejects_a_stype_absent_from_the_frame(multimodal_dataset):
    with pytest.raises(ValueError, match="eeg"):
        build_stype_encoder_dict(
            multimodal_dataset.tensor_frame, embedding_dim=8,
            col_stats=multimodal_dataset.col_stats, include=[stype.eeg])


def test_filtering_everything_out_is_an_error(multimodal_dataset):
    with pytest.raises(ValueError, match="No encoders were built"):
        build_stype_encoder_dict(
            multimodal_dataset.tensor_frame, embedding_dim=8,
            col_stats=multimodal_dataset.col_stats,
            exclude=list(multimodal_dataset.tensor_frame.feat_dict))


def test_overrides_bypass_inference(multimodal_dataset):
    from medtabx.encoders import NumericEncoder

    override = NumericEncoder(embedding_dim=8, num_cols=2)
    encoders = build_stype_encoder_dict(
        multimodal_dataset.tensor_frame, embedding_dim=8,
        col_stats=multimodal_dataset.col_stats,
        overrides={stype.numerical: override})
    assert encoders[stype.numerical] is override


def test_categorical_without_col_stats_says_what_is_missing(
        multimodal_dataset):
    with pytest.raises(ValueError, match="col_stats"):
        build_stype_encoder_dict(multimodal_dataset.tensor_frame,
                                 embedding_dim=8)


def test_ads_benchmark_does_not_model_the_identifier_column():
    from pathlib import Path

    from medtabx.benchmarks import build_ads_benchmark_dataset

    root = Path(__file__).parent.parent / "examples" / "data" / "ads_case"
    dataset = build_ads_benchmark_dataset(
        root, target_col="target_has_stroke").materialize()

    modelled = {
        col
        for cols in dataset.tensor_frame.col_names_dict.values()
        for col in cols
    }
    assert "subject_id" not in modelled
