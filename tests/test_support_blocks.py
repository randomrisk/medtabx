"""Support-block tests for model registry and text backends."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import medtabx.core.stype as stype
from medtabx.data import Dataset
from medtabx.data.io_tabular import load_csv
from medtabx.data.medical_mappers import ReportTextTensorMapper
from medtabx.models import get_model_class, list_supported_models
from medtabx.text import embed_texts


def test_text_embedding_backends_and_report_mapper():
    texts = ["Left MCA stroke on DWI.", "No perfusion deficit identified."]

    deterministic = embed_texts(texts, backend="deterministic", embedding_dim=16)
    callable_backend = embed_texts(
        texts,
        backend=lambda rows: [[len(row), row.lower().count("stroke")] for row in rows],
    )
    mapped = ReportTextTensorMapper(embedding_dim=16).forward(pd.Series(texts))

    assert deterministic.shape == (2, 16)
    assert callable_backend.shape == (2, 2)
    assert isinstance(mapped, torch.Tensor)
    assert mapped.shape == (2, 16)


def test_model_registry_lists_and_loads_supported_models():
    expected = {
        "FTTransformer",
        "ResNet",
        "TabNet",
        "TabTransformer",
        "Trompt",
        "ExcelFormer",
        "MLP",
        "MultimodalBaseline",
    }
    assert expected.issubset(set(list_supported_models()))

    for name in expected:
        cls = get_model_class(name)
        assert cls.__name__ == name


def test_tabular_model_smoke_suite():
    data_root = Path(__file__).parent.parent / "examples" / "data"
    csv_path = data_root / "tabular" / "diabetes_sample.csv"
    df = load_csv(str(csv_path))

    col_to_stype = {
        "patient_id": stype.categorical,
        "age": stype.numerical,
        "gender": stype.categorical,
        "bmi": stype.numerical,
        "blood_pressure": stype.numerical,
        "glucose": stype.numerical,
        "insulin": stype.numerical,
        "diabetes": stype.categorical,
    }

    dataset = Dataset(df=df, col_to_stype=col_to_stype, target_col="diabetes")
    dataset.materialize()
    tf = dataset.tensor_frame
    col_stats = dataset.col_stats
    col_names_dict = tf.col_names_dict

    ft = get_model_class("FTTransformer")(
        channels=16,
        out_channels=2,
        num_layers=2,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    )
    resnet = get_model_class("ResNet")(
        channels=16,
        out_channels=2,
        num_layers=2,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    )
    tabnet = get_model_class("TabNet")(
        out_channels=2,
        num_layers=2,
        split_feat_channels=8,
        split_attn_channels=8,
        gamma=1.3,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    )
    tabtransformer = get_model_class("TabTransformer")(
        channels=8,
        out_channels=2,
        num_layers=2,
        num_heads=2,
        encoder_pad_size=2,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    )
    trompt = get_model_class("Trompt")(
        channels=8,
        out_channels=2,
        num_prompts=4,
        num_layers=2,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
    )

    numerical_cols = ["age", "bmi", "blood_pressure", "glucose", "insulin"]
    numeric_df = df[numerical_cols + ["diabetes"]].copy()
    numeric_stypes = {col: stype.numerical for col in numerical_cols}
    numeric_stypes["diabetes"] = stype.categorical
    numeric_dataset = Dataset(df=numeric_df, col_to_stype=numeric_stypes, target_col="diabetes")
    numeric_dataset.materialize()
    numeric_tf = numeric_dataset.tensor_frame
    excel = get_model_class("ExcelFormer")(
        in_channels=8,
        out_channels=2,
        num_cols=len(numerical_cols),
        num_layers=2,
        num_heads=2,
        col_stats=numeric_dataset.col_stats,
        col_names_dict=numeric_tf.col_names_dict,
    )

    outputs = [
        ft(tf),
        resnet(tf),
        tabnet(tf),
        tabtransformer(tf),
        trompt(tf),
        excel(numeric_tf),
    ]

    for out in outputs:
        assert out.shape[0] == len(df)
        assert out.shape[-1] == 2
