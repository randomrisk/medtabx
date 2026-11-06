from __future__ import annotations

from pathlib import Path

import pandas as pd

import medtabx
from medtabx.core.schema import DatasetSchema
from medtabx.data import Dataset
from medtabx.features.ads_features import build_ads_feature_table_from_root
from medtabx.features.harmonization import build_harmonized_subject_view
from medtabx.transforms import apply_schema_transforms


#: Columns that identify a row rather than describe it. Never modelled.
IDENTIFIER_COLS = {"subject_id"}

RAW_COLS = {
    "subject_id",
    "dwi_path",
    "pwi_path",
    "stroke_mask_path",
    "hp_mask_path",
    "dwi_report_text",
    "pwi_report_text",
}

def build_ads_benchmark_dataframe(dataset_root: str | Path) -> pd.DataFrame:
    df = build_ads_feature_table_from_root(dataset_root)
    df = df.copy()
    df["target_has_stroke"] = (df["stroke_mask_voxels"] > 0).astype(int)
    df["target_brain_ttp_mean"] = df["brain_ttp_mean"].astype(float)
    return df


def build_ads_harmonized_benchmark_dataframe(
    dataset_root: str | Path,
    *,
    reference_col: str = "dwi_path",
    harmonize_cols: list[str] | None = None,
    output_root: str | Path | None = None,
) -> pd.DataFrame:
    """Build ADS benchmark dataframe with selected raw paths harmonized to a reference.

    This is a non-registration path. It uses subject manifests plus direct
    resampling into the chosen reference image space.
    """
    dataset_root = Path(dataset_root)
    df = build_ads_benchmark_dataframe(dataset_root)
    harmonize_cols = ["stroke_mask_path", "hp_mask_path"] if harmonize_cols is None else harmonize_cols

    rows = []
    for _, row in df.iterrows():
        subject_root = dataset_root / str(row["subject_id"])
        reference_path = Path(row[reference_col])
        reference_relpath = str(reference_path.resolve().relative_to(subject_root.resolve()))

        path_columns = {col: row[col] for col in [reference_col, *harmonize_cols] if col in row}
        subject_output_dir = None
        if output_root is not None:
            subject_output_dir = Path(output_root) / str(row["subject_id"])

        harmonized = build_harmonized_subject_view(
            subject_root,
            reference_relpath=reference_relpath,
            path_columns=path_columns,
            include_columns=harmonize_cols,
            output_dir=subject_output_dir,
        )
        updated_row = row.to_dict()
        updated_row.update(harmonized)
        rows.append(updated_row)
    return pd.DataFrame(rows)


def build_ads_benchmark_stype_map(df: pd.DataFrame) -> dict[str, medtabx.core.stype]:
    base_col_to_stype: dict[str, medtabx.core.stype] = {
        "dwi_path": medtabx.image_3d,
        "pwi_path": medtabx.fmri,
        "stroke_mask_path": medtabx.mask_3d,
        "hp_mask_path": medtabx.mask_3d,
        "dwi_report_text": medtabx.report_text,
        "pwi_report_text": medtabx.report_text,
        "target_has_stroke": medtabx.categorical,
        "target_brain_ttp_mean": medtabx.numerical,
    }
    col_to_stype = {col: stype for col, stype in base_col_to_stype.items() if col in df.columns}
    for col in df.columns:
        # subject_id identifies the row, it is not evidence about the outcome.
        # It used to be typed as categorical and was only harmless because
        # hand-written encoder dicts happened to leave it out.
        if col in col_to_stype or col in IDENTIFIER_COLS:
            continue
        col_to_stype[col] = medtabx.tabular_derived
    return col_to_stype


def build_ads_benchmark_schema(
    df: pd.DataFrame,
    preprocessing=None,
) -> DatasetSchema:
    col_to_stype = build_ads_benchmark_stype_map(df)
    derived_cols = [c for c in df.columns if c not in RAW_COLS | {"target_has_stroke", "target_brain_ttp_mean"}]
    schema = DatasetSchema(
        col_to_stype=col_to_stype,
        target_cols=["target_has_stroke", "target_brain_ttp_mean"],
        col_to_space={
            "dwi_path": "dwi",
            "stroke_mask_path": "dwi",
            "pwi_path": "pwi",
            "hp_mask_path": "pwi",
        },
        col_to_modality_config={
            "dwi_path": {"target_shape": (64, 64, 32)},
            "stroke_mask_path": {"target_shape": (64, 64, 32)},
            "hp_mask_path": {"target_shape": (64, 64, 32)},
        },
        modality_groups={
            "raw_imaging": ["dwi_path", "pwi_path"],
            "masks": ["stroke_mask_path", "hp_mask_path"],
            "reports": ["dwi_report_text", "pwi_report_text"],
            "derived": derived_cols,
        },
    )
    return apply_schema_transforms(schema, preprocessing)


def build_ads_benchmark_dataset(
    dataset_root: str | Path,
    target_col: str = "target_has_stroke",
    preprocessing=None,
    harmonize_reference_col: str | None = None,
    harmonize_cols: list[str] | None = None,
    harmonized_output_root: str | Path | None = None,
) -> Dataset:
    if harmonize_reference_col is None:
        df = build_ads_benchmark_dataframe(dataset_root)
    else:
        df = build_ads_harmonized_benchmark_dataframe(
            dataset_root,
            reference_col=harmonize_reference_col,
            harmonize_cols=harmonize_cols,
            output_root=harmonized_output_root,
        )
    schema = build_ads_benchmark_schema(df, preprocessing=preprocessing)
    col_to_stype = build_ads_benchmark_stype_map(df)
    return Dataset(
        df=df,
        col_to_stype=col_to_stype,
        target_col=target_col,
        col_to_space=schema.col_to_space,
        col_to_modality_kwargs=schema.col_to_modality_config,
    )
