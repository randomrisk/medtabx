"""Tests for registry-backed modality loading, feature extraction, and ADS benchmark recipe."""

from __future__ import annotations

from pathlib import Path

import medtabx
from medtabx.benchmarks import (
    build_ads_benchmark_dataframe,
    build_ads_benchmark_dataset,
    build_ads_harmonized_benchmark_dataframe,
    build_ads_benchmark_schema,
)
from medtabx.data import get_modality_loader, load_modality
from medtabx.features import (
    build_harmonized_subject_view,
    get_feature_extractor,
    harmonize_subject_to_reference,
    run_feature_extractor,
)
from medtabx.transforms import apply_schema_transforms, build_transform


def test_modality_loader_registry_and_feature_registry():
    fixture_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    dwi_path = fixture_root / "imaging" / "sub-02e8eb42_DWI.nii.gz"
    report_path = fixture_root / "text" / "sub-02e8eb42_dwi_report.txt"

    image_loader = get_modality_loader(medtabx.image_3d)
    report_loader = get_modality_loader(medtabx.report_text)
    assert image_loader is not None
    assert report_loader is not None

    dwi = load_modality(medtabx.image_3d, str(dwi_path))
    report = load_modality(medtabx.report_text, str(report_path))
    assert dwi.ndim == 3
    assert isinstance(report, str)
    assert len(report) > 0

    extractor = get_feature_extractor("ads_subject_feature_row")
    assert extractor is not None
    row = run_feature_extractor("ads_subject_feature_row", fixture_root)
    assert row["subject_id"] == "sub-02e8eb42"
    assert "brain_ttp_mean" in row


def test_ads_benchmark_recipe():
    dataset_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case"
    df = build_ads_benchmark_dataframe(dataset_root)
    schema = build_ads_benchmark_schema(df)
    dataset = build_ads_benchmark_dataset(dataset_root, target_col="target_has_stroke")
    assert schema.col_to_space["stroke_mask_path"] == "dwi"
    assert schema.col_to_space["hp_mask_path"] == "pwi"
    assert "dwi" in schema.space_groups
    assert "pwi" in schema.space_groups
    assert dataset.space_groups["dwi"] == ["dwi_path", "stroke_mask_path"]
    dataset.materialize()

    assert len(df) == 1
    assert "target_has_stroke" in df.columns
    assert "target_brain_ttp_mean" in df.columns
    assert "raw_imaging" in schema.modality_groups
    assert dataset.target_col == "target_has_stroke"
    assert medtabx.image_3d in dataset.tensor_frame.feat_dict
    assert medtabx.fmri in dataset.tensor_frame.feat_dict
    assert medtabx.mask_3d in dataset.tensor_frame.feat_dict
    assert dataset.tensor_frame.feat_dict[medtabx.mask_3d].shape[1] == 2

    dwi_only = build_ads_benchmark_dataset(dataset_root, target_col="target_has_stroke").space_select("dwi")
    assert "hp_mask_path" not in dwi_only.feat_cols
    assert "stroke_mask_path" in dwi_only.feat_cols


def test_transform_registry_driven_benchmark_preprocessing(tmp_path):
    dataset_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case"
    df = build_ads_benchmark_dataframe(dataset_root)

    atlas_path = tmp_path / "dummy_atlas.nii.gz"
    atlas_path.write_bytes(b"not-used-here")

    harmonized = build_transform("space_harmonized", image_shape=(48, 48, 24))
    atlas_roi = build_transform("atlas_roi", atlas_path=str(atlas_path), min_label=2)

    schema = build_ads_benchmark_schema(
        df,
        preprocessing=[
            harmonized,
            atlas_roi,
        ],
    )
    assert schema.col_to_modality_config["dwi_path"]["target_shape"] == (48, 48, 24)
    assert schema.col_to_modality_config["stroke_mask_path"]["target_shape"] == (48, 48, 24)
    assert schema.col_to_modality_config["pwi_path"]["atlas_path"] == str(atlas_path)
    assert schema.col_to_modality_config["pwi_path"]["min_label"] == 2

    reapplied = apply_schema_transforms(
        schema,
        [("atlas_roi", {"column": "pwi_path", "min_label": 3})],
    )
    assert reapplied.col_to_modality_config["pwi_path"]["min_label"] == 3


def test_subject_harmonization_helpers_and_harmonized_benchmark(tmp_path):
    subject_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    outputs = harmonize_subject_to_reference(
        subject_root,
        "imaging/sub-02e8eb42_DWI.nii.gz",
        include_paths=["masks/sub-02e8eb42_hp_mask.nii.gz"],
        output_dir=tmp_path / "harmonized_subject",
    )
    hp_out = Path(outputs["masks/sub-02e8eb42_hp_mask.nii.gz"])
    assert hp_out.exists()

    view = build_harmonized_subject_view(
        subject_root,
        reference_relpath="imaging/sub-02e8eb42_DWI.nii.gz",
        path_columns={
            "dwi_path": str(subject_root / "imaging" / "sub-02e8eb42_DWI.nii.gz"),
            "hp_mask_path": str(subject_root / "masks" / "sub-02e8eb42_hp_mask.nii.gz"),
        },
        include_columns=["hp_mask_path"],
        output_dir=tmp_path / "harmonized_view",
    )
    assert view["dwi_path"].endswith("sub-02e8eb42_DWI.nii.gz")
    assert "harmonized_view" in view["hp_mask_path"]

    dataset_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case"
    hdf = build_ads_harmonized_benchmark_dataframe(
        dataset_root,
        reference_col="dwi_path",
        harmonize_cols=["hp_mask_path"],
        output_root=tmp_path / "benchmark_harmonized",
    )
    assert "benchmark_harmonized" in hdf.loc[0, "hp_mask_path"]

    dataset = build_ads_benchmark_dataset(
        dataset_root,
        target_col="target_has_stroke",
        harmonize_reference_col="dwi_path",
        harmonize_cols=["hp_mask_path"],
        harmonized_output_root=tmp_path / "dataset_harmonized",
    )
    dataset.materialize()
    assert medtabx.mask_3d in dataset.tensor_frame.feat_dict
