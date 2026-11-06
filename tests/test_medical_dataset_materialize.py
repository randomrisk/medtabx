"""Tests for medical-path Dataset materialization."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import medtabx
from medtabx.data import Dataset


def test_dataset_materialize_medical_paths():
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
    tf = dataset.tensor_frame

    assert medtabx.image_3d in tf.feat_dict
    assert medtabx.mask_3d in tf.feat_dict
    assert medtabx.fmri in tf.feat_dict
    assert medtabx.vital_sign in tf.feat_dict
    assert medtabx.embedding in tf.feat_dict
    assert medtabx.numerical in tf.feat_dict

    assert tf.feat_dict[medtabx.image_3d].ndim == 6
    assert tf.feat_dict[medtabx.mask_3d].ndim == 6
    assert tf.feat_dict[medtabx.fmri].ndim == 4
    assert tf.feat_dict[medtabx.vital_sign].ndim == 3
    assert tf.feat_dict[medtabx.embedding].shape[-1] == 16
    assert tf.col_names_dict[medtabx.embedding] == ["report_text"]
    assert "derived_volume" in tf.col_names_dict[medtabx.numerical]


def test_dataset_materialize_with_atlas_configurable_fmri(tmp_path):
    fixture_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    pwi_path = fixture_root / "imaging" / "sub-02e8eb42_PWI.nii.gz"
    pwi_img = nib.load(str(pwi_path))
    pwi = pwi_img.get_fdata()

    atlas = np.zeros(pwi.shape[:3], dtype=np.int16)
    thirds = atlas.shape[0] // 3
    atlas[:thirds, :, :] = 1
    atlas[thirds : 2 * thirds, :, :] = 2
    atlas[2 * thirds :, :, :] = 3
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(atlas, pwi_img.affine), str(atlas_path))

    df = pd.DataFrame({"pwi_path": [str(pwi_path)]})
    dataset = Dataset(
        df=df,
        col_to_stype={"pwi_path": medtabx.fmri},
        col_to_modality_kwargs={"pwi_path": {"atlas_path": str(atlas_path), "min_label": 1}},
    )
    dataset.materialize()
    tf = dataset.tensor_frame
    assert medtabx.fmri in tf.feat_dict
    assert tf.feat_dict[medtabx.fmri].shape[2] == 3
