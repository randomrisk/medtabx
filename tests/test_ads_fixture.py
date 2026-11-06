"""Smoke tests for the curated ADS multimodal fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def test_ads_fixture_manifest_and_files():
    fixture_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    manifest = json.loads((fixture_root / "manifest.json").read_text())

    assert manifest["subject_id"] == "sub-02e8eb42"
    assert len(manifest["artifacts"]) >= 16

    for artifact in manifest["artifacts"]:
        path = fixture_root / artifact["path"]
        assert path.exists(), f"Missing fixture artifact: {path}"


def test_ads_fixture_loaders():
    try:
        import nibabel as nib
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    fixture_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"

    dwi = nib.load(str(fixture_root / "imaging" / "sub-02e8eb42_DWI.nii.gz")).get_fdata()
    pwi = nib.load(str(fixture_root / "imaging" / "sub-02e8eb42_PWI.nii.gz")).get_fdata()
    ttp = nib.load(str(fixture_root / "perfusion_maps" / "sub-02e8eb42_ttp.nii.gz")).get_fdata()
    cbf = nib.load(str(fixture_root / "perfusion_maps" / "sub-02e8eb42_cbf.nii.gz")).get_fdata()
    stroke_mask = nib.load(str(fixture_root / "masks" / "sub-02e8eb42_stroke_mask.nii.gz")).get_fdata()
    hp_mask = nib.load(str(fixture_root / "masks" / "sub-02e8eb42_hp_mask.nii.gz")).get_fdata()
    brain_mask = nib.load(str(fixture_root / "metadata" / "sub-02e8eb42_brain_mask_from_pwi.nii.gz")).get_fdata()

    dwi_qfv = pd.read_csv(fixture_root / "tabular" / "sub-02e8eb42_VASCULAR_QFV.csv")
    hp_qfv = pd.read_csv(fixture_root / "tabular" / "sub-02e8eb42_Vascular_HPQFV.csv")

    dwi_report = (fixture_root / "text" / "sub-02e8eb42_dwi_report.txt").read_text()
    pwi_meta = json.loads((fixture_root / "metadata" / "sub-02e8eb42_PWI.json").read_text())

    assert dwi.ndim == 3
    assert pwi.ndim == 4
    assert ttp.ndim == 3
    assert cbf.ndim == 3
    assert stroke_mask.ndim == 3
    assert hp_mask.ndim == 3
    assert brain_mask.ndim == 3
    assert dwi_qfv.shape[0] == 1
    assert hp_qfv.shape[0] == 1
    assert "subject_id" in dwi_qfv.columns
    assert len(dwi_report) > 0
    assert isinstance(pwi_meta, dict)
