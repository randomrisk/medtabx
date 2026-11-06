"""Tests for lesion and atlas-backed QFV feature extraction."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _openads_available():
    """Skip when OpenADS is not cached and cannot be downloaded."""
    from medtabx.features.openads_bridge import get_openads_root
    try:
        get_openads_root()
    except RuntimeError as exc:
        pytest.skip(f"OpenADS unavailable: {exc}")


def test_openads_bridge_imports():
    from medtabx.features.atlas import load_openads_feature_metadata
    from medtabx.features.openads_bridge import ensure_openads_on_path, get_openads_atlas_dir

    src_root = ensure_openads_on_path()
    atlas_dir = get_openads_atlas_dir()
    metadata = load_openads_feature_metadata()

    assert src_root.exists()
    assert atlas_dir.exists()
    assert "Arterial" in metadata
    assert "QFV_names" in metadata["Arterial"]


def test_openads_stroke_feature_wrapper(tmp_path):
    try:
        import nibabel as nib
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    from medtabx.features.lesion import (
        calculate_openads_stroke_features,
        save_openads_feature_tables,
    )
    from medtabx.features.openads_bridge import get_openads_atlas_dir

    atlas_dir = get_openads_atlas_dir()
    vascular_atlas = atlas_dir / "ArterialAtlas_MNI182.nii.gz"
    atlas_img = nib.load(str(vascular_atlas))
    shape = atlas_img.shape
    affine = atlas_img.affine
    header = atlas_img.header.copy()

    stroke = np.zeros(shape, dtype=np.float32)
    mask = np.ones(shape, dtype=np.float32)
    adc = np.zeros(shape, dtype=np.float32)

    hit = tuple(s // 2 for s in shape)
    stroke[hit] = 1.0
    adc[hit] = 1.0

    stroke_path = tmp_path / "stroke.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    adc_path = tmp_path / "adc.nii.gz"

    nib.save(nib.Nifti1Image(stroke, affine, header), str(stroke_path))
    nib.save(nib.Nifti1Image(mask, affine, header), str(mask_path))
    nib.save(nib.Nifti1Image(adc, affine, header), str(adc_path))

    results = calculate_openads_stroke_features(
        stroke_img_path=str(stroke_path),
        mask_raw_mni_path=str(mask_path),
        adc_mni_path=str(adc_path),
    )

    assert "feature_vectors" in results
    assert "probability_vectors" in results
    assert "qfv_features" in results
    assert "stroke_logml" in results
    assert results["feature_vectors"]["vascular"].ndim == 1
    assert len(results["qfv_features"]) == 7

    output_dir = tmp_path / "tables"
    save_openads_feature_tables(results, subject_id="sub-test", output_dir=str(output_dir))
    assert (output_dir / "sub-test_VASCULAR_QFV.csv").exists()
