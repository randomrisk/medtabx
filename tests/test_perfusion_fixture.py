"""Tests for perfusion metadata and summary helpers."""

from __future__ import annotations

from pathlib import Path


def test_perfusion_sidecar_and_summary():
    from medtabx.features.perfusion import (
        extract_roi_time_series,
        infer_perfusion_parameters,
        load_perfusion_sidecar,
        resample_to_reference,
        resample_mask_to_map,
        summarize_maps_with_mask,
        summarize_maps_with_mask_paths,
        summarize_perfusion_series,
        summarize_roi_time_series,
    )
    import nibabel as nib
    import numpy as np

    fixture_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    perf_path = fixture_root / "imaging" / "sub-02e8eb42_PWI.nii.gz"
    sidecar_path = fixture_root / "metadata" / "sub-02e8eb42_PWI.json"

    sidecar = load_perfusion_sidecar(str(sidecar_path))
    params = infer_perfusion_parameters(str(sidecar_path), None, None)
    summary = summarize_perfusion_series(str(perf_path), str(sidecar_path))
    ttp = nib.load(str(fixture_root / "perfusion_maps" / "sub-02e8eb42_ttp.nii.gz")).get_fdata()
    cbf = nib.load(str(fixture_root / "perfusion_maps" / "sub-02e8eb42_cbf.nii.gz")).get_fdata()
    brain_mask = nib.load(
        str(fixture_root / "metadata" / "sub-02e8eb42_brain_mask_from_pwi.nii.gz")
    ).get_fdata()
    map_summary = summarize_maps_with_mask({"ttp": ttp, "cbf": cbf}, brain_mask, prefix="brain")
    path_summary = summarize_maps_with_mask_paths(
        {
            "ttp": str(fixture_root / "perfusion_maps" / "sub-02e8eb42_ttp.nii.gz"),
            "cbf": str(fixture_root / "perfusion_maps" / "sub-02e8eb42_cbf.nii.gz"),
        },
        str(fixture_root / "metadata" / "sub-02e8eb42_brain_mask_from_pwi.nii.gz"),
        prefix="brain_path",
    )

    assert sidecar["RepetitionTime"] == 1.35
    assert abs(params["scan_interval"] - 1.35) < 1e-6
    assert abs(params["echo_time"] - 0.03) < 1e-6
    assert summary["num_timepoints"] > 1
    assert summary["shape"][-1] == summary["num_timepoints"]
    assert len(summary["mean_signal_curve"]) == summary["num_timepoints"]
    assert summary["has_slice_timing"] is True
    assert "brain_ttp_mean" in map_summary
    assert "brain_cbf_max" in map_summary
    assert "brain_path_ttp_mean" in path_summary

    synthetic_perf = np.zeros((4, 4, 2, 5), dtype=float)
    synthetic_perf[0:2, :, :, :] = 1.0
    synthetic_perf[2:4, :, :, :] = np.arange(5, dtype=float)
    synthetic_atlas = np.zeros((4, 4, 2), dtype=int)
    synthetic_atlas[0:2, :, :] = 1
    synthetic_atlas[2:4, :, :] = 2
    roi_series = extract_roi_time_series(synthetic_perf, synthetic_atlas)
    roi_summary = summarize_roi_time_series(roi_series)
    assert set(roi_series) == {1, 2}
    assert len(roi_series[1]) == 5
    assert "roi_2_ttp_index" in roi_summary

    # Resampling check with a deliberately lower-resolution mask grid.
    highres_map_path = fixture_root / "perfusion_maps" / "sub-02e8eb42_ttp.nii.gz"
    highres_map = nib.load(str(highres_map_path))
    small_mask = np.ones((10, 64, 64), dtype=np.uint8)
    affine = highres_map.affine.copy()
    affine[0, 0] *= 2
    affine[1, 1] *= 2
    affine[2, 2] *= 2
    tmp_mask_path = fixture_root / "metadata" / "tmp_test_resample_mask.nii.gz"
    nib.save(nib.Nifti1Image(small_mask, affine), str(tmp_mask_path))
    try:
        resampled = resample_mask_to_map(str(tmp_mask_path), str(highres_map_path))
        assert resampled.shape == highres_map.shape
    finally:
        tmp_mask_path.unlink(missing_ok=True)

    # Generic reference-space resampling for images/maps.
    reference_path = fixture_root / "perfusion_maps" / "sub-02e8eb42_ttp.nii.gz"
    moving_path = fixture_root / "perfusion_maps" / "sub-02e8eb42_cbf.nii.gz"
    resampled_map = resample_to_reference(str(moving_path), str(reference_path), kind="map")
    assert resampled_map.shape == highres_map.shape

    # Known affine transform path: apply a small translation before resampling.
    translated = resample_to_reference(
        str(moving_path),
        str(reference_path),
        kind="image",
        transform=np.array(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert translated.shape == highres_map.shape
