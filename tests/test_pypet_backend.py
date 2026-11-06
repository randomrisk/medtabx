"""Tests for the vendored PyPeT backend integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_pypet_runtime_config():
    from medtabx.features.third_party.pypet import config as pypet_config

    pypet_config.set_runtime_config(
        image_type="mrp",
        scan_interval=1.35,
        echo_time=0.03,
        debug=False,
    )

    assert pypet_config.IMAGE_TYPE == "mrp"
    assert abs(pypet_config.SCAN_INTERVAL - 1.35) < 1e-6
    assert abs(pypet_config.ECHO_TIME - 0.03) < 1e-6
    assert pypet_config.PROJECTION == "min"


def test_pypet_backend_synthetic(tmp_path):
    try:
        import nibabel as nib
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    from medtabx.features.perfusion import process_perfusion

    timepoints, depth, height, width = 16, 4, 8, 8
    series = np.full((width, height, depth, timepoints), 100.0, dtype=np.float32)
    background_curve = np.array(
        [0, 0, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0, 0], dtype=np.float32
    )
    bolus_curve = np.array(
        [0, 0, 0, 0, 10, 25, 50, 90, 120, 80, 40, 15, 5, 0, 0, 0],
        dtype=np.float32,
    )

    for t in range(timepoints):
        series[..., t] += background_curve[t]
        series[2:6, 2:6, 1:3, t] += bolus_curve[t]

    mask = np.ones((width, height, depth), dtype=np.uint8)
    perf_path = tmp_path / "synthetic_ctp.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(series, np.eye(4)), str(perf_path))
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

    maps = process_perfusion(
        perf_path=str(perf_path),
        mask_path=str(mask_path),
        image_type="ctp",
        scan_interval=1.0,
        backend="pypet",
        motion_correction=False,
        aif_kwargs={
            "min_vol_thres": 1,
            "max_vol_thres": 500,
            "smoothness_threshold": 10.0,
        },
    )

    assert set(maps) == {"cbf", "cbv", "mtt", "ttp", "tmax"}
    assert maps["ttp"].shape == (depth, height, width)
