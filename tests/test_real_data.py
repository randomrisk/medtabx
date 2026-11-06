"""Pytest-style real data loading checks for MedTabX sample assets."""

from pathlib import Path

import pytest


_TEST_DATA = Path(__file__).parent / "data"
_EXAMPLE_DATA = Path(__file__).parent.parent / "examples" / "data"
BASE = _TEST_DATA if _TEST_DATA.exists() else _EXAMPLE_DATA


def test_tabular_data():
    from medtabx.data.io_tabular import infer_column_types, load_csv

    csv_path = BASE / "tabular/diabetes_sample.csv"
    df = load_csv(str(csv_path))

    assert df.shape == (10, 8)
    col_types = infer_column_types(df)
    assert isinstance(col_types, dict)
    assert len(col_types) >= 5


def test_nifti_data():
    try:
        from medtabx.data.io_nifti import load_nifti
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    nifti_path = BASE / "nifti/brain_mri_sample.nii"
    img, affine = load_nifti(str(nifti_path), return_affine=True)

    assert img.shape == (160, 160, 35)
    assert affine.shape == (4, 4)


def test_dicom_data():
    try:
        from medtabx.data.io_dicom import load_dicom_file
    except ImportError as exc:
        pytest.skip(f"pydicom unavailable: {exc}")

    dicom_path = BASE / "dicom/chest_xray_sample.dcm"
    img, metadata = load_dicom_file(str(dicom_path))

    assert img.shape == (1024, 1024)
    assert isinstance(metadata, dict)
    assert "Modality" in metadata


def test_eeg_data():
    try:
        from medtabx.data.io_eeg import load_eeg
    except ImportError as exc:
        pytest.skip(f"mne unavailable: {exc}")

    eeg_path = BASE / "eeg/chb01_01.edf"
    data, metadata = load_eeg(str(eeg_path))

    assert data.shape[0] == metadata["n_channels"]
    assert data.shape[1] == metadata["n_times"]
    assert metadata["duration"] > 0


def test_ecg_data():
    from medtabx.data.io_ecg import load_ecg_wfdb

    ecg_path = BASE / "ecg/ecg_sample"
    data, metadata = load_ecg_wfdb(str(ecg_path))

    assert data.shape == (1000, 12)
    assert metadata["fs"] == 100.0
    assert metadata["sig_len"] == 1000


def test_fmri_data():
    try:
        from medtabx.data.io_fmri import load_fmri
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    fmri_path = BASE / "fmri/fmri_timeseries.nii.gz"
    data, affine = load_fmri(str(fmri_path), return_affine=True)

    assert data.shape == (128, 96, 24, 2)
    assert affine.shape == (4, 4)


def test_vitals_data():
    from medtabx.data.io_vitals import load_vital_signs

    vitals_path = BASE / "vitals/vitals_timeseries.csv"
    df = load_vital_signs(str(vitals_path))

    assert df.shape == (10, 7)
    assert len(df.columns) == 7
