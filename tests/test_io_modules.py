"""Pytest-style checks for MedTabX I/O modules on sample assets."""

from pathlib import Path

import pytest

_TEST_DATA = Path(__file__).parent / "data"
_EXAMPLE_DATA = Path(__file__).parent.parent / "examples" / "data"
DATA_ROOT = _TEST_DATA if _TEST_DATA.exists() else _EXAMPLE_DATA


def test_tabular():
    from medtabx.data.io_tabular import infer_column_types, load_csv

    csv_path = DATA_ROOT / "tabular/diabetes_sample.csv"
    df = load_csv(str(csv_path))

    assert not df.empty
    assert df.shape[1] > 0
    col_types = infer_column_types(df)
    assert isinstance(col_types, dict)
    assert len(col_types) > 0


def test_nifti():
    try:
        from medtabx.data.io_nifti import load_nifti
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    nifti_path = DATA_ROOT / "nifti/brain_mri_sample.nii.gz"
    if not nifti_path.exists():
        nifti_path = DATA_ROOT / "nifti/brain_mri_sample.nii"
    img, affine = load_nifti(str(nifti_path), return_affine=True)

    assert img.ndim in (3, 4)
    assert affine.shape == (4, 4)


def test_dicom():
    try:
        from medtabx.data.io_dicom import PYDICOM_AVAILABLE, load_dicom_file
    except ImportError as exc:
        pytest.skip(f"pydicom unavailable: {exc}")
    if not PYDICOM_AVAILABLE:
        pytest.skip("pydicom unavailable")

    dicom_path = DATA_ROOT / "dicom/chest_xray_sample.dcm"
    img, metadata = load_dicom_file(str(dicom_path))

    assert img.ndim == 2
    assert img.shape[0] > 0 and img.shape[1] > 0
    assert isinstance(metadata, dict)


def test_eeg():
    try:
        from medtabx.data.io_eeg import MNE_AVAILABLE, load_eeg
    except ImportError as exc:
        pytest.skip(f"mne unavailable: {exc}")
    if not MNE_AVAILABLE:
        pytest.skip("mne unavailable")

    eeg_path = DATA_ROOT / "eeg/eeg_sample.edf"
    if not eeg_path.exists():
        eeg_path = DATA_ROOT / "eeg/chb01_01.edf"
    data, metadata = load_eeg(str(eeg_path))

    assert data.ndim == 2
    assert metadata["n_channels"] == data.shape[0]
    assert metadata["n_times"] == data.shape[1]


def test_ecg():
    from medtabx.data.io_ecg import load_ecg_wfdb

    ecg_path = DATA_ROOT / "ecg/ecg_sample"
    data, metadata = load_ecg_wfdb(str(ecg_path))

    assert data.ndim == 2
    assert metadata["sig_len"] == data.shape[0]
    assert len(metadata["sig_name"]) == data.shape[1]


def test_fmri():
    try:
        from medtabx.data.io_fmri import load_fmri
    except ImportError as exc:
        pytest.skip(f"nibabel unavailable: {exc}")

    fmri_path = DATA_ROOT / "fmri/fmri_timeseries.nii.gz"
    data, affine = load_fmri(str(fmri_path), return_affine=True)

    assert data.ndim == 4
    assert data.shape[3] > 0
    assert affine.shape == (4, 4)


def test_vitals():
    from medtabx.data.io_vitals import load_vital_signs

    vitals_path = DATA_ROOT / "vitals/vitals_timeseries.csv"
    df = load_vital_signs(str(vitals_path))

    assert not df.empty
    assert df.shape[1] > 0
