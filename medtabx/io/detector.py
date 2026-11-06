"""Modality and format detection for IO resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .errors import UnsupportedModalityError


_MODALITY_BY_EXT: Dict[str, str] = {
    ".csv": "tabular",
    ".parquet": "tabular",
    ".nii": "nifti",
    ".dcm": "dicom",
    ".edf": "eeg",
    ".fif": "eeg",
    ".bdf": "eeg",
    ".set": "eeg",
}

_FORMAT_BY_EXT: Dict[str, str] = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".nii": "nifti",
    ".dcm": "dicom",
    ".edf": "edf",
    ".fif": "fif",
    ".bdf": "bdf",
    ".set": "set",
}


def _is_nifti_gz(path: Path) -> bool:
    return path.name.lower().endswith(".nii.gz")


def detect_format(path: str) -> str:
    p = Path(path)

    if p.is_dir():
        if list(p.glob("*.dcm")):
            return "dicom_series"
        raise UnsupportedModalityError(f"Unsupported directory format: {path}")

    if _is_nifti_gz(p):
        return "nifti_gz"

    ext = p.suffix.lower()
    if ext in _FORMAT_BY_EXT:
        return _FORMAT_BY_EXT[ext]

    # WFDB record can be provided without extension; keep as explicit fallback.
    if p.with_suffix(".hea").exists() and p.with_suffix(".dat").exists():
        return "wfdb"

    raise UnsupportedModalityError(f"Unsupported file format: {path}")


def detect_modality(path: str) -> str:
    p = Path(path)

    if p.is_dir():
        if list(p.glob("*.dcm")):
            return "dicom"
        raise UnsupportedModalityError(f"Unsupported directory modality: {path}")

    if _is_nifti_gz(p):
        return "nifti"

    ext = p.suffix.lower()
    if ext in _MODALITY_BY_EXT:
        return _MODALITY_BY_EXT[ext]

    if p.with_suffix(".hea").exists() and p.with_suffix(".dat").exists():
        return "ecg"

    raise UnsupportedModalityError(f"Unable to detect modality for: {path}")
