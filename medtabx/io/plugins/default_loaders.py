"""Default loader plugins backed by medtabx.data.io_* modules."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

from ..base_loader import BaseLoader
from ..contracts import ChannelInfo, GeometryInfo, MedSample, TimeInfo
from ..registry import LoaderRegistry


def _io_module(name: str):
    return import_module(f"medtabx.data.{name}")


class PandasTabularLoader(BaseLoader):
    name = "pandas"
    supported_modalities = ["tabular"]
    supported_formats = ["csv", "parquet"]
    priority = 10
    version_module = "pandas"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def can_load(cls, path: str) -> bool:
        lower = path.lower()
        return lower.endswith(".csv") or lower.endswith(".parquet")

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path, "size": Path(path).stat().st_size}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        io_tabular = _io_module("io_tabular")
        opts = options or {}
        if path.lower().endswith(".csv"):
            data = io_tabular.load_csv(path, **opts)
            fmt = "csv"
        else:
            data = io_tabular.load_parquet(path, **opts)
            fmt = "parquet"

        return MedSample(
            modality="tabular",
            data=data,
            meta={"source_format": fmt},
        )


class NibabelNiftiLoader(BaseLoader):
    name = "nibabel"
    supported_modalities = ["nifti"]
    supported_formats = ["nifti", "nifti_gz"]
    priority = 10
    version_module = "nibabel"

    @classmethod
    def is_available(cls) -> bool:
        return bool(getattr(_io_module("io_nifti"), "NIBABEL_AVAILABLE", False))

    @classmethod
    def can_load(cls, path: str) -> bool:
        lower = path.lower()
        return lower.endswith(".nii") or lower.endswith(".nii.gz")

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        del options
        io_nifti = _io_module("io_nifti")
        image, affine = io_nifti.load_nifti(path, return_affine=True)
        return MedSample(
            modality="nifti",
            data=image,
            meta={"source_format": "nifti"},
            geometry=GeometryInfo(affine=affine.tolist()),
        )


class NibabelFmriLoader(BaseLoader):
    name = "nibabel_fmri"
    supported_modalities = ["fmri"]
    supported_formats = ["nifti", "nifti_gz"]
    priority = 10
    version_module = "nibabel"

    @classmethod
    def is_available(cls) -> bool:
        return bool(getattr(_io_module("io_fmri"), "NIBABEL_AVAILABLE", False))

    @classmethod
    def can_load(cls, path: str) -> bool:
        lower = path.lower()
        return lower.endswith(".nii") or lower.endswith(".nii.gz")

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        del options
        io_fmri = _io_module("io_fmri")
        data, affine = io_fmri.load_fmri(path, return_affine=True)
        return MedSample(
            modality="fmri",
            data=data,
            meta={"source_format": "nifti"},
            geometry=GeometryInfo(affine=affine.tolist()),
            time=TimeInfo(time_axis_index=3),
        )


class PydicomFileLoader(BaseLoader):
    name = "pydicom_file"
    supported_modalities = ["dicom"]
    supported_formats = ["dicom"]
    priority = 10
    version_module = "pydicom"

    @classmethod
    def is_available(cls) -> bool:
        return bool(getattr(_io_module("io_dicom"), "PYDICOM_AVAILABLE", False))

    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).is_file() and path.lower().endswith(".dcm")

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        del options
        io_dicom = _io_module("io_dicom")
        image, metadata = io_dicom.load_dicom_file(path)
        return MedSample(modality="dicom", data=image, meta=dict(metadata))


class PydicomSeriesLoader(BaseLoader):
    name = "pydicom_series"
    supported_modalities = ["dicom"]
    supported_formats = ["dicom_series"]
    priority = 20
    version_module = "pydicom"

    @classmethod
    def is_available(cls) -> bool:
        return bool(getattr(_io_module("io_dicom"), "PYDICOM_AVAILABLE", False))

    @classmethod
    def can_load(cls, path: str) -> bool:
        p = Path(path)
        return p.is_dir() and bool(list(p.glob("*.dcm")))

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        p = Path(path)
        return {"path": path, "num_files": len(list(p.glob("*.dcm")))}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        del options
        io_dicom = _io_module("io_dicom")
        volume, metadata = io_dicom.load_dicom_series(path)
        return MedSample(modality="dicom", data=volume, meta=dict(metadata))


class MneEegLoader(BaseLoader):
    name = "mne"
    supported_modalities = ["eeg"]
    supported_formats = ["edf", "fif", "bdf", "set"]
    priority = 10
    version_module = "mne"

    @classmethod
    def is_available(cls) -> bool:
        return bool(getattr(_io_module("io_eeg"), "MNE_AVAILABLE", False))

    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).suffix.lower() in {".edf", ".fif", ".bdf", ".set"}

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        io_eeg = _io_module("io_eeg")
        opts = options or {}
        data, metadata = io_eeg.load_eeg(path, preload=bool(opts.get("preload", True)))
        return MedSample(
            modality="eeg",
            data=data,
            meta=dict(metadata),
            time=TimeInfo(
                sampling_rate=metadata.get("sfreq"),
                duration=metadata.get("duration"),
                time_axis_index=1,
            ),
            channels=ChannelInfo(
                names=metadata.get("ch_names"),
                count=metadata.get("n_channels"),
                axis_index=0,
            ),
        )


class WfdbEcgLoader(BaseLoader):
    name = "wfdb"
    supported_modalities = ["ecg"]
    supported_formats = ["wfdb"]
    priority = 10
    version_module = "wfdb"

    @classmethod
    def is_available(cls) -> bool:
        # Always available: uses wfdb package when healthy, otherwise local fallback parser.
        _io_module("io_ecg")
        return True

    @classmethod
    def can_load(cls, path: str) -> bool:
        p = Path(path)
        return p.with_suffix(".hea").exists() and p.with_suffix(".dat").exists()

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        io_ecg = _io_module("io_ecg")
        opts = options or {}
        data, metadata = io_ecg.load_ecg_wfdb(path, channels=opts.get("channels"))
        return MedSample(
            modality="ecg",
            data=data,
            meta=dict(metadata),
            time=TimeInfo(sampling_rate=metadata.get("fs"), time_axis_index=0),
            channels=ChannelInfo(names=metadata.get("sig_name"), axis_index=1),
        )


class VitalsCsvLoader(BaseLoader):
    name = "vitals_csv"
    supported_modalities = ["vital_sign"]
    supported_formats = ["csv"]
    priority = 10
    version_module = "pandas"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def can_load(cls, path: str) -> bool:
        return path.lower().endswith(".csv")

    @classmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        return {"path": path, "size": Path(path).stat().st_size}

    @classmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        io_vitals = _io_module("io_vitals")
        opts = options or {}
        data = io_vitals.load_vital_signs(
            path,
            time_column=str(opts.get("time_column", "time")),
            value_columns=opts.get("value_columns"),
        )
        return MedSample(modality="vital_sign", data=data, meta={"source_format": "csv"})


def register_default_loaders(registry: LoaderRegistry) -> None:
    for loader_cls in [
        PandasTabularLoader,
        NibabelNiftiLoader,
        NibabelFmriLoader,
        PydicomFileLoader,
        PydicomSeriesLoader,
        MneEegLoader,
        WfdbEcgLoader,
        VitalsCsvLoader,
    ]:
        registry.register(loader_cls)
