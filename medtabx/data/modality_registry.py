from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import medtabx
from medtabx.core.registry import modality_loader_registry
from medtabx.data.io_ecg import load_ecg_wfdb
from medtabx.data.io_eeg import load_eeg
from medtabx.data.io_fmri import load_fmri
from medtabx.data.io_nifti import load_nifti
from medtabx.data.io_vitals import load_vital_signs


def _read_text_or_identity(value: Any) -> str:
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            return path.read_text()
        return value
    return str(value)


@modality_loader_registry.register(medtabx.image_3d.value)
def load_image_3d(value: str):
    return load_nifti(str(value))


@modality_loader_registry.register(medtabx.mask_3d.value)
def load_mask_3d(value: str):
    return load_nifti(str(value))


@modality_loader_registry.register(medtabx.timeseries_4d.value)
@modality_loader_registry.register(medtabx.fmri.value)
def load_timeseries_4d(value: str):
    return load_fmri(str(value))


@modality_loader_registry.register(medtabx.vital_sign.value)
def load_vitals(value: str):
    return load_vital_signs(str(value))


@modality_loader_registry.register(medtabx.eeg.value)
def load_eeg_modality(value: str):
    return load_eeg(str(value))


@modality_loader_registry.register(medtabx.ecg.value)
def load_ecg_modality(value: str):
    return load_ecg_wfdb(str(value))


@modality_loader_registry.register(medtabx.report_text.value)
def load_report_text(value: Any):
    return _read_text_or_identity(value)


@modality_loader_registry.register(medtabx.numerical.value)
@modality_loader_registry.register(medtabx.tabular_derived.value)
def load_scalar(value: Any):
    return value


def get_modality_loader(stype_or_name: medtabx.core.stype | str) -> Callable[..., Any]:
    name = stype_or_name.value if hasattr(stype_or_name, "value") else str(stype_or_name)
    return modality_loader_registry.get(name)


def load_modality(stype_or_name: medtabx.core.stype | str, value: Any):
    loader = get_modality_loader(stype_or_name)
    return loader(value)
