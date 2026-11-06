from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from medtabx.data.io_ecg import load_ecg_wfdb
from medtabx.data.io_eeg import load_eeg
from medtabx.data.io_fmri import load_fmri
from medtabx.data.io_nifti import load_nifti
from medtabx.data.io_vitals import load_vital_signs
from medtabx.data.mapper import TensorMapper
from medtabx.text import get_text_embedding_backend


def _resize_nd(
    array: np.ndarray,
    target_shape: tuple[int, ...] | None,
    mode: str,
) -> np.ndarray:
    if target_shape is None or tuple(array.shape) == tuple(target_shape):
        return array.astype(np.float32, copy=False)

    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported resize ndim={tensor.ndim} for shape {array.shape}.")

    resized = F.interpolate(
        tensor,
        size=target_shape,
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    if array.ndim == 3:
        return resized.squeeze(0).squeeze(0).cpu().numpy()
    return resized.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()


def _pad_1d_series_list(series_list: list[np.ndarray]) -> Tensor:
    max_len = max(series.shape[-1] for series in series_list)
    num_rows = len(series_list)
    padded = np.zeros((num_rows, max_len), dtype=np.float32)
    for i, series in enumerate(series_list):
        padded[i, : series.shape[-1]] = series.astype(np.float32, copy=False)
    return torch.from_numpy(padded)


def _pad_2d_series_list(series_list: list[np.ndarray]) -> Tensor:
    max_channels = max(series.shape[0] for series in series_list)
    max_len = max(series.shape[-1] for series in series_list)
    num_rows = len(series_list)
    padded = np.zeros((num_rows, max_channels, max_len), dtype=np.float32)
    for i, series in enumerate(series_list):
        channels, length = series.shape
        padded[i, :channels, :length] = series.astype(np.float32, copy=False)
    return torch.from_numpy(padded)


class NiftiVolumeTensorMapper(TensorMapper):
    def __init__(
        self,
        target_shape: tuple[int, int, int] | None = None,
        interpolation: str = "trilinear",
    ):
        self.target_shape = target_shape
        self.interpolation = interpolation

    def forward(self, ser: pd.Series, *, device: torch.device | None = None) -> Tensor:
        arrays = []
        for value in ser.tolist():
            array = load_nifti(str(value))
            if array.ndim != 3:
                raise ValueError(f"Expected 3D NIfTI volume, got shape {array.shape}.")
            array = _resize_nd(array, self.target_shape, self.interpolation)
            arrays.append(array.astype(np.float32, copy=False))

        stacked = np.stack(arrays, axis=0)[:, None, :, :, :]
        return torch.from_numpy(stacked).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError("Path-based medical tensor mappers are not invertible.")


class MaskVolumeTensorMapper(NiftiVolumeTensorMapper):
    def __init__(self, target_shape: tuple[int, int, int] | None = None):
        super().__init__(target_shape=target_shape, interpolation="nearest")


class FMRITensorMapper(TensorMapper):
    """Map 4D perfusion/fMRI paths into lightweight ROI time-series tensors.

    Default behavior extracts two coarse ROIs by splitting the first spatial
    axis into left/right hemispheres and averaging over voxels. This is a
    stable fallback for raw 4D series when no atlas is supplied yet.
    """

    def __init__(
        self,
        atlas_path: str | None = None,
        min_label: int = 1,
    ):
        self.atlas_path = atlas_path
        self.min_label = min_label

    @staticmethod
    def _maybe_align_atlas(atlas: np.ndarray, data: np.ndarray) -> np.ndarray:
        if atlas.shape == data.shape[:3]:
            return atlas
        if atlas.shape == data.shape[:3][::-1]:
            return np.transpose(atlas, (2, 1, 0))
        raise ValueError(
            f"Atlas shape {atlas.shape} does not match data spatial shape {data.shape[:3]}."
        )

    def forward(self, ser: pd.Series, *, device: torch.device | None = None) -> Tensor:
        atlas = None
        if self.atlas_path is not None:
            atlas = load_nifti(self.atlas_path)
            if atlas.ndim != 3:
                raise ValueError(f"Expected 3D atlas, got shape {atlas.shape}.")

        roi_series_list: list[np.ndarray] = []
        max_timepoints = 0

        for value in ser.tolist():
            data = load_fmri(str(value))
            if atlas is not None:
                from medtabx.features.perfusion import extract_roi_time_series

                aligned_atlas = self._maybe_align_atlas(atlas, data)
                roi_dict = extract_roi_time_series(data, aligned_atlas, min_label=self.min_label)
                if not roi_dict:
                    raise ValueError("Atlas-based ROI extraction returned no regions.")
                roi_series = np.stack(
                    [roi_dict[key] for key in sorted(roi_dict.keys())],
                    axis=0,
                ).astype(np.float32, copy=False)
            else:
                mid = data.shape[0] // 2
                left = data[:mid, ...].mean(axis=(0, 1, 2))
                right = data[mid:, ...].mean(axis=(0, 1, 2))
                roi_series = np.stack([left, right], axis=0).astype(np.float32, copy=False)
            roi_series_list.append(roi_series)
            max_timepoints = max(max_timepoints, roi_series.shape[-1])

        max_rois = max(roi_series.shape[0] for roi_series in roi_series_list)
        padded = np.zeros((len(roi_series_list), max_rois, max_timepoints), dtype=np.float32)
        for i, roi_series in enumerate(roi_series_list):
            padded[i, : roi_series.shape[0], : roi_series.shape[-1]] = roi_series
        return torch.from_numpy(padded).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError("fMRI path mappers are not invertible.")


class ReportTextTensorMapper(TensorMapper):
    """Pluggable text/LLM embedding mapper for raw report text."""

    def __init__(
        self,
        embedding_dim: int = 16,
        backend: str | object = "deterministic",
        backend_kwargs: dict | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}

    def forward(self, ser: pd.Series, *, device: torch.device | None = None) -> Tensor:
        backend_kwargs = dict(self.backend_kwargs)
        backend_kwargs.setdefault("embedding_dim", self.embedding_dim)
        embedded = get_text_embedding_backend(
            self.backend,
            **backend_kwargs,
        ).embed([str(text) for text in ser.tolist()])
        return torch.from_numpy(embedded).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError("Raw report text embeddings are not invertible.")


class VitalSignTensorMapper(TensorMapper):
    def forward(self, ser: pd.Series, *, device: torch.device | None = None) -> Tensor:
        series_list: list[np.ndarray] = []
        for value in ser.tolist():
            df = load_vital_signs(str(value))
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                raise ValueError(f"No numeric vital-sign channels found in {value}.")
            series = numeric.to_numpy(dtype=np.float32).T.reshape(-1)
            series_list.append(series)
        return _pad_1d_series_list(series_list).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError("Vital-sign path mappers are not invertible.")


class EEGTensorMapper(TensorMapper):
    def forward(self, ser: pd.Series, *, device: torch.device | None = None) -> Tensor:
        series_list = [load_eeg(str(value))[0].astype(np.float32, copy=False) for value in ser.tolist()]
        return _pad_2d_series_list(series_list).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError("EEG path mappers are not invertible.")


class ECGTensorMapper(TensorMapper):
    def forward(self, ser: pd.Series, *, device: torch.device | None = None) -> Tensor:
        series_list = []
        for value in ser.tolist():
            record_path = str(Path(value))
            data, _ = load_ecg_wfdb(record_path)
            series_list.append(data.astype(np.float32, copy=False).T)
        return _pad_2d_series_list(series_list).to(device)

    def backward(self, tensor: Tensor) -> pd.Series:
        raise NotImplementedError("ECG path mappers are not invertible.")
