"""Perfusion feature extraction - main processing pipeline."""

from __future__ import annotations

import json
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk


def load_perfusion_sidecar(sidecar_path: str) -> Dict[str, object]:
    """Load a perfusion JSON sidecar."""
    return json.loads(Path(sidecar_path).read_text())


def infer_perfusion_parameters(
    sidecar_path: Optional[str] = None,
    scan_interval: Optional[float] = None,
    echo_time: Optional[float] = None,
) -> Dict[str, float]:
    """Infer perfusion timing parameters from explicit values and/or JSON metadata."""
    metadata: Dict[str, object] = {}
    if sidecar_path is not None:
        metadata = load_perfusion_sidecar(sidecar_path)

    resolved_scan_interval = (
        float(metadata.get("RepetitionTime", 1.5))
        if scan_interval is None
        else float(scan_interval)
    )
    resolved_echo_time = (
        float(metadata.get("EchoTime", 0.03))
        if echo_time is None
        else float(echo_time)
    )
    return {
        "scan_interval": resolved_scan_interval,
        "echo_time": resolved_echo_time,
    }


def summarize_perfusion_series(
    perf_path: str,
    sidecar_path: Optional[str] = None,
) -> Dict[str, object]:
    """Create lightweight descriptive summaries from a raw 4D perfusion series."""
    import nibabel as nib

    metadata = load_perfusion_sidecar(sidecar_path) if sidecar_path is not None else {}
    perf = nib.load(perf_path).get_fdata()
    mean_curve = perf.mean(axis=(0, 1, 2))
    peak_index = int(np.argmax(mean_curve))

    return {
        "shape": tuple(int(v) for v in perf.shape),
        "num_timepoints": int(perf.shape[-1]),
        "mean_signal_curve": mean_curve.astype(float),
        "peak_timepoint_index": peak_index,
        "repetition_time": float(metadata.get("RepetitionTime", 1.5)),
        "echo_time": float(metadata.get("EchoTime", 0.03)),
        "has_slice_timing": "SliceTiming" in metadata,
    }


def extract_roi_time_series(
    perf_4d: np.ndarray,
    atlas_3d: np.ndarray,
    min_label: int = 1,
) -> Dict[int, np.ndarray]:
    """Extract mean ROI time series from a 4D perfusion array and a 3D label atlas.

    Expected array conventions:
    - perf_4d: `(x, y, z, t)` or `(z, y, x, t)`
    - atlas_3d: matching spatial shape in the same orientation
    """
    if perf_4d.ndim != 4:
        raise ValueError(f"perf_4d must be 4D. Got shape {perf_4d.shape}.")
    if atlas_3d.ndim != 3:
        raise ValueError(f"atlas_3d must be 3D. Got shape {atlas_3d.shape}.")
    if perf_4d.shape[:3] != atlas_3d.shape:
        raise ValueError(
            f"Spatial dimensions must match. Got perf {perf_4d.shape[:3]} and atlas {atlas_3d.shape}."
        )

    roi_series: Dict[int, np.ndarray] = {}
    labels = np.unique(atlas_3d)
    labels = labels[labels >= min_label]
    for label in labels:
        roi_mask = atlas_3d == label
        if not np.any(roi_mask):
            continue
        roi_values = perf_4d[roi_mask, :]
        roi_series[int(label)] = roi_values.mean(axis=0)
    return roi_series


def summarize_roi_time_series(roi_series: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Create tabular summaries from ROI time series."""
    summary: Dict[str, float] = {}
    for label, series in roi_series.items():
        summary[f"roi_{label}_mean"] = float(np.mean(series))
        summary[f"roi_{label}_std"] = float(np.std(series))
        summary[f"roi_{label}_peak"] = float(np.max(series))
        summary[f"roi_{label}_ttp_index"] = float(np.argmax(series))
    return summary


def summarize_maps_with_mask(
    maps: Dict[str, np.ndarray],
    mask: np.ndarray,
    prefix: str = "lesion",
) -> Dict[str, float]:
    """Summarize 3D perfusion maps inside a binary mask."""
    summary: Dict[str, float] = {}
    for name, arr in maps.items():
        if arr.shape == mask.shape:
            mask_bool = mask > 0.5
        elif arr.shape == mask.shape[::-1]:
            # Common case: nibabel-loaded NIfTI masks are (x, y, z) while
            # SimpleITK/PyPeT arrays are (z, y, x).
            mask_bool = np.transpose(mask > 0.5, (2, 1, 0))
        else:
            raise ValueError(
                f"Map and mask shapes must match or be reversed axis order. "
                f"Got map {arr.shape} and mask {mask.shape}."
            )
        if not np.any(mask_bool):
            raise ValueError("Mask contains no positive voxels.")
        values = arr[mask_bool]
        summary[f"{prefix}_{name}_mean"] = float(np.mean(values))
        summary[f"{prefix}_{name}_std"] = float(np.std(values))
        summary[f"{prefix}_{name}_min"] = float(np.min(values))
        summary[f"{prefix}_{name}_max"] = float(np.max(values))
    return summary


def load_nifti_array_and_image(path: str):
    """Load a NIfTI image and return both image object and float data array."""
    import nibabel as nib

    img = nib.load(path)
    return img, img.get_fdata()


def _coerce_affine_transform(transform) -> np.ndarray | None:
    if transform is None:
        return None
    if isinstance(transform, np.ndarray):
        matrix = transform
    else:
        try:
            matrix = np.asarray(transform, dtype=float)
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported transform type: {type(transform)}") from exc
    if matrix.shape != (4, 4):
        raise ValueError(f"Affine transform must have shape (4, 4), got {matrix.shape}.")
    return matrix


def resample_to_reference(
    moving_path: str,
    reference_path: str,
    *,
    kind: str = "image",
    transform=None,
    return_image: bool = False,
):
    """Resample a NIfTI image into a declared reference image space.

    This function does not run registration. It supports:
    - grid-only resampling when the images are already in the same physical space
    - applying a known affine transform before resampling

    Args:
        moving_path: Image or mask to resample.
        reference_path: Reference image defining the target grid/space.
        kind: ``"image"``, ``"map"``, or ``"mask"``. Controls interpolation.
        transform: Optional known 4x4 affine transform applied to the moving
            image before resampling. This is not registration; it assumes the
            transform is already known.
        return_image: If True, return a nibabel image object instead of an array.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    if kind not in {"image", "map", "mask"}:
        raise ValueError(f"Unsupported kind '{kind}'. Expected image, map, or mask.")

    reference_img = nib.load(reference_path)
    moving_img = nib.load(moving_path)

    affine_transform = _coerce_affine_transform(transform)
    if affine_transform is not None:
        transformed_affine = affine_transform @ moving_img.affine
        moving_img = nib.Nifti1Image(
            moving_img.get_fdata(),
            transformed_affine,
            moving_img.header,
        )

    order = 0 if kind == "mask" else 1
    resampled = resample_from_to(moving_img, reference_img, order=order)
    return resampled if return_image else resampled.get_fdata()


def resample_mask_to_map(mask_path: str, map_path: str) -> np.ndarray:
    """Resample a mask NIfTI into a map NIfTI space using nearest-neighbor interpolation."""
    return resample_to_reference(mask_path, map_path, kind="mask")


def summarize_maps_with_mask_paths(
    map_paths: Dict[str, str],
    mask_path: str,
    prefix: str = "lesion",
) -> Dict[str, float]:
    """Load NIfTI maps plus a mask path, resample the mask to map space, and summarize."""
    first_map_name = next(iter(map_paths))
    reference_map_path = map_paths[first_map_name]
    mask = resample_mask_to_map(mask_path, reference_map_path)

    maps: Dict[str, np.ndarray] = {}
    for name, path in map_paths.items():
        _, arr = load_nifti_array_and_image(path)
        maps[name] = arr
    return summarize_maps_with_mask(maps, mask, prefix=prefix)


def process_perfusion_native(
    perf_path: str,
    mask_path: Optional[str] = None,
    image_type: str = "ctp",
    scan_interval: Optional[float] = 1.5,
    echo_time: Optional[float] = 0.03,
    method: str = "bcSVD1",
    sidecar_path: Optional[str] = None,
    motion_correction: bool = True,
    aif_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """Native MedTabX perfusion-map generation.

    Note:
        The native preprocessing path does not perform motion correction, so
        ``motion_correction`` is accepted for signature parity with the pypet
        backend but has no effect here. Use ``backend="pypet"`` for motion
        correction.
    """
    aif_kwargs = aif_kwargs or {}
    from .perfusion_utils.load_and_preprocess import (
        load_and_preprocess_perf_map,
        load_and_preprocess_raw_perf,
    )
    from .perfusion_utils.generate_mask import generate_mask_ctp, generate_mask_mrp
    from .perfusion_utils.generate_perfusion_maps import (
        extract_ctc,
        generate_perfusion_maps,
        generate_ttp,
    )
    from .perfusion_utils.determine_aif import determine_aif
    from .perfusion_utils.post_process import post_process_perfusion_map

    params = infer_perfusion_parameters(
        sidecar_path=sidecar_path,
        scan_interval=scan_interval,
        echo_time=echo_time,
    )

    # Step 1: Load and preprocess
    volume_list, time_index = load_and_preprocess_raw_perf(perf_path)
    time_index = [i * params["scan_interval"] for i in time_index]

    # Step 2: Generate or load brain mask
    if mask_path is None:
        if image_type == "ctp":
            mask, mask_path = generate_mask_ctp(volume_list[0], perf_path)
        else:
            mask, mask_path = generate_mask_mrp(volume_list[0], perf_path)
    else:
        mask = load_and_preprocess_perf_map(mask_path)

    # Step 3: Extract contrast time curves
    ctc_volumes, s0_index, bolus_data = extract_ctc(
        volume_list, mask, image_type, params["echo_time"]
    )

    # Step 4: Generate TTP map
    ttp = generate_ttp(ctc_volumes, time_index, s0_index, mask)

    # Step 5: Determine AIF
    aif_properties, aif_mask = determine_aif(
        ctc_volumes, time_index, mask, ttp, **aif_kwargs
    )

    # Step 6: Generate perfusion maps via deconvolution
    mtt, cbv, cbf, tmax = generate_perfusion_maps(
        ctc_volumes, time_index, mask, aif_properties, method=method
    )

    # Step 7: Post-process
    ttp = post_process_perfusion_map(ttp, mask, "TTP")
    cbf = post_process_perfusion_map(cbf, mask, "CBF")
    cbv = post_process_perfusion_map(cbv, mask, "CBV")
    mtt = post_process_perfusion_map(mtt, mask, "MTT")
    tmax = post_process_perfusion_map(tmax, mask, "TMAX")

    return {"cbf": cbf, "cbv": cbv, "mtt": mtt, "ttp": ttp, "tmax": tmax}


def process_perfusion_pypet(*args, **kwargs) -> Dict[str, np.ndarray]:
    """Run the vendored PyPeT backend and return maps in memory."""
    perf_path = kwargs["perf_path"]
    mask_path = kwargs.get("mask_path")
    image_type = kwargs.get("image_type", "ctp")
    sidecar_path = kwargs.get("sidecar_path")
    motion_correction = kwargs.get("motion_correction", True)
    aif_kwargs = kwargs.get("aif_kwargs") or {}
    params = infer_perfusion_parameters(
        sidecar_path=sidecar_path,
        scan_interval=kwargs.get("scan_interval"),
        echo_time=kwargs.get("echo_time"),
    )
    method = kwargs.get("method", "bcSVD1")

    try:
        from .third_party.pypet import config as pypet_config
        from .third_party.pypet.utils import determine_aif as determine_aif_mod
        from .third_party.pypet.utils import generate_mask as generate_mask_mod
        from .third_party.pypet.utils import generate_perfusion_maps as perf_maps_mod
        from .third_party.pypet.utils import load_and_preprocess as preprocess_mod
        from .third_party.pypet.utils import post_process as postprocess_mod
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Vendored PyPeT backend dependencies are incomplete. "
            "Expected ants, SimpleITK, scipy, scikit-image, and related modules."
        ) from exc

    pypet_config.set_runtime_config(
        image_type=image_type,
        scan_interval=params["scan_interval"],
        echo_time=params["echo_time"],
        debug=False,
    )

    # Reload modules so their `from ..config import *` bindings pick up runtime values.
    preprocess_mod = importlib.reload(preprocess_mod)
    generate_mask_mod = importlib.reload(generate_mask_mod)
    perf_maps_mod = importlib.reload(perf_maps_mod)
    determine_aif_mod = importlib.reload(determine_aif_mod)
    postprocess_mod = importlib.reload(postprocess_mod)

    volume_list, time_index = preprocess_mod.load_and_preprocess_raw_perf(
        perf_path,
        apply_motion_correction=motion_correction,
    )

    if mask_path is None:
        if image_type == "ctp":
            mask, mask_path = generate_mask_mod.generate_mask_ctp(volume_list[0], perf_path)
        else:
            mask, mask_path = generate_mask_mod.generate_mask_mrp(volume_list[0], perf_path)
    else:
        mask = preprocess_mod.load_and_preprocess_perf_map(mask_path)

    ctc_volumes, s0_index, bolus_data = perf_maps_mod.extract_ctc(volume_list, mask)
    ttp = perf_maps_mod.generate_ttp(ctc_volumes, time_index, s0_index, mask)
    try:
        aif_properties, aif_mask = determine_aif_mod.determine_aif(
            ctc_volumes, time_index, mask, ttp, **aif_kwargs
        )
    except ValueError:
        ctc_array = np.stack(ctc_volumes, axis=0)
        mask_bool = mask.astype(bool)
        voxel_curves = ctc_array[:, mask_bool]
        if voxel_curves.size == 0:
            raise
        peak_idx = int(np.argmax(voxel_curves.max(axis=0)))
        peak_curve = voxel_curves[:, peak_idx]
        aif_properties = determine_aif_mod.fit_gamma_variate(
            np.asarray(time_index), peak_curve
        )
        aif_mask = np.zeros_like(mask, dtype=bool)
        masked_indices = np.argwhere(mask_bool)
        chosen_index = masked_indices[peak_idx]
        aif_mask[tuple(chosen_index)] = True
    mtt, cbv, cbf, tmax = perf_maps_mod.generate_perfusion_maps(
        ctc_volumes, time_index, mask, aif_properties, method=method
    )

    ttp = postprocess_mod.post_process_perfusion_map(ttp, mask, "TTP")
    cbf = postprocess_mod.post_process_perfusion_map(cbf, mask, "CBF")
    cbv = postprocess_mod.post_process_perfusion_map(cbv, mask, "CBV")
    mtt = postprocess_mod.post_process_perfusion_map(mtt, mask, "MTT")
    tmax = postprocess_mod.post_process_perfusion_map(tmax, mask, "TMAX")

    return {"cbf": cbf, "cbv": cbv, "mtt": mtt, "ttp": ttp, "tmax": tmax}


def process_perfusion(
    perf_path: str,
    mask_path: Optional[str] = None,
    image_type: str = "ctp",
    scan_interval: Optional[float] = 1.5,
    echo_time: Optional[float] = 0.03,
    method: str = "bcSVD1",
    backend: str = "native",
    sidecar_path: Optional[str] = None,
    motion_correction: bool = True,
    aif_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """Process perfusion scan and generate perfusion maps.

    Args:
        perf_path: Path to 4D perfusion NIfTI
        mask_path: Optional brain mask path
        image_type: 'ctp' or 'mrp'
        scan_interval: Time between volumes (seconds)
        echo_time: Echo time for MRP (seconds)
        method: Deconvolution method
        backend: ``native`` or ``pypet``

    Returns:
        Dictionary with perfusion maps: cbf, cbv, mtt, ttp, tmax
    """
    if backend == "native":
        return process_perfusion_native(
            perf_path=perf_path,
            mask_path=mask_path,
            image_type=image_type,
            scan_interval=scan_interval,
            echo_time=echo_time,
            method=method,
            sidecar_path=sidecar_path,
            motion_correction=motion_correction,
            aif_kwargs=aif_kwargs,
        )
    if backend == "pypet":
        return process_perfusion_pypet(
            perf_path=perf_path,
            mask_path=mask_path,
            image_type=image_type,
            scan_interval=scan_interval,
            echo_time=echo_time,
            method=method,
            sidecar_path=sidecar_path,
            motion_correction=motion_correction,
            aif_kwargs=aif_kwargs,
        )
    raise ValueError(f"Unsupported perfusion backend: {backend}")


def save_perfusion_maps(
    maps: Dict[str, np.ndarray],
    reference_img_path: str,
    output_dir: str,
    subject_id: str
):
    """Save perfusion maps as NIfTI files.

    Args:
        maps: Dictionary with perfusion maps
        reference_img_path: Reference image for affine
        output_dir: Output directory
        subject_id: Subject identifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ref_img = sitk.ReadImage(reference_img_path)

    for map_name, map_data in maps.items():
        img = sitk.GetImageFromArray(map_data)
        img.CopyInformation(ref_img)
        sitk.WriteImage(img, str(output_path / f"{subject_id}_{map_name}.nii.gz"))
