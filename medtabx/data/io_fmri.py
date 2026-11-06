"""
I/O module for fMRI (functional MRI) data.

Supports:
- NIfTI format (.nii, .nii.gz)
- 4D time series data
- ROI extraction
- Time series extraction
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import warnings

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not installed. Install: pip install nibabel")


def load_fmri(
    filepath: str,
    return_affine: bool = False
) -> Tuple[np.ndarray, ...]:
    """Load 4D fMRI data.

    Args:
        filepath: Path to fMRI file (.nii or .nii.gz)
        return_affine: If True, return affine matrix

    Returns:
        Tuple of (data_array, [affine])
        - data_array: [X, Y, Z, T] where T is time

    Example:
        >>> fmri_data = load_fmri('bold.nii.gz')
        >>> print(fmri_data.shape)  # (64, 64, 40, 200)
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel required")

    nii = nib.load(filepath)
    data = nii.get_fdata()

    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")

    if return_affine:
        return data, nii.affine
    return data
