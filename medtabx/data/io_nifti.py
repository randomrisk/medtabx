"""
I/O module for NIfTI medical imaging format.

NIfTI (Neuroimaging Informatics Technology Initiative) is a common format
for storing 3D medical images, especially in neuroimaging (MRI, fMRI).

Supports:
- .nii files (uncompressed)
- .nii.gz files (compressed)
- 3D and 4D volumes
- Metadata extraction (affine, header info)
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
    warnings.warn(
        "nibabel not installed. Install with: pip install nibabel"
    )


def load_nifti(
    filepath: str,
    return_affine: bool = False,
    return_header: bool = False
) -> Tuple[np.ndarray, ...]:
    """Load NIfTI file (.nii or .nii.gz).

    Args:
        filepath: Path to NIfTI file
        return_affine: If True, return affine transformation matrix
        return_header: If True, return header information

    Returns:
        Tuple containing:
        - image_data: numpy array [D, H, W] or [D, H, W, T] for 4D
        - affine: (optional) 4x4 affine matrix
        - header: (optional) NIfTI header object

    Raises:
        ImportError: If nibabel is not installed
        FileNotFoundError: If file doesn't exist

    Example:
        >>> # Load image only
        >>> img = load_nifti('scan.nii.gz')
        >>> print(img.shape)  # (256, 256, 256)

        >>> # Load with affine
        >>> img, affine = load_nifti('scan.nii.gz', return_affine=True)
        >>> print(affine.shape)  # (4, 4)
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel required. Install: pip install nibabel")

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load NIfTI file
    nii_img = nib.load(filepath)
    image_data = nii_img.get_fdata()

    results = [image_data]

    if return_affine:
        results.append(nii_img.affine)

    if return_header:
        results.append(nii_img.header)

    return tuple(results) if len(results) > 1 else results[0]


def save_nifti(
    image_data: np.ndarray,
    filepath: str,
    affine: Optional[np.ndarray] = None,
    header: Optional[Any] = None
) -> None:
    """Save numpy array as NIfTI file.

    Args:
        image_data: Image array to save
        filepath: Output path (.nii or .nii.gz)
        affine: Optional affine transformation matrix
        header: Optional NIfTI header

    Example:
        >>> img = np.random.randn(256, 256, 256)
        >>> save_nifti(img, 'output.nii.gz')
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel required")

    if affine is None:
        affine = np.eye(4)

    nii_img = nib.Nifti1Image(image_data, affine, header)
    nib.save(nii_img, filepath)
