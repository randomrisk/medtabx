"""
I/O module for DICOM medical imaging format.

DICOM (Digital Imaging and Communications in Medicine) is the standard
format for medical imaging data including CT, MRI, X-ray, ultrasound.

Supports:
- Single DICOM files (.dcm)
- DICOM series (multiple slices)
- Metadata extraction
- Hounsfield Unit (HU) conversion for CT
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import warnings

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    warnings.warn(
        "pydicom not installed. Install with: pip install pydicom"
    )


def load_dicom_file(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load single DICOM file.

    Args:
        filepath: Path to DICOM file

    Returns:
        Tuple of (image_array, metadata_dict)

    Example:
        >>> img, meta = load_dicom_file('slice001.dcm')
        >>> print(img.shape)  # (512, 512)
        >>> print(meta['PatientID'])
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom required. Install: pip install pydicom")

    dcm = pydicom.dcmread(filepath)
    image = dcm.pixel_array.astype(np.float32)

    # Extract metadata
    metadata = {
        'PatientID': getattr(dcm, 'PatientID', None),
        'StudyDate': getattr(dcm, 'StudyDate', None),
        'Modality': getattr(dcm, 'Modality', None),
        'SliceThickness': getattr(dcm, 'SliceThickness', None),
        'PixelSpacing': getattr(dcm, 'PixelSpacing', None),
    }

    return image, metadata


def load_dicom_series(directory: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM series from directory.

    Args:
        directory: Directory containing DICOM slices

    Returns:
        Tuple of (3D_volume, metadata_dict)

    Example:
        >>> volume, meta = load_dicom_series('patient001/ct_scan/')
        >>> print(volume.shape)  # (100, 512, 512)
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom required")

    # Get all DICOM files
    dicom_files = sorted(Path(directory).glob('*.dcm'))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    # Load first to get dimensions
    first_dcm = pydicom.dcmread(str(dicom_files[0]))
    slices = [first_dcm]

    # Load remaining slices
    for f in dicom_files[1:]:
        slices.append(pydicom.dcmread(str(f)))

    # Sort by slice location
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Stack into 3D volume
    volume = np.stack([s.pixel_array for s in slices], axis=0)

    # Extract metadata from first slice
    metadata = {
        'PatientID': getattr(first_dcm, 'PatientID', None),
        'Modality': getattr(first_dcm, 'Modality', None),
        'SliceThickness': getattr(first_dcm, 'SliceThickness', None),
        'PixelSpacing': getattr(first_dcm, 'PixelSpacing', None),
        'NumSlices': len(slices),
    }

    return volume.astype(np.float32), metadata
