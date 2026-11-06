"""Load and preprocess perfusion data."""
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def reorient_to_ras(image):
    """Reorient image to RAS orientation."""
    return sitk.DICOMOrient(image, 'RAS')


def load_and_preprocess_raw_perf(perf_path):
    """Load and preprocess raw 4D perfusion data.

    Args:
        perf_path: Path to 4D perfusion NIfTI

    Returns:
        volume_list: List of 3D volumes
        time_index: Time indices for each volume
    """
    img = sitk.ReadImage(perf_path)
    img = reorient_to_ras(img)
    data = sitk.GetArrayFromImage(img)

    # Convert to list of 3D volumes
    volume_list = [data[i] for i in range(data.shape[0])]
    time_index = list(range(len(volume_list)))

    return volume_list, time_index


def load_and_preprocess_perf_map(map_path):
    """Load and preprocess perfusion map.

    Args:
        map_path: Path to perfusion map NIfTI

    Returns:
        data: 3D array
    """
    img = sitk.ReadImage(map_path)
    img = reorient_to_ras(img)
    data = sitk.GetArrayFromImage(img)
    return data
