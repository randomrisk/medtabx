"""Manual feature extraction for imaging data."""
import numpy as np


def extract_imaging_features(data: np.ndarray) -> dict:
    """Extract statistical features from 3D medical images."""
    return {
        'intensity_mean': np.mean(data),
        'intensity_std': np.std(data),
        'intensity_max': np.max(data),
        'intensity_min': np.min(data),
    }
