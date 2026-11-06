"""Manual feature extraction for time series data."""
import numpy as np


def extract_timeseries_features(data: np.ndarray) -> dict:
    """Extract statistical features from time series data."""
    return {
        'signal_mean': np.mean(data),
        'signal_std': np.std(data),
        'signal_max': np.max(data),
        'signal_min': np.min(data),
    }
