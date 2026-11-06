"""Tabular data transforms."""
import numpy as np
from typing import Dict, Any
from .base import FittableTransform


class StandardScaler(FittableTransform):
    """Standardize numerical features to zero mean and unit variance."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> 'StandardScaler':
        """Fit scaler to data."""
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Transform must be fitted before use")
        return (data - self.mean_) / self.std_
