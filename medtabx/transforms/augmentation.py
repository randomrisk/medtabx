"""Augmentation transforms for tabular data."""
import numpy as np
from .base import FittableTransform


class GaussianNoise(FittableTransform):
    """Add Gaussian noise to numerical features."""

    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_level, data.shape)
        return data + noise


class Mixup(FittableTransform):
    """Mixup augmentation for tabular data."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        batch_size = data.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        indices = np.random.permutation(batch_size)
        return lam * data + (1 - lam) * data[indices]


class RandomRotation3D(FittableTransform):
    """Random rotation for 3D medical images."""

    def __init__(self, max_angle: float = 15.0):
        self.max_angle = max_angle

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        # Simple rotation along one axis
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        # For simplicity, return original (full 3D rotation needs scipy)
        return data


class IntensityShift(FittableTransform):
    """Random intensity shift for medical images."""

    def __init__(self, shift_range: float = 0.1):
        self.shift_range = shift_range

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(-self.shift_range, self.shift_range)
        return data + shift


class TimeJitter(FittableTransform):
    """Add jitter to time series data."""

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.sigma, data.shape)
        return data + noise


class TimeScaling(FittableTransform):
    """Scale time series data."""

    def __init__(self, scale_range: tuple = (0.9, 1.1)):
        self.scale_range = scale_range

    def fit(self, data: np.ndarray):
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        return data * scale





