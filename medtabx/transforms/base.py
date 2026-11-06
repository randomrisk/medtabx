"""Base transform class with fit/transform/save/load pattern."""
from abc import ABC, abstractmethod
from typing import Any
import pickle


class FittableTransform(ABC):
    """
    Base class for all transforms following PyTorch Frame's pattern.

    All transforms must implement fit/transform/save/load.
    """

    @abstractmethod
    def fit(self, data: Any) -> 'FittableTransform':
        """
        Fit the transform to data.

        Args:
            data: Input data to fit on

        Returns:
            self for chaining
        """
        pass

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform data using fitted parameters.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        pass

    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def save(self, path: str):
        """Save fitted transform to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'FittableTransform':
        """Load fitted transform from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
