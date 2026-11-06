"""
Regression metrics for MedTabX.

Used by continuous medical targets such as perfusion summary values, lesion
volumes and other derived biomarkers.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import AccumulatingMetric, Metric
from .functional import r2_score


class _StreamingRegressionMetric(Metric):
    """Shared accumulation for error metrics that reduce to a running sum."""

    def reset(self):
        self.total_error = 0.0
        self.total = 0

    def _errors(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError

    def update(self, predictions: Tensor, targets: Tensor):
        """Update with predictions and targets.

        Args:
            predictions: [B] or [B, 1] continuous predictions
            targets: [B] or [B, 1] continuous targets
        """
        predictions = predictions.detach().reshape(-1).to(torch.float64)
        targets = targets.detach().reshape(-1).to(torch.float64)
        if predictions.numel() != targets.numel():
            raise ValueError(
                f"predictions and targets must have the same number of "
                f"elements, but got {predictions.numel()} and "
                f"{targets.numel()}.")
        self.total_error += self._errors(predictions, targets).sum().item()
        self.total += targets.numel()


class MAE(_StreamingRegressionMetric):
    """Mean absolute error."""

    def __init__(self):
        super().__init__(name='mae')

    def _errors(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets).abs()

    def compute(self) -> float:
        if self.total == 0:
            return float('nan')
        return self.total_error / self.total


class MSE(_StreamingRegressionMetric):
    """Mean squared error."""

    def __init__(self, name: str = 'mse'):
        super().__init__(name=name)

    def _errors(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets)**2

    def compute(self) -> float:
        if self.total == 0:
            return float('nan')
        return self.total_error / self.total


class RMSE(MSE):
    """Root mean squared error."""

    def __init__(self):
        super().__init__(name='rmse')

    def compute(self) -> float:
        mse = super().compute()
        if mse != mse:  # NaN
            return mse
        return mse**0.5


class R2(AccumulatingMetric):
    """Coefficient of determination.

    Unlike the error metrics this needs the target mean over the whole split,
    so predictions are buffered until :meth:`compute`.
    """

    def __init__(self):
        super().__init__(name='r2', from_logits=False)

    def compute(self) -> float:
        if self.is_empty:
            return float('nan')
        return r2_score(self.scores, self.targets)
