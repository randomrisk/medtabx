"""
Base metrics class for MedTabX.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import Tensor


class Metric(ABC):
    """Base class for all metrics.

    Args:
        name (str): Name of the metric
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def update(self, predictions: Tensor, targets: Tensor):
        """Update metric with new predictions and targets."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> float:
        """Compute the final metric value."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset metric state."""
        raise NotImplementedError


class AccumulatingMetric(Metric):
    """Base class for metrics that need the whole prediction set at once.

    Ranking and calibration metrics (AUROC, average precision, ECE) cannot be
    reduced incrementally, so this base buffers every batch it is given on the
    CPU and concatenates at :meth:`compute` time.

    Args:
        name (str): Name of the metric
        from_logits (bool): Whether :meth:`update` receives unnormalized
            logits rather than probabilities. (default: :obj:`True`)
    """

    def __init__(self, name: str, from_logits: bool = True):
        self.from_logits = from_logits
        super().__init__(name)

    def reset(self):
        self._scores: list[Tensor] = []
        self._targets: list[Tensor] = []

    def update(self, predictions: Tensor, targets: Tensor):
        """Buffer a batch of predictions and targets.

        Args:
            predictions: Model output, kept in whatever shape the model emits
            targets: [B] targets
        """
        self._scores.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu().reshape(-1))

    @property
    def scores(self) -> Tensor:
        """All buffered predictions, concatenated along the batch dim."""
        if not self._scores:
            return torch.empty(0)
        return torch.cat(self._scores, dim=0)

    @property
    def targets(self) -> Tensor:
        """All buffered targets, concatenated along the batch dim."""
        if not self._targets:
            return torch.empty(0)
        return torch.cat(self._targets, dim=0)

    @property
    def is_empty(self) -> bool:
        return not self._scores
