"""
Calibration metrics for MedTabX.

A clinically useful risk model has to be calibrated, not only discriminative:
a predicted 20% risk should correspond to roughly 20% observed events. AUROC
is invariant to any monotone rescaling of the scores and therefore cannot see
miscalibration at all, so these metrics are reported alongside it.
"""

from __future__ import annotations

from torch import Tensor

from .base import AccumulatingMetric
from .functional import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    normalize_binary_scores,
    normalize_multiclass_scores,
    reliability_curve,
)


class _CalibrationMetric(AccumulatingMetric):
    """Shared probability normalization for calibration metrics.

    Args:
        name (str): Name of the metric
        task (str): Either :obj:`"binary"` or :obj:`"multiclass"`.
            (default: :obj:`"binary"`)
        from_logits (bool): Whether updates receive unnormalized logits.
            (default: :obj:`True`)
    """

    def __init__(self, name: str, task: str = 'binary',
                 from_logits: bool = True):
        if task not in ('binary', 'multiclass'):
            raise ValueError(f"Unknown task '{task}' for {name}; expected "
                             f"'binary' or 'multiclass'.")
        self.task = task
        super().__init__(name, from_logits=from_logits)

    def probabilities(self) -> Tensor:
        if self.task == 'binary':
            return normalize_binary_scores(self.scores, self.from_logits)
        return normalize_multiclass_scores(self.scores, self.from_logits)


class BrierScore(_CalibrationMetric):
    """Brier score (lower is better).

    Args:
        task (str): Either :obj:`"binary"` or :obj:`"multiclass"`.
            (default: :obj:`"binary"`)
        from_logits (bool): Whether updates receive unnormalized logits.
            (default: :obj:`True`)
    """

    def __init__(self, task: str = 'binary', from_logits: bool = True):
        super().__init__('brier', task=task, from_logits=from_logits)

    def compute(self) -> float:
        if self.is_empty:
            return float('nan')
        return brier_score(self.probabilities(), self.targets)


class ExpectedCalibrationError(_CalibrationMetric):
    """Expected calibration error (lower is better).

    Args:
        num_bins (int): Number of equal-width confidence bins.
            (default: :obj:`15`)
        task (str): Either :obj:`"binary"` or :obj:`"multiclass"`.
            (default: :obj:`"binary"`)
        from_logits (bool): Whether updates receive unnormalized logits.
            (default: :obj:`True`)
    """

    def __init__(self, num_bins: int = 15, task: str = 'binary',
                 from_logits: bool = True):
        self.num_bins = num_bins
        super().__init__('ece', task=task, from_logits=from_logits)

    def compute(self) -> float:
        if self.is_empty:
            return float('nan')
        return expected_calibration_error(self.probabilities(), self.targets,
                                          num_bins=self.num_bins)

    def reliability_curve(self) -> dict[str, list[float]]:
        """Per-bin confidence and accuracy, for a reliability diagram."""
        return reliability_curve(self.probabilities(), self.targets,
                                 num_bins=self.num_bins)


class MaximumCalibrationError(_CalibrationMetric):
    """Largest confidence-accuracy gap over any non-empty confidence bin.

    Args:
        num_bins (int): Number of equal-width confidence bins.
            (default: :obj:`15`)
        task (str): Either :obj:`"binary"` or :obj:`"multiclass"`.
            (default: :obj:`"binary"`)
        from_logits (bool): Whether updates receive unnormalized logits.
            (default: :obj:`True`)
    """

    def __init__(self, num_bins: int = 15, task: str = 'binary',
                 from_logits: bool = True):
        self.num_bins = num_bins
        super().__init__('mce', task=task, from_logits=from_logits)

    def compute(self) -> float:
        if self.is_empty:
            return float('nan')
        return maximum_calibration_error(self.probabilities(), self.targets,
                                         num_bins=self.num_bins)
