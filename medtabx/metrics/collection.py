"""
Grouping of metrics into a single update/compute unit.

The training loop evaluates a whole panel of metrics per split, so it needs
one object it can update per batch and reduce once at the end of an epoch.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from torch import Tensor

from medtabx.typing import TaskType

from .base import Metric
from .calibration import BrierScore, ExpectedCalibrationError
from .classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    BalancedAccuracy,
    F1,
    Precision,
    Recall,
)
from .regression import MAE, R2, RMSE

__all__ = ["MetricCollection", "default_metrics", "LOWER_IS_BETTER",
           "metric_mode"]

#: Metric names for which a smaller value is a better model. Used to pick the
#: comparison direction when a metric is monitored for model selection.
LOWER_IS_BETTER = frozenset(
    {"loss", "mae", "mse", "rmse", "brier", "ece", "mce"})


def metric_mode(name: str) -> str:
    r"""Return :obj:`"min"` or :obj:`"max"` for a metric name.

    Args:
        name (str): Metric name, optionally prefixed with a split, e.g.
            :obj:`"val_auroc"`.
    """
    bare = name.split("_", 1)[1] if name.startswith(
        ("train_", "val_", "test_")) else name
    return "min" if bare in LOWER_IS_BETTER else "max"


class MetricCollection:
    r"""A set of metrics updated and computed together.

    .. code-block:: python

        from medtabx.metrics import MetricCollection, Accuracy, AUROC

        metrics = MetricCollection([Accuracy(), AUROC()])
        metrics.update(logits, targets)
        print(metrics.compute())
        >>> {'accuracy': 0.82, 'auroc': 0.88}

    Args:
        metrics (iterable or dict): The metrics to group. An iterable keys
            each metric by its own :obj:`name`; a dict keys them explicitly.
    """

    def __init__(self, metrics: Iterable[Metric] | Mapping[str, Metric]):
        if isinstance(metrics, Mapping):
            self.metrics: dict[str, Metric] = dict(metrics)
        else:
            self.metrics = {}
            for metric in metrics:
                if metric.name in self.metrics:
                    raise ValueError(
                        f"Duplicate metric name '{metric.name}'. Pass a dict "
                        f"to give the metrics distinct keys.")
                self.metrics[metric.name] = metric

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update every metric with the same batch."""
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Reduce every metric to a float."""
        return {name: metric.compute() for name, metric in
                self.metrics.items()}

    def reset(self) -> None:
        """Reset every metric's accumulated state."""
        for metric in self.metrics.values():
            metric.reset()

    def __len__(self) -> int:
        return len(self.metrics)

    def __contains__(self, name: str) -> bool:
        return name in self.metrics

    def __getitem__(self, name: str) -> Metric:
        return self.metrics[name]

    def __iter__(self):
        return iter(self.metrics.values())

    def __repr__(self) -> str:
        names = ", ".join(self.metrics)
        return f"{self.__class__.__name__}({names})"


def default_metrics(task_type: TaskType | str,
                    num_classes: int | None = None,
                    from_logits: bool = True) -> MetricCollection:
    r"""The standard metric panel for a task type.

    Classification panels pair discrimination (AUROC, average precision) with
    calibration (Brier, ECE), since a model can rank cases well and still emit
    risks that are unusable at the bedside.

    Args:
        task_type (TaskType or str): The task to build a panel for.
        num_classes (int, optional): Required for multiclass classification.
            (default: :obj:`None`)
        from_logits (bool): Whether the metrics will receive unnormalized
            logits. (default: :obj:`True`)
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    if task_type == TaskType.BINARY_CLASSIFICATION:
        return MetricCollection([
            Accuracy(from_logits=from_logits),
            BalancedAccuracy(num_classes=2, from_logits=from_logits),
            Precision(from_logits=from_logits),
            Recall(from_logits=from_logits),
            F1(from_logits=from_logits),
            AUROC(task="binary", from_logits=from_logits),
            AveragePrecision(task="binary", from_logits=from_logits),
            BrierScore(task="binary", from_logits=from_logits),
            ExpectedCalibrationError(task="binary", from_logits=from_logits),
        ])

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if num_classes is None:
            raise ValueError(
                "num_classes is required for a multiclass metric panel.")
        return MetricCollection([
            Accuracy(from_logits=from_logits),
            BalancedAccuracy(num_classes=num_classes,
                             from_logits=from_logits),
            AUROC(task="multiclass", from_logits=from_logits),
            AveragePrecision(task="multiclass", from_logits=from_logits),
            BrierScore(task="multiclass", from_logits=from_logits),
            ExpectedCalibrationError(task="multiclass",
                                     from_logits=from_logits),
        ])

    if task_type == TaskType.REGRESSION:
        return MetricCollection([MAE(), RMSE(), R2()])

    raise ValueError(
        f"No default metric panel is defined for task type '{task_type}'.")
