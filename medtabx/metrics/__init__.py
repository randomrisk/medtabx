r"""Evaluation metrics for MedTabX.

The panel covers discrimination, calibration and regression error, plus
bootstrap confidence intervals, and depends on :mod:`torch` only.
"""

from .base import AccumulatingMetric, Metric
from .bootstrap import BootstrapResult, bootstrap_ci, bootstrap_metric
from .calibration import (
    BrierScore,
    ExpectedCalibrationError,
    MaximumCalibrationError,
)
from .classification import (
    AUROC,
    Accuracy,
    AveragePrecision,
    BalancedAccuracy,
    F1,
    Precision,
    Recall,
    Specificity,
)
from .collection import (
    LOWER_IS_BETTER,
    MetricCollection,
    default_metrics,
    metric_mode,
)
from .regression import MAE, MSE, R2, RMSE

__all__ = [
    # Base
    "Metric",
    "AccumulatingMetric",
    # Classification
    "Accuracy",
    "BalancedAccuracy",
    "Precision",
    "Recall",
    "F1",
    "Specificity",
    "AUROC",
    "AveragePrecision",
    # Calibration
    "BrierScore",
    "ExpectedCalibrationError",
    "MaximumCalibrationError",
    # Regression
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    # Grouping and selection
    "MetricCollection",
    "default_metrics",
    "metric_mode",
    "LOWER_IS_BETTER",
    # Uncertainty
    "bootstrap_ci",
    "bootstrap_metric",
    "BootstrapResult",
]
