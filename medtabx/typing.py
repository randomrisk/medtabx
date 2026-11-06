from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypeAlias

import pandas as pd
import torch
from torch import Tensor

from medtabx.data.multi_embedding_tensor import MultiEmbeddingTensor
from medtabx.data.multi_nested_tensor import MultiNestedTensor

WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
WITH_PT24 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 4


class Metric(Enum):
    r"""The metric.

    The names mirror the implementations in :mod:`medtabx.metrics`, which is
    where the metrics are actually computed; this enum only names them so
    that a task type can declare which ones apply to it.

    Attributes:
        ACCURACY: accuracy
        BALANCED_ACCURACY: mean of per-class recall
        F1: F1 score
        ROCAUC: area under the ROC curve
        AUPRC: average precision
        BRIER: Brier score
        ECE: expected calibration error
        RMSE: rmse
        MAE: mae
        R2: coefficient of determination
    """
    ACCURACY = 'accuracy'
    BALANCED_ACCURACY = 'balanced_accuracy'
    F1 = 'f1'
    ROCAUC = 'rocauc'
    AUPRC = 'auprc'
    BRIER = 'brier'
    ECE = 'ece'
    RMSE = 'rmse'
    MAE = 'mae'
    R2 = 'r2'

    def supports_task_type(self, task_type: TaskType) -> bool:
        return self in task_type.supported_metrics


class TaskType(Enum):
    r"""The type of the task.

    Attributes:
        REGRESSION: Regression task.
        MULTICLASS_CLASSIFICATION: Multi-class classification task.
        BINARY_CLASSIFICATION: Binary classification task.
    """
    REGRESSION = 'regression'
    MULTICLASS_CLASSIFICATION = 'multiclass_classification'
    BINARY_CLASSIFICATION = 'binary_classification'
    MULTILABEL_CLASSIFICATION = 'multilabel_classification'

    @property
    def is_classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION,
                        TaskType.MULTICLASS_CLASSIFICATION)

    @property
    def is_regression(self) -> bool:
        return self == TaskType.REGRESSION

    @property
    def supported_metrics(self) -> list[Metric]:
        if self == TaskType.REGRESSION:
            return [Metric.RMSE, Metric.MAE, Metric.R2]
        elif self == TaskType.BINARY_CLASSIFICATION:
            return [
                Metric.ACCURACY, Metric.BALANCED_ACCURACY, Metric.F1,
                Metric.ROCAUC, Metric.AUPRC, Metric.BRIER, Metric.ECE
            ]
        elif self == TaskType.MULTICLASS_CLASSIFICATION:
            return [
                Metric.ACCURACY, Metric.BALANCED_ACCURACY, Metric.ROCAUC,
                Metric.AUPRC, Metric.BRIER, Metric.ECE
            ]
        else:
            return []


class NAStrategy(Enum):
    r"""Strategy for dealing with NaN values in columns.

    Attributes:
        MEAN: Replaces NaN values with the mean of a
            :obj:`torch_frame.numerical` column.
        ZEROS: Replaces NaN values with zeros in a
            :obj:`torch_frame.numerical` column.
        MOST_FREQUENT: Replaces NaN values with the most frequent category of a
            :obj:`torch_frame.categorical` column.
    """
    MEAN = 'mean'
    MOST_FREQUENT = 'most_frequent'
    ZEROS = 'zeros'
    OLDEST_TIMESTAMP = 'oldest_timestamp'
    NEWEST_TIMESTAMP = 'newest_timestamp'
    MEDIAN_TIMESTAMP = 'median_timestamp'

    @property
    def is_categorical_strategy(self) -> bool:
        return self == NAStrategy.MOST_FREQUENT

    @property
    def is_multicategorical_strategy(self) -> bool:
        return self == NAStrategy.ZEROS

    @property
    def is_numerical_strategy(self) -> bool:
        return self in [NAStrategy.MEAN, NAStrategy.ZEROS]

    @property
    def is_timestamp_strategy(self) -> bool:
        return self in [
            NAStrategy.NEWEST_TIMESTAMP,
            NAStrategy.OLDEST_TIMESTAMP,
            NAStrategy.MEDIAN_TIMESTAMP,
        ]


Series: TypeAlias = pd.Series
DataFrame: TypeAlias = pd.DataFrame

IndexSelectType: TypeAlias = int | list[int] | range | slice | Tensor
ColumnSelectType: TypeAlias = str | list[str]
TextTokenizationMapping: TypeAlias = Mapping[str, Tensor]
TextTokenizationOutputs: TypeAlias = \
    list[TextTokenizationMapping] | TextTokenizationMapping
TensorData: TypeAlias = (Tensor | MultiNestedTensor | MultiEmbeddingTensor
                         | dict[str, MultiNestedTensor])
