"""
Classification metrics for MedTabX.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import AccumulatingMetric, Metric
from .functional import (
    binary_auroc,
    predicted_labels,
    binary_average_precision,
    multiclass_auroc,
    multiclass_average_precision,
    normalize_binary_scores,
    normalize_multiclass_scores,
)


class Accuracy(Metric):
    """Accuracy metric for classification."""

    def __init__(self, from_logits: bool = True,
                 threshold: float = 0.5):
        self.from_logits = from_logits
        self.threshold = threshold
        super().__init__(name='accuracy')

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: Tensor, targets: Tensor):
        """Update with predictions and targets.

        Args:
            predictions: [B, num_classes] logits, [B, 1] or [B] binary
                scores, or [B] class indices
            targets: [B] class indices
        """
        preds = predicted_labels(predictions, from_logits=self.from_logits,
                                 threshold=self.threshold)

        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Precision(Metric):
    """Precision metric for binary classification."""

    def __init__(self, from_logits: bool = True,
                 threshold: float = 0.5):
        self.from_logits = from_logits
        self.threshold = threshold
        super().__init__(name='precision')

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0

    def update(self, predictions: Tensor, targets: Tensor):
        preds = predicted_labels(predictions, from_logits=self.from_logits,
                                 threshold=self.threshold)

        self.true_positives += ((preds == 1) & (targets == 1)).sum().item()
        self.false_positives += ((preds == 1) & (targets == 0)).sum().item()

    def compute(self) -> float:
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom


class Recall(Metric):
    """Recall metric for binary classification."""

    def __init__(self, from_logits: bool = True,
                 threshold: float = 0.5):
        self.from_logits = from_logits
        self.threshold = threshold
        super().__init__(name='recall')

    def reset(self):
        self.true_positives = 0
        self.false_negatives = 0

    def update(self, predictions: Tensor, targets: Tensor):
        preds = predicted_labels(predictions, from_logits=self.from_logits,
                                 threshold=self.threshold)

        self.true_positives += ((preds == 1) & (targets == 1)).sum().item()
        self.false_negatives += ((preds == 0) & (targets == 1)).sum().item()

    def compute(self) -> float:
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom


class F1(Metric):
    """F1 score for binary classification."""

    def __init__(self, from_logits: bool = True,
                 threshold: float = 0.5):
        self.from_logits = from_logits
        self.threshold = threshold
        super().__init__(name='f1')

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, predictions: Tensor, targets: Tensor):
        preds = predicted_labels(predictions, from_logits=self.from_logits,
                                 threshold=self.threshold)

        self.true_positives += ((preds == 1) & (targets == 1)).sum().item()
        self.false_positives += ((preds == 1) & (targets == 0)).sum().item()
        self.false_negatives += ((preds == 0) & (targets == 1)).sum().item()

    def compute(self) -> float:
        denom = 2 * self.true_positives + self.false_positives + \
            self.false_negatives
        if denom == 0:
            return 0.0
        return 2 * self.true_positives / denom


class Specificity(Metric):
    """Specificity (true negative rate) for binary classification."""

    def __init__(self, from_logits: bool = True,
                 threshold: float = 0.5):
        self.from_logits = from_logits
        self.threshold = threshold
        super().__init__(name='specificity')

    def reset(self):
        self.true_negatives = 0
        self.false_positives = 0

    def update(self, predictions: Tensor, targets: Tensor):
        preds = predicted_labels(predictions, from_logits=self.from_logits,
                                 threshold=self.threshold)

        self.true_negatives += ((preds == 0) & (targets == 0)).sum().item()
        self.false_positives += ((preds == 1) & (targets == 0)).sum().item()

    def compute(self) -> float:
        denom = self.true_negatives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_negatives / denom


class BalancedAccuracy(Metric):
    """Mean of per-class recall.

    Preferable to plain accuracy on the imbalanced outcome labels that are
    common in clinical cohorts, where a majority-class predictor can score a
    high accuracy while never identifying a positive case.

    Args:
        num_classes (int): Number of classes. (default: :obj:`2`)
    """

    def __init__(self, num_classes: int = 2, from_logits: bool = True,
                 threshold: float = 0.5):
        self.num_classes = num_classes
        self.from_logits = from_logits
        self.threshold = threshold
        super().__init__(name='balanced_accuracy')

    def reset(self):
        self.correct_per_class = torch.zeros(self.num_classes)
        self.total_per_class = torch.zeros(self.num_classes)

    def update(self, predictions: Tensor, targets: Tensor):
        preds = predicted_labels(predictions, from_logits=self.from_logits,
                                 threshold=self.threshold)

        targets = targets.reshape(-1)
        for class_idx in range(self.num_classes):
            in_class = targets == class_idx
            self.total_per_class[class_idx] += in_class.sum().item()
            self.correct_per_class[class_idx] += (
                (preds == class_idx) & in_class).sum().item()

    def compute(self) -> float:
        present = self.total_per_class > 0
        if not bool(present.any()):
            return 0.0
        recalls = (self.correct_per_class[present] /
                   self.total_per_class[present])
        return float(recalls.mean().item())


class AUROC(AccumulatingMetric):
    """Area under the ROC curve.

    Args:
        task (str): Either :obj:`"binary"` or :obj:`"multiclass"`. The
            multiclass variant is macro-averaged one-vs-rest.
            (default: :obj:`"binary"`)
        from_logits (bool): Whether updates receive unnormalized logits.
            (default: :obj:`True`)
    """

    def __init__(self, task: str = 'binary', from_logits: bool = True):
        if task not in ('binary', 'multiclass'):
            raise ValueError(f"Unknown task '{task}' for AUROC; expected "
                             f"'binary' or 'multiclass'.")
        self.task = task
        super().__init__(name='auroc', from_logits=from_logits)

    def compute(self) -> float:
        if self.is_empty:
            return float('nan')
        if self.task == 'binary':
            scores = normalize_binary_scores(self.scores, self.from_logits)
            return binary_auroc(scores, self.targets)
        probs = normalize_multiclass_scores(self.scores, self.from_logits)
        return multiclass_auroc(probs, self.targets)


class AveragePrecision(AccumulatingMetric):
    """Average precision, i.e. area under the precision-recall curve.

    Reported as ``auprc``. On the low-prevalence outcomes typical of medical
    cohorts this separates models that AUROC rates as near-identical.

    Args:
        task (str): Either :obj:`"binary"` or :obj:`"multiclass"`. The
            multiclass variant is macro-averaged one-vs-rest.
            (default: :obj:`"binary"`)
        from_logits (bool): Whether updates receive unnormalized logits.
            (default: :obj:`True`)
    """

    def __init__(self, task: str = 'binary', from_logits: bool = True):
        if task not in ('binary', 'multiclass'):
            raise ValueError(f"Unknown task '{task}' for AveragePrecision; "
                             f"expected 'binary' or 'multiclass'.")
        self.task = task
        super().__init__(name='auprc', from_logits=from_logits)

    def compute(self) -> float:
        if self.is_empty:
            return float('nan')
        if self.task == 'binary':
            scores = normalize_binary_scores(self.scores, self.from_logits)
            return binary_average_precision(scores, self.targets)
        probs = normalize_multiclass_scores(self.scores, self.from_logits)
        return multiclass_average_precision(probs, self.targets)
