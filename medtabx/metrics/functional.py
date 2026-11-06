"""Functional metric implementations.

These are written against :mod:`torch` only. ``scikit-learn`` is an optional
extra of this project (it ships with the tree baselines), so the metric stack
deliberately avoids depending on it: evaluating a model must work in a
``medtabx[core]`` install.

All functions return plain Python floats and return ``float('nan')`` when a
metric is undefined for the given inputs (for example AUROC on a split that
contains a single class), rather than raising. Callers that need to
distinguish "undefined" from "bad" should check with :func:`math.isnan`.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "predicted_labels",
    "normalize_binary_scores",
    "normalize_multiclass_scores",
    "binary_auroc",
    "binary_average_precision",
    "multiclass_auroc",
    "multiclass_average_precision",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
    "r2_score",
]


def _flatten(x: Tensor) -> Tensor:
    return x.detach().reshape(-1).to(torch.float64)


def normalize_binary_scores(scores: Tensor, from_logits: bool = True) -> Tensor:
    r"""Reduce model output to a 1-D positive-class score.

    Accepts :obj:`[B]`, :obj:`[B, 1]` and :obj:`[B, 2]` outputs, which is the
    range of shapes the binary decoders in this repository can produce.

    Args:
        scores (Tensor): Raw model output.
        from_logits (bool): Whether :obj:`scores` are unnormalized logits. If
            :obj:`True`, a sigmoid (single-column input) or a softmax
            (two-column input) is applied. (default: :obj:`True`)
    """
    scores = scores.detach()
    if scores.ndim == 2 and scores.size(1) == 2:
        if from_logits:
            return scores.softmax(dim=1)[:, 1].to(torch.float64)
        return scores[:, 1].to(torch.float64)
    if scores.ndim == 2 and scores.size(1) == 1:
        scores = scores.squeeze(1)
    if scores.ndim != 1:
        raise ValueError(
            f"Expected a binary score of shape [B], [B, 1] or [B, 2], but got "
            f"{list(scores.shape)}.")
    if from_logits:
        return scores.sigmoid().to(torch.float64)
    return scores.to(torch.float64)


def normalize_multiclass_scores(scores: Tensor,
                                from_logits: bool = True) -> Tensor:
    r"""Reduce model output to a :obj:`[B, C]` probability matrix."""
    scores = scores.detach()
    if scores.ndim != 2:
        raise ValueError(
            f"Expected a multiclass score of shape [B, C], but got "
            f"{list(scores.shape)}.")
    if from_logits:
        return scores.softmax(dim=1).to(torch.float64)
    return scores.to(torch.float64)


def predicted_labels(scores: Tensor, from_logits: bool = True,
                     threshold: float = 0.5) -> Tensor:
    r"""Reduce model output to hard class predictions.

    Handles the three shapes a classification head in this repository can
    emit: :obj:`[B, C]` logits are arg-maxed, a :obj:`[B, 1]` or :obj:`[B]`
    float output is a single binary score and gets thresholded, and an
    integer :obj:`[B]` tensor is taken to be class indices already.

    Args:
        scores (Tensor): Raw model output, or class indices.
        from_logits (bool): Whether a single-column float output is an
            unnormalized logit rather than a probability.
            (default: :obj:`True`)
        threshold (float): Decision threshold on the probability scale, used
            only for single-column output. (default: :obj:`0.5`)
    """
    scores = scores.detach()
    if scores.ndim == 2 and scores.size(1) > 1:
        return scores.argmax(dim=1)
    if scores.ndim == 2:
        scores = scores.squeeze(1)
    if scores.ndim != 1:
        raise ValueError(
            f"Expected scores of shape [B], [B, 1] or [B, C], but got "
            f"{list(scores.shape)}.")
    if not scores.is_floating_point():
        return scores.to(torch.long)
    probs = scores.sigmoid() if from_logits else scores
    return (probs >= threshold).to(torch.long)


def _average_ranks(scores: Tensor) -> Tensor:
    r"""Ranks in :obj:`[1, n]`, with tied scores sharing their average rank.

    This is the tie handling the Mann-Whitney form of AUROC requires; without
    it, a model that emits a constant score would not score 0.5.
    """
    num = scores.numel()
    sorted_scores, order = torch.sort(scores)
    _, inverse, counts = torch.unique(sorted_scores, return_inverse=True,
                                      return_counts=True)
    end = torch.cumsum(counts, dim=0)
    start = end - counts
    # Mean of the 1-based rank span [start + 1, end] covered by each tie group.
    group_rank = (start + 1 + end).to(torch.float64) / 2.0
    ranks = torch.empty(num, dtype=torch.float64)
    ranks[order] = group_rank[inverse]
    return ranks


def binary_auroc(scores: Tensor, targets: Tensor) -> float:
    r"""Area under the ROC curve for binary targets.

    Computed through the Mann-Whitney U statistic, so :obj:`scores` only needs
    to be monotonically related to the positive-class probability.

    Args:
        scores (Tensor): 1-D positive-class scores of shape :obj:`[B]`.
        targets (Tensor): Binary targets of shape :obj:`[B]`.
    """
    scores = _flatten(scores)
    targets = _flatten(targets)
    positive = targets == 1
    num_pos = int(positive.sum().item())
    num_neg = int(targets.numel() - num_pos)
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    rank_sum = ranks[positive].sum().item()
    return (rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)


def binary_average_precision(scores: Tensor, targets: Tensor) -> float:
    r"""Average precision (area under the precision-recall curve).

    Uses the step-wise sum :math:`\sum_n (R_n - R_{n-1}) P_n` over distinct
    score thresholds, which is the same estimator as
    :func:`sklearn.metrics.average_precision_score` and does not apply the
    optimistic linear interpolation of a trapezoidal PR-AUC.

    Args:
        scores (Tensor): 1-D positive-class scores of shape :obj:`[B]`.
        targets (Tensor): Binary targets of shape :obj:`[B]`.
    """
    scores = _flatten(scores)
    targets = _flatten(targets)
    num_pos = float((targets == 1).sum().item())
    if num_pos == 0 or num_pos == targets.numel():
        return float("nan")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = (targets[order] == 1).to(torch.float64)

    true_pos = torch.cumsum(sorted_targets, dim=0)
    false_pos = torch.cumsum(1.0 - sorted_targets, dim=0)

    # Collapse runs of equal scores: a threshold cannot split a tie group.
    boundary = torch.nonzero(sorted_scores[1:] != sorted_scores[:-1]).flatten()
    last = torch.tensor([sorted_scores.numel() - 1])
    idx = torch.cat([boundary, last])

    true_pos = true_pos[idx]
    false_pos = false_pos[idx]
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / num_pos
    prev_recall = torch.cat([torch.zeros(1, dtype=recall.dtype), recall[:-1]])
    return float(((recall - prev_recall) * precision).sum().item())


def _macro_one_vs_rest(probs: Tensor, targets: Tensor,
                       binary_fn) -> tuple[float, list[float]]:
    probs = probs.detach().to(torch.float64)
    targets = targets.detach().reshape(-1).to(torch.long)
    per_class: list[float] = []
    for class_idx in range(probs.size(1)):
        value = binary_fn(probs[:, class_idx],
                          (targets == class_idx).to(torch.float64))
        per_class.append(value)
    defined = [v for v in per_class if v == v]  # drop NaNs
    if not defined:
        return float("nan"), per_class
    return sum(defined) / len(defined), per_class


def multiclass_auroc(probs: Tensor, targets: Tensor) -> float:
    r"""Macro-averaged one-vs-rest AUROC.

    Classes that are absent from :obj:`targets` are undefined one-vs-rest and
    are skipped rather than counted as zero.

    Args:
        probs (Tensor): Class probabilities of shape :obj:`[B, C]`.
        targets (Tensor): Class indices of shape :obj:`[B]`.
    """
    return _macro_one_vs_rest(probs, targets, binary_auroc)[0]


def multiclass_average_precision(probs: Tensor, targets: Tensor) -> float:
    r"""Macro-averaged one-vs-rest average precision.

    Args:
        probs (Tensor): Class probabilities of shape :obj:`[B, C]`.
        targets (Tensor): Class indices of shape :obj:`[B]`.
    """
    return _macro_one_vs_rest(probs, targets, binary_average_precision)[0]


def brier_score(probs: Tensor, targets: Tensor) -> float:
    r"""Brier score (lower is better).

    For a 1-D :obj:`probs` this is the standard binary Brier score
    :math:`\frac{1}{N}\sum (p_i - y_i)^2`. For a :obj:`[B, C]` :obj:`probs`
    this is the multiclass sum-of-squares form
    :math:`\frac{1}{N}\sum_i \sum_c (p_{ic} - y_{ic})^2`, whose range is
    :obj:`[0, 2]`.

    Args:
        probs (Tensor): Probabilities of shape :obj:`[B]` or :obj:`[B, C]`.
        targets (Tensor): Binary targets or class indices of shape :obj:`[B]`.
    """
    probs = probs.detach().to(torch.float64)
    targets = targets.detach().reshape(-1)
    if probs.numel() == 0:
        return float("nan")

    if probs.ndim == 1:
        return float(((probs - targets.to(torch.float64))**2).mean().item())

    one_hot = torch.zeros_like(probs)
    one_hot[torch.arange(probs.size(0)), targets.to(torch.long)] = 1.0
    return float(((probs - one_hot)**2).sum(dim=1).mean().item())


def _confidence_and_correctness(probs: Tensor,
                                targets: Tensor) -> tuple[Tensor, Tensor]:
    probs = probs.detach().to(torch.float64)
    targets = targets.detach().reshape(-1)
    if probs.ndim == 1:
        predicted = (probs >= 0.5).to(torch.long)
        confidence = torch.where(predicted == 1, probs, 1.0 - probs)
    else:
        confidence, predicted = probs.max(dim=1)
    correct = (predicted == targets.to(torch.long)).to(torch.float64)
    return confidence, correct


def reliability_curve(probs: Tensor, targets: Tensor,
                      num_bins: int = 15) -> dict[str, list[float]]:
    r"""Bin-level accuracy and confidence, for plotting a reliability diagram.

    Bins are equal-width over :obj:`[0, 1]` in predicted confidence. Empty bins
    are omitted.

    Args:
        probs (Tensor): Probabilities of shape :obj:`[B]` or :obj:`[B, C]`.
        targets (Tensor): Binary targets or class indices of shape :obj:`[B]`.
        num_bins (int): Number of confidence bins. (default: :obj:`15`)

    Returns:
        dict: Keys :obj:`"bin_lower"`, :obj:`"bin_upper"`,
        :obj:`"confidence"`, :obj:`"accuracy"` and :obj:`"count"`.
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive (got {num_bins})")

    confidence, correct = _confidence_and_correctness(probs, targets)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)

    curve: dict[str, list[float]] = {
        "bin_lower": [],
        "bin_upper": [],
        "confidence": [],
        "accuracy": [],
        "count": [],
    }
    for i in range(num_bins):
        lower, upper = edges[i], edges[i + 1]
        # Lower-exclusive bins, with the first bin closed on the left so that
        # a confidence of exactly 0 is not dropped.
        in_bin = (confidence > lower) & (confidence <= upper)
        if i == 0:
            in_bin |= confidence == lower
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        curve["bin_lower"].append(float(lower.item()))
        curve["bin_upper"].append(float(upper.item()))
        curve["confidence"].append(float(confidence[in_bin].mean().item()))
        curve["accuracy"].append(float(correct[in_bin].mean().item()))
        curve["count"].append(float(count))
    return curve


def expected_calibration_error(probs: Tensor, targets: Tensor,
                               num_bins: int = 15) -> float:
    r"""Expected calibration error (lower is better).

    The count-weighted mean absolute gap between confidence and accuracy over
    equal-width confidence bins.

    Args:
        probs (Tensor): Probabilities of shape :obj:`[B]` or :obj:`[B, C]`.
        targets (Tensor): Binary targets or class indices of shape :obj:`[B]`.
        num_bins (int): Number of confidence bins. (default: :obj:`15`)
    """
    curve = reliability_curve(probs, targets, num_bins=num_bins)
    total = sum(curve["count"])
    if total == 0:
        return float("nan")
    gap = sum(
        count * abs(acc - conf) for count, acc, conf in zip(
            curve["count"], curve["accuracy"], curve["confidence"]))
    return gap / total


def maximum_calibration_error(probs: Tensor, targets: Tensor,
                              num_bins: int = 15) -> float:
    r"""Largest confidence-accuracy gap over any non-empty bin."""
    curve = reliability_curve(probs, targets, num_bins=num_bins)
    if not curve["count"]:
        return float("nan")
    return max(
        abs(acc - conf)
        for acc, conf in zip(curve["accuracy"], curve["confidence"]))


def r2_score(predictions: Tensor, targets: Tensor) -> float:
    r"""Coefficient of determination.

    Returns :obj:`nan` when the targets have zero variance, since
    :math:`R^2` is undefined there.
    """
    predictions = _flatten(predictions)
    targets = _flatten(targets)
    if targets.numel() == 0:
        return float("nan")
    ss_tot = ((targets - targets.mean())**2).sum()
    if ss_tot == 0:
        return float("nan")
    ss_res = ((targets - predictions)**2).sum()
    return float((1.0 - ss_res / ss_tot).item())
