"""
Bootstrap confidence intervals for MedTabX metrics.

A single point estimate on a medical test split of a few hundred subjects
carries substantial sampling error, and reporting it alone invites
over-reading small differences between models. Every metric in an evaluation
report should therefore be accompanied by an interval.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

__all__ = ["BootstrapResult", "bootstrap_ci", "bootstrap_metric"]


@dataclass
class BootstrapResult:
    """Outcome of a bootstrap resampling run.

    Attributes:
        point: The metric computed once on the full sample.
        lower: Lower percentile bound.
        upper: Upper percentile bound.
        std: Standard deviation across resamples (the bootstrap standard
            error).
        confidence_level: The nominal coverage, e.g. :obj:`0.95`.
        num_resamples: Number of resamples that produced a defined value.
        num_skipped: Number of resamples discarded because the metric was
            undefined on them, typically a resample that drew a single class.
        values: The per-resample metric values that were kept.
    """

    point: float
    lower: float
    upper: float
    std: float
    confidence_level: float
    num_resamples: int
    num_skipped: int
    values: list[float] = field(default_factory=list, repr=False)

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten into a metrics dictionary, e.g. for a results table."""
        return {
            f"{prefix}point": self.point,
            f"{prefix}lower": self.lower,
            f"{prefix}upper": self.upper,
            f"{prefix}std": self.std,
        }

    def __str__(self) -> str:
        pct = int(round(self.confidence_level * 100))
        return (f"{self.point:.4f} "
                f"[{pct}% CI {self.lower:.4f}-{self.upper:.4f}]")


def bootstrap_ci(
    metric_fn: Callable[[Tensor, Tensor], float],
    scores: Tensor,
    targets: Tensor,
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int | None = None,
    stratified: bool = True,
) -> BootstrapResult:
    r"""Percentile bootstrap confidence interval for a metric.

    Args:
        metric_fn (callable): Takes :obj:`(scores, targets)` and returns a
            float. Any of the functions in :mod:`medtabx.metrics.functional`
            works directly.
        scores (Tensor): Predictions, indexed along the first dimension.
        targets (Tensor): Targets of shape :obj:`[B]`.
        num_resamples (int): Number of bootstrap resamples.
            (default: :obj:`1000`)
        confidence_level (float): Nominal coverage. (default: :obj:`0.95`)
        seed (int, optional): Seed for the resampling generator, for
            reproducible intervals. (default: :obj:`None`)
        stratified (bool): Resample within each target class so that every
            resample preserves the observed class prevalence. This keeps
            ranking metrics defined on imbalanced splits, and is the usual
            choice for classification. Ignored for float targets.
            (default: :obj:`True`)

    Returns:
        BootstrapResult: The point estimate and its interval.
    """
    if num_resamples <= 0:
        raise ValueError(
            f"num_resamples must be positive (got {num_resamples})")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must lie in (0, 1) (got {confidence_level})")

    targets = targets.detach().reshape(-1)
    num_samples = targets.numel()
    if scores.shape[0] != num_samples:
        raise ValueError(
            f"scores and targets must agree on the batch dimension, but got "
            f"{scores.shape[0]} and {num_samples}.")
    if num_samples == 0:
        nan = float("nan")
        return BootstrapResult(nan, nan, nan, nan, confidence_level, 0, 0)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Stratification only makes sense for a small set of discrete labels.
    strata: list[Tensor] | None = None
    if stratified and not targets.is_floating_point():
        strata = [
            torch.nonzero(targets == value).flatten()
            for value in torch.unique(targets)
        ]

    values: list[float] = []
    num_skipped = 0
    for _ in range(num_resamples):
        if strata is not None:
            index = torch.cat([
                group[torch.randint(len(group), (len(group), ),
                                    generator=generator)] for group in strata
            ])
        else:
            index = torch.randint(num_samples, (num_samples, ),
                                  generator=generator)

        value = metric_fn(scores[index], targets[index])
        if value != value:  # NaN: undefined on this resample
            num_skipped += 1
            continue
        values.append(float(value))

    point = float(metric_fn(scores, targets))
    if not values:
        warnings.warn(
            "Every bootstrap resample produced an undefined metric value; "
            "returning a NaN interval. This usually means the split is too "
            "small or contains a single class.", stacklevel=2)
        nan = float("nan")
        return BootstrapResult(point, nan, nan, nan, confidence_level, 0,
                               num_skipped)
    if num_skipped:
        warnings.warn(
            f"{num_skipped} of {num_resamples} bootstrap resamples produced "
            f"an undefined metric value and were discarded; the interval is "
            f"based on the remaining {len(values)}.", stacklevel=2)

    sampled = torch.tensor(values, dtype=torch.float64)
    tail = (1.0 - confidence_level) / 2.0
    quantiles = torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    lower, upper = torch.quantile(sampled, quantiles).tolist()
    std = float(sampled.std(unbiased=True).item()) if len(values) > 1 else 0.0

    return BootstrapResult(
        point=point,
        lower=float(lower),
        upper=float(upper),
        std=std,
        confidence_level=confidence_level,
        num_resamples=len(values),
        num_skipped=num_skipped,
        values=values,
    )


def bootstrap_metric(
    metric_fns: dict[str, Callable[[Tensor, Tensor], float]],
    scores: Tensor,
    targets: Tensor,
    **kwargs,
) -> dict[str, BootstrapResult]:
    r"""Run :func:`bootstrap_ci` for several metrics over the same predictions.

    Args:
        metric_fns (dict): Maps a metric name to a
            :obj:`(scores, targets) -> float` callable.
        scores (Tensor): Predictions, indexed along the first dimension.
        targets (Tensor): Targets of shape :obj:`[B]`.
        **kwargs: Forwarded to :func:`bootstrap_ci`.
    """
    return {
        name: bootstrap_ci(fn, scores, targets, **kwargs)
        for name, fn in metric_fns.items()
    }
