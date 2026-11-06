"""
Early stopping for MedTabX training runs.
"""

from __future__ import annotations

import math

__all__ = ["EarlyStopping"]


class EarlyStopping:
    r"""Track a monitored value and decide when training should stop.

    .. code-block:: python

        stopper = EarlyStopping(patience=10, mode="max")
        for epoch in range(epochs):
            score = validate()
            if stopper.step(score):
                save_best()
            if stopper.should_stop:
                break

    Args:
        patience (int): Number of consecutive non-improving epochs to tolerate
            before :obj:`should_stop` becomes :obj:`True`.
        mode (str): :obj:`"max"` if a larger monitored value is better,
            :obj:`"min"` if smaller is better. (default: :obj:`"max"`)
        min_delta (float): Minimum change that counts as an improvement. Guards
            against stopping being deferred by noise-level gains.
            (default: :obj:`0.0`)
    """

    def __init__(self, patience: int, mode: str = "max",
                 min_delta: float = 0.0):
        if patience < 0:
            raise ValueError(f"patience must be non-negative (got {patience})")
        if mode not in ("min", "max"):
            raise ValueError(
                f"mode must be 'min' or 'max' (got '{mode}')")
        if min_delta < 0:
            raise ValueError(
                f"min_delta must be non-negative (got {min_delta})")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.reset()

    def reset(self) -> None:
        """Clear all tracked state."""
        self.best_value = -math.inf if self.mode == "max" else math.inf
        self.best_step = -1
        self.num_bad_steps = 0
        self.should_stop = False
        self._step_count = 0

    def is_improvement(self, value: float) -> bool:
        """Whether :obj:`value` beats the best seen by more than min_delta."""
        if value != value:  # NaN never counts as an improvement
            return False
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def step(self, value: float) -> bool:
        r"""Record one monitored value.

        Args:
            value (float): The monitored value for this epoch. A :obj:`NaN` is
                treated as a non-improvement rather than an error, so an
                undefined metric on a degenerate validation split does not
                crash a run.

        Returns:
            bool: :obj:`True` if this value is a new best.
        """
        improved = self.is_improvement(value)
        if improved:
            self.best_value = value
            self.best_step = self._step_count
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1
            if self.num_bad_steps > self.patience:
                self.should_stop = True
        self._step_count += 1
        return improved

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(patience={self.patience}, "
                f"mode='{self.mode}', best_value={self.best_value:.6f}, "
                f"best_step={self.best_step})")
