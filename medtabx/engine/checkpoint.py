"""
Checkpoint saving and loading for MedTabX training runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    r"""Write a training checkpoint.

    Args:
        path (str or Path): Destination file. Parent directories are created.
        model (Module): The model whose :obj:`state_dict` is stored.
        optimizer (Optimizer, optional): Optimizer state, needed to resume a
            run exactly. (default: :obj:`None`)
        scheduler (optional): Scheduler state. (default: :obj:`None`)
        epoch (int, optional): The epoch this checkpoint was taken at.
            (default: :obj:`None`)
        metrics (dict, optional): Validation metrics at this epoch, so a saved
            model carries the score it was selected on. (default: :obj:`None`)
        config (dict, optional): Serialized run configuration.
            (default: :obj:`None`)
        extra (dict, optional): Any further payload. (default: :obj:`None`)

    Returns:
        Path: The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: Any = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    r"""Read a checkpoint written by :func:`save_checkpoint`.

    Args:
        path (str or Path): Checkpoint file.
        model (Module, optional): If given, its weights are loaded in place.
            (default: :obj:`None`)
        optimizer (Optimizer, optional): If given, its state is restored.
            (default: :obj:`None`)
        scheduler (optional): If given, its state is restored.
            (default: :obj:`None`)
        map_location: Passed to :func:`torch.load`. (default: :obj:`"cpu"`)
        strict (bool): Passed to :meth:`load_state_dict`.
            (default: :obj:`True`)

    Returns:
        dict: The full checkpoint payload.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at '{path}'.")

    # The payload holds run metadata alongside tensors, so it cannot be read
    # under the weights_only default that torch >= 2.6 applies.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload
