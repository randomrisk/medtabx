"""
Training loop for MedTabX models.

The trainer consumes anything that yields :class:`TensorFrame` batches -- a
:class:`~medtabx.data.Dataset`, a materialized :class:`TensorFrame`, or a
ready-made :class:`~medtabx.data.DataLoader` -- and drives any model whose
``forward`` maps a :class:`TensorFrame` to a ``[batch_size, out_channels]``
tensor. That covers both the tabular models and the multimodal baseline.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

from medtabx.data import Dataset, TensorFrame
from medtabx.data.loader import DataLoader
from medtabx.metrics import MetricCollection, default_metrics, metric_mode
from medtabx.typing import TaskType

from .checkpoint import load_checkpoint, save_checkpoint
from .early_stopping import EarlyStopping

__all__ = ["Trainer", "TrainerConfig", "TrainHistory"]

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


@dataclass
class TrainerConfig:
    r"""Configuration for :class:`Trainer`.

    Attributes:
        epochs: Maximum number of passes over the training split.
        batch_size: Mini-batch size used when the trainer has to build its own
            loader. Ignored if a :class:`DataLoader` is passed in directly.
        lr: Learning rate.
        weight_decay: Weight decay passed to the optimizer.
        optimizer: One of :obj:`"adamw"`, :obj:`"adam"`, :obj:`"sgd"`,
            :obj:`"rmsprop"`.
        scheduler: One of :obj:`None`, :obj:`"cosine"`, :obj:`"plateau"` or
            :obj:`"step"`.
        scheduler_kwargs: Extra keyword arguments for the scheduler.
        grad_clip: If set, clip the gradient norm to this value.
        device: Torch device string. Defaults to CUDA when available.
        seed: Seed applied to torch before training, for reproducible runs.
        num_workers: Dataloader worker processes.
        monitor: Validation key used for model selection and early stopping.
            Defaults to :obj:`"loss"`, which is always defined; set it to
            :obj:`"auroc"` or another panel metric to select on that instead.
        monitor_mode: :obj:`"min"`, :obj:`"max"` or :obj:`"auto"`. Under
            :obj:`"auto"` the direction is inferred from the metric name.
        early_stopping_patience: Epochs without improvement to tolerate.
            :obj:`None` disables early stopping.
        early_stopping_min_delta: Smallest change counted as an improvement.
        checkpoint_path: Where to write the best checkpoint. :obj:`None`
            keeps the best weights in memory only.
        restore_best: Reload the best weights when :meth:`fit` returns.
        verbose: Print a line per epoch.
        log_every: Print every n-th epoch when verbose.
    """

    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str | None = None
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    grad_clip: float | None = None
    device: str | None = None
    seed: int | None = None
    num_workers: int = 0
    monitor: str = "loss"
    monitor_mode: str = "auto"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    checkpoint_path: str | None = None
    restore_best: bool = True
    verbose: bool = True
    log_every: int = 1

    def resolved_monitor_mode(self) -> str:
        """The comparison direction for :obj:`monitor`."""
        if self.monitor_mode != "auto":
            return self.monitor_mode
        return metric_mode(self.monitor)


@dataclass
class TrainHistory:
    r"""Per-epoch records produced by :meth:`Trainer.fit`.

    Each record is a flat dict with a :obj:`"epoch"` key and metrics prefixed
    by split, e.g. :obj:`"train_loss"` and :obj:`"val_auroc"`.
    """

    records: list[dict[str, float]] = field(default_factory=list)

    def append(self, record: dict[str, float]) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, float]:
        return self.records[index]

    def __iter__(self):
        return iter(self.records)

    def keys(self) -> list[str]:
        """Union of the keys across all records, in first-seen order."""
        seen: dict[str, None] = {}
        for record in self.records:
            for key in record:
                seen.setdefault(key, None)
        return list(seen)

    def values_of(self, key: str) -> list[float]:
        """The per-epoch series for one key, skipping epochs that lack it."""
        return [r[key] for r in self.records if key in r]

    def best(self, key: str,
             mode: str = "auto") -> tuple[int, float] | tuple[None, None]:
        r"""The epoch and value where :obj:`key` was best.

        Args:
            key (str): The record key to search.
            mode (str): :obj:`"min"`, :obj:`"max"` or :obj:`"auto"`.
                (default: :obj:`"auto"`)
        """
        if mode == "auto":
            mode = metric_mode(key)
        best_epoch: int | None = None
        best_value: float | None = None
        for record in self.records:
            value = record.get(key)
            if value is None or value != value:  # missing or NaN
                continue
            if (best_value is None or (mode == "max" and value > best_value)
                    or (mode == "min" and value < best_value)):
                best_value = value
                best_epoch = int(record["epoch"])
        if best_value is None:
            return None, None
        return best_epoch, best_value

    def to_dataframe(self):
        """Return the history as a :class:`pandas.DataFrame`."""
        import pandas as pd
        return pd.DataFrame(self.records)

    def to_csv(self, path: str | Path) -> Path:
        """Write the history to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path


class Trainer:
    r"""Supervised training loop over :class:`TensorFrame` batches.

    .. code-block:: python

        from medtabx.engine import Trainer, TrainerConfig

        trainer = Trainer(
            model,
            task_type=dataset.task_type,
            config=TrainerConfig(epochs=50, monitor="auroc",
                                 early_stopping_patience=10),
        )
        history = trainer.fit(train_dataset, val_dataset)
        print(trainer.evaluate(test_dataset))

    Args:
        model (Module): Maps a :class:`TensorFrame` to
            :obj:`[batch_size, out_channels]`.
        task_type (TaskType or str): Determines the default loss and metric
            panel.
        config (TrainerConfig, optional): Training configuration. A default
            configuration is used when omitted. (default: :obj:`None`)
        loss_fn (callable, optional): Overrides the task default. Called as
            :obj:`loss_fn(output, target)`. (default: :obj:`None`)
        optimizer (Optimizer, optional): A pre-built optimizer. When omitted
            one is built from the config. (default: :obj:`None`)
        metrics (MetricCollection, optional): Overrides the default panel for
            the task. (default: :obj:`None`)
        num_classes (int, optional): Required for multiclass tasks when
            :obj:`metrics` is not given. (default: :obj:`None`)
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: TaskType | str,
        config: TrainerConfig | None = None,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        metrics: MetricCollection | None = None,
        num_classes: int | None = None,
    ) -> None:
        self.config = config or TrainerConfig()
        self.task_type = (TaskType(task_type)
                          if isinstance(task_type, str) else task_type)
        self.num_classes = num_classes

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        self.device = torch.device(self.config.device if self.config.device
                                   else ("cuda" if torch.cuda.is_available()
                                         else "cpu"))
        self.model = model.to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer or self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._metrics_template = metrics
        self.history = TrainHistory()
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self._best_state: dict[str, Tensor] | None = None

    # Construction helpers ####################################################

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.config.optimizer.lower()
        if name not in _OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer '{self.config.optimizer}'. Available: "
                f"{sorted(_OPTIMIZERS)}.")
        return _OPTIMIZERS[name](self.model.parameters(), lr=self.config.lr,
                                 weight_decay=self.config.weight_decay)

    def _build_scheduler(self):
        name = self.config.scheduler
        if name is None:
            return None
        kwargs = dict(self.config.scheduler_kwargs)
        if name == "cosine":
            kwargs.setdefault("T_max", self.config.epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, **kwargs)
        if name == "plateau":
            kwargs.setdefault("mode", self.config.resolved_monitor_mode())
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **kwargs)
        if name == "step":
            kwargs.setdefault("step_size", max(1, self.config.epochs // 3))
            return torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        raise ValueError(
            f"Unknown scheduler '{name}'. Available: 'cosine', 'plateau', "
            f"'step'.")

    def _build_metrics(self) -> MetricCollection:
        if self._metrics_template is not None:
            return copy.deepcopy(self._metrics_template)
        return default_metrics(self.task_type, num_classes=self.num_classes)

    def _as_loader(self, data: Dataset | TensorFrame | DataLoader,
                   shuffle: bool) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, Dataset):
            data = data.materialize().tensor_frame
        if not isinstance(data, TensorFrame):
            raise TypeError(
                f"Expected a Dataset, TensorFrame or DataLoader, but got "
                f"{type(data).__name__}.")
        return DataLoader(data, batch_size=self.config.batch_size,
                          shuffle=shuffle,
                          num_workers=self.config.num_workers)

    # Loss and targets ########################################################

    def _prepare_target(self, y: Tensor, output: Tensor) -> Tensor:
        r"""Cast and reshape the batch target to match the model output."""
        if y is None:
            raise ValueError(
                "The batch carries no target. Build the Dataset with a "
                "'target_col' so that TensorFrame.y is populated.")
        if self.task_type.is_regression:
            return y.reshape(output.shape).to(output.dtype)
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return y.reshape(output.shape).to(output.dtype)
        if (self.task_type == TaskType.BINARY_CLASSIFICATION
                and output.ndim == 2 and output.size(1) == 1):
            # Single-logit binary head trained with BCE.
            return y.reshape(output.shape).to(output.dtype)
        return y.reshape(-1).to(torch.long)

    def compute_loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Loss for one batch, using the task default unless overridden."""
        if self.loss_fn is not None:
            return self.loss_fn(output, target)

        if self.task_type.is_regression:
            return nn.functional.mse_loss(output, target)
        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            return nn.functional.binary_cross_entropy_with_logits(
                output, target)
        if output.ndim == 2 and output.size(1) == 1:
            return nn.functional.binary_cross_entropy_with_logits(
                output, target)
        return nn.functional.cross_entropy(output, target)

    # Core loops ##############################################################

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training pass and return the mean loss per sample."""
        self.model.train()
        total_loss = 0.0
        total = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(batch)
            target = self._prepare_target(batch.y, output)
            loss = self.compute_loss(output, target)
            loss.backward()
            if self.config.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config.grad_clip)
            self.optimizer.step()

            batch_size = output.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        return total_loss / total if total else float("nan")

    @torch.no_grad()
    def predict(
        self, data: Dataset | TensorFrame | DataLoader
    ) -> tuple[Tensor, Tensor]:
        r"""Run the model over a split without updating it.

        Returns:
            tuple: :obj:`(outputs, targets)`, both on the CPU. Outputs are the
            raw model outputs, i.e. logits for classification heads.
        """
        loader = self._as_loader(data, shuffle=False)
        self.model.eval()
        outputs: list[Tensor] = []
        targets: list[Tensor] = []

        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            outputs.append(output.detach().cpu())
            if batch.y is not None:
                targets.append(batch.y.detach().cpu().reshape(-1))

        if not outputs:
            return torch.empty(0), torch.empty(0)
        return (torch.cat(outputs, dim=0),
                torch.cat(targets, dim=0) if targets else torch.empty(0))

    @torch.no_grad()
    def evaluate(self, data: Dataset | TensorFrame | DataLoader,
                 metrics: MetricCollection | None = None) -> dict[str, float]:
        r"""Compute the loss and the metric panel over a split.

        Args:
            data: The split to evaluate.
            metrics (MetricCollection, optional): Overrides the trainer's
                panel for this call. (default: :obj:`None`)
        """
        loader = self._as_loader(data, shuffle=False)
        panel = metrics if metrics is not None else self._build_metrics()
        panel.reset()

        self.model.eval()
        total_loss = 0.0
        total = 0

        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            target = self._prepare_target(batch.y, output)
            loss = self.compute_loss(output, target)

            batch_size = output.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            # Metrics take the raw output and a flat target vector; the loss
            # target may have been reshaped to the output for BCE/MSE.
            panel.update(output.detach().cpu(),
                         batch.y.detach().cpu().reshape(-1))

        results = {"loss": total_loss / total if total else float("nan")}
        results.update(panel.compute())
        return results

    def fit(
        self,
        train_data: Dataset | TensorFrame | DataLoader,
        val_data: Dataset | TensorFrame | DataLoader | None = None,
    ) -> TrainHistory:
        r"""Train for :obj:`config.epochs`, with optional validation.

        Model selection, early stopping and checkpointing all key off
        :obj:`config.monitor` on the validation split. Without a validation
        split the training loss is monitored instead.

        Args:
            train_data: The training split.
            val_data (optional): The validation split. (default: :obj:`None`)

        Returns:
            TrainHistory: One record per epoch.
        """
        train_loader = self._as_loader(train_data, shuffle=True)
        val_loader = (self._as_loader(val_data, shuffle=False)
                      if val_data is not None else None)

        mode = self.config.resolved_monitor_mode()
        stopper = None
        if self.config.early_stopping_patience is not None:
            stopper = EarlyStopping(
                patience=self.config.early_stopping_patience, mode=mode,
                min_delta=self.config.early_stopping_min_delta)

        self.best_value = -math.inf if mode == "max" else math.inf
        self.best_epoch = None

        for epoch in range(1, self.config.epochs + 1):
            started = time.time()
            train_loss = self.train_epoch(train_loader)

            record: dict[str, float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                record.update(
                    {f"val_{k}": v for k, v in val_results.items()})
                monitored = val_results.get(self.config.monitor)
                if monitored is None:
                    raise KeyError(
                        f"monitor='{self.config.monitor}' is not produced by "
                        f"the metric panel. Available keys: "
                        f"{sorted(val_results)}.")
            else:
                monitored = train_loss

            record["time"] = time.time() - started
            self.history.append(record)
            self._step_scheduler(monitored)

            improved = self._track_best(monitored, mode, epoch, record)
            if self.config.verbose and epoch % self.config.log_every == 0:
                self._log_epoch(record, improved)

            if stopper is not None:
                stopper.step(monitored)
                if stopper.should_stop:
                    if self.config.verbose:
                        print(f"Early stopping at epoch {epoch}: "
                              f"{self.config.monitor} has not improved for "
                              f"{stopper.patience} epochs.")
                    break

        if self.config.restore_best and self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return self.history

    # Bookkeeping #############################################################

    def _step_scheduler(self, monitored: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            # A NaN monitored value would be counted as a non-improvement and
            # could trigger a spurious LR drop, so skip the step instead.
            if monitored == monitored:
                self.scheduler.step(monitored)
        else:
            self.scheduler.step()

    def _track_best(self, monitored: float, mode: str, epoch: int,
                    record: dict[str, float]) -> bool:
        if monitored != monitored:  # NaN
            return False
        improved = (monitored > self.best_value if mode == "max" else
                    monitored < self.best_value)
        if not improved:
            return False

        self.best_value = monitored
        self.best_epoch = epoch
        self._best_state = {
            k: v.detach().cpu().clone()
            for k, v in self.model.state_dict().items()
        }
        if self.config.checkpoint_path is not None:
            save_checkpoint(
                self.config.checkpoint_path,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=record,
                config=asdict(self.config),
            )
        return True

    def _log_epoch(self, record: dict[str, float], improved: bool) -> None:
        parts = [f"epoch {int(record['epoch']):>4d}"]
        parts.append(f"train_loss {record['train_loss']:.4f}")
        for key, value in record.items():
            if key.startswith("val_"):
                parts.append(f"{key} {value:.4f}")
        parts.append(f"{record['time']:.1f}s")
        if improved:
            parts.append("*")
        print("  ".join(parts))

    # Checkpointing ###########################################################

    def save_checkpoint(self, path: str | Path, **kwargs) -> Path:
        """Write the current model and optimizer state to :obj:`path`."""
        return save_checkpoint(path, self.model, optimizer=self.optimizer,
                               scheduler=self.scheduler,
                               epoch=self.best_epoch,
                               config=asdict(self.config), **kwargs)

    def load_checkpoint(self, path: str | Path,
                        strict: bool = True) -> dict[str, Any]:
        """Restore model and optimizer state from :obj:`path`."""
        return load_checkpoint(path, self.model, optimizer=self.optimizer,
                               scheduler=self.scheduler,
                               map_location=self.device, strict=strict)
