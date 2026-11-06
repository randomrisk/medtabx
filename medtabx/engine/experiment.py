"""
Config-driven experiment runner.

Wraps dataset construction, model construction, training and evaluation into
one reproducible unit so that a run is described by a file that can be
committed next to its results, rather than by a shell invocation that is not
recoverable afterwards.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import medtabx
from medtabx.core.stype import stype
from medtabx.data import Dataset
from medtabx.metrics import bootstrap_ci
from medtabx.metrics.functional import (
    binary_auroc,
    binary_average_precision,
    normalize_binary_scores,
    r2_score,
)
from medtabx.typing import TaskType
from medtabx.utils.split import generate_random_split

from .trainer import Trainer, TrainerConfig

__all__ = [
    "ExperimentConfig",
    "build_dataset",
    "build_model",
    "load_experiment",
    "run_experiment",
]

#: Filenames written into a run directory by :func:`run_experiment`.
CONFIG_FILENAME = "config.json"
COL_STATS_FILENAME = "col_stats.pt"
CHECKPOINT_FILENAME = "best.pt"
HISTORY_FILENAME = "history.csv"
METRICS_FILENAME = "metrics.json"

#: Models that :func:`build_model` can construct from a config alone. The
#: multimodal baseline needs per-stype encoder instances, so it has to be
#: assembled through the Python API instead.
TABULAR_MODELS = ("FTTransformer", "ResNet", "TabNet", "TabTransformer",
                  "Trompt", "ExcelFormer", "MLP")


@dataclass
class ExperimentConfig:
    r"""A complete description of one training run.

    Attributes:
        csv: Path to the feature table.
        target_col: Name of the target column.
        col_to_stype: Maps each modelled column to a semantic type name.
            Columns absent from this mapping are ignored.
        name: Run name, used for the default output directory.
        split_col: Column holding a precomputed split (0/1/2). When absent, a
            random split is generated from the ratios below.
        train_ratio: Training fraction for a generated split.
        val_ratio: Validation fraction for a generated split.
        split_seed: Seed for a generated split.
        model: Model class name, one of :obj:`TABULAR_MODELS`.
        model_kwargs: Extra constructor arguments for the model.
        trainer: Fields of :class:`TrainerConfig`.
        bootstrap_resamples: Resamples for the test-set confidence intervals.
            Set to 0 to skip.
        confidence_level: Nominal coverage of those intervals.
        bootstrap_seed: Seed for the bootstrap generator.
        output_dir: Where artifacts are written.
    """

    csv: str
    target_col: str
    col_to_stype: dict[str, str]
    name: str = "run"
    split_col: str | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    split_seed: int = 0
    model: str = "FTTransformer"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    bootstrap_seed: int = 0
    output_dir: str | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a JSON file."""
        payload = json.loads(Path(path).read_text())
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unknown keys in experiment config '{path}': "
                f"{sorted(unknown)}. Known keys: "
                f"{sorted(cls.__dataclass_fields__)}.")
        return cls(**payload)

    def to_json(self, path: str | Path) -> Path:
        """Write the resolved config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        return Path("runs") / self.name

    def trainer_config(self) -> TrainerConfig:
        """Build the :class:`TrainerConfig` for this run."""
        unknown = set(self.trainer) - set(TrainerConfig.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"Unknown trainer keys: {sorted(unknown)}. Known keys: "
                f"{sorted(TrainerConfig.__dataclass_fields__)}.")
        return TrainerConfig(**self.trainer)


def build_dataset(config: ExperimentConfig,
                  col_stats: dict[str, Any] | None = None,
                  require_target: bool = True) -> Dataset:
    r"""Construct and materialize the dataset described by :obj:`config`.

    When :obj:`config.split_col` is unset, a reproducible random split is
    generated and added to the frame so that the run is repeatable.

    Args:
        config (ExperimentConfig): The run description.
        col_stats (dict, optional): Column statistics from a previous
            materialization. Passing the training statistics is what keeps
            normalization and category indices identical when scoring a new
            cohort; recomputing them on the new frame would silently shift
            the model's inputs. (default: :obj:`None`)
        require_target (bool): When :obj:`False`, a missing target column is
            filled with a placeholder so that an unlabelled frame can still be
            materialized for inference. (default: :obj:`True`)
    """
    csv_path = Path(config.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No feature table at '{csv_path}'.")
    df = pd.read_csv(csv_path)

    if config.target_col not in config.col_to_stype:
        raise ValueError(
            f"target_col '{config.target_col}' must appear in col_to_stype so "
            f"that its semantic type determines the task.")

    missing = [c for c in config.col_to_stype if c not in df.columns]
    if missing == [config.target_col] and not require_target:
        df[config.target_col] = _placeholder_target(config, col_stats)
        missing = []
    if missing:
        raise ValueError(
            f"Columns declared in col_to_stype are absent from '{csv_path}': "
            f"{missing}.")

    col_to_stype = {
        col: stype(name)
        for col, name in config.col_to_stype.items()
    }

    split_col = config.split_col
    if split_col is None:
        split_col = "_medtabx_split"
        df[split_col] = generate_random_split(
            len(df), seed=config.split_seed, train_ratio=config.train_ratio,
            val_ratio=config.val_ratio)

    keep = list(config.col_to_stype) + [split_col]
    dataset = Dataset(df[keep], col_to_stype=col_to_stype,
                      target_col=config.target_col, split_col=split_col)
    return dataset.materialize(col_stats=col_stats)


def _placeholder_target(config: ExperimentConfig,
                        col_stats: dict[str, Any] | None):
    """A stand-in target column for scoring an unlabelled frame.

    Inference never reads ``TensorFrame.y``, but materialization requires the
    target column to exist and to hold a value the training statistics know.
    """
    if col_stats is None or config.target_col not in col_stats:
        raise ValueError(
            f"Target column '{config.target_col}' is absent from the input "
            f"table and no training col_stats were supplied to fill it.")

    from medtabx.data.stats import StatType

    target_stats = col_stats[config.target_col]
    if StatType.COUNT in target_stats:
        return target_stats[StatType.COUNT][0][0]
    if StatType.MEAN in target_stats:
        return target_stats[StatType.MEAN]
    raise ValueError(
        f"Cannot synthesize a placeholder for target column "
        f"'{config.target_col}' from the available statistics: "
        f"{list(target_stats)}.")


def build_model(name: str, dataset: Dataset, **kwargs) -> torch.nn.Module:
    r"""Construct a tabular model sized for :obj:`dataset`.

    :obj:`col_stats`, :obj:`col_names_dict` and :obj:`out_channels` are filled
    in from the dataset unless they are given explicitly.

    Args:
        name (str): Model class name, one of :obj:`TABULAR_MODELS`.
        dataset (Dataset): A materialized dataset with a target column.
        **kwargs: Forwarded to the model constructor.
    """
    if name not in TABULAR_MODELS:
        raise ValueError(
            f"'{name}' cannot be built from a config. Config-driven runs "
            f"support {list(TABULAR_MODELS)}; MultimodalBaseline needs "
            f"per-stype encoder instances and has to be assembled through "
            f"the Python API.")

    from medtabx.models import get_model_class

    task_type = dataset.task_type
    if task_type.is_regression:
        out_channels = 1
    else:
        out_channels = dataset.num_classes

    kwargs.setdefault("channels", 128)
    kwargs.setdefault("num_layers", 3)
    kwargs.setdefault("out_channels", out_channels)
    kwargs.setdefault("col_stats", dataset.col_stats)
    kwargs.setdefault("col_names_dict",
                      dataset.tensor_frame.col_names_dict)
    return get_model_class(name)(**kwargs)


def _test_set_intervals(config: ExperimentConfig, task_type: TaskType,
                        outputs: torch.Tensor,
                        targets: torch.Tensor) -> dict[str, dict[str, float]]:
    """Bootstrap confidence intervals for the main test metrics."""
    if config.bootstrap_resamples <= 0 or targets.numel() == 0:
        return {}

    if task_type == TaskType.BINARY_CLASSIFICATION:
        scores = normalize_binary_scores(outputs)
        metric_fns = {
            "auroc": binary_auroc,
            "auprc": binary_average_precision,
        }
    elif task_type.is_regression:
        scores = outputs.reshape(-1)
        metric_fns = {"r2": r2_score}
    else:
        # Macro one-vs-rest metrics over resampled multiclass splits are not
        # reliable on the cohort sizes this runner targets.
        return {}

    intervals: dict[str, dict[str, float]] = {}
    for metric_name, fn in metric_fns.items():
        result = bootstrap_ci(fn, scores, targets,
                              num_resamples=config.bootstrap_resamples,
                              confidence_level=config.confidence_level,
                              seed=config.bootstrap_seed)
        intervals[metric_name] = {
            "point": result.point,
            "lower": result.lower,
            "upper": result.upper,
            "std": result.std,
            "confidence_level": result.confidence_level,
            "num_resamples": result.num_resamples,
        }
    return intervals


def run_experiment(config: ExperimentConfig,
                   output_dir: str | Path | None = None) -> dict[str, Any]:
    r"""Train and evaluate one configuration, writing artifacts to disk.

    Writes :obj:`config.json`, :obj:`history.csv`, :obj:`metrics.json` and the
    best checkpoint into the output directory.

    Args:
        config (ExperimentConfig): The run to execute.
        output_dir (str or Path, optional): Overrides
            :obj:`config.output_dir`. (default: :obj:`None`)

    Returns:
        dict: The results payload that was written to :obj:`metrics.json`.
    """
    out_dir = Path(output_dir) if output_dir else config.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)
    task_type = dataset.task_type
    train_set, val_set, test_set = dataset.split()

    split_sizes = {
        "train": len(train_set),
        "val": len(val_set),
        "test": len(test_set),
    }
    for split_name, size in split_sizes.items():
        if size == 0:
            raise ValueError(
                f"The '{split_name}' split is empty. Adjust train_ratio / "
                f"val_ratio, or the split column, before training.")

    model = build_model(config.model, dataset, **config.model_kwargs)

    trainer_config = config.trainer_config()
    if trainer_config.checkpoint_path is None:
        trainer_config.checkpoint_path = str(out_dir / CHECKPOINT_FILENAME)

    num_classes = None if task_type.is_regression else dataset.num_classes
    trainer = Trainer(model, task_type=task_type, config=trainer_config,
                      num_classes=num_classes)

    history = trainer.fit(train_set, val_set)
    history.to_csv(out_dir / HISTORY_FILENAME)
    # Scoring a new cohort later has to reuse these, not recompute them.
    torch.save(dataset.col_stats, out_dir / COL_STATS_FILENAME)

    results: dict[str, Any] = {
        "name": config.name,
        "task_type": task_type.value,
        "model": config.model,
        "split_sizes": split_sizes,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": trainer.best_epoch,
        "best_val_monitor": trainer.best_value,
        "monitor": trainer_config.monitor,
        "epochs_run": len(history),
        "train": trainer.evaluate(train_set),
        "val": trainer.evaluate(val_set),
        "test": trainer.evaluate(test_set),
        "environment": {
            "medtabx": medtabx.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(trainer.device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    outputs, targets = trainer.predict(test_set)
    intervals = _test_set_intervals(config, task_type, outputs, targets)
    if intervals:
        results["test_intervals"] = intervals

    config.to_json(out_dir / CONFIG_FILENAME)
    (out_dir / METRICS_FILENAME).write_text(
        json.dumps(results, indent=2, default=str))

    if trainer_config.verbose:
        print(f"\nArtifacts written to {out_dir}")
    return results


def load_experiment(
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
    csv: str | Path | None = None,
    device: str | None = None,
    require_target: bool = True,
) -> tuple[ExperimentConfig, Dataset, Trainer]:
    r"""Rebuild a finished run so it can be evaluated or applied to new data.

    Args:
        run_dir (str or Path): A directory written by :func:`run_experiment`.
        checkpoint (str or Path, optional): Weights to load. Defaults to
            :obj:`best.pt` inside :obj:`run_dir`. (default: :obj:`None`)
        csv (str or Path, optional): Score a different feature table than the
            one the run was trained on. It must carry the same feature
            columns. (default: :obj:`None`)
        device (str, optional): Overrides the device recorded in the run's
            trainer config. (default: :obj:`None`)
        require_target (bool): See :func:`build_dataset`.
            (default: :obj:`True`)

    Returns:
        tuple: :obj:`(config, dataset, trainer)`, with the checkpoint weights
        already loaded into the trainer's model.
    """
    run_dir = Path(run_dir)
    config = ExperimentConfig.from_json(run_dir / CONFIG_FILENAME)

    col_stats_path = run_dir / COL_STATS_FILENAME
    if not col_stats_path.exists():
        raise FileNotFoundError(
            f"No column statistics at '{col_stats_path}'. A run directory "
            f"written by run_experiment always contains them; without them "
            f"the model would be fed differently normalized inputs.")
    col_stats = torch.load(col_stats_path, weights_only=False)

    if csv is not None:
        config.csv = str(csv)

    dataset = build_dataset(config, col_stats=col_stats,
                            require_target=require_target)
    model = build_model(config.model, dataset, **config.model_kwargs)

    trainer_config = config.trainer_config()
    if device is not None:
        trainer_config.device = device
    num_classes = (None if dataset.task_type.is_regression else
                   dataset.num_classes)
    trainer = Trainer(model, task_type=dataset.task_type,
                      config=trainer_config, num_classes=num_classes)

    checkpoint_path = (Path(checkpoint) if checkpoint is not None else
                       run_dir / CHECKPOINT_FILENAME)
    trainer.load_checkpoint(checkpoint_path)
    return config, dataset, trainer
