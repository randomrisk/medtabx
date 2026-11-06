"""Tests for the MedTabX training engine."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from medtabx import stype
from medtabx.data import Dataset
from medtabx.engine import (
    EarlyStopping,
    Trainer,
    TrainerConfig,
    load_checkpoint,
    save_checkpoint,
)
from medtabx.models import MLP
from medtabx.typing import TaskType
from medtabx.utils.split import generate_random_split


def _classification_dataset(num_rows: int = 300, seed: int = 0) -> Dataset:
    """A separable binary cohort, so a working loop must reach a high AUROC."""
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 10, num_rows)
    severity = rng.normal(8, 3, num_rows)
    logit = 0.08 * (age - 65) + 0.45 * (severity - 8)
    label = (rng.random(num_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "age": age,
        "severity": severity,
        "outcome": pd.Categorical(label),
    })
    df["split"] = generate_random_split(num_rows, seed=seed, train_ratio=0.6,
                                        val_ratio=0.2)
    dataset = Dataset(df, col_to_stype={
        "age": stype.numerical,
        "severity": stype.numerical,
        "outcome": stype.categorical,
    }, target_col="outcome", split_col="split")
    return dataset.materialize()


def _regression_dataset(num_rows: int = 300, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, num_rows)
    x2 = rng.normal(0, 1, num_rows)
    target = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.2, num_rows)

    df = pd.DataFrame({"x1": x1, "x2": x2, "ttp": target})
    df["split"] = generate_random_split(num_rows, seed=seed, train_ratio=0.6,
                                        val_ratio=0.2)
    dataset = Dataset(df, col_to_stype={
        "x1": stype.numerical,
        "x2": stype.numerical,
        "ttp": stype.numerical,
    }, target_col="ttp", split_col="split")
    return dataset.materialize()


def _build_model(dataset: Dataset, out_channels: int) -> MLP:
    return MLP(channels=32, out_channels=out_channels, num_layers=2,
               col_stats=dataset.col_stats,
               col_names_dict=dataset.tensor_frame.col_names_dict)


def test_trainer_reduces_loss_and_learns_a_separable_signal():
    dataset = _classification_dataset()
    train_set, val_set, test_set = dataset.split()
    model = _build_model(dataset, dataset.num_classes)

    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=30, lr=1e-2, seed=0,
                                           verbose=False))
    history = trainer.fit(train_set, val_set)

    assert len(history) == 30
    first_loss = history[0]["train_loss"]
    last_loss = history[-1]["train_loss"]
    assert last_loss < first_loss

    results = trainer.evaluate(test_set)
    assert results["auroc"] > 0.7
    assert 0.0 <= results["accuracy"] <= 1.0


def test_trainer_supports_regression():
    dataset = _regression_dataset()
    train_set, val_set, test_set = dataset.split()
    model = _build_model(dataset, out_channels=1)

    trainer = Trainer(model, task_type=TaskType.REGRESSION,
                      config=TrainerConfig(epochs=40, lr=1e-2, monitor="rmse",
                                           seed=0, verbose=False))
    trainer.fit(train_set, val_set)

    results = trainer.evaluate(test_set)
    assert set(results) == {"loss", "mae", "rmse", "r2"}
    assert results["r2"] > 0.8


def test_trainer_selects_and_restores_the_best_epoch():
    dataset = _classification_dataset()
    train_set, val_set = dataset.get_split("train"), dataset.get_split("val")
    model = _build_model(dataset, dataset.num_classes)

    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=15, lr=1e-2,
                                           monitor="auroc", seed=0,
                                           verbose=False))
    history = trainer.fit(train_set, val_set)

    best_epoch, best_value = history.best("val_auroc")
    assert trainer.best_epoch == best_epoch
    assert trainer.best_value == pytest.approx(best_value)

    # restore_best defaults to True, so the model in hand must be the one
    # that produced the best validation score, not the last epoch's.
    restored = trainer.evaluate(val_set)
    assert restored["auroc"] == pytest.approx(best_value)


def test_trainer_early_stops_when_the_monitor_plateaus():
    dataset = _classification_dataset()
    train_set, val_set = dataset.get_split("train"), dataset.get_split("val")
    model = _build_model(dataset, dataset.num_classes)

    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=200, lr=1e-2,
                                           monitor="auroc",
                                           early_stopping_patience=3, seed=0,
                                           verbose=False))
    history = trainer.fit(train_set, val_set)
    assert len(history) < 200


def test_trainer_rejects_an_unknown_monitor():
    dataset = _classification_dataset()
    train_set, val_set = dataset.get_split("train"), dataset.get_split("val")
    model = _build_model(dataset, dataset.num_classes)

    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=1, monitor="not_a_metric",
                                           verbose=False))
    with pytest.raises(KeyError, match="not_a_metric"):
        trainer.fit(train_set, val_set)


def test_trainer_predict_returns_aligned_outputs_and_targets():
    dataset = _classification_dataset()
    test_set = dataset.get_split("test")
    model = _build_model(dataset, dataset.num_classes)

    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=1, verbose=False))
    outputs, targets = trainer.predict(test_set)

    assert outputs.shape == (len(test_set), dataset.num_classes)
    assert targets.shape == (len(test_set), )


def test_trainer_accepts_dataset_tensor_frame_and_loader():
    from medtabx.data.loader import DataLoader

    dataset = _classification_dataset()
    test_set = dataset.get_split("test")
    model = _build_model(dataset, dataset.num_classes)
    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=1, verbose=False))

    from_dataset = trainer.evaluate(test_set)
    from_frame = trainer.evaluate(test_set.tensor_frame)
    from_loader = trainer.evaluate(
        DataLoader(test_set.tensor_frame, batch_size=16))

    assert from_dataset["loss"] == pytest.approx(from_frame["loss"])
    assert from_dataset["loss"] == pytest.approx(from_loader["loss"])


def test_trainer_writes_a_checkpoint_that_round_trips(tmp_path):
    dataset = _classification_dataset()
    train_set, val_set = dataset.get_split("train"), dataset.get_split("val")
    checkpoint = tmp_path / "best.pt"

    model = _build_model(dataset, dataset.num_classes)
    trainer = Trainer(model, task_type=dataset.task_type,
                      config=TrainerConfig(epochs=5, lr=1e-2,
                                           checkpoint_path=str(checkpoint),
                                           seed=0, verbose=False))
    trainer.fit(train_set, val_set)
    assert checkpoint.exists()

    before = trainer.evaluate(val_set)

    fresh_model = _build_model(dataset, dataset.num_classes)
    fresh_trainer = Trainer(fresh_model, task_type=dataset.task_type,
                            config=TrainerConfig(epochs=1, verbose=False))
    fresh_trainer.load_checkpoint(checkpoint)
    after = fresh_trainer.evaluate(val_set)

    assert before["loss"] == pytest.approx(after["loss"])
    assert before["auroc"] == pytest.approx(after["auroc"])


def test_checkpoint_carries_run_metadata(tmp_path):
    dataset = _classification_dataset()
    model = _build_model(dataset, dataset.num_classes)
    path = tmp_path / "ckpt.pt"

    save_checkpoint(path, model, epoch=7, metrics={"val_auroc": 0.83})
    payload = load_checkpoint(path)

    assert payload["epoch"] == 7
    assert payload["metrics"]["val_auroc"] == pytest.approx(0.83)


def test_trainer_raises_without_a_target_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=32)})
    dataset = Dataset(df, col_to_stype={"x": stype.numerical}).materialize()

    model = MLP(channels=8, out_channels=2, num_layers=1,
                col_stats=dataset.col_stats,
                col_names_dict=dataset.tensor_frame.col_names_dict)
    trainer = Trainer(model, task_type=TaskType.BINARY_CLASSIFICATION,
                      config=TrainerConfig(epochs=1, verbose=False))

    with pytest.raises(ValueError, match="target_col"):
        trainer.fit(dataset)


def test_early_stopping_counts_consecutive_non_improvements():
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) is True
    assert stopper.step(0.6) is True
    assert stopper.step(0.55) is False
    assert stopper.should_stop is False
    stopper.step(0.54)
    assert stopper.should_stop is False
    stopper.step(0.53)
    assert stopper.should_stop is True
    assert stopper.best_value == pytest.approx(0.6)
    assert stopper.best_step == 1


def test_early_stopping_treats_nan_as_no_improvement():
    stopper = EarlyStopping(patience=5, mode="min")
    assert stopper.step(1.0) is True
    assert stopper.step(float("nan")) is False
    assert stopper.best_value == pytest.approx(1.0)


def test_train_history_reports_best_epoch_and_series():
    from medtabx.engine.trainer import TrainHistory

    history = TrainHistory()
    history.append({"epoch": 1, "val_auroc": 0.70, "val_loss": 0.6})
    history.append({"epoch": 2, "val_auroc": 0.82, "val_loss": 0.5})
    history.append({"epoch": 3, "val_auroc": float("nan"), "val_loss": 0.7})

    assert history.best("val_auroc") == (2, 0.82)
    assert history.best("val_loss")[0] == 2
    assert history.values_of("val_loss") == [0.6, 0.5, 0.7]
    assert "val_auroc" in history.keys()
