"""Tests for the streaming metric classes, collection, and bootstrap CIs."""

from __future__ import annotations

import math

import pytest
import torch

from medtabx.metrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    BalancedAccuracy,
    BrierScore,
    ExpectedCalibrationError,
    F1,
    MAE,
    R2,
    RMSE,
    MetricCollection,
    Specificity,
    bootstrap_ci,
    default_metrics,
    metric_mode,
)
from medtabx.metrics.functional import binary_auroc
from medtabx.typing import TaskType


def test_streaming_metrics_are_batch_order_invariant():
    torch.manual_seed(0)
    logits = torch.randn(64, 2)
    targets = torch.randint(0, 2, (64, ))

    whole = Accuracy()
    whole.update(logits, targets)

    batched = Accuracy()
    for start in range(0, 64, 7):
        batched.update(logits[start:start + 7], targets[start:start + 7])

    assert whole.compute() == pytest.approx(batched.compute())


def test_accumulating_metrics_are_batch_order_invariant():
    torch.manual_seed(0)
    logits = torch.randn(64, 2)
    targets = torch.randint(0, 2, (64, ))

    whole = AUROC()
    whole.update(logits, targets)

    batched = AUROC()
    for start in range(0, 64, 9):
        batched.update(logits[start:start + 9], targets[start:start + 9])

    assert whole.compute() == pytest.approx(batched.compute())


def test_reset_clears_accumulated_state():
    metric = AUROC()
    metric.update(torch.randn(10, 2), torch.randint(0, 2, (10, )))
    metric.reset()
    assert metric.is_empty
    assert math.isnan(metric.compute())


def test_threshold_metrics_handle_single_logit_head():
    # A [B, 1] head is a binary head; arg-maxing it would always predict 0.
    logits = torch.tensor([[3.0], [-3.0], [2.0], [-2.0]])
    targets = torch.tensor([1, 0, 1, 0])
    for metric in (Accuracy(), F1(), BalancedAccuracy(), Specificity()):
        metric.update(logits, targets)
        assert metric.compute() == pytest.approx(1.0), metric.name


def test_balanced_accuracy_penalizes_majority_predictor():
    # 90/10 imbalance, model always predicts the majority class.
    targets = torch.cat([torch.zeros(90), torch.ones(10)]).to(torch.long)
    logits = torch.zeros(100, 2)
    logits[:, 0] = 1.0

    accuracy, balanced = Accuracy(), BalancedAccuracy(num_classes=2)
    accuracy.update(logits, targets)
    balanced.update(logits, targets)

    assert accuracy.compute() == pytest.approx(0.9)
    assert balanced.compute() == pytest.approx(0.5)


def test_auprc_separates_models_that_auroc_rates_alike():
    metric = AveragePrecision()
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    metric.update(scores, torch.tensor([1, 0, 1, 0]))
    assert 0.0 < metric.compute() <= 1.0


def test_calibration_metrics_see_what_auroc_cannot():
    # Ranking is perfect but the probabilities are wildly overconfident, so
    # AUROC is 1.0 while the calibration metrics are poor.
    probs = torch.tensor([0.99, 0.99, 0.98, 0.01])
    targets = torch.tensor([1, 1, 0, 0])

    auroc = AUROC(from_logits=False)
    brier = BrierScore(from_logits=False)
    ece = ExpectedCalibrationError(from_logits=False)
    for metric in (auroc, brier, ece):
        metric.update(probs, targets)

    assert auroc.compute() == pytest.approx(1.0)
    assert brier.compute() > 0.2
    assert ece.compute() > 0.2


def test_regression_metrics_closed_form():
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 2.0, 5.0])

    mae, rmse, r2 = MAE(), RMSE(), R2()
    for metric in (mae, rmse, r2):
        metric.update(predictions, targets)

    assert mae.compute() == pytest.approx((1 + 0 + 2) / 3)
    assert rmse.compute() == pytest.approx(math.sqrt((1 + 0 + 4) / 3))
    assert r2.compute() == pytest.approx(1 - 5 / 6)


def test_metric_collection_groups_and_resets():
    collection = MetricCollection([Accuracy(), AUROC()])
    collection.update(torch.randn(20, 2), torch.randint(0, 2, (20, )))
    results = collection.compute()
    assert set(results) == {"accuracy", "auroc"}

    collection.reset()
    assert math.isnan(collection["auroc"].compute())


def test_metric_collection_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate metric name"):
        MetricCollection([Accuracy(), Accuracy()])


def test_default_metrics_cover_discrimination_and_calibration():
    binary = default_metrics(TaskType.BINARY_CLASSIFICATION)
    assert {"auroc", "auprc", "brier", "ece"} <= set(binary.metrics)

    regression = default_metrics(TaskType.REGRESSION)
    assert set(regression.metrics) == {"mae", "rmse", "r2"}

    with pytest.raises(ValueError, match="num_classes is required"):
        default_metrics(TaskType.MULTICLASS_CLASSIFICATION)


def test_metric_mode_picks_the_optimization_direction():
    assert metric_mode("auroc") == "max"
    assert metric_mode("val_auroc") == "max"
    assert metric_mode("loss") == "min"
    assert metric_mode("val_rmse") == "min"
    assert metric_mode("test_ece") == "min"


def test_bootstrap_ci_brackets_the_point_estimate():
    torch.manual_seed(0)
    targets = (torch.rand(400) < 0.3).to(torch.long)
    scores = targets.float() * 1.2 + torch.randn(400)

    result = bootstrap_ci(binary_auroc, scores, targets, num_resamples=300,
                          seed=0)
    assert result.lower <= result.point <= result.upper
    assert result.std > 0
    assert result.num_resamples == 300
    assert result.num_skipped == 0


def test_bootstrap_ci_is_reproducible_under_a_seed():
    torch.manual_seed(0)
    targets = (torch.rand(200) < 0.4).to(torch.long)
    scores = targets.float() + torch.randn(200)

    first = bootstrap_ci(binary_auroc, scores, targets, num_resamples=100,
                         seed=7)
    second = bootstrap_ci(binary_auroc, scores, targets, num_resamples=100,
                          seed=7)
    assert first.lower == pytest.approx(second.lower)
    assert first.upper == pytest.approx(second.upper)


def test_bootstrap_ci_degrades_to_nan_on_a_single_class_split():
    with pytest.warns(UserWarning, match="undefined metric value"):
        result = bootstrap_ci(binary_auroc, torch.rand(20),
                              torch.zeros(20, dtype=torch.long),
                              num_resamples=20, seed=0)
    assert math.isnan(result.point)
    assert math.isnan(result.lower)
    assert result.num_skipped == 20
