"""Correctness tests for the functional metric implementations.

The reference values are computed by hand or from closed forms so that the
suite does not depend on scikit-learn, which is an optional extra.
"""

from __future__ import annotations

import math

import pytest
import torch

from medtabx.metrics.functional import (
    binary_auroc,
    binary_average_precision,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_auroc,
    normalize_binary_scores,
    normalize_multiclass_scores,
    predicted_labels,
    r2_score,
    reliability_curve,
)


def test_binary_auroc_perfect_and_inverted():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    assert binary_auroc(scores, targets) == pytest.approx(1.0)
    assert binary_auroc(-scores, targets) == pytest.approx(0.0)


def test_binary_auroc_handles_ties():
    # Every score identical: the model carries no information at all, which
    # only comes out as 0.5 if tied ranks are averaged.
    scores = torch.full((6, ), 0.42)
    targets = torch.tensor([0, 1, 0, 1, 0, 1])
    assert binary_auroc(scores, targets) == pytest.approx(0.5)

    # One tie group spanning a positive and a negative.
    scores = torch.tensor([0.0, 1.0, 1.0, 2.0])
    targets = torch.tensor([0, 0, 1, 1])
    # Pairs (neg, pos): (0.0, 1.0)->1, (0.0, 2.0)->1, (1.0, 1.0)->0.5,
    # (1.0, 2.0)->1  =>  3.5 / 4
    assert binary_auroc(scores, targets) == pytest.approx(3.5 / 4)


def test_binary_auroc_undefined_on_single_class():
    assert math.isnan(binary_auroc(torch.rand(8), torch.zeros(8)))
    assert math.isnan(binary_auroc(torch.rand(8), torch.ones(8)))


def test_binary_average_precision_matches_manual_sum():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    targets = torch.tensor([1, 0, 1, 0])
    # Ranked: hit, miss, hit, miss. AP = (1/1 + 2/3) / 2
    assert binary_average_precision(scores, targets) == pytest.approx(
        (1.0 + 2.0 / 3.0) / 2.0)


def test_binary_average_precision_perfect_ranking():
    scores = torch.tensor([0.1, 0.4, 0.8, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    assert binary_average_precision(scores, targets) == pytest.approx(1.0)


def test_multiclass_auroc_skips_absent_classes():
    # Class 2 never appears, so one-vs-rest is undefined for it and it must
    # be dropped from the macro average rather than counted as zero.
    probs = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
    ])
    targets = torch.tensor([0, 0, 1, 1])
    assert multiclass_auroc(probs, targets) == pytest.approx(1.0)


def test_brier_score_binary_and_multiclass():
    probs = torch.tensor([1.0, 0.0])
    targets = torch.tensor([1, 0])
    assert brier_score(probs, targets) == pytest.approx(0.0)

    probs = torch.tensor([0.5, 0.5])
    assert brier_score(probs, targets) == pytest.approx(0.25)

    probs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert brier_score(probs, targets) == pytest.approx(0.0)


def test_expected_calibration_error_extremes():
    # Confident and always right.
    probs = torch.tensor([1.0, 1.0, 0.0, 0.0])
    targets = torch.tensor([1, 1, 0, 0])
    assert expected_calibration_error(probs, targets) == pytest.approx(0.0)

    # Equally confident and always wrong.
    assert expected_calibration_error(probs, 1 - targets) == pytest.approx(1.0)


def test_maximum_calibration_error_is_at_least_ece():
    torch.manual_seed(0)
    probs = torch.rand(200)
    targets = (torch.rand(200) < probs).to(torch.long)
    ece = expected_calibration_error(probs, targets)
    mce = maximum_calibration_error(probs, targets)
    assert mce >= ece - 1e-12


def test_reliability_curve_bins_cover_every_sample():
    torch.manual_seed(0)
    probs = torch.rand(500)
    targets = (torch.rand(500) < probs).to(torch.long)
    curve = reliability_curve(probs, targets, num_bins=10)
    assert sum(curve["count"]) == 500
    assert len(curve["accuracy"]) == len(curve["confidence"])


def test_r2_score_perfect_and_undefined():
    y = torch.tensor([1.0, 2.0, 3.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    # Predicting the mean gives R2 = 0.
    assert r2_score(torch.full((3, ), 2.0), y) == pytest.approx(0.0)
    # Zero-variance targets make R2 undefined.
    assert math.isnan(r2_score(torch.rand(3), torch.ones(3)))


def test_normalize_binary_scores_shapes():
    logits = torch.tensor([0.0, 2.0])
    assert torch.allclose(normalize_binary_scores(logits),
                          logits.sigmoid().double())
    assert normalize_binary_scores(logits.reshape(-1, 1)).shape == (2, )
    two_col = torch.tensor([[0.0, 0.0], [0.0, 10.0]])
    probs = normalize_binary_scores(two_col)
    assert probs[0] == pytest.approx(0.5)
    assert probs[1] > 0.99


def test_normalize_multiclass_scores_sums_to_one():
    probs = normalize_multiclass_scores(torch.randn(5, 4))
    assert torch.allclose(probs.sum(dim=1), torch.ones(5, dtype=torch.float64))


def test_predicted_labels_covers_every_head_shape():
    # [B, C] logits -> argmax
    assert predicted_labels(torch.tensor([[1.0, -1.0],
                                          [-1.0, 1.0]])).tolist() == [0, 1]
    # [B, 1] logits -> threshold at 0 on the logit scale
    assert predicted_labels(torch.tensor([[2.0], [-2.0]])).tolist() == [1, 0]
    # [B] integer -> already class indices
    assert predicted_labels(torch.tensor([0, 1, 1])).tolist() == [0, 1, 1]
    # [B] probabilities
    assert predicted_labels(torch.tensor([0.7, 0.3]),
                            from_logits=False).tolist() == [1, 0]
