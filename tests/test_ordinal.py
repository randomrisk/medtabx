"""Tests for the ordinal (full-scale mRS) metrics and harness."""

from __future__ import annotations

import numpy as np
import pytest

from medtabx.benchmarks.ordinal_bench import (
    CumulativeLink,
    Multinomial,
    run_ordinal_cv,
    stratification_label,
)
from medtabx.benchmarks.typed_bench import TypedData
from medtabx.metrics.ordinal import (
    cumulative_targets,
    distribution,
    expected_level,
    monotone_repair,
    ordinal_c_index,
    ordinal_metrics,
    quadratic_weighted_kappa,
)


def _ordinal_data(n=120, n_levels=5, seed=0):
    """Levels driven by a latent score, so ordering is genuinely informative."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    edges = np.quantile(latent, np.linspace(0, 1, n_levels + 1)[1:-1])
    y_ord = np.searchsorted(edges, latent)
    num = np.column_stack([latent + rng.normal(0, .5, n),
                           latent + rng.normal(0, .5, n),
                           rng.normal(size=n)])
    data = TypedData(y=(y_ord > n_levels // 2).astype(int), num=num,
                     num_cols=["a", "b", "c"],
                     num_modality=["sig", "sig", "noise"])
    return data, y_ord


# -- metric identities -----------------------------------------------------


def test_cumulative_targets_are_nested():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    t = cumulative_targets(y, 7)
    assert t.shape == (7, 6)
    assert np.all(np.diff(t, axis=1) <= 0)          # non-increasing in k
    assert t[:, 2].tolist() == [0, 0, 0, 1, 1, 1, 1]   # the mRS>2 dichotomy


def test_distribution_recovers_exact_levels_from_exact_cumulatives():
    y = np.array([0, 3, 6, 4])
    p = distribution(cumulative_targets(y, 7).astype(float))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.argmax(axis=1).tolist() == y.tolist()
    assert np.allclose(expected_level(p), y)


def test_monotone_repair_fixes_only_incoherent_rows():
    cum = np.array([[0.9, 0.8, 0.7],        # already coherent
                    [0.5, 0.9, 0.1]])       # P(y>1) > P(y>0): impossible
    fixed, violated = monotone_repair(cum)
    assert np.allclose(fixed[0], cum[0])
    assert np.all(np.diff(fixed, axis=1) <= 1e-12)
    assert violated == 0.5


def test_ordinal_c_index_equals_auroc_when_there_are_two_levels():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    score = y + rng.normal(0, 1.2, 200)
    assert ordinal_c_index(y, score) == pytest.approx(
        roc_auc_score(y, score), abs=1e-9)


def test_c_index_and_kappa_reward_getting_the_order_right():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    assert ordinal_c_index(y, y.astype(float)) == 1.0
    assert ordinal_c_index(y, -y.astype(float)) == 0.0
    assert quadratic_weighted_kappa(y, y, 7) == pytest.approx(1.0)


def test_a_near_miss_scores_better_than_a_far_miss():
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    near = np.clip(y + 1, 0, 6)
    far = (6 - y)
    assert (quadratic_weighted_kappa(y, near, 7)
            > quadratic_weighted_kappa(y, far, 7))


def test_perfect_prediction_scores_perfectly_everywhere():
    y = np.array([0, 1, 2, 3, 4, 5, 6] * 4)
    m = ordinal_metrics(y, cumulative_targets(y, 7).astype(float), 7)
    assert m["qwk"] == pytest.approx(1.0)
    assert m["c_index"] == pytest.approx(1.0)
    assert m["mae_levels"] == pytest.approx(0.0)
    assert m["acc"] == pytest.approx(1.0)
    assert m["monotonicity_violated"] == 0.0


def test_metrics_report_the_conventional_dichotomy_unchanged():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    y = rng.integers(0, 7, 150)
    cum = np.clip(cumulative_targets(y, 7) + rng.normal(0, .3, (150, 6)), 0, 1)
    m = ordinal_metrics(y, cum, 7)
    assert m["auroc_dichotomy"] == pytest.approx(
        roc_auc_score((y > 2).astype(int), cum[:, 2]))


# -- stratification --------------------------------------------------------


def test_stratification_merges_rare_levels_upward_without_touching_labels():
    y = np.array([0] * 44 + [1] * 24 + [2] * 11 + [3] * 12 + [4] * 18
                 + [5] * 3 + [6] * 17)
    s = stratification_label(y, 5)
    assert (s == 5).sum() == 0                 # mRS 5 merged away
    assert (s == 6).sum() == 20                # into mRS 6: severe or dead
    assert np.bincount(s)[np.bincount(s) > 0].min() >= 5
    assert (y == 5).sum() == 3                 # original array untouched


def test_stratification_is_a_no_op_when_every_level_is_frequent():
    y = np.repeat(np.arange(4), 10)
    assert np.array_equal(stratification_label(y, 5), y)


# -- the two strategies ----------------------------------------------------


def test_cumulative_link_fits_one_model_per_threshold():
    data, y_ord = _ordinal_data(n_levels=5)
    m = CumulativeLink("lr", n_levels=5).fit(data, y_ord)
    assert len(m.models) == 4
    cum = m.predict_cumulative(data)
    assert cum.shape == (len(data), 4)
    assert np.all((cum >= 0) & (cum <= 1))


def test_cumulative_link_handles_a_threshold_with_one_class():
    data, y_ord = _ordinal_data(n_levels=5)
    y_ord = np.minimum(y_ord, 2)               # nothing above level 2
    m = CumulativeLink("lr", n_levels=5).fit(data, y_ord)
    cum = m.predict_cumulative(data)
    assert np.allclose(cum[:, 3], 0.0)         # degenerate threshold
    assert np.isfinite(cum).all()


def test_multinomial_control_returns_coherent_cumulatives():
    data, y_ord = _ordinal_data(n_levels=5)
    m = Multinomial("lr", n_levels=5).fit(data, y_ord)
    cum = m.predict_cumulative(data)
    assert cum.shape == (len(data), 4)
    # assembled from one distribution, so monotone by construction
    assert np.all(np.diff(cum, axis=1) <= 1e-9)
    assert monotone_repair(cum)[1] == 0.0


def test_both_strategies_beat_chance_on_an_ordered_signal():
    data, y_ord = _ordinal_data(n=200, n_levels=5, seed=4)
    for model in (CumulativeLink("lr", 5), Multinomial("lr", 5)):
        cum = model.fit(data, y_ord).predict_cumulative(data)
        m = ordinal_metrics(y_ord, cum, 5)
        assert m["c_index"] > 0.7, model.name
        assert m["qwk"] > 0.4, model.name


def test_run_ordinal_cv_pairs_strategies_on_identical_folds():
    data, y_ord = _ordinal_data(n=120, n_levels=5)
    rows = run_ordinal_cv(data, y_ord, ["lr"], {"full": [], "no noise": ["noise"]},
                          multinomial_bases=("lr",), n_levels=5, n_splits=3,
                          repeats=1, log=lambda *_: None)
    frame_keys = {(r["repeat"], r["fold"], r["method"], r["scenario"])
                  for r in rows}
    assert len(rows) == 3 * 2 * 2 == len(frame_keys)
    assert {r["strategy"] for r in rows} == {"cumulative", "multinomial"}
    assert all(np.isfinite(r["qwk"]) for r in rows)


def test_run_ordinal_cv_rejects_a_misaligned_label():
    data, y_ord = _ordinal_data(n=60, n_levels=4)
    with pytest.raises(ValueError, match="rows"):
        run_ordinal_cv(data, y_ord[:-1], ["lr"], {"full": []},
                       n_levels=4, n_splits=3, repeats=1,
                       log=lambda *_: None)


# -- the stacked (proportional / partial proportional odds) strategies -----


def test_proportional_odds_cannot_violate_monotonicity():
    """Shared coefficients plus ordered intercepts make the cumulative
    probabilities coherent by construction. This is the property that
    motivates the strategy, so it is asserted rather than hoped for."""
    from medtabx.benchmarks.ordinal_bench import StackedOrdinal
    data, y_ord = _ordinal_data(n=200, n_levels=5, seed=2)
    cum = StackedOrdinal((), 5).fit(data, y_ord).predict_cumulative(data)
    assert monotone_repair(cum)[1] == 0.0
    assert cum.shape == (len(data), 4)


def test_partial_proportional_odds_frees_only_the_named_modalities():
    from medtabx.benchmarks.ordinal_bench import StackedOrdinal
    data, y_ord = _ordinal_data(n=150, n_levels=5)
    po = StackedOrdinal((), 5).fit(data, y_ord)
    partial = StackedOrdinal(("sig",), 5).fit(data, y_ord)
    free = StackedOrdinal("all", 5).fit(data, y_ord)
    n_po = po.model.coef_.size
    # each freed column adds one coefficient per threshold
    assert partial.model.coef_.size > n_po
    assert free.model.coef_.size > partial.model.coef_.size
    assert po.strategy == "proportional"
    assert partial.strategy == "partial"
    assert "sig" in partial.name


def test_stacked_reshape_keeps_patients_aligned():
    """The stacked design is threshold-major; a wrong reshape would silently
    scramble patients across thresholds."""
    from medtabx.benchmarks.ordinal_bench import StackedOrdinal
    data, y_ord = _ordinal_data(n=120, n_levels=5, seed=7)
    m = StackedOrdinal((), 5).fit(data, y_ord)
    cum_all = m.predict_cumulative(data)
    # predicting one patient at a time must give the same rows
    for i in (0, 17, 60, 119):
        one = m.predict_cumulative(data.subset(np.array([i])))
        assert np.allclose(one[0], cum_all[i], atol=1e-8)


def test_stacked_beats_chance_and_handles_a_missing_modality():
    from medtabx.benchmarks.ordinal_bench import StackedOrdinal
    data, y_ord = _ordinal_data(n=200, n_levels=5, seed=5)
    m = StackedOrdinal((), 5).fit(data, y_ord)
    assert ordinal_metrics(y_ord, m.predict_cumulative(data), 5)["c_index"] > 0.7
    dropped = m.predict_cumulative(data.drop(["noise"]))
    assert np.isfinite(dropped).all()
    assert dropped.shape == (len(data), 4)


def test_auto_selection_finds_the_modality_whose_informativeness_drifts():
    """Ground-truth check: only the modality built to be informative at the
    severe end should be freed, and the choice must use the training rows
    only."""
    from medtabx.benchmarks.ordinal_bench import StackedOrdinal
    rng = np.random.default_rng(0)
    n, K = 240, 5
    latent = rng.normal(size=n)
    y_ord = np.searchsorted(np.quantile(latent, [.2, .4, .6, .8]), latent)
    drift = np.where(y_ord >= 3, rng.normal(2, 1, n), rng.normal(0, 1, n))
    num = np.column_stack([latent + rng.normal(0, .6, n), drift,
                           rng.normal(size=n)])
    data = TypedData(y=(y_ord > 2).astype(int), num=num,
                     num_cols=["a", "b", "c"],
                     num_modality=["flat", "drift", "noise"])
    m = StackedOrdinal("auto", K).fit(data, y_ord)
    assert "drift" in m.selected
    assert "noise" not in m.selected
    assert m.strategy == "partial"
    cum = m.predict_cumulative(data)
    assert np.isfinite(cum).all() and cum.shape == (n, K - 1)
