"""Tests for the failure-aware extensions: modality selection,
stacked router, evidential fusion, test-time corruption and zoo ids."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from medtabx.analysis.perturb import corrupt
from medtabx.auto.select import GreedyModalitySelector, ModalitySelected
from medtabx.auto.zoo import (
    create_model,
    label,
    list_models,
    parse,
    split_subset,
)
from medtabx.benchmarks.typed_bench import TypedData
from medtabx.fusion import (
    EvidentialFusion,
    ModalityExperts,
    StackedRouter,
    late_mean,
)
from medtabx.fusion.evidential import combine, opinion, probability


def _data(n=120, seed=0, noise_width=40, with_image=False):
    """``clin`` and ``labs`` carry y; ``noise`` is wide and carries nothing.

    With ``with_image`` an image modality ``scan`` (weak signal) is added.
    """
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    clin = rng.normal(size=(n, 3)) + 1.5 * y[:, None]
    labs = rng.normal(size=(n, 2)) + 1.0 * y[:, None]
    noise = rng.normal(size=(n, noise_width))
    num = np.concatenate([clin, labs, noise], axis=1)
    mods = ["clin"] * 3 + ["labs"] * 2 + ["noise"] * noise_width
    d = TypedData(y=y, num=num, num_cols=[f"c{j}" for j in range(len(mods))],
                  num_modality=mods)
    if with_image:
        img = rng.normal(size=(n, 1, 2, 8)) + 0.5 * y[:, None, None, None]
        d.img, d.img_present, d.img_modality = img, np.ones((n, 1), bool), \
            ["scan"]
    return d


# Shared experts ###########################################################


def test_experts_out_of_fold_matrix_marks_absent_modalities():
    d = _data()
    d.num[:10, 3:5] = np.nan                       # labs missing for 10
    e = ModalityExperts("lr", inner_splits=3).fit(d)
    assert e.oof_.shape == (len(d), len(e.mods))
    labs = e.mods.index("labs")
    assert np.isnan(e.oof_[:10, labs]).all()
    assert np.isfinite(e.oof_[10:, labs]).all()
    p = e.predict_matrix(d)
    assert np.isnan(p[:10, labs]).all()


def test_late_mean_ignores_absent_and_falls_back_to_half():
    p = np.array([[0.2, 0.6], [np.nan, 0.9], [np.nan, np.nan]])
    assert np.allclose(late_mean(p), [0.4, 0.9, 0.5])


# Stacked router ###########################################################


def test_router_with_zero_blend_is_exactly_late_fusion():
    d, held = _data(seed=0), _data(seed=1)
    router = StackedRouter(blend=0.0, include_concat=False,
                           inner_splits=3).fit(d)
    late = create_model("lr@late").fit(d, seed=0)
    assert np.allclose(router.predict(held), late.predict(held))


def test_router_predicts_probabilities_and_handles_a_missing_modality():
    d, held = _data(seed=0), _data(seed=1)
    router = StackedRouter(inner_splits=3).fit(d)
    p = router.predict(held)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(held.y, p) > 0.75
    p_missing = router.predict(held.drop(["clin"]))
    assert np.isfinite(p_missing).all()


# Evidential fusion ########################################################


def test_vacuous_opinion_leaves_the_other_unchanged():
    p, ev = np.array([0.8, 0.3]), np.array([5.0, 2.0])
    o = opinion(p, ev)
    vac = opinion(np.array([0.5, 0.5]), np.zeros(2))
    for a, b in zip(combine(o, vac), o):
        assert np.allclose(a, b)
    assert np.allclose(probability(vac), 0.5)


def test_dempster_rule_matches_hand_computation_and_is_symmetric():
    o1 = (np.array([0.2]), np.array([0.6]), np.array([0.2]))
    o2 = (np.array([0.5]), np.array([0.3]), np.array([0.2]))
    conflict = 0.2 * 0.3 + 0.6 * 0.5
    b0 = (0.2 * 0.5 + 0.2 * 0.2 + 0.5 * 0.2) / (1 - conflict)
    b1 = (0.6 * 0.3 + 0.6 * 0.2 + 0.3 * 0.2) / (1 - conflict)
    u = 0.2 * 0.2 / (1 - conflict)
    for got, want in zip(combine(o1, o2), (b0, b1, u)):
        assert np.allclose(got, want)
    for a, b in zip(combine(o1, o2), combine(o2, o1)):
        assert np.allclose(a, b)
    assert np.allclose(sum(combine(o1, o2)), 1.0)


def test_agreeing_opinions_are_more_confident_than_either():
    o = opinion(np.array([0.8]), np.array([4.0]))
    both = combine(o, o)
    assert probability(both)[0] > probability(o)[0]
    assert both[2][0] < o[2][0]


def test_evidential_fusion_trusts_informative_modalities_more():
    d, held = _data(seed=0), _data(seed=1)
    m = EvidentialFusion(inner_splits=3).fit(d)
    assert m.reliability["clin"] > m.reliability["noise"]
    w, mods = m.weights(held)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert w[:, mods.index("clin")].mean() > w[:, mods.index("noise")].mean()
    p = m.predict(held)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(held.y, p) > 0.75
    assert m.strength_ in m.strength_grid


def test_evidential_absent_modality_has_no_evidence():
    d = _data()
    m = EvidentialFusion(inner_splits=3, strength=10.0).fit(d)
    w, mods = m.weights(d.drop(["labs"]))
    assert np.allclose(w[:, mods.index("labs")], 0.0)


# Modality selection #######################################################


def test_permutation_filter_keeps_signal_and_drops_wide_noise():
    d = _data(n=160, noise_width=60)
    sel = GreedyModalitySelector(permutation_filter=True, inner_splits=3,
                                 inner_repeats=1)
    chosen = sel.select(d, seed=0)
    assert "clin" in chosen
    assert "noise" not in chosen
    assert not sel.fallback_
    taken = [r for r in sel.trace_ if r["taken"]]
    assert [r["modality"] for r in taken] == chosen
    assert all(np.isfinite(r["auroc_shuffled"]) for r in sel.trace_)


def test_selection_falls_back_to_one_modality_when_nothing_beats_chance():
    rng = np.random.default_rng(3)
    n = 80
    d = TypedData(y=np.array([0, 1] * (n // 2)),
                  num=rng.normal(size=(n, 4)), num_cols=list("abcd"),
                  num_modality=["u", "u", "v", "v"])
    sel = GreedyModalitySelector(eps=0.2, inner_splits=3, inner_repeats=1)
    chosen = sel.select(d)
    assert len(chosen) == 1 and sel.fallback_


def test_max_one_modality_picks_the_strongest_single_source():
    d = _data(n=160)
    sel = GreedyModalitySelector(max_modalities=1, inner_splits=3,
                                 inner_repeats=1)
    assert sel.select(d) == ["clin"]


def test_selector_refits_the_final_model_on_the_chosen_modalities():
    d, held = _data(seed=0), _data(seed=1)
    sel = GreedyModalitySelector(final="lr@late", permutation_filter=True,
                                 inner_splits=3, inner_repeats=1).fit(d)
    direct = create_model("lr@late").fit(d.select(sel.selected), seed=0)
    assert np.allclose(sel.predict(held), direct.predict(
        held.select(sel.selected)))


def test_selector_works_with_an_image_modality():
    d = _data(with_image=True)
    sel = GreedyModalitySelector(permutation_filter=True, inner_splits=3,
                                 inner_repeats=1, img_components=4)
    chosen = sel.select(d)
    assert set(chosen) <= {"clin", "labs", "noise", "scan"}


# Corruption ###############################################################


def test_swap_changes_only_the_chosen_rows_of_the_chosen_modality():
    d = _data()
    out, mask = corrupt(d, "labs", 0.25, kind="swap", seed=0)
    assert mask.sum() == round(0.25 * len(d))
    labs = [j for j, m in enumerate(d.num_modality) if m == "labs"]
    other = [j for j, m in enumerate(d.num_modality) if m != "labs"]
    assert np.array_equal(out.num[:, other], d.num[:, other])
    assert np.array_equal(out.num[~mask][:, labs], d.num[~mask][:, labs])
    assert not np.array_equal(out.num[mask][:, labs], d.num[mask][:, labs])
    assert np.array_equal(out.y, d.y)
    assert np.array_equal(d.num, _data().num)      # input untouched


def test_corruption_skips_patients_without_the_modality():
    d = _data()
    d.num[:30, 3:5] = np.nan
    out, mask = corrupt(d, "labs", 1.0, kind="noise", seed=1)
    assert not mask[:30].any() and mask[30:].all()
    assert np.isnan(out.num[:30, 3:5]).all()


def test_zero_fraction_changes_nothing_and_image_swap_works():
    d = _data(with_image=True)
    out, mask = corrupt(d, "scan", 0.0)
    assert not mask.any() and out.img is d.img
    out, mask = corrupt(d, "scan", 0.5, kind="swap", seed=2)
    assert np.array_equal(out.img[~mask], d.img[~mask])
    assert not np.array_equal(out.img[mask], d.img[mask])


def test_corrupt_rejects_unknown_inputs():
    with pytest.raises(KeyError):
        corrupt(_data(), "nope", 0.5)
    with pytest.raises(ValueError):
        corrupt(_data(), "labs", 0.5, kind="blur")


# Zoo ids ##################################################################


def test_zoo_parses_subsets_prefixes_and_opt_in_fusions():
    assert split_subset("lr@late[clin+labs]") == ("lr@late", ["clin", "labs"])
    assert parse("lr@late[clin+labs]")[1] == "late"
    assert parse("selp:tabpfn@late")[1] == "late"
    assert parse("lr@stack")[1] == "stack"
    assert parse("lr@ds")[1] == "ds"
    with pytest.raises(ValueError):
        parse("pf_ft@stack")                       # whole-input model
    assert "{clin+labs}" in label("lr@late[clin+labs]")
    assert "permutation filter" in label("selp:lr@late")
    assert "permutation filter" not in label("sel:lr@late")
    assert label("aug:lr@late").endswith("+ mod. dropout")


def test_zoo_builds_the_new_models():
    assert isinstance(create_model("lr@stack"), StackedRouter)
    assert isinstance(create_model("lr@ds"), EvidentialFusion)
    assert isinstance(create_model("lr@late[clin]"), ModalitySelected)
    sel = create_model("selp:lr@late")
    assert isinstance(sel, GreedyModalitySelector)
    assert sel.permutation_filter and sel.final == "lr@late"


def test_default_search_space_is_unchanged():
    ids = list_models(_data(), include_unavailable=True)
    assert not any(i.endswith(("@stack", "@ds")) for i in ids)
    assert not any(i.startswith(("sel:", "selp:")) or "[" in i for i in ids)


def test_new_models_run_through_the_cv_harness():
    from medtabx.auto.zoo import ensure_registered
    from medtabx.benchmarks.typed_bench import run_cv
    ids = ["lr@stack", "lr@ds", "selp:lr@late", "lr@late[clin+labs]"]
    ensure_registered(ids, img_components=4)
    rows = run_cv(_data(), ids, {"full": [], "no clin": ["clin"]},
                  n_splits=3, repeats=1, log=lambda *_: None)
    assert len(rows) == len(ids) * 3 * 2
    assert np.isfinite([r["auroc"] for r in rows]).all()


def test_autorecipe_can_select_modalities_first():
    from medtabx.auto.recommend import AutoRecipe
    d, held = _data(n=160, noise_width=60, seed=0), _data(n=80, seed=1,
                                                          noise_width=60)
    auto = AutoRecipe(budget=6, n_splits=3, repeats=1,
                      models=["lr", "lr@late"], select_modalities=True)
    auto.fit(d, seed=0)
    assert auto.selected is not None and "noise" not in auto.selected
    assert np.isfinite(auto.predict(held)).all()
    plain = AutoRecipe(budget=6, n_splits=3, repeats=1,
                       models=["lr", "lr@late"]).fit(d, seed=0)
    assert plain.selected is None


# TabPFN paths (skipped where tabpfn is not installed) #####################


@pytest.mark.skipif(__import__("importlib").util.find_spec("tabpfn") is None,
                    reason="tabpfn not installed")
def test_tabpfn_paths_run_end_to_end():
    d, held = _data(n=100, seed=0, with_image=True), _data(n=60, seed=1,
                                                           with_image=True)
    for mid in ("tabpfn@stack", "tabpfn@ds", "selp:tabpfn@late",
                "tabpfn@late[clin+scan]"):
        m = create_model(mid, img_components=4).fit(d, seed=0)
        p = m.predict(held)
        assert np.isfinite(p).all() and roc_auc_score(held.y, p) > 0.7, mid
    e = ModalityExperts("tabpfn", img_components=4).fit(d, oof=False)
    assert np.isfinite(e.predict_matrix(held)).all()
