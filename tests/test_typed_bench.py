"""Tests for the typed multimodal comparison harness."""

from __future__ import annotations

import numpy as np
import pytest

from medtabx.benchmarks.typed_bench import (
    METHODS,
    TypedData,
    evaluate_scenarios,
    run_cv,
    run_group_shift,
)


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    num = rng.normal(size=(n, 3)) + y[:, None]
    num[rng.random((n, 3)) < 0.1] = np.nan
    cat = rng.integers(0, 2, (n, 1))
    img = (rng.normal(size=(n, 2, 1, 16)) + y[:, None, None, None]).astype(np.float16)
    present = np.ones((n, 2), bool)
    present[::7, 1] = False
    return TypedData(y=y, num=num, num_cols=["a", "b", "c"],
                     num_modality=["demo", "demo", "labs"], cat=cat,
                     cat_cols=["sex"], cat_modality=["demo"],
                     cat_cardinality=[2], img=img, img_present=present,
                     img_modality=["dwi", "flair"],
                     groups=np.array([1, 1, 2, 2] * (n // 4)))


def test_drop_marks_whole_modalities_missing_without_mutating():
    d = _data()
    out = d.drop(["labs", "flair"])
    assert np.isnan(out.num[:, 2]).all()
    assert not out.img_present[:, 1].any()
    assert d.img_present[:, 1].sum() > 0  # original untouched


def test_select_keeps_only_requested_sources():
    d = _data().select(["demo", "dwi"])
    assert d.num.shape[1] == 2 and d.cat.shape[1] == 1
    assert d.img_modality == ["dwi"] and d.img.shape[1] == 1
    only_img = _data().select(["flair"])
    assert only_img.num is None and only_img.cat is None


def test_subset_slices_every_array():
    d = _data().subset(np.arange(10))
    assert len(d) == 10 and d.img.shape[0] == 10 and d.groups.shape == (10,)


@pytest.mark.parametrize("name", ["lr", "gbdt", "late", "mlp", "medtabx"])
def test_methods_learn_a_separable_signal(name):
    d = _data(80)
    tr, te = np.arange(0, 60), np.arange(60, 80)
    model = METHODS[name]().fit(d.subset(tr), seed=0)
    rows = evaluate_scenarios(model, d.subset(te),
                              {"full": [], "no dwi": ["dwi"]})
    assert rows[0]["auroc"] > 0.8
    assert all(0 <= p <= 1 for p in rows[1]["_pred"])


def test_run_cv_and_group_shift_rows():
    d = _data()
    rows = run_cv(d, ["lr"], {"full": []}, n_splits=3, repeats=1,
                  log=lambda s: None)
    assert len(rows) == 3 and {"auroc", "auprc", "brier"} <= set(rows[0])
    rows = run_group_shift(d, ["lr"], {"full": []}, 1, 2, seeds=2,
                           log=lambda s: None)
    assert len(rows) == 2 and rows[0]["split"] == "1->2"


def test_typed_stacking_learns_and_handles_missing_modality():
    d = _data(80)
    tr, te = np.arange(0, 60), np.arange(60, 80)
    model = METHODS["stack_typed"]().fit(d.subset(tr), seed=0)
    rows = evaluate_scenarios(model, d.subset(te),
                              {"full": [], "no dwi": ["dwi"]})
    assert rows[0]["auroc"] > 0.8
    assert np.isfinite(rows[1]["_pred"]).all()


def test_subset_ensemble_never_uses_an_absent_modality():
    from medtabx.benchmarks.typed_bench import ModalitySubsetEnsemble

    d = _data(80)
    tr, te = np.arange(0, 60), np.arange(60, 80)
    model = ModalitySubsetEnsemble("s", base="lr", n_members=8).fit(
        d.subset(tr), seed=0)
    subsets = [s for s, _ in model.members]
    assert ("dwi", ) in subsets and len(subsets) >= 5
    rows = evaluate_scenarios(model, d.subset(te),
                              {"full": [], "no dwi": ["dwi"]})
    assert rows[0]["auroc"] > 0.8
    # With DWI absent, a member that needs DWI must not contribute: the
    # prediction equals the same ensemble restricted to DWI-free members.
    test = d.subset(te).drop(["dwi"])
    kept = [(s, m) for s, m in model.members if "dwi" not in s]
    full_members = model.members
    model.members = kept
    restricted = model.predict(test)
    model.members = full_members
    assert np.allclose(model.predict(test), restricted)


def test_singletons_only_is_late_fusion():
    from medtabx.benchmarks.typed_bench import ModalitySubsetEnsemble

    model = ModalitySubsetEnsemble("late", base="lr",
                                   singletons_only=True).fit(_data(60), 0)
    assert all(len(s) == 1 for s, _ in model.members)
