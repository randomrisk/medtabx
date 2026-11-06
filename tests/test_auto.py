"""MedTabX AutoRecipe: contract builder, model zoo, compare_models, recommender."""

import json

import numpy as np
import pandas as pd
import pytest

import medtabx.core.stype as st
from medtabx.auto import (
    AutoRecipe,
    Objective,
    build_frame,
    compare_models,
    create_model,
    describe,
    feasible,
    flat_view,
    list_models,
    prior,
    register_base,
    token_view,
)
from medtabx.auto import zoo
from medtabx.auto.recommend import portfolio, sh_cost, sh_schedule

SKLEARN = ["lr", "lr@late", "rf", "rf@late", "hgb", "et@subset"]


def _df(n=160, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "age": rng.normal(65, 10, n) + 5 * y,
        "sex": rng.choice(["F", "M"], n),
        "nihss": rng.integers(0, 25, n).astype(float) + 3 * y,
        "glucose": rng.normal(6, 1, n),
        "inr": rng.normal(1, 0.1, n),
        "htn": rng.integers(0, 2, n),
        "label": y,
        "site": rng.integers(1, 3, n),
    })
    df.loc[rng.random(n) < 0.4, ["glucose", "inr"]] = np.nan  # labs missing
    df.loc[rng.random(n) < 0.3, "inr"] = 0.0
    img = rng.normal(size=(n, 8, 20)).astype(np.float32)
    img[:, :, 0] += y[:, None]
    img[rng.random(n) < 0.2] = np.nan                   # dwi absent
    return df, {"dwi": img}


MODS = {"demographics": ["age", "sex"], "presentation": ["nihss", "htn"],
        "labs": ["glucose", "inr"]}


@pytest.fixture(scope="module")
def data():
    df, img = _df()
    d, _ = build_frame(df, "label", MODS, images=img, group="site",
                       zero_is_missing=["inr"])
    return d


def test_build_frame_contract(data):
    assert data.num_cols == ["age", "nihss", "glucose", "inr"]
    assert data.cat_cols == ["sex", "htn"]
    assert data.cat_cardinality == [2, 2]
    assert data.img.shape == (160, 1, 8, 20)
    assert 0.1 < 1 - data.img_present.mean() < 0.3
    assert np.isnan(data.num[:, 3]).mean() > 0.4        # zero -> missing
    assert data.modalities == ["demographics", "presentation", "labs", "dwi"]
    assert data.groups is not None


def test_builder_transform_new_data_and_stypes():
    df, img = _df()
    d, b = build_frame(df, "label", MODS, images=img,
                       stypes={"htn": st.numerical, "age": st.lab_value})
    assert "htn" in d.num_cols
    new = df.iloc[:5].copy()
    new.loc[new.index[0], "sex"] = "unknown-level"
    d2 = b.transform(new, images={"dwi": img["dwi"][:5]})
    assert d2.cat[0, d2.cat_cols.index("sex")] == -1     # unseen -> missing
    with pytest.raises(ValueError):
        build_frame(df, "label", {"x": ["age"]}, stypes={"age": st.image_3d})


def test_views_and_describe(data):
    m = describe(data)
    assert m["n"] == 160 and m["n_modalities"] == 4
    assert m["value_types"] == ["num", "cat", "img"]
    assert m["modality_missing"]["labs"] > 0.3
    x, groups = flat_view(data)
    assert x.shape[0] == 160 and len(groups) == x.shape[1]
    h, missing = token_view(data, d_model=16)
    assert h.shape == (160, data.schema().num_tokens, 16)   # H [B, N, C]
    assert missing.shape == h.shape[:2]


def test_zoo_ids_and_feasibility(data):
    ids = list_models(data)
    assert {"lr", "lr@late", "rf@subset", "hgb"} <= set(ids)
    one = data.select(["dwi"])
    assert not feasible("lr@late", describe(one))[0]
    ok, why = feasible("tabpfn", {**describe(data), "n": 50_000})
    assert not ok
    with pytest.raises(ValueError):
        zoo.parse("pf_ft@late")


@pytest.mark.parametrize("mid", SKLEARN)
def test_every_fusion_fits_and_handles_missing(data, mid):
    tr, te = data.subset(np.arange(120)), data.subset(np.arange(120, 160))
    m = create_model(mid).fit(tr, seed=0)
    for dropped in ([], ["labs"], ["dwi", "labs"]):
        p = m.predict(te.drop(dropped) if dropped else te)
        assert p.shape == (40,) and np.all((p >= 0) & (p <= 1))


def test_register_custom_learner(data):
    from sklearn.naive_bayes import GaussianNB
    register_base("gnb", lambda seed: GaussianNB(), label="Naive Bayes")
    try:
        assert {"gnb", "gnb@late", "gnb@subset"} <= set(list_models(data))
        board = compare_models(data, ["gnb", "gnb@late"], n_splits=3,
                               log=None)
        assert set(board.table.index) == {"gnb", "gnb@late"}
    finally:
        zoo.BASES.pop("gnb")


def test_compare_models_leaderboard(data):
    board = compare_models(data, ["lr", "lr@late", "hgb"], n_splits=3,
                           log=None)
    t = board.table
    assert list(t.index) == list(t.sort_values("auroc",
                                               ascending=False).index)
    for c in ("auroc", "auprc", "brier", "ece", "auroc_missing",
              "missing_drop", "fit_s"):
        assert c in t
    assert (t.folds == 3).all()
    assert board.best in {"lr", "lr@late", "hgb"}
    # 4 modalities -> full + 4 "no <modality>" scenarios per fold
    assert board.rows.groupby("model").size().eq(3 * 5).all()


def test_compare_records_failures(data):
    def boom(seed):
        raise RuntimeError("broken learner")
    register_base("boom", boom)
    try:
        board = compare_models(data, ["lr", "boom"], n_splits=2, log=None)
        assert "broken learner" in board.table.loc["boom", "error"]
        assert board.best == "lr"
    finally:
        zoo.BASES.pop("boom")


def test_prior_rules(data):
    meta = describe(data)
    cands = {c.model: c for c in prior(meta, Objective.preset("balanced"))}
    # small n, multimodal, missing modalities: structured fusion preferred
    assert cands["rf@late"].prior > cands["rf"].prior
    assert any("structured_fusion_missing" in r
               for r in cands["lr@late"].reasons)
    if "pf_ft" in cands:
        assert cands["pf_ft"].prior < cands["lr"].prior
    fast = prior(meta, Objective.preset("fast"))
    assert all(zoo.parse(c.model)[0].cost <= 1.0 for c in fast)


def test_successive_halving_schedule_and_portfolio(data):
    assert sh_schedule(9, 10, eta=3) == [(9, 2), (3, 6), (1, 10)]
    assert sh_cost(9, 10, eta=3) == 9 * 2 + 3 * 4 + 1 * 4
    cands = prior(describe(data), Objective())
    pool = portfolio(cands, 4)
    assert len(pool) == 4
    assert len({c.family for c in pool}) == min(4, len({c.family
                                                          for c in cands}))


def test_autorecipe_end_to_end(data, tmp_path):
    hist = tmp_path / "history.jsonl"
    auto = AutoRecipe(budget=30, n_splits=3, repeats=1, models=SKLEARN,
                      history=hist)
    auto.fit(data.subset(np.arange(120)), seed=0)
    fits = sum(c.folds for c in auto.candidates)
    assert fits <= 30
    assert auto.best.model in SKLEARN
    assert auto.best.rung == max(c.rung for c in auto.candidates)
    p = auto.predict(data.subset(np.arange(120, 160)))
    assert p.shape == (40,)
    assert "Recommended:" in auto.explain()
    rec = json.loads(hist.read_text().splitlines()[-1])
    assert rec["best"] == auto.best.model and rec["meta"]["n"] == 120


def test_autorecipe_runs_in_cv_harness(data):
    from medtabx.benchmarks.typed_bench import METHODS, run_cv
    METHODS["auto_test"] = lambda: AutoRecipe(budget=12, n_splits=2,
                                              repeats=1, models=SKLEARN[:3])
    try:
        rows = run_cv(data, ["auto_test"], {"full": []}, n_splits=2,
                      repeats=1, log=lambda s: None)
        assert len(rows) == 2 and all(0 <= r["auroc"] <= 1 for r in rows)
    finally:
        METHODS.pop("auto_test")
