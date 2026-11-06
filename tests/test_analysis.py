"""Mechanistic analysis: information decomposition, alignment, robustness."""

import numpy as np
import pytest

from medtabx.analysis.alignment import (branch_report, js_divergence,
                                        pairwise_js, probe_branches)
from medtabx.analysis.info import cross_fit_ce, decompose, shuffled
from medtabx.analysis.robustness import (DropoutAugmented, concat,
                                         drop_per_row, observed_modalities,
                                         random_modality_dropout,
                                         reliance_profile)
from medtabx.auto import build_frame, create_model
from medtabx.benchmarks.typed_bench import TypedData


def make(n=240, seed=0, informative_image=True):
    """Two clinical modalities carrying signal, one image modality that is
    informative or pure noise, and missing labs."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "age": rng.normal(70, 10, n) + 6 * y,
        "nihss": rng.normal(10, 5, n) + 4 * y,
        "glucose": rng.normal(6, 1, n) + y,
        "inr": rng.normal(1, 0.2, n),
        "label": y,
    })
    df.loc[rng.random(n) < 0.3, ["glucose", "inr"]] = np.nan
    img = rng.normal(size=(n, 4, 24)).astype(np.float32)
    if informative_image:
        img[:, :, 0] += 1.5 * y[:, None]
    img[rng.random(n) < 0.15] = np.nan
    d, _ = build_frame(df, "label",
                       {"demographics": ["age"], "presentation": ["nihss"],
                        "labs": ["glucose", "inr"]}, images={"dwi": img})
    return d


@pytest.fixture(scope="module")
def data():
    return make()


# Information ##############################################################


def test_cross_fit_ce_baseline_and_signal(data):
    h_y, auroc = cross_fit_ce(np.zeros((len(data), 0)), data.y)
    assert 0.6 < h_y <= np.log(2) + 1e-6      # no features -> ~H(Y)
    assert abs(auroc - 0.5) < 0.05     # constant prediction: no ranking
    x = np.nan_to_num(data.num)
    ce, auroc = cross_fit_ce(x, data.y)
    assert ce < h_y and auroc > 0.7           # real features carry signal


def test_shuffled_control_destroys_information(data):
    x = np.nan_to_num(data.num)
    _, auroc = cross_fit_ce(shuffled(x, 0), data.y)
    assert auroc < 0.65
    assert shuffled(x, 0).shape == x.shape


def test_decompose_informative_image(data):
    d = decompose(data, ["demographics", "presentation", "labs"], ["dwi"])
    assert d.i_a > 0 and d.i_b > 0
    assert d.unique_b > 0                      # image adds on top of clinical
    assert d.auroc_ab > d.auroc_ab_shuffled_b  # real beats same-width noise
    assert d.information_gain_net > 0
    assert set(d.as_dict()) > {"unique_b", "redundancy", "synergy",
                               "dimensional_cost"}


def test_decompose_noise_image_is_pure_capacity_cost():
    noise = make(informative_image=False, seed=3)
    d = decompose(noise, ["demographics", "presentation", "labs"], ["dwi"],
                  img_components=16)
    # no content: real image behaves like its shuffled control ...
    assert abs(d.information_gain_net) < 0.05
    # ... and widening the model costs something relative to clinical alone
    assert d.auroc_ab <= d.auroc_a + 0.02


# Alignment ################################################################


def test_js_divergence_properties():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(js_divergence(p, p), 0.0, atol=1e-9)
    assert js_divergence(np.array([0.01]), np.array([0.99]))[0] > 0.4
    a, b = np.array([0.2, 0.8]), np.array([0.6, 0.4])
    assert np.allclose(js_divergence(a, b), js_divergence(b, a))


def test_pairwise_js_respects_presence():
    probs = {"a": np.array([0.1, 0.9]), "b": np.array([0.9, 0.1])}
    present = {"a": np.array([True, True]), "b": np.array([True, False])}
    mean, pairs = pairwise_js(probs, present)
    assert set(pairs) == {"a|b"}
    assert mean == pytest.approx(pairs["a|b"])
    assert mean == pytest.approx(js_divergence(np.array([0.1]),
                                               np.array([0.9]))[0])


def test_probe_branches_and_report(data):
    probs, present = probe_branches(data, n_splits=3)
    assert set(probs) == set(data.modalities)
    assert not present["dwi"].all()            # absent images stay absent
    assert np.isnan(probs["dwi"][~present["dwi"]]).all()
    rep = branch_report(probs, present, data.y,
                        fused=np.nanmean(np.vstack(
                            [np.where(present[m], probs[m], np.nan)
                             for m in probs]), axis=0))
    assert rep.gap >= 0
    assert rep.best_modality in data.modalities
    assert not np.isnan(rep.js_mean)
    assert "fusion_gain" in rep.as_dict()


def test_alignment_loss_and_schedule(data):
    from medtabx.benchmarks.typed_bench import TypedTransformerMethod
    plain = TypedTransformerMethod("plain")
    m = TypedTransformerMethod("js", head="gated_experts",
                               align_weight=0.5, align="js")
    m.max_epochs, m.patience = 3, 3
    m.track_epochs = True
    m.fit(data, seed=0)
    assert len(m.history) >= 1
    h = m.history[0]
    assert "js_mean" in h and "gap" in h and h["alpha"] == 0.5
    assert 0 <= m.predict(data).min() <= m.predict(data).max() <= 1
    # schedules
    model = m.model
    model.config.align_schedule = "decay"
    a1, a2 = m._align_alpha(model, 1), m._align_alpha(model, 50)
    assert a1 > a2 > 0                       # alignment first, fusion later
    model.config.align_schedule = "stepwise"
    assert m._align_alpha(model, 1) == 0.5
    assert m._align_alpha(model, m.max_epochs) == 0.0
    assert plain._align_alpha(plain, 1) is None   # off by default


def test_prediction_alignment_requires_trainable_branches():
    """JS alignment on a head whose output does not decompose per modality
    would be vacuous (experts get no task gradient), so it is refused."""
    from medtabx.models.typed_token import (TypedTokenConfig,
                                            TypedTokenSchema,
                                            TypedTokenTransformer)
    schema = TypedTokenSchema(num_modality=["a", "b"], cat_modality=[],
                              cat_cardinality=[])
    with pytest.raises(ValueError, match="gated_experts"):
        TypedTokenTransformer(schema, TypedTokenConfig(
            head="cls", align_weight=0.5, align="js"))
    # representation alignment is fine on any head
    TypedTokenTransformer(schema, TypedTokenConfig(
        head="cls", align_weight=0.5, align="rep"))


def test_epoch_metrics_without_expert_heads(data):
    """Branch curves must also work for a model with no alignment (no expert
    modules): the branches come from masking to one modality."""
    from medtabx.benchmarks.typed_bench import TypedTransformerMethod
    m = TypedTransformerMethod("plain")
    m.max_epochs, m.patience, m.track_epochs = 2, 2, True
    m.fit(data, seed=0)
    h = m.history[-1]
    assert not hasattr(m.model, "n_modalities")     # no experts built
    for mod in data.modalities:
        assert f"auroc_{mod}" in h
    assert not np.isnan(h["js_mean"]) and h["gap"] >= 0
    assert 0.0 <= h["fused_auroc"] <= 1.0


def test_semi_supervised_uses_unlabeled_rows(data):
    from medtabx.benchmarks.typed_bench import TypedTransformerMethod
    m = TypedTransformerMethod("ssl", head="gated_experts",
                               align_weight=0.5, align="js",
                               ssl_entropy=0.1)
    m.max_epochs, m.patience = 3, 3
    m.unlabeled = data.subset(np.arange(120, 240))
    m.fit(data.subset(np.arange(120)), seed=0)
    assert m._ssl is not None
    assert m.predict(data).shape == (len(data),)


# Robustness ###############################################################


def test_drop_per_row_and_observed(data):
    keep = {"labs": np.zeros(len(data), bool)}
    out = drop_per_row(data, keep)
    j = [i for i, m in enumerate(data.num_modality) if m == "labs"]
    assert np.isnan(out.num[:, j]).all()
    assert not np.isnan(data.num[:, j]).all()      # original untouched
    obs = observed_modalities(data)
    assert obs["demographics"].all()
    assert obs["dwi"].mean() < 1.0


def test_random_modality_dropout_keeps_one(data):
    rng = np.random.default_rng(0)
    out = random_modality_dropout(data, p=0.9, rng=rng, keep_one=True)
    obs = observed_modalities(out)
    any_left = np.any([obs[m] for m in out.modalities], axis=0)
    assert any_left.all()
    assert sum(obs[m].mean() for m in out.modalities) < \
        sum(observed_modalities(data)[m].mean() for m in data.modalities)


def test_concat_stacks_rows(data):
    both = concat([data, data])
    assert len(both) == 2 * len(data)
    assert both.img.shape[0] == 2 * data.img.shape[0]
    assert both.num_cols == data.num_cols


def test_dropout_augmentation_helps_a_model_that_never_saw_missing(data):
    train, test = data.subset(np.arange(180)), data.subset(np.arange(180, 240))
    plain = create_model("lr").fit(train, seed=0)
    aug = DropoutAugmented(lambda: create_model("lr"), p=0.4,
                           n_aug=3).fit(train, seed=0)
    from sklearn.metrics import roc_auc_score
    missing = test.drop(["labs"])
    a = roc_auc_score(test.y, plain.predict(missing))
    b = roc_auc_score(test.y, aug.predict(missing))
    assert b >= a - 0.02          # augmentation must not hurt much ...
    assert "modality dropout" in aug.name
    full_a = roc_auc_score(test.y, plain.predict(test))
    full_b = roc_auc_score(test.y, aug.predict(test))
    assert full_b >= full_a - 0.05   # ... and keeps full-input performance


def test_aug_prefix_model_id(data):
    m = create_model("aug:lr@late")
    assert isinstance(m, DropoutAugmented)
    m.fit(data.subset(np.arange(200)), seed=0)
    p = m.predict(data.subset(np.arange(200, 240)).drop(["dwi"]))
    assert p.shape == (40,) and np.all((p >= 0) & (p <= 1))


def test_reliance_profile(data):
    m = create_model("lr").fit(data, seed=0)
    rel = reliance_profile(m, data)
    assert set(rel) == set(data.modalities)
    assert all(v >= 0 for v in rel.values())
    assert max(rel.values()) > 0
