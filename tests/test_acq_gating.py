"""Tests for the acquisition-condition gate (medtabx.auto.gating)."""

from __future__ import annotations

import numpy as np
import pytest

from medtabx.auto.gating import AcquisitionGatedFusion
from medtabx.auto.zoo import create_model, label, list_models, parse
from medtabx.benchmarks.typed_bench import TypedData


def _data(n=80, seed=0, informative=("labs",)):
    """Two numeric modalities; only those named in ``informative`` carry y."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    signal = {m: (y if m in informative else 0) for m in ("demo", "labs")}
    num = np.column_stack([
        rng.normal(size=n) + signal["demo"],
        rng.normal(size=n) + signal["demo"],
        rng.normal(size=n) + 2 * signal["labs"],
        rng.normal(size=n) + 2 * signal["labs"],
    ])
    return TypedData(y=y, num=num, num_cols=["a", "b", "c", "d"],
                     num_modality=["demo", "demo", "labs", "labs"])


def test_zero_exponents_reproduce_uniform_late_fusion():
    d = _data()
    g = AcquisitionGatedFusion("g", w_reliability=0, w_quality=0).fit(d)
    w, mods = g.weights(d)
    assert len(mods) == 2
    assert np.allclose(w, 0.5)


def test_reliability_term_prefers_the_informative_modality():
    d = _data(informative=("labs",))
    g = AcquisitionGatedFusion("g", w_reliability=1, w_quality=0).fit(d)
    w, mods = g.weights(d)
    labs = mods.index("labs")
    assert g.reliability["labs"] > g.reliability["demo"]
    assert w[:, labs].mean() > 0.5


def test_weights_are_per_patient_only_when_quality_is_on():
    d = _data()
    flat = AcquisitionGatedFusion("g", w_reliability=1, w_quality=0).fit(d)
    w_flat, _ = flat.weights(d)
    assert np.allclose(w_flat, w_flat[0])          # same for every patient
    gated = AcquisitionGatedFusion("g", w_reliability=1, w_quality=1).fit(d)
    w_gated, _ = gated.weights(d)
    assert w_gated.std(axis=0).max() > 1e-6        # varies by patient


def test_an_outlying_acquisition_is_downweighted_for_that_patient_only():
    d = _data()
    g = AcquisitionGatedFusion("g", w_reliability=0, w_quality=1).fit(d)
    corrupt = TypedData(**{**d.__dict__, "num": d.num.copy()})
    corrupt.num[0, 2:] *= 40                       # patient 0's labs go wild
    w, mods = g.weights(corrupt)
    labs = mods.index("labs")
    assert w[0, labs] < w[1:, labs].min()
    assert np.allclose(w.sum(axis=1), 1.0)


def test_absent_modality_gets_zero_weight_and_the_rest_renormalize():
    d = _data()
    g = AcquisitionGatedFusion("g").fit(d)
    w, mods = g.weights(d.drop(["demo"]))
    assert np.allclose(w[:, mods.index("demo")], 0.0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_predictions_are_probabilities_and_beat_chance_on_separable_data():
    from sklearn.metrics import roc_auc_score
    d, held = _data(seed=0), _data(seed=1)
    g = AcquisitionGatedFusion("g").fit(d)
    p = g.predict(held)
    assert p.shape == (len(held),) and np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(held.y, p) > 0.7


def test_zoo_exposes_acq_as_a_fusion_without_disturbing_the_others():
    assert parse("lr@acq")[1] == "acq"
    assert parse("lr@late")[1] == "late"           # existing ids unchanged
    assert "[acq]" in label("lr@acq")
    m = create_model("lr@acq", img_components=4)
    assert isinstance(m, AcquisitionGatedFusion)
    ids = list_models(_data(), include_unavailable=True)
    assert {"lr", "lr@late", "lr@subset", "lr@acq"} <= set(ids)


def test_gate_settings_reach_the_model_through_create_model():
    m = create_model("lr@acq", gate_quality=0.0, gate_temperature=2.0)
    assert m.w_quality == 0.0 and m.temperature == 2.0


def test_runs_through_the_cv_harness():
    from medtabx.benchmarks.typed_bench import run_cv
    from medtabx.auto.zoo import ensure_registered
    ensure_registered(["lr@acq"], img_components=4)
    out = run_cv(_data(), ["lr@acq"], {"all": []}, n_splits=3, repeats=1,
                 log=lambda *_: None)
    assert len(out) == 3
    assert np.isfinite([r["auroc"] for r in out]).all()
