"""Tests for tree-based baseline adapters."""

from pathlib import Path

import numpy as np

from medtabx.baselines import available_tree_backends, create_tree_model


def test_hist_gbdt_classifier_and_persistence(tmp_path: Path):
    x = np.array([[0.0, 0.0], [0.2, 0.1], [0.8, 0.9], [1.0, 1.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    model = create_tree_model("hist_gbdt_classifier", max_depth=3, random_state=0)
    model.fit(x, y)
    preds = model.predict(x)
    probs = model.predict_proba(x)

    assert preds.shape == (4,)
    assert probs.shape == (4, 2)

    path = tmp_path / "hist_gbdt.joblib"
    model.save(str(path))
    assert path.exists()


def test_hist_gbdt_regressor():
    x = np.array([[0.0], [0.5], [1.0], [1.5]], dtype=np.float32)
    y = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)

    model = create_tree_model("hist_gbdt_regressor", max_depth=3, random_state=0)
    model.fit(x, y)
    preds = model.predict(x)

    assert preds.shape == (4,)
    assert available_tree_backends()["hist_gbdt_classifier"] is True
