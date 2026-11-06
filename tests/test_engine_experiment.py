"""End-to-end tests for the config-driven experiment runner and CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from medtabx.engine.experiment import (
    ExperimentConfig,
    build_model,
    load_experiment,
    run_experiment,
)


def _write_cohort(path, num_rows: int = 300, seed: int = 0,
                  with_target: bool = True):
    rng = np.random.default_rng(seed)
    age = rng.normal(66, 10, num_rows)
    severity = rng.normal(8, 3, num_rows)
    logit = 0.08 * (age - 66) + 0.45 * (severity - 8)
    frame = {"age": age, "severity": severity}
    if with_target:
        frame["outcome"] = (rng.random(num_rows) <
                            1 / (1 + np.exp(-logit))).astype(int)
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def _config(tmp_path, **overrides) -> ExperimentConfig:
    csv = _write_cohort(tmp_path / "cohort.csv")
    payload = {
        "name": "unit_run",
        "csv": str(csv),
        "target_col": "outcome",
        "col_to_stype": {
            "age": "numerical",
            "severity": "numerical",
            "outcome": "categorical",
        },
        "model": "MLP",
        "model_kwargs": {"channels": 16, "num_layers": 2},
        "trainer": {"epochs": 5, "lr": 1e-2, "seed": 0, "verbose": False},
        "bootstrap_resamples": 50,
        "output_dir": str(tmp_path / "run"),
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)


def test_run_experiment_writes_every_artifact(tmp_path):
    config = _config(tmp_path)
    results = run_experiment(config)

    run_dir = tmp_path / "run"
    for filename in ("config.json", "history.csv", "metrics.json",
                     "col_stats.pt", "best.pt"):
        assert (run_dir / filename).exists(), filename

    assert results["task_type"] == "binary_classification"
    assert sum(results["split_sizes"].values()) == 300
    assert {"train", "val", "test"} <= set(results)
    assert "auroc" in results["test"]
    assert results["test_intervals"]["auroc"]["lower"] <= \
        results["test_intervals"]["auroc"]["point"]

    written = json.loads((run_dir / "metrics.json").read_text())
    assert written["name"] == "unit_run"

    history = pd.read_csv(run_dir / "history.csv")
    assert len(history) == 5
    assert "val_auroc" in history.columns


def test_run_experiment_is_reproducible_under_a_seed(tmp_path):
    first = run_experiment(_config(tmp_path, output_dir=str(tmp_path / "a")))
    second = run_experiment(_config(tmp_path, output_dir=str(tmp_path / "b")))
    assert first["test"]["auroc"] == pytest.approx(second["test"]["auroc"])


def test_load_experiment_reproduces_the_recorded_test_metrics(tmp_path):
    config = _config(tmp_path)
    results = run_experiment(config)

    _, dataset, trainer = load_experiment(tmp_path / "run")
    reloaded = trainer.evaluate(dataset.get_split("test"))

    # Reusing the stored column statistics is what makes this hold; if they
    # were recomputed the inputs would be normalized differently.
    assert reloaded["auroc"] == pytest.approx(results["test"]["auroc"])
    assert reloaded["loss"] == pytest.approx(results["test"]["loss"])


def test_load_experiment_scores_an_unlabelled_cohort(tmp_path):
    run_experiment(_config(tmp_path))
    new_csv = _write_cohort(tmp_path / "new.csv", num_rows=40, seed=5,
                            with_target=False)

    _, dataset, trainer = load_experiment(tmp_path / "run", csv=new_csv,
                                          require_target=False)
    outputs, _ = trainer.predict(dataset)
    assert outputs.shape[0] == 40


def test_regression_run_reports_regression_metrics(tmp_path):
    rng = np.random.default_rng(0)
    num_rows = 300
    x1, x2 = rng.normal(size=num_rows), rng.normal(size=num_rows)
    csv = tmp_path / "reg.csv"
    pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "ttp": 2 * x1 - 1.5 * x2 + rng.normal(0, 0.2, num_rows),
    }).to_csv(csv, index=False)

    config = ExperimentConfig(
        name="reg", csv=str(csv), target_col="ttp",
        col_to_stype={"x1": "numerical", "x2": "numerical",
                      "ttp": "numerical"},
        model="MLP", model_kwargs={"channels": 16, "num_layers": 2},
        trainer={"epochs": 20, "lr": 1e-2, "monitor": "rmse", "seed": 0,
                 "verbose": False},
        bootstrap_resamples=50, output_dir=str(tmp_path / "reg_run"))

    results = run_experiment(config)
    assert results["task_type"] == "regression"
    assert set(results["test"]) == {"loss", "mae", "rmse", "r2"}
    assert results["test"]["r2"] > 0.8
    assert "r2" in results["test_intervals"]


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "csv": "x.csv", "target_col": "y", "col_to_stype": {},
        "epoch_count": 10,
    }))
    with pytest.raises(ValueError, match="Unknown keys"):
        ExperimentConfig.from_json(path)


def test_config_rejects_unknown_trainer_keys(tmp_path):
    config = _config(tmp_path, trainer={"epochs": 1, "learning_rate": 0.1})
    with pytest.raises(ValueError, match="Unknown trainer keys"):
        config.trainer_config()


def test_build_model_refuses_the_multimodal_baseline(tmp_path):
    from medtabx.engine.experiment import build_dataset

    dataset = build_dataset(_config(tmp_path))
    with pytest.raises(ValueError, match="MultimodalBaseline"):
        build_model("MultimodalBaseline", dataset)


def test_train_cli_runs_a_config(tmp_path, capsys):
    from medtabx.cli.train import main

    config = _config(tmp_path)
    config_path = tmp_path / "experiment.json"
    config.to_json(config_path)

    exit_code = main(["--config", str(config_path), "--epochs", "3",
                      "--quiet", "--output-dir", str(tmp_path / "cli_run")])
    assert exit_code == 0
    assert "Test metrics:" in capsys.readouterr().out
    assert (tmp_path / "cli_run" / "metrics.json").exists()


def test_evaluate_cli_scores_a_finished_run(tmp_path, capsys):
    from medtabx.cli.evaluate import main

    run_experiment(_config(tmp_path))
    exit_code = main(["--run-dir", str(tmp_path / "run"), "--split", "test",
                      "--bootstrap", "50", "--output",
                      str(tmp_path / "eval.json")])
    assert exit_code == 0
    assert "auroc" in capsys.readouterr().out

    payload = json.loads((tmp_path / "eval.json").read_text())
    assert payload["split"] == "test"
    assert "intervals" in payload


def test_predict_cli_writes_per_row_predictions(tmp_path):
    from medtabx.cli.inference import main

    run_experiment(_config(tmp_path))
    new_csv = _write_cohort(tmp_path / "new.csv", num_rows=25, seed=3,
                            with_target=False)
    output = tmp_path / "preds.csv"

    exit_code = main(["--run-dir", str(tmp_path / "run"), "--csv",
                      str(new_csv), "--output", str(output)])
    assert exit_code == 0

    predictions = pd.read_csv(output)
    assert len(predictions) == 25
    assert list(predictions.columns) == ["row", "probability", "prediction"]
    assert predictions["probability"].between(0, 1).all()


def test_cli_reports_a_missing_run_directory(tmp_path, capsys):
    from medtabx.cli.evaluate import main

    assert main(["--run-dir", str(tmp_path / "nope")]) == 1
    assert "no run directory" in capsys.readouterr().err
