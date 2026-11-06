"""Example: train, evaluate and score a model with the MedTabX engine.

Runs the full supervised loop on a small synthetic stroke-outcome cohort, so
it needs no external data and finishes in seconds. The same code path works
on a real feature table -- swap the synthetic frame for
``pd.read_csv(...)``.

The equivalent command-line flow, once an experiment config exists, is::

    medtabx-train    --config experiment.json
    medtabx-evaluate --run-dir runs/synthetic_stroke --split test --bootstrap 1000
    medtabx-predict  --run-dir runs/synthetic_stroke --csv new_cohort.csv \
        --output predictions.csv
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medtabx import stype
from medtabx.data import Dataset
from medtabx.engine import Trainer, TrainerConfig
from medtabx.metrics import bootstrap_ci
from medtabx.metrics.functional import (
    binary_auroc,
    binary_average_precision,
    normalize_binary_scores,
)
from medtabx.models import FTTransformer
from medtabx.utils.split import generate_random_split


def synthetic_cohort(num_subjects: int = 800, seed: int = 0) -> pd.DataFrame:
    """A toy cohort with an outcome that depends on age, severity and sex."""
    rng = np.random.default_rng(seed)
    age = rng.normal(68, 11, num_subjects)
    nihss = rng.normal(9, 4, num_subjects)
    sex = rng.integers(0, 2, num_subjects)

    logit = 0.04 * (age - 68) + 0.30 * (nihss - 9) + 0.7 * sex - 0.4
    outcome = (rng.random(num_subjects) <
               1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "age": age,
        "nihss": nihss,
        "sex": pd.Categorical(sex),
        "outcome": pd.Categorical(outcome),
    })
    df["split"] = generate_random_split(num_subjects, seed=seed,
                                        train_ratio=0.7, val_ratio=0.15)
    return df


def main() -> None:
    print("=" * 70)
    print("MedTabX training engine")
    print("=" * 70)

    dataset = Dataset(
        synthetic_cohort(),
        col_to_stype={
            "age": stype.numerical,
            "nihss": stype.numerical,
            "sex": stype.categorical,
            "outcome": stype.categorical,
        },
        target_col="outcome",
        split_col="split",
    ).materialize()

    train_set, val_set, test_set = dataset.split()
    print(f"\ntask: {dataset.task_type.value}")
    print(f"splits: train={len(train_set)} val={len(val_set)} "
          f"test={len(test_set)}")

    with tempfile.TemporaryDirectory() as run_dir:
        model = FTTransformer(
            channels=64,
            out_channels=dataset.num_classes,
            num_layers=2,
            col_stats=dataset.col_stats,
            col_names_dict=dataset.tensor_frame.col_names_dict,
        )

        trainer = Trainer(
            model,
            task_type=dataset.task_type,
            config=TrainerConfig(
                epochs=40,
                batch_size=64,
                lr=5e-3,
                # Select on discrimination rather than on the loss.
                monitor="auroc",
                scheduler="cosine",
                early_stopping_patience=10,
                checkpoint_path=str(Path(run_dir) / "best.pt"),
                seed=0,
                log_every=5,
            ),
        )

        print("\ntraining")
        history = trainer.fit(train_set, val_set)
        print(f"\nbest epoch {trainer.best_epoch} of {len(history)} "
              f"(val auroc {trainer.best_value:.4f})")

        print("\ntest metrics")
        for name, value in trainer.evaluate(test_set).items():
            print(f"  {name:20s} {value:.4f}")

        # A point estimate on a few hundred subjects is not worth reporting
        # on its own, so attach an interval to the headline metrics.
        outputs, targets = trainer.predict(test_set)
        probabilities = normalize_binary_scores(outputs)
        print("\ntest metrics with 95% bootstrap intervals")
        for name, fn in (("auroc", binary_auroc),
                         ("auprc", binary_average_precision)):
            result = bootstrap_ci(fn, probabilities, targets,
                                  num_resamples=1000, seed=0)
            print(f"  {name:20s} {result}")

    print("\n" + "=" * 70)
    print("done")
    print("=" * 70)


if __name__ == "__main__":
    main()
