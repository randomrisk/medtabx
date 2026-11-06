"""
Inference entry point for MedTabX models.

Applies a trained run to a feature table and writes per-row predictions. The
target column may be absent; the training column statistics stored in the run
directory are reused so that inputs are normalized exactly as during training.

.. code-block:: bash

    medtabx-predict --run-dir runs/ads_stroke --csv new_cohort.csv \\
        --output predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

from medtabx.engine.experiment import load_experiment
from medtabx.metrics.functional import (
    normalize_binary_scores,
    normalize_multiclass_scores,
    predicted_labels,
)
from medtabx.typing import TaskType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a trained MedTabX model over a feature table.')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Run directory written by medtabx-train')
    parser.add_argument('--csv', type=str, required=True,
                        help='Feature table to score')
    parser.add_argument('--output', type=str, required=True,
                        help='Destination CSV for the predictions')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load (default: best.pt in the '
                             'run directory)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for binary classification')
    parser.add_argument('--device', type=str, default=None,
                        help='Override the device')
    return parser.parse_args(argv)


def _prediction_frame(task_type: TaskType, outputs: torch.Tensor,
                      threshold: float) -> pd.DataFrame:
    """Turn raw model output into a readable prediction table."""
    if task_type.is_regression:
        return pd.DataFrame({'prediction': outputs.reshape(-1).numpy()})

    if task_type == TaskType.BINARY_CLASSIFICATION:
        probability = normalize_binary_scores(outputs)
        return pd.DataFrame({
            'probability': probability.numpy(),
            'prediction': predicted_labels(outputs,
                                           threshold=threshold).numpy(),
        })

    probs = normalize_multiclass_scores(outputs)
    frame = pd.DataFrame(
        probs.numpy(),
        columns=[f'probability_{i}' for i in range(probs.size(1))])
    frame['prediction'] = probs.argmax(dim=1).numpy()
    return frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"error: no run directory at '{run_dir}'", file=sys.stderr)
        return 1
    if not Path(args.csv).exists():
        print(f"error: no feature table at '{args.csv}'", file=sys.stderr)
        return 1

    config, dataset, trainer = load_experiment(run_dir,
                                               checkpoint=args.checkpoint,
                                               csv=args.csv,
                                               device=args.device,
                                               require_target=False)

    outputs, _ = trainer.predict(dataset)
    frame = _prediction_frame(dataset.task_type, outputs, args.threshold)
    frame.insert(0, 'row', range(len(frame)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} predictions to {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
