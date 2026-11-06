"""
Evaluation entry point for MedTabX models.

Reloads a finished run and scores one of its splits, or a new feature table
that carries the same columns.

.. code-block:: bash

    medtabx-evaluate --run-dir runs/ads_stroke --split test
    medtabx-evaluate --run-dir runs/ads_stroke --csv external_cohort.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medtabx.engine.experiment import load_experiment
from medtabx.metrics import bootstrap_ci
from medtabx.metrics.functional import (
    binary_auroc,
    binary_average_precision,
    normalize_binary_scores,
    r2_score,
)
from medtabx.typing import TaskType


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a trained MedTabX model.')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Run directory written by medtabx-train')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to load (default: best.pt in the '
                             'run directory)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to score')
    parser.add_argument('--csv', type=str, default=None,
                        help='Score this feature table instead of the one '
                             'the run was trained on; implies --split all')
    parser.add_argument('--bootstrap', type=int, default=0,
                        help='Bootstrap resamples for confidence intervals '
                             '(0 disables)')
    parser.add_argument('--confidence-level', type=float, default=0.95,
                        help='Nominal coverage of the intervals')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the bootstrap generator')
    parser.add_argument('--device', type=str, default=None,
                        help='Override the device')
    parser.add_argument('--output', type=str, default=None,
                        help='Write the metrics to this JSON file')
    return parser.parse_args(argv)


def _interval_metrics(task_type: TaskType):
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return {'auroc': binary_auroc, 'auprc': binary_average_precision}
    if task_type.is_regression:
        return {'r2': r2_score}
    return {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"error: no run directory at '{run_dir}'", file=sys.stderr)
        return 1

    config, dataset, trainer = load_experiment(run_dir,
                                               checkpoint=args.checkpoint,
                                               csv=args.csv,
                                               device=args.device)

    split = 'all' if args.csv is not None else args.split
    data = dataset if split == 'all' else dataset.get_split(split)
    if len(data) == 0:
        print(f"error: the '{split}' split is empty", file=sys.stderr)
        return 1

    results = {'run': config.name, 'split': split, 'num_rows': len(data),
               'metrics': trainer.evaluate(data)}

    print(f"{config.name} / {split} (n={len(data)})")
    for key, value in results['metrics'].items():
        print(f"  {key:20s} {value:.4f}")

    if args.bootstrap > 0:
        outputs, targets = trainer.predict(data)
        metric_fns = _interval_metrics(dataset.task_type)
        if not metric_fns:
            print("  (confidence intervals are not reported for this task "
                  "type)")
        intervals = {}
        for name, fn in metric_fns.items():
            scores = (normalize_binary_scores(outputs)
                      if dataset.task_type == TaskType.BINARY_CLASSIFICATION
                      else outputs.reshape(-1))
            result = bootstrap_ci(fn, scores, targets,
                                  num_resamples=args.bootstrap,
                                  confidence_level=args.confidence_level,
                                  seed=args.seed)
            intervals[name] = result.as_dict()
            print(f"  {name:20s} {result}")
        results['intervals'] = intervals

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
