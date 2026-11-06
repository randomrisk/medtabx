"""
Training entry point for MedTabX models.

A run is described by a JSON experiment config; the flags below override
individual fields so that sweeps do not need one file per variant.

.. code-block:: bash

    medtabx-train --config experiment.json
    medtabx-train --config experiment.json --epochs 100 --lr 3e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medtabx.engine.experiment import ExperimentConfig, run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a MedTabX model from an experiment config.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON experiment config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override the output directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Override the run name')
    parser.add_argument('--model', type=str, default=None,
                        help='Override the model class name')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override the learning rate')
    parser.add_argument('--monitor', type=str, default=None,
                        help='Override the metric used for model selection')
    parser.add_argument('--patience', type=int, default=None,
                        help='Override the early-stopping patience')
    parser.add_argument('--device', type=str, default=None,
                        help='Override the device, e.g. cpu or cuda:0')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the training seed')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress the per-epoch log')
    return parser.parse_args(argv)


def apply_overrides(config: ExperimentConfig,
                    args: argparse.Namespace) -> ExperimentConfig:
    """Apply the command-line overrides onto a loaded config."""
    for attr in ('name', 'model', 'output_dir'):
        value = getattr(args, attr)
        if value is not None:
            setattr(config, attr, value)

    trainer_overrides = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'monitor': args.monitor,
        'early_stopping_patience': args.patience,
        'device': args.device,
        'seed': args.seed,
    }
    for key, value in trainer_overrides.items():
        if value is not None:
            config.trainer[key] = value
    if args.quiet:
        config.trainer['verbose'] = False
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"error: no experiment config at '{config_path}'",
              file=sys.stderr)
        return 1

    config = apply_overrides(ExperimentConfig.from_json(config_path), args)
    results = run_experiment(config, output_dir=args.output_dir)

    print("\nTest metrics:")
    for key, value in results['test'].items():
        print(f"  {key:20s} {value:.4f}")
    for metric, interval in results.get('test_intervals', {}).items():
        level = int(round(interval['confidence_level'] * 100))
        print(f"  {metric:20s} {interval['point']:.4f} "
              f"[{level}% CI {interval['lower']:.4f}-{interval['upper']:.4f}]")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
