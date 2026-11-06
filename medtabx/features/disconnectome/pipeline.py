"""Python helpers and CLI for connectome pipeline outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .graph_metrics import export_graph_artifacts


def _read_labels_file(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None

    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                labels.append(line.split("\t")[-1].strip())
            elif "," in line:
                labels.append(line.split(",")[-1].strip())
            else:
                labels.append(line)
    return labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process connectome matrices into downstream graph artifacts."
    )
    parser.add_argument("--matrix", required=True, help="Path to CSV/TSV connectivity matrix.")
    parser.add_argument("--output-dir", required=True, help="Directory to write graph artifacts.")
    parser.add_argument("--labels-file", help="Optional node labels file (one label per line).")
    parser.add_argument("--prefix", default="connectome", help="Artifact filename prefix.")
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Minimum edge weight to keep in the exported edge list.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    labels = _read_labels_file(args.labels_file)
    export_graph_artifacts(
        args.matrix,
        args.output_dir,
        labels=labels,
        prefix=args.prefix,
        min_weight=args.min_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
