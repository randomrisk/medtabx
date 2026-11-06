"""Utilities for connectome matrices and graph-style downstream features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _infer_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return "\t"
    return ","


def _coerce_matrix(matrix: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(matrix, np.ndarray):
        arr = np.asarray(matrix, dtype=float)
    else:
        arr = load_connectivity_matrix(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape {arr.shape}.")
    return arr


def _normalize_labels(labels: Sequence[str] | None, size: int) -> list[str]:
    if labels is None:
        return [f"node_{idx:03d}" for idx in range(size)]
    if len(labels) != size:
        raise ValueError(f"Expected {size} labels, got {len(labels)}.")
    return [str(label) for label in labels]


def load_connectivity_matrix(
    path: str | Path,
    delimiter: str | None = None,
    zero_diagonal: bool = True,
) -> np.ndarray:
    """Load a square connectome matrix from CSV/TSV text."""

    matrix_path = Path(path)
    arr = np.loadtxt(matrix_path, delimiter=delimiter or _infer_delimiter(matrix_path))
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = np.atleast_2d(arr)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape {arr.shape}.")
    if zero_diagonal:
        arr = arr.copy()
        np.fill_diagonal(arr, 0.0)
    return arr


def matrix_to_edge_table(
    matrix: str | Path | np.ndarray,
    labels: Sequence[str] | None = None,
    min_weight: float = 0.0,
    upper_triangle: bool = True,
) -> pd.DataFrame:
    """Convert an adjacency matrix to a tidy edge table."""

    arr = _coerce_matrix(matrix)
    names = _normalize_labels(labels, arr.shape[0])
    rows: list[dict[str, object]] = []

    for src_idx in range(arr.shape[0]):
        dst_start = src_idx + 1 if upper_triangle else 0
        for dst_idx in range(dst_start, arr.shape[1]):
            weight = float(arr[src_idx, dst_idx])
            if src_idx == dst_idx or weight <= min_weight:
                continue
            rows.append(
                {
                    "source_index": src_idx,
                    "source_label": names[src_idx],
                    "target_index": dst_idx,
                    "target_label": names[dst_idx],
                    "weight": weight,
                }
            )

    return pd.DataFrame(rows)


def compute_node_strengths(
    matrix: str | Path | np.ndarray,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute weighted node strengths from a connectivity matrix."""

    arr = _coerce_matrix(matrix)
    names = _normalize_labels(labels, arr.shape[0])
    strengths = arr.sum(axis=1)
    degrees = (arr > 0).sum(axis=1)
    return pd.DataFrame(
        {
            "node_index": np.arange(arr.shape[0], dtype=int),
            "node_label": names,
            "strength": strengths.astype(float),
            "degree": degrees.astype(int),
        }
    )


def summarize_connectivity_matrix(
    matrix: str | Path | np.ndarray,
    prefix: str = "connectome",
) -> dict[str, float | int | bool]:
    """Generate compact scalar features for downstream tabular models."""

    arr = _coerce_matrix(matrix)
    upper = arr[np.triu_indices_from(arr, k=1)]
    nonzero = upper[upper > 0]
    strengths = arr.sum(axis=1)
    degrees = (arr > 0).sum(axis=1)
    total_possible_edges = arr.shape[0] * (arr.shape[0] - 1) / 2
    density = float(nonzero.size / total_possible_edges) if total_possible_edges else 0.0

    summary: dict[str, float | int | bool] = {
        f"{prefix}_num_nodes": int(arr.shape[0]),
        f"{prefix}_num_edges": int(nonzero.size),
        f"{prefix}_density": density,
        f"{prefix}_total_weight": float(nonzero.sum()) if nonzero.size else 0.0,
        f"{prefix}_mean_edge_weight": float(nonzero.mean()) if nonzero.size else 0.0,
        f"{prefix}_max_edge_weight": float(nonzero.max()) if nonzero.size else 0.0,
        f"{prefix}_mean_node_strength": float(strengths.mean()) if strengths.size else 0.0,
        f"{prefix}_max_node_strength": float(strengths.max()) if strengths.size else 0.0,
        f"{prefix}_mean_degree": float(degrees.mean()) if degrees.size else 0.0,
        f"{prefix}_max_degree": int(degrees.max()) if degrees.size else 0,
        f"{prefix}_symmetric": bool(np.allclose(arr, arr.T)),
    }
    return summary


def export_graph_artifacts(
    matrix: str | Path | np.ndarray,
    output_dir: str | Path,
    *,
    labels: Sequence[str] | None = None,
    prefix: str = "connectome",
    min_weight: float = 0.0,
) -> dict[str, str]:
    """Write downstream-ready matrix, edge, node, and summary artifacts."""

    arr = _coerce_matrix(matrix)
    names = _normalize_labels(labels, arr.shape[0])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_df = pd.DataFrame(arr, index=names, columns=names)
    matrix_path = out_dir / f"{prefix}_matrix.tsv"
    edge_path = out_dir / f"{prefix}_edges.csv"
    node_path = out_dir / f"{prefix}_node_strengths.csv"
    summary_path = out_dir / f"{prefix}_summary.json"

    matrix_df.to_csv(matrix_path, sep="\t", index=True)
    matrix_to_edge_table(arr, labels=names, min_weight=min_weight).to_csv(edge_path, index=False)
    compute_node_strengths(arr, labels=names).to_csv(node_path, index=False)

    summary = summarize_connectivity_matrix(arr, prefix=prefix)
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    return {
        "matrix_tsv": str(matrix_path),
        "edges_csv": str(edge_path),
        "node_strengths_csv": str(node_path),
        "summary_json": str(summary_path),
    }
