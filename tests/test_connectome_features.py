"""Tests for connectome/disconnectome feature helpers."""

from __future__ import annotations

import json

import numpy as np

from medtabx.features import (
    export_graph_artifacts,
    get_feature_extractor,
    load_connectivity_matrix,
    matrix_to_edge_table,
    run_feature_extractor,
    summarize_connectivity_matrix,
)


def test_connectome_helpers_and_registry(tmp_path):
    matrix = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 3.5],
            [0.0, 3.5, 0.0],
        ]
    )
    matrix_path = tmp_path / "connectome.csv"
    np.savetxt(matrix_path, matrix, delimiter=",")

    loaded = load_connectivity_matrix(matrix_path)
    assert loaded.shape == (3, 3)
    assert np.allclose(loaded, matrix)

    edges = matrix_to_edge_table(loaded, labels=["A", "B", "C"])
    assert list(edges["source_label"]) == ["A", "B"]
    assert list(edges["target_label"]) == ["B", "C"]

    summary = summarize_connectivity_matrix(loaded, prefix="demo")
    assert summary["demo_num_nodes"] == 3
    assert summary["demo_num_edges"] == 2
    assert summary["demo_symmetric"] is True

    artifacts = export_graph_artifacts(loaded, tmp_path / "graph", labels=["A", "B", "C"], prefix="demo")
    assert (tmp_path / "graph" / "demo_matrix.tsv").exists()
    assert (tmp_path / "graph" / "demo_edges.csv").exists()
    assert (tmp_path / "graph" / "demo_node_strengths.csv").exists()
    assert (tmp_path / "graph" / "demo_summary.json").exists()
    assert artifacts["matrix_tsv"].endswith("demo_matrix.tsv")

    summary_json = json.loads((tmp_path / "graph" / "demo_summary.json").read_text())
    assert summary_json["demo_total_weight"] == 5.5

    extractor = get_feature_extractor("connectome_summary")
    assert extractor is not None
    registry_summary = run_feature_extractor("connectome_summary", matrix_path, prefix="registry")
    assert registry_summary["registry_num_edges"] == 2
