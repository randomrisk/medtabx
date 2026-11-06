"""Tests for ADS feature-table CLI workflows."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_ads_feature_discovery_and_cli(tmp_path):
    from medtabx.features.ads_features import (
        build_ads_feature_table_from_root,
        discover_ads_subject_roots,
    )

    dataset_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case"
    subject_roots = discover_ads_subject_roots(dataset_root)
    assert len(subject_roots) == 1

    df = build_ads_feature_table_from_root(dataset_root)
    assert len(df) == 1

    output_csv = tmp_path / "ads_features.csv"
    cmd = [
        sys.executable,
        "medtabx/cli/build_ads_features.py",
        "--root",
        str(dataset_root),
        "--output",
        str(output_csv),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, check=True, capture_output=True, text=True)
    assert output_csv.exists()
    assert "subjects=1" in result.stdout
