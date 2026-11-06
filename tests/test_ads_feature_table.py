"""Tests for ADS feature table builders."""

from __future__ import annotations

from pathlib import Path


def test_ads_subject_feature_row_and_table():
    from medtabx.features.ads_features import (
        build_ads_feature_table,
        build_ads_subject_feature_row,
    )

    subject_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    row = build_ads_subject_feature_row(subject_root)
    table = build_ads_feature_table([subject_root])

    assert row["subject_id"] == "sub-02e8eb42"
    assert "dwi_path" in row
    assert "pwi_report_text" in row
    assert "brain_ttp_mean" in row
    assert "dwi_vascular_qfv_volml" in row
    assert "pwi_hp_qfv_volml" in row
    assert table.shape[0] == 1
    assert table.iloc[0]["subject_id"] == "sub-02e8eb42"
