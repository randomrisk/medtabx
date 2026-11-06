"""Feature table builders for curated ADS/OpenADS multimodal subjects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import nibabel as nib
import pandas as pd

from .perfusion import summarize_maps_with_mask_paths, summarize_perfusion_series


def _flatten_single_row_csv(path: Path, prefix: str) -> Dict[str, object]:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected a single-row feature CSV at {path}, got {len(df)} rows.")
    row = df.iloc[0].to_dict()
    flattened: Dict[str, object] = {}
    for key, value in row.items():
        if key == "subject_id":
            continue
        flattened[f"{prefix}_{key}"] = value
    return flattened


def build_ads_subject_feature_row(subject_root: str | Path) -> Dict[str, object]:
    """Build one mixed raw-plus-derived feature row from an ADS fixture subject."""
    subject_root = Path(subject_root)
    subject_id = subject_root.name

    pwi_path = subject_root / "imaging" / f"{subject_id}_PWI.nii.gz"
    dwi_path = subject_root / "imaging" / f"{subject_id}_DWI.nii.gz"
    pwi_json_path = subject_root / "metadata" / f"{subject_id}_PWI.json"
    brain_mask_path = subject_root / "metadata" / f"{subject_id}_brain_mask_from_pwi.nii.gz"
    stroke_mask_path = subject_root / "masks" / f"{subject_id}_stroke_mask.nii.gz"
    hp_mask_path = subject_root / "masks" / f"{subject_id}_hp_mask.nii.gz"

    map_paths = {
        "ttp": str(subject_root / "perfusion_maps" / f"{subject_id}_ttp.nii.gz"),
        "tmax": str(subject_root / "perfusion_maps" / f"{subject_id}_tmax.nii.gz"),
        "cbf": str(subject_root / "perfusion_maps" / f"{subject_id}_cbf.nii.gz"),
        "cbv": str(subject_root / "perfusion_maps" / f"{subject_id}_cbv.nii.gz"),
        "mtt": str(subject_root / "perfusion_maps" / f"{subject_id}_mtt.nii.gz"),
    }

    perf_summary = summarize_perfusion_series(str(pwi_path), str(pwi_json_path))
    brain_map_summary = summarize_maps_with_mask_paths(
        map_paths,
        str(brain_mask_path),
        prefix="brain",
    )

    dwi_qfv = _flatten_single_row_csv(
        subject_root / "tabular" / f"{subject_id}_VASCULAR_QFV.csv",
        prefix="dwi_vascular_qfv",
    )
    hp_qfv = _flatten_single_row_csv(
        subject_root / "tabular" / f"{subject_id}_Vascular_HPQFV.csv",
        prefix="pwi_hp_qfv",
    )

    dwi_report = (subject_root / "text" / f"{subject_id}_dwi_report.txt").read_text()
    pwi_report = (subject_root / "text" / f"{subject_id}_pwi_report.txt").read_text()
    pwi_meta = json.loads(pwi_json_path.read_text())

    stroke_mask = nib.load(str(stroke_mask_path)).get_fdata()
    hp_mask = nib.load(str(hp_mask_path)).get_fdata()

    row: Dict[str, object] = {
        "subject_id": subject_id,
        "dwi_path": str(dwi_path),
        "pwi_path": str(pwi_path),
        "stroke_mask_path": str(stroke_mask_path),
        "hp_mask_path": str(hp_mask_path),
        "dwi_report_text": dwi_report,
        "pwi_report_text": pwi_report,
        "pwi_repetition_time": float(pwi_meta.get("RepetitionTime", 1.5)),
        "pwi_echo_time": float(pwi_meta.get("EchoTime", 0.03)),
        "pwi_num_timepoints": int(perf_summary["num_timepoints"]),
        "pwi_peak_timepoint_index": int(perf_summary["peak_timepoint_index"]),
        "stroke_mask_voxels": int((stroke_mask > 0.5).sum()),
        "hp_mask_voxels": int((hp_mask > 0.5).sum()),
    }
    row.update(brain_map_summary)
    row.update(dwi_qfv)
    row.update(hp_qfv)
    return row


def build_ads_feature_table(subject_roots: list[str | Path]) -> pd.DataFrame:
    """Build a cohort-style feature table from multiple ADS fixture subjects."""
    rows = [build_ads_subject_feature_row(subject_root) for subject_root in subject_roots]
    return pd.DataFrame(rows)


def discover_ads_subject_roots(dataset_root: str | Path) -> list[Path]:
    """Discover ADS subject directories under a dataset root.

    A valid subject directory currently means:
    - directory name starts with ``sub-``
    - contains a ``manifest.json`` file
    """
    dataset_root = Path(dataset_root)
    subject_roots: list[Path] = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and child.name.startswith("sub-") and (child / "manifest.json").exists():
            subject_roots.append(child)
    return subject_roots


def build_ads_feature_table_from_root(dataset_root: str | Path) -> pd.DataFrame:
    """Discover ADS subjects under ``dataset_root`` and build a merged feature table."""
    subject_roots = discover_ads_subject_roots(dataset_root)
    if not subject_roots:
        raise ValueError(f"No ADS subject directories found under {dataset_root}.")
    return build_ads_feature_table(subject_roots)
