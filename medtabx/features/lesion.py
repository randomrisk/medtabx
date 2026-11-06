"""Lesion feature extraction for stroke and hyperperfusion masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np

from .openads_bridge import ensure_openads_on_path, get_openads_atlas_dir


def calculate_lesion_qfv(
    lesion_path: str,
    vascular_atlas_path: str,
    volume_table_path: str,
    watershed_atlas_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Calculate a compact vascular QFV vector from a lesion mask."""
    import pandas as pd

    lesion_img = nib.load(lesion_path)
    lesion = nib.as_closest_canonical(lesion_img).get_fdata().squeeze()
    atlas = nib.as_closest_canonical(nib.load(vascular_atlas_path)).get_fdata().squeeze()
    volumes = pd.read_pickle(volume_table_path).T.reset_index()[0].values

    voxel_ml = np.prod(lesion_img.header.get_zooms()[:3]) / 1000.0
    lesion_ml = np.sum(lesion > 0.5) * voxel_ml
    lesion_logml = np.log10((lesion_ml + 1.0) / 1000.0)

    max_label = int(atlas.max())
    counts = np.zeros(max_label, dtype=int)
    for i in range(max_label):
        roi_mask = atlas == (i + 1)
        counts[i] = int(np.sum(lesion[roi_mask] > 0.5))

    qfv = np.zeros(16, dtype=int)
    qfv[0] = counts[0] + counts[2]
    qfv[1] = counts[1] + counts[3]
    qfv[2] = np.sum(counts[6:16:2])
    qfv[3] = np.sum(counts[7:16:2])
    qfv[4] = counts[16] + counts[18]
    qfv[5] = counts[17] + counts[19]
    qfv[6] = counts[26] + counts[28]
    qfv[7] = counts[27] + counts[29]
    qfv[8] = counts[24]
    qfv[9] = counts[25]
    qfv[10] = counts[4]
    qfv[11] = counts[5]
    qfv[12] = counts[20] + counts[22]
    qfv[13] = counts[21] + counts[23]
    qfv[14] = counts[30]
    qfv[15] = counts[31]

    qfv_vols = np.zeros(16, dtype=float)
    qfv_vols[0] = volumes[0] + volumes[2]
    qfv_vols[1] = volumes[1] + volumes[3]
    qfv_vols[2] = np.sum(volumes[6:16:2])
    qfv_vols[3] = np.sum(volumes[7:16:2])
    qfv_vols[4] = volumes[16] + volumes[18]
    qfv_vols[5] = volumes[17] + volumes[19]
    qfv_vols[6] = volumes[26] + volumes[28]
    qfv_vols[7] = volumes[27] + volumes[29]
    qfv_vols[8] = volumes[24]
    qfv_vols[9] = volumes[25]
    qfv_vols[10] = volumes[4]
    qfv_vols[11] = volumes[5]
    qfv_vols[12] = volumes[20] + volumes[22]
    qfv_vols[13] = volumes[21] + volumes[23]
    qfv_vols[14] = volumes[30]
    qfv_vols[15] = volumes[31]

    qfv_probs = np.round(qfv / (qfv_vols + 1e-6), 4)
    qfv_features = np.concatenate([[lesion_logml], qfv_probs])

    return {"qfv_features": qfv_features, "lesion_volume_ml": lesion_ml}


def calculate_openads_stroke_features(
    stroke_img_path: str,
    mask_raw_mni_path: str,
    adc_mni_path: str,
    atlas_dir: Optional[str] = None,
    adc_threshold: float = 0.5490,
    precision: int = 7,
) -> Dict[str, Any]:
    """Calculate atlas-backed lesion and QFV features with OpenADS.

    OpenADS is downloaded on first use; see ``medtabx.features.openads_bridge``.
    """
    ensure_openads_on_path()
    from ads.pipelines.reporting_qfv_calculation import StrokeQFVCalculator

    template_dir = str(Path(atlas_dir) if atlas_dir is not None else get_openads_atlas_dir())
    calculator = StrokeQFVCalculator(template_dir=template_dir)
    results = calculator.calculate(
        stroke_img_path=stroke_img_path,
        mask_raw_mni_path=mask_raw_mni_path,
        adc_mni_path=adc_mni_path,
        adc_threshold=adc_threshold,
        precision=precision,
    )
    return {
        "atlas_dir": template_dir,
        "feature_vectors": results["FV"],
        "probability_vectors": results["probability_vectors"],
        "qfv_features": results["qfv_features"],
        "stroke_logml": results["stroke_logml"],
    }


def calculate_openads_hyperperfusion_features(
    hp_img_path: str,
    mask_raw_mni_path: str,
    adc_mni_path: str,
    atlas_dir: Optional[str] = None,
    adc_threshold: float = 0.5490,
    precision: int = 7,
) -> Dict[str, Any]:
    """Reuse the lesion feature stack for hyperperfusion masks.

    OpenADS uses the same atlas/QFV machinery for lesion-style masks. This
    wrapper keeps the MedTabX API explicit about the intended mask semantics.
    """
    return calculate_openads_stroke_features(
        stroke_img_path=hp_img_path,
        mask_raw_mni_path=mask_raw_mni_path,
        adc_mni_path=adc_mni_path,
        atlas_dir=atlas_dir,
        adc_threshold=adc_threshold,
        precision=precision,
    )


def save_openads_feature_tables(
    results: Dict[str, Any],
    subject_id: str,
    output_dir: str,
    atlas_dir: Optional[str] = None,
    mode: str = "stroke",
) -> None:
    """Save OpenADS feature outputs to CSV tables.

    Args:
        results: Output from ``calculate_openads_stroke_features`` or
            ``calculate_openads_hyperperfusion_features``.
        subject_id: Subject identifier used in output filenames.
        output_dir: Destination directory.
        atlas_dir: Optional atlas bundle directory.
        mode: Either ``stroke`` or ``hp``. This controls the file stem suffixes
            used by OpenADS conventions.
    """
    ensure_openads_on_path()
    from ads.pipelines.reporting_qfv_calculation import StrokeQFVCalculator

    template_dir = str(Path(atlas_dir) if atlas_dir is not None else get_openads_atlas_dir())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    calculator = StrokeQFVCalculator(template_dir=template_dir)

    qfv_suffix = "QFV"
    lesionload_suffix = "lesionload"
    qfv_stem_map = None
    lesionload_stem_map = None
    if mode.lower() == "hp":
        qfv_suffix = "HPQFV"
        lesionload_suffix = "HPload"
        qfv_stem_map = {
            "Vascular": "Vascular",
            "Lobe": "Lobe",
            "Aspects": "ASPECTS",
            "AspectsPC": "ASPECTSPC",
            "BPM": "BPM_TYPE1",
        }
        lesionload_stem_map = {
            "vascular": "Vascular",
            "lobe": "Lobe",
            "aspects": "ASPECTS",
            "aspectpc": "ASPECTSPC",
            "bpm_type1": "BPM_TYPE1",
            "Ventricles": "VENTRICLES",
        }

    calculator.save_QFV_to_csv(
        {
            "FV": results["feature_vectors"],
            "qfv_features": results["qfv_features"],
            "probability_vectors": results["probability_vectors"],
        },
        subject_id=subject_id,
        output_dir=output_dir,
        qfv_suffix=qfv_suffix,
        lesionload_suffix=lesionload_suffix,
        qfv_stem_map=qfv_stem_map,
        lesionload_stem_map=lesionload_stem_map,
    )
