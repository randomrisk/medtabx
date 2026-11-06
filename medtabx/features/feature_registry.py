from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from medtabx.core.registry import feature_extractor_registry

from .ads_features import (
    build_ads_feature_table,
    build_ads_feature_table_from_root,
    build_ads_subject_feature_row,
)
from .disconnectome import (
    export_graph_artifacts,
    load_connectivity_matrix,
    summarize_connectivity_matrix,
)
from .lesion import (
    calculate_openads_hyperperfusion_features,
    calculate_openads_stroke_features,
)
from .perfusion import (
    extract_roi_time_series,
    summarize_maps_with_mask_paths,
    summarize_perfusion_series,
    summarize_roi_time_series,
)


@feature_extractor_registry.register("ads_subject_feature_row")
def extract_ads_subject_feature_row(subject_root: str | Path):
    return build_ads_subject_feature_row(subject_root)


@feature_extractor_registry.register("ads_feature_table")
def extract_ads_feature_table(subject_roots):
    return build_ads_feature_table(subject_roots)


@feature_extractor_registry.register("ads_feature_table_from_root")
def extract_ads_feature_table_from_root(dataset_root: str | Path):
    return build_ads_feature_table_from_root(dataset_root)


@feature_extractor_registry.register("perfusion_series_summary")
def extract_perfusion_series_summary(perf_path: str, sidecar_path: str | None = None):
    return summarize_perfusion_series(perf_path, sidecar_path)


@feature_extractor_registry.register("perfusion_mask_summary")
def extract_perfusion_mask_summary(map_paths, mask_path: str, prefix: str = "lesion"):
    return summarize_maps_with_mask_paths(map_paths, mask_path, prefix=prefix)


@feature_extractor_registry.register("roi_time_series")
def extract_roi_series(perf_4d, atlas_3d, min_label: int = 1):
    return extract_roi_time_series(perf_4d, atlas_3d, min_label=min_label)


@feature_extractor_registry.register("roi_time_series_summary")
def extract_roi_series_summary(roi_series):
    return summarize_roi_time_series(roi_series)


@feature_extractor_registry.register("stroke_qfv")
def extract_stroke_qfv(mask_path: str, atlas_dir: str | None = None):
    return calculate_openads_stroke_features(mask_path, atlas_dir=atlas_dir)


@feature_extractor_registry.register("hyperperfusion_qfv")
def extract_hyperperfusion_qfv(mask_path: str, atlas_dir: str | None = None):
    return calculate_openads_hyperperfusion_features(mask_path, atlas_dir=atlas_dir)


@feature_extractor_registry.register("connectome_matrix")
def extract_connectome_matrix(matrix_path: str | Path):
    return load_connectivity_matrix(matrix_path)


@feature_extractor_registry.register("connectome_summary")
def extract_connectome_summary(matrix_path: str | Path, prefix: str = "connectome"):
    return summarize_connectivity_matrix(matrix_path, prefix=prefix)


@feature_extractor_registry.register("connectome_graph_artifacts")
def extract_connectome_graph_artifacts(
    matrix_path: str | Path,
    output_dir: str | Path,
    prefix: str = "connectome",
):
    return export_graph_artifacts(matrix_path, output_dir, prefix=prefix)


def get_feature_extractor(name: str) -> Callable[..., Any]:
    return feature_extractor_registry.get(name)


def run_feature_extractor(name: str, *args, **kwargs):
    extractor = get_feature_extractor(name)
    return extractor(*args, **kwargs)
