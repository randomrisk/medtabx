"""Manual feature extraction module."""
from .tabular import extract_tabular_features
from .imaging import extract_imaging_features
from .timeseries import extract_timeseries_features
from .atlas import load_openads_feature_metadata
from .harmonization import (
    build_harmonized_subject_view,
    harmonize_subject_to_reference,
    load_subject_manifest,
)
from .disconnectome import (
    compute_node_strengths,
    export_graph_artifacts,
    load_connectivity_matrix,
    matrix_to_edge_table,
    summarize_connectivity_matrix,
)
from .ads_features import (
    build_ads_feature_table,
    build_ads_feature_table_from_root,
    build_ads_subject_feature_row,
    discover_ads_subject_roots,
)
from .feature_registry import get_feature_extractor, run_feature_extractor
from .perfusion import (
    extract_roi_time_series,
    resample_to_reference,
    summarize_maps_with_mask_paths,
    summarize_perfusion_series,
    resample_mask_to_map,
    summarize_maps_with_mask,
    summarize_roi_time_series,
)
from .lesion import (
    calculate_lesion_qfv,
    calculate_openads_hyperperfusion_features,
    calculate_openads_stroke_features,
    save_openads_feature_tables,
)

__all__ = [
    'extract_tabular_features',
    'extract_imaging_features',
    'extract_timeseries_features',
    'build_ads_subject_feature_row',
    'build_ads_feature_table',
    'discover_ads_subject_roots',
    'build_ads_feature_table_from_root',
    'load_connectivity_matrix',
    'matrix_to_edge_table',
    'compute_node_strengths',
    'summarize_connectivity_matrix',
    'export_graph_artifacts',
    'get_feature_extractor',
    'run_feature_extractor',
    'load_openads_feature_metadata',
    'load_subject_manifest',
    'build_harmonized_subject_view',
    'harmonize_subject_to_reference',
    'extract_roi_time_series',
    'resample_to_reference',
    'resample_mask_to_map',
    'summarize_perfusion_series',
    'summarize_maps_with_mask',
    'summarize_maps_with_mask_paths',
    'summarize_roi_time_series',
    'calculate_lesion_qfv',
    'calculate_openads_stroke_features',
    'calculate_openads_hyperperfusion_features',
    'save_openads_feature_tables',
]
