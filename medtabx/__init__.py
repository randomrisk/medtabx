"""
MedTabX: Unified deep learning framework for medical data.

Extends PyTorch Frame to support medical imaging, time series, and medical codes.
"""

# Core abstractions
from .core import stype, Registry, DatasetSchema
from .core import (
    encoder_registry,
    interaction_registry,
    decoder_registry,
    transform_registry,
    modality_loader_registry,
    feature_extractor_registry,
    model_registry,
    tree_model_registry,
    text_embedding_backend_registry,
)

# Stype aliases (PyTorch Frame style)
from .core.stype import (
    numerical,
    categorical,
    multicategorical,
    text_embedded,
    text_tokenized,
    sequence_numerical,
    timestamp,
    image_embedded,
    embedding,
    # Medical extensions
    image_3d,
    mask_3d,
    timeseries_4d,
    report_text,
    tabular_derived,
    eeg,
    ecg,
    fmri,
    vital_sign,
    medical_code,
    lab_value,
    medication,
)

# Data structures
try:
    from .data import TensorFrame, Dataset, DataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name != "torch":
        raise
    TensorFrame = None
    Dataset = None
    DataLoader = None

# Unified IO
from .io import MedSample, load, load_batch, inspect
from .benchmarks import (
    build_ads_benchmark_dataframe,
    build_ads_benchmark_dataset,
    build_ads_harmonized_benchmark_dataframe,
    build_ads_benchmark_schema,
    build_ads_benchmark_stype_map,
)
from .baselines import (
    TreeModelAdapter,
    available_tree_backends,
    create_tree_model,
    get_tree_model_class,
)
from .text import (
    CallableTextEmbeddingBackend,
    DeterministicTextEmbeddingBackend,
    TextEmbeddingBackend,
    TransformersTextEmbeddingBackend,
    embed_texts,
    get_text_embedding_backend,
)

# Type definitions
try:
    from .typing import TaskType, Metric, NAStrategy
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name != "torch":
        raise
    TaskType = None
    Metric = None
    NAStrategy = None

__version__ = "0.1.0"

__all__ = [
    # Core
    "stype",
    "Registry",
    "DatasetSchema",
    "encoder_registry",
    "interaction_registry",
    "decoder_registry",
    "transform_registry",
    "modality_loader_registry",
    "feature_extractor_registry",
    "model_registry",
    "tree_model_registry",
    "text_embedding_backend_registry",
    # Stype aliases
    "numerical",
    "categorical",
    "multicategorical",
    "text_embedded",
    "text_tokenized",
    "sequence_numerical",
    "timestamp",
    "image_embedded",
    "embedding",
    "image_3d",
    "mask_3d",
    "timeseries_4d",
    "report_text",
    "tabular_derived",
    "eeg",
    "ecg",
    "fmri",
    "vital_sign",
    "medical_code",
    "lab_value",
    "medication",
    # Data
    "TensorFrame",
    "Dataset",
    "DataLoader",
    "MedSample",
    "load",
    "load_batch",
    "inspect",
    "build_ads_benchmark_dataframe",
    "build_ads_harmonized_benchmark_dataframe",
    "build_ads_benchmark_stype_map",
    "build_ads_benchmark_schema",
    "build_ads_benchmark_dataset",
    "TreeModelAdapter",
    "available_tree_backends",
    "create_tree_model",
    "get_tree_model_class",
    "TextEmbeddingBackend",
    "DeterministicTextEmbeddingBackend",
    "TransformersTextEmbeddingBackend",
    "CallableTextEmbeddingBackend",
    "get_text_embedding_backend",
    "embed_texts",
    # Types
    "TaskType",
    "Metric",
    "NAStrategy",
]
