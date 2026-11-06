"""Data loading and processing for MedTabX."""

try:
    from .tensor_frame import TensorFrame
    from .dataset import Dataset
    from .loader import DataLoader
    from .modality_registry import get_modality_loader, load_modality
    from .mapper import (
        TensorMapper,
        NumericalTensorMapper,
        CategoricalTensorMapper,
        MultiCategoricalTensorMapper,
        TimestampTensorMapper,
        EmbeddingTensorMapper,
    )
    from .stats import StatType
    from .multi_tensor import _MultiTensor
    from .multi_nested_tensor import MultiNestedTensor
    from .multi_embedding_tensor import MultiEmbeddingTensor
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name != "torch":
        raise

    TensorFrame = None
    Dataset = None
    DataLoader = None
    get_modality_loader = None
    load_modality = None
    TensorMapper = None
    NumericalTensorMapper = None
    CategoricalTensorMapper = None
    MultiCategoricalTensorMapper = None
    TimestampTensorMapper = None
    EmbeddingTensorMapper = None
    StatType = None
    _MultiTensor = None
    MultiNestedTensor = None
    MultiEmbeddingTensor = None

__all__ = [
    "TensorFrame",
    "Dataset",
    "DataLoader",
    "get_modality_loader",
    "load_modality",
    "TensorMapper",
    "NumericalTensorMapper",
    "CategoricalTensorMapper",
    "MultiCategoricalTensorMapper",
    "TimestampTensorMapper",
    "EmbeddingTensorMapper",
    "StatType",
    "_MultiTensor",
    "MultiNestedTensor",
    "MultiEmbeddingTensor",
]
