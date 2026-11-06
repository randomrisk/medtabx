"""Tree-based and classical baseline adapters."""

from .tree import (
    TreeModelAdapter,
    available_tree_backends,
    create_tree_model,
    get_tree_model_class,
)

__all__ = [
    "TreeModelAdapter",
    "available_tree_backends",
    "create_tree_model",
    "get_tree_model_class",
]
