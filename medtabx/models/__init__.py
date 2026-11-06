"""Model package with lazy imports.

Some legacy model modules still depend on internal paths that have not been
ported into this repository. Importing ``medtabx.models`` should not fail when
the caller only needs a working subset such as ``FTTransformer``.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Trompt",
    "FTTransformer",
    "ExcelFormer",
    "TabNet",
    "ResNet",
    "TabTransformer",
    "MLP",
    "MultimodalBaseline",
    "TypedTokenTransformer",
    "TypedTokenConfig",
    "TypedTokenSchema",
    "create_model",
    "get_model_class",
    "list_supported_models",
]

_MODULE_BY_NAME = {
    "Trompt": "medtabx.models.trompt",
    "FTTransformer": "medtabx.models.ft_transformer",
    "ExcelFormer": "medtabx.models.excelformer",
    "TabNet": "medtabx.models.tabnet",
    "ResNet": "medtabx.models.resnet",
    "TabTransformer": "medtabx.models.tab_transformer",
    "MLP": "medtabx.models.mlp",
    "MultimodalBaseline": "medtabx.models.multimodal_baseline",
    "TypedTokenTransformer": "medtabx.models.typed_token",
    "TypedTokenConfig": "medtabx.models.typed_token",
    "TypedTokenSchema": "medtabx.models.typed_token",
}


def __getattr__(name: str):
    if name in {"create_model", "get_model_class", "list_supported_models"}:
        module = import_module("medtabx.models.model_registry")
        return getattr(module, name)
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_BY_NAME[name])
    return getattr(module, name)
