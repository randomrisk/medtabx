"""Registry-backed access to tabular and multimodal model constructors."""

from __future__ import annotations

from importlib import import_module

from medtabx.core.registry import model_registry

_MODEL_IMPORTS = {
    "FTTransformer": "medtabx.models.ft_transformer:FTTransformer",
    "ResNet": "medtabx.models.resnet:ResNet",
    "TabNet": "medtabx.models.tabnet:TabNet",
    "TabTransformer": "medtabx.models.tab_transformer:TabTransformer",
    "Trompt": "medtabx.models.trompt:Trompt",
    "ExcelFormer": "medtabx.models.excelformer:ExcelFormer",
    "MLP": "medtabx.models.mlp:MLP",
    "MultimodalBaseline": "medtabx.models.multimodal_baseline:MultimodalBaseline",
}


def _load_symbol(spec: str):
    module_name, symbol_name = spec.split(":")
    module = import_module(module_name)
    return getattr(module, symbol_name)


for _name, _spec in _MODEL_IMPORTS.items():
    if _name in model_registry:
        continue

    @model_registry.register(_name)
    class _LazyModelProxy:
        _spec = _spec

        def __new__(cls, *args, **kwargs):
            target = _load_symbol(cls._spec)
            return target(*args, **kwargs)


def get_model_class(name: str):
    registered = model_registry.get(name)
    if hasattr(registered, "_spec"):
        return _load_symbol(registered._spec)
    return registered


def create_model(name: str, **kwargs):
    return get_model_class(name)(**kwargs)


def list_supported_models() -> list[str]:
    return sorted(model_registry.list())
