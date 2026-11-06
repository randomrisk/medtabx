"""Public IO APIs for MedTabX."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .capabilities import package_version
from .contracts import MedSample
from .registry import LoaderRegistry
from .reporting import build_load_event
from .resolver import LoaderResolver

_DEFAULT_REGISTRY: Optional[LoaderRegistry] = None
_DEFAULT_RESOLVER: Optional[LoaderResolver] = None


def get_default_registry() -> LoaderRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from .plugins import register_default_loaders

        _DEFAULT_REGISTRY = LoaderRegistry()
        register_default_loaders(_DEFAULT_REGISTRY)
    return _DEFAULT_REGISTRY


def get_default_resolver() -> LoaderResolver:
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = LoaderResolver(get_default_registry())
    return _DEFAULT_RESOLVER


def load(
    path: str,
    modality: Optional[str] = None,
    backend: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> MedSample:
    """Load one asset into the canonical MedSample contract."""
    resolver = get_default_resolver()
    loader_cls, detected_modality, detected_format = resolver.resolve(
        path=path,
        modality=modality,
        backend=backend,
        options=options,
    )
    sample = loader_cls.load(path, options=options)
    if not isinstance(sample, MedSample):
        raise TypeError(f"Loader '{loader_cls.name}' must return MedSample")

    backend_version = package_version(loader_cls.version_module or loader_cls.name)
    event = build_load_event(
        path=path,
        modality=detected_modality,
        fmt=detected_format,
        backend=loader_cls.name,
        backend_version=backend_version,
    )
    sample.meta.update(event)
    sample.modality = sample.modality or detected_modality
    return sample.with_defaults()


def inspect(
    path: str,
    modality: Optional[str] = None,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Return lightweight metadata for an asset without full decode when possible."""
    resolver = get_default_resolver()
    loader_cls, detected_modality, detected_format = resolver.resolve(
        path=path,
        modality=modality,
        backend=backend,
    )
    payload = loader_cls.probe(path)
    payload.update(
        {
            "detected_modality": detected_modality,
            "detected_format": detected_format,
            "loader_backend": loader_cls.name,
        }
    )
    return payload


def load_batch(
    paths: Iterable[str],
    modality: Optional[str] = None,
    backend: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> List[MedSample]:
    """Load multiple paths with shared routing options."""
    return [load(p, modality=modality, backend=backend, options=options) for p in paths]
