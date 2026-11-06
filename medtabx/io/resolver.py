"""Backend resolver for MedTabX IO."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from .base_loader import BaseLoader
from .detector import detect_format, detect_modality
from .errors import (
    Attempt,
    BackendNotAvailableError,
    LoadFailedError,
    UnsupportedModalityError,
)
from .registry import LoaderRegistry


class LoaderResolver:
    """Resolve the best backend for a path/modality request."""

    def __init__(self, registry: LoaderRegistry):
        self.registry = registry

    def resolve(
        self,
        path: str,
        modality: Optional[str] = None,
        backend: Optional[str] = None,
        options: Optional[Dict[str, object]] = None,
    ) -> Tuple[Type[BaseLoader], str, str]:
        del options
        detected_modality = modality or detect_modality(path)
        try:
            detected_format = detect_format(path)
        except UnsupportedModalityError:
            # Explicit modality can still be resolvable via backend `can_load`.
            if modality is not None:
                detected_format = "unknown"
            else:
                raise

        candidates = self.registry.get_candidates(detected_modality)
        if not candidates:
            raise UnsupportedModalityError(
                f"No loaders registered for modality '{detected_modality}'"
            )

        if backend:
            backend_key = backend.lower()
            if not self.registry.has(backend_key):
                raise BackendNotAvailableError(
                    f"Backend '{backend}' is not registered. Available: {self.registry.names()}"
                )

            loader_cls = self.registry.get(backend_key)
            if detected_modality not in loader_cls.supported_modalities:
                raise BackendNotAvailableError(
                    f"Backend '{backend}' does not support modality '{detected_modality}'"
                )
            if not loader_cls.is_available() or not loader_cls.health_check():
                raise BackendNotAvailableError(
                    f"Backend '{backend}' is not available in this environment"
                )
            if not loader_cls.can_load(path):
                raise BackendNotAvailableError(
                    f"Backend '{backend}' cannot load this input: {path}"
                )
            return loader_cls, detected_modality, detected_format

        attempts = []
        for loader_cls in candidates:
            if not loader_cls.is_available():
                attempts.append(Attempt(loader_cls.name, "dependency unavailable"))
                continue
            if not loader_cls.health_check():
                attempts.append(Attempt(loader_cls.name, "health check failed"))
                continue
            if not loader_cls.can_load(path):
                attempts.append(Attempt(loader_cls.name, "can_load returned False"))
                continue

            return loader_cls, detected_modality, detected_format

        raise LoadFailedError(
            message=f"Unable to resolve loader for '{path}' ({detected_modality})",
            attempts=attempts,
        )
