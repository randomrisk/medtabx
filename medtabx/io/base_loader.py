"""Base loader interface for MedTabX IO plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .contracts import MedSample


class BaseLoader(ABC):
    """Contract for all IO backends."""

    name: str = "base"
    supported_modalities: List[str] = []
    supported_formats: List[str] = []
    priority: int = 100
    version_module: Optional[str] = None

    @classmethod
    def health_check(cls) -> bool:
        return True

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if required dependency stack is available."""

    @classmethod
    @abstractmethod
    def can_load(cls, path: str) -> bool:
        """Fast path suitability check."""

    @classmethod
    @abstractmethod
    def probe(cls, path: str) -> Dict[str, Any]:
        """Return lightweight metadata without full decode when possible."""

    @classmethod
    @abstractmethod
    def load(cls, path: str, options: Optional[Dict[str, Any]] = None) -> MedSample:
        """Decode and return a canonical MedSample."""
