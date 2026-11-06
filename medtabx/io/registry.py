"""Loader registry for IO backends."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Type

from .base_loader import BaseLoader


class LoaderRegistry:
    """Deterministic registry by modality and backend name."""

    def __init__(self) -> None:
        self._by_modality: DefaultDict[str, List[Type[BaseLoader]]] = defaultdict(list)
        self._by_name: Dict[str, Type[BaseLoader]] = {}

    def register(self, loader_cls: Type[BaseLoader]) -> None:
        name = loader_cls.name.lower()
        if name in self._by_name:
            raise ValueError(f"Loader '{loader_cls.name}' already registered")

        self._by_name[name] = loader_cls
        for modality in loader_cls.supported_modalities:
            self._by_modality[modality].append(loader_cls)
            self._by_modality[modality].sort(key=lambda c: (c.priority, c.name.lower()))

    def get(self, name: str) -> Type[BaseLoader]:
        return self._by_name[name.lower()]

    def has(self, name: str) -> bool:
        return name.lower() in self._by_name

    def get_candidates(self, modality: str) -> List[Type[BaseLoader]]:
        return list(self._by_modality.get(modality, []))

    def names(self) -> List[str]:
        return sorted(self._by_name.keys())
