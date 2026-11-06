"""Structured load reporting helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


def build_load_event(
    path: str,
    modality: str,
    fmt: str,
    backend: str,
    backend_version: str,
) -> Dict[str, Any]:
    return {
        "source_path": path,
        "detected_modality": modality,
        "detected_format": fmt,
        "source_format": fmt,
        "loader_backend": backend,
        "loader_backend_version": backend_version,
        "load_timestamp": datetime.now(timezone.utc).isoformat(),
        "warnings": [],
    }
