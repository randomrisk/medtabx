"""Canonical IO contracts for MedTabX."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class GeometryInfo:
    """Spatial metadata for volumetric/planar imaging."""

    spacing: Optional[List[float]] = None
    origin: Optional[List[float]] = None
    direction: Optional[List[float]] = None
    affine: Optional[List[List[float]]] = None
    orientation: Optional[str] = None


@dataclass
class TimeInfo:
    """Temporal metadata for signals/time series."""

    sampling_rate: Optional[float] = None
    start_time: Optional[str] = None
    duration: Optional[float] = None
    time_axis_index: Optional[int] = None


@dataclass
class ChannelInfo:
    """Channel metadata for multi-channel modalities."""

    names: Optional[List[str]] = None
    count: Optional[int] = None
    axis_index: Optional[int] = None


@dataclass
class MedSample:
    """Canonical in-memory sample contract returned by IO load APIs."""

    modality: str
    data: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    geometry: Optional[GeometryInfo] = None
    time: Optional[TimeInfo] = None
    channels: Optional[ChannelInfo] = None

    def with_defaults(self) -> "MedSample":
        """Ensure required provenance keys exist with sensible defaults."""
        self.meta.setdefault("load_timestamp", datetime.now(timezone.utc).isoformat())
        self.meta.setdefault("warnings", [])
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "modality": self.modality,
            "data": self.data,
            "meta": self.meta,
            "geometry": asdict(self.geometry) if self.geometry else None,
            "time": asdict(self.time) if self.time else None,
            "channels": asdict(self.channels) if self.channels else None,
        }
