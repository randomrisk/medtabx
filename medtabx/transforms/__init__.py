"""Transform exports and registry helpers."""

from __future__ import annotations

from medtabx.core.registry import transform_registry

from .augmentation import (
    GaussianNoise,
    IntensityShift,
    Mixup,
    RandomRotation3D,
    TimeJitter,
    TimeScaling,
)
from .base import FittableTransform
from .medical import AtlasROIPipeline, MedicalDefaultPipeline, SpaceHarmonizedPipeline
from .tabular import StandardScaler


def build_transform(name: str, **kwargs):
    transform_cls = transform_registry.get(name)
    return transform_cls(**kwargs)


def apply_schema_transforms(schema, transforms):
    if transforms is None:
        return schema
    items = transforms if isinstance(transforms, list) else [transforms]
    updated = schema
    for item in items:
        if isinstance(item, str):
            transform = build_transform(item)
        elif isinstance(item, tuple):
            name, kwargs = item
            transform = build_transform(name, **kwargs)
        else:
            transform = item
        updated = transform.fit(updated).transform(updated)
    return updated


__all__ = [
    "FittableTransform",
    "StandardScaler",
    "GaussianNoise",
    "Mixup",
    "RandomRotation3D",
    "IntensityShift",
    "TimeJitter",
    "TimeScaling",
    "MedicalDefaultPipeline",
    "SpaceHarmonizedPipeline",
    "AtlasROIPipeline",
    "build_transform",
    "apply_schema_transforms",
]
