from __future__ import annotations

import copy
from typing import Any

from medtabx.core.registry import transform_registry
from medtabx.core.schema import DatasetSchema

from .base import FittableTransform


class SchemaTransform(FittableTransform):
    """Base transform for schema-level preprocessing plans."""

    def fit(self, data: Any) -> "SchemaTransform":
        return self


@transform_registry.register("medical_default")
class MedicalDefaultPipeline(SchemaTransform):
    def transform(self, data: DatasetSchema) -> DatasetSchema:
        return copy.deepcopy(data)


@transform_registry.register("space_harmonized")
class SpaceHarmonizedPipeline(SchemaTransform):
    """Ensure common target shapes for raw medical imaging columns."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (64, 64, 32),
        mask_shape: tuple[int, int, int] | None = None,
    ):
        self.image_shape = image_shape
        self.mask_shape = image_shape if mask_shape is None else mask_shape

    def transform(self, data: DatasetSchema) -> DatasetSchema:
        schema = copy.deepcopy(data)
        for col, col_stype in schema.col_to_stype.items():
            if col_stype.value == "image_3d":
                schema.col_to_modality_config.setdefault(col, {})
                schema.col_to_modality_config[col]["target_shape"] = self.image_shape
            elif col_stype.value == "mask_3d":
                schema.col_to_modality_config.setdefault(col, {})
                schema.col_to_modality_config[col]["target_shape"] = self.mask_shape
        return schema


@transform_registry.register("atlas_roi")
class AtlasROIPipeline(SchemaTransform):
    """Configure atlas-based ROI extraction for a 4D imaging column."""

    def __init__(
        self,
        column: str = "pwi_path",
        atlas_path: str | None = None,
        min_label: int = 1,
    ):
        self.column = column
        self.atlas_path = atlas_path
        self.min_label = min_label

    def transform(self, data: DatasetSchema) -> DatasetSchema:
        schema = copy.deepcopy(data)
        schema.col_to_modality_config.setdefault(self.column, {})
        if self.atlas_path is not None:
            schema.col_to_modality_config[self.column]["atlas_path"] = self.atlas_path
        schema.col_to_modality_config[self.column]["min_label"] = self.min_label
        return schema
