"""
DatasetSchema: Column definitions, modality grouping, and target specification.
"""
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from .stype import stype


@dataclass
class DatasetSchema:
    """
    Schema defining the structure of a MedTabX dataset.

    Attributes:
        col_to_stype: Mapping from column name to semantic type
        target_cols: List of target column names
        path_cols: Columns containing file paths (for lazy loading)
        modality_groups: Optional grouping of columns by modality
        col_to_space: Optional mapping from column name to spatial/reference space
        col_to_modality_config: Optional per-column loader/materialization config
    """
    col_to_stype: Dict[str, stype]
    target_cols: List[str] = field(default_factory=list)
    path_cols: Set[str] = field(default_factory=set)
    modality_groups: Optional[Dict[str, List[str]]] = None
    col_to_space: Dict[str, str] = field(default_factory=dict)
    col_to_modality_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate schema after initialization."""
        # Ensure target columns exist
        for target in self.target_cols:
            if target not in self.col_to_stype:
                raise ValueError(f"Target column '{target}' not in schema")

        for col in self.col_to_space:
            if col not in self.col_to_stype:
                raise ValueError(f"Space metadata references unknown column '{col}'")

        for col in self.col_to_modality_config:
            if col not in self.col_to_stype:
                raise ValueError(f"Modality config references unknown column '{col}'")

        # Auto-detect path columns from stype
        for col, col_stype in self.col_to_stype.items():
            if col_stype.is_medical_imaging or col_stype.is_medical_timeseries:
                self.path_cols.add(col)

    @property
    def feature_cols(self) -> List[str]:
        """Get all feature column names (excluding targets)."""
        return [col for col in self.col_to_stype.keys()
                if col not in self.target_cols]

    @property
    def num_cols(self) -> int:
        """Total number of feature columns."""
        return len(self.feature_cols)

    @property
    def space_groups(self) -> Dict[str, List[str]]:
        """Group columns by declared spatial/reference space."""
        groups: Dict[str, List[str]] = {}
        for col, space in self.col_to_space.items():
            groups.setdefault(space, []).append(col)
        return groups
