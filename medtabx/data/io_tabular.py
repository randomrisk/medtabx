"""
I/O module for standard tabular data types.

This module inherits PyTorch Frame's handling for:
- numerical: Numerical columns
- categorical: Categorical columns
- multicategorical: Multi-categorical columns
- text_embedded: Pre-computed text embeddings
- text_tokenized: Tokenized text
- sequence_numerical: Numerical sequences
- timestamp: Timestamp columns
- image_embedded: Pre-computed image embeddings

All standard tabular data loading is delegated to PyTorch Frame's
Dataset and TensorMapper classes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..core.stype import stype


def load_csv(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """Load CSV file into pandas DataFrame.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments passed to pd.read_csv

    Returns:
        DataFrame with loaded data

    Example:
        >>> df = load_csv('data.csv')
        >>> df = load_csv('data.csv', sep='\t', encoding='utf-8')
    """
    return pd.read_csv(filepath, **kwargs)


def load_parquet(
    filepath: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """Load Parquet file into pandas DataFrame.

    Args:
        filepath: Path to Parquet file
        **kwargs: Additional arguments passed to pd.read_parquet

    Returns:
        DataFrame with loaded data

    Example:
        >>> df = load_parquet('data.parquet')
    """
    return pd.read_parquet(filepath, **kwargs)


def infer_column_types(
    df: pd.DataFrame,
    categorical_threshold: float = 0.05
) -> Dict[str, stype]:
    """Infer semantic types for DataFrame columns.

    Uses heuristics to automatically detect column types:
    - Numerical: numeric dtypes
    - Categorical: low cardinality (<5% unique values)
    - Text: string columns with high cardinality
    - Timestamp: datetime dtypes

    Args:
        df: Input DataFrame
        categorical_threshold: Threshold for categorical detection
            (unique_values / total_values)

    Returns:
        Dictionary mapping column names to stypes

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> col_types = infer_column_types(df)
        >>> print(col_types)
        {'age': stype.numerical, 'gender': stype.categorical, ...}
    """
    col_to_stype = {}

    for col in df.columns:
        # Skip if all null
        if df[col].isna().all():
            continue

        dtype = df[col].dtype

        # Numerical
        if pd.api.types.is_numeric_dtype(dtype):
            col_to_stype[col] = stype.numerical

        # Timestamp
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_to_stype[col] = stype.timestamp

        # Categorical or text
        elif pd.api.types.is_string_dtype(dtype) or dtype == 'object':
            unique_ratio = df[col].nunique() / len(df[col])

            if unique_ratio < categorical_threshold:
                col_to_stype[col] = stype.categorical
            else:
                col_to_stype[col] = stype.text_embedded

    return col_to_stype


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> None:
    """Validate DataFrame structure.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Raises:
        ValueError: If validation fails

    Example:
        >>> validate_dataframe(df, required_columns=['age', 'gender'])
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
