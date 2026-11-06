"""
I/O module for medical time series data (vital signs, lab values).

Supports:
- CSV format
- Time-aligned data
- Missing value handling
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path


def load_vital_signs(
    filepath: str,
    time_column: str = 'time',
    value_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load vital signs time series.

    Args:
        filepath: Path to CSV file
        time_column: Name of time column
        value_columns: List of value columns to load

    Returns:
        DataFrame with time-indexed vital signs

    Example:
        >>> vitals = load_vital_signs('vitals.csv')
        >>> print(vitals.columns)
        ['heart_rate', 'blood_pressure', 'temperature']
    """
    df = pd.read_csv(filepath)

    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)

    if value_columns:
        df = df[value_columns]

    return df
