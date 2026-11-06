"""Manual feature extraction for tabular data."""
import numpy as np
import pandas as pd


def extract_tabular_features(df: pd.DataFrame) -> dict:
    """Extract statistical features from tabular data."""
    features = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_max'] = df[col].max()
        features[f'{col}_min'] = df[col].min()
    return features
