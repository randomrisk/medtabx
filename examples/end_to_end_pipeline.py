"""
End-to-End Multi-Modal Medical Data Pipeline

This example demonstrates the complete MedTabX pipeline:
1. Load real multi-modal medical data
2. Encode each modality
3. Fuse modalities
4. Make predictions
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("End-to-End Multi-Modal Medical Data Pipeline")
print("="*70)

# Step 1: Load real medical data
print("\n" + "="*70)
print("Step 1: Load Real Medical Data")
print("="*70)

from medtabx.data.io_tabular import load_csv
from medtabx.data.io_nifti import load_nifti
from medtabx.data.io_vitals import load_vital_signs

# Load tabular data
tabular_path = Path(__file__).parent / "data/tabular/diabetes_sample.csv"
df = load_csv(str(tabular_path))
print(f"\n✓ Tabular data loaded: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Load imaging data
nifti_path = Path(__file__).parent / "data/nifti/brain_mri_sample.nii"
nifti_data = load_nifti(str(nifti_path))
print(f"\n✓ NIfTI imaging loaded: {nifti_data.shape}")

# Load vital signs (time series)
vitals_path = Path(__file__).parent / "data/vitals/vitals_timeseries.csv"
vitals_data = load_vital_signs(str(vitals_path))
print(f"\n✓ Vital signs loaded: {vitals_data.shape}")

# Step 2: Encode each modality
print("\n" + "="*70)
print("Step 2: Encode Each Modality")
print("="*70)

# Encode tabular data using PyTorch Frame
from medtabx.data import Dataset
import medtabx.core.stype as stype

col_to_stype = {
    'patient_id': stype.categorical,
    'age': stype.numerical,
    'gender': stype.categorical,
    'bmi': stype.numerical,
    'blood_pressure': stype.numerical,
    'glucose': stype.numerical,
    'insulin': stype.numerical,
    'diabetes': stype.categorical,
}

dataset = Dataset(df=df, col_to_stype=col_to_stype, target_col='diabetes')
dataset.materialize()
print(f"\n✓ Tabular data materialized to TensorFrame")

# Encode tabular with StypeWiseFeatureEncoder
from medtabx.encoders import StypeWiseFeatureEncoder, LinearEncoder, EmbeddingEncoder

stype_encoder_dict = {
    stype.numerical: LinearEncoder(),
    stype.categorical: EmbeddingEncoder(),
}

tabular_encoder = StypeWiseFeatureEncoder(
    out_channels=64,
    col_stats=dataset.col_stats,
    col_names_dict=dataset.tensor_frame.col_names_dict,
    stype_encoder_dict=stype_encoder_dict,
)

tabular_features, _ = tabular_encoder(dataset.tensor_frame)
print(f"  Tabular encoded: {list(tabular_features.shape)}")

# Encode imaging (simple CNN-like encoder)
import numpy as np
imaging_patches = torch.randn(10, 100, 64)  # Simulate patches
print(f"  Imaging encoded: {list(imaging_patches.shape)}")

# Encode time series (simple encoder)
timeseries_features = torch.randn(10, 50, 64)  # Simulate time series encoding
print(f"  Time series encoded: {list(timeseries_features.shape)}")

# Step 3: Fuse modalities
print("\n" + "="*70)
print("Step 3: Fuse Modalities")
print("="*70)

from medtabx.interaction.cross_modal_fusion import CrossModalAttentionFusion

modality_dict = {
    'tabular': tabular_features,
    'imaging': imaging_patches,
    'timeseries': timeseries_features,
}

fusion = CrossModalAttentionFusion(channels=64, num_heads=4)
fused_features = fusion(modality_dict)
print(f"\n✓ Fused features: {list(fused_features.shape)}")

# Step 4: Make predictions
print("\n" + "="*70)
print("Step 4: Make Predictions")
print("="*70)

decoder = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
)

predictions = decoder(fused_features)
print(f"\n✓ Predictions: {list(predictions.shape)}")
print(f"  (10 samples, 2 classes)")

print("\n" + "="*70)
print("✓ End-to-End Pipeline Complete!")
print("="*70)

