"""
Feature extraction from different medical data modalities.

This script demonstrates how to extract features from:
1. Tabular data -> numerical/categorical features
2. NIfTI imaging -> volume statistics, ROI features
3. DICOM imaging -> image statistics, metadata
4. EEG -> spectral features, statistical features
5. fMRI -> time series features, connectivity
6. Vital signs -> time series statistics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add medtabx to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("MedTabX Feature Extraction Examples")
print("="*70)


# ============================================================================
# 1. Tabular Data Features
# ============================================================================

def extract_tabular_features():
    """Extract features from tabular CSV data."""
    print("\n" + "="*70)
    print("1. Tabular Data Feature Extraction")
    print("="*70)
    
    from medtabx.data.io_tabular import load_csv, infer_column_types
    
    csv_path = Path(__file__).parent / "data/tabular/diabetes_sample.csv"
    df = load_csv(str(csv_path))
    
    print(f"\nOriginal data shape: {df.shape}")
    
    # Separate numerical and categorical columns
    numerical_cols = ['age', 'bmi', 'blood_pressure', 'glucose', 'insulin']
    categorical_cols = ['gender', 'diabetes']
    
    # Extract numerical features
    numerical_features = df[numerical_cols].values
    print(f"\nNumerical features shape: {numerical_features.shape}")
    print(f"Numerical feature names: {numerical_cols}")
    print(f"Sample values:\n{numerical_features[:3]}")
    
    # Extract categorical features (one-hot encoding)
    categorical_features = pd.get_dummies(df[categorical_cols])
    print(f"\nCategorical features (one-hot) shape: {categorical_features.shape}")
    print(f"Categorical feature names: {list(categorical_features.columns)}")
    
    # Combined features
    all_features = np.concatenate([
        numerical_features,
        categorical_features.values
    ], axis=1)
    
    print(f"\nCombined features shape: {all_features.shape}")
    print(f"Total features: {all_features.shape[1]}")
    
    return all_features



# ============================================================================
# 2. NIfTI Imaging Features
# ============================================================================

def extract_nifti_features():
    """Extract features from NIfTI brain imaging."""
    print("\n" + "="*70)
    print("2. NIfTI Brain Imaging Feature Extraction")
    print("="*70)
    
    from medtabx.data.io_nifti import load_nifti
    
    nifti_path = Path(__file__).parent / "data/nifti/brain_mri_sample.nii"
    img = load_nifti(str(nifti_path))
    
    print(f"\nImage shape: {img.shape}")
    
    # Extract statistical features
    features = {
        'mean': np.mean(img),
        'std': np.std(img),
        'min': np.min(img),
        'max': np.max(img),
        'median': np.median(img),
        'q25': np.percentile(img, 25),
        'q75': np.percentile(img, 75),
    }
    
    print(f"\nStatistical features:")
    for name, value in features.items():
        print(f"  {name:10s}: {value:.2f}")
    
    # Flatten for feature vector
    feature_vector = np.array(list(features.values()))
    print(f"\nFeature vector shape: {feature_vector.shape}")
    
    return feature_vector



# ============================================================================
# 3. DICOM Imaging Features
# ============================================================================

def extract_dicom_features():
    """Extract features from DICOM medical imaging."""
    print("\n" + "="*70)
    print("3. DICOM Medical Imaging Feature Extraction")
    print("="*70)
    
    from medtabx.data.io_dicom import load_dicom_file
    
    dicom_path = Path(__file__).parent / "data/dicom/chest_xray_sample.dcm"
    img, metadata = load_dicom_file(str(dicom_path))
    
    print(f"\nImage shape: {img.shape}")
    
    # Extract image features
    features = {
        'mean_intensity': np.mean(img),
        'std_intensity': np.std(img),
        'min_intensity': np.min(img),
        'max_intensity': np.max(img),
    }
    
    print(f"\nImage features:")
    for name, value in features.items():
        print(f"  {name:20s}: {value:.2f}")
    
    feature_vector = np.array(list(features.values()))
    print(f"\nFeature vector shape: {feature_vector.shape}")
    
    return feature_vector



# ============================================================================
# 4. EEG Features
# ============================================================================

def extract_eeg_features():
    """Extract features from EEG data."""
    print("\n" + "="*70)
    print("4. EEG Data Feature Extraction")
    print("="*70)
    
    from medtabx.data.io_eeg import load_eeg
    
    eeg_path = Path(__file__).parent / "data/eeg/chb01_01.edf"
    data, metadata = load_eeg(str(eeg_path))
    
    print(f"\nEEG data shape: {data.shape}")
    print(f"Sampling rate: {metadata['sfreq']} Hz")
    
    # Extract features from first 10 seconds
    n_samples = int(10 * metadata['sfreq'])
    data_segment = data[:, :n_samples]
    
    features = {}
    for ch_idx in range(min(5, data.shape[0])):  # First 5 channels
        ch_name = metadata['ch_names'][ch_idx]
        ch_data = data_segment[ch_idx]
        
        features[f'{ch_name}_mean'] = np.mean(ch_data)
        features[f'{ch_name}_std'] = np.std(ch_data)
        features[f'{ch_name}_max'] = np.max(ch_data)
        features[f'{ch_name}_min'] = np.min(ch_data)
    
    print(f"\nExtracted {len(features)} features from {min(5, data.shape[0])} channels")
    print(f"Sample features:")
    for i, (name, value) in enumerate(list(features.items())[:8]):
        print(f"  {name:20s}: {value:.4f}")
    
    feature_vector = np.array(list(features.values()))
    print(f"\nFeature vector shape: {feature_vector.shape}")
    
    return feature_vector



# ============================================================================
# 5. fMRI Features
# ============================================================================

def extract_fmri_features():
    """Extract features from fMRI data."""
    print("\n" + "="*70)
    print("5. fMRI Data Feature Extraction")
    print("="*70)
    
    from medtabx.data.io_fmri import load_fmri
    
    fmri_path = Path(__file__).parent / "data/fmri/fmri_timeseries.nii.gz"
    data = load_fmri(str(fmri_path))
    
    print(f"\nfMRI data shape: {data.shape} (X, Y, Z, Time)")
    
    # Extract mean time series from center voxel
    center = tuple(s // 2 for s in data.shape[:3])
    time_series = data[center[0], center[1], center[2], :]
    
    features = {
        'mean_signal': np.mean(time_series),
        'std_signal': np.std(time_series),
        'max_signal': np.max(time_series),
        'min_signal': np.min(time_series),
    }
    
    print(f"\nTime series features:")
    for name, value in features.items():
        print(f"  {name:20s}: {value:.2f}")
    
    feature_vector = np.array(list(features.values()))
    print(f"\nFeature vector shape: {feature_vector.shape}")
    
    return feature_vector



# ============================================================================
# 6. Vital Signs Features
# ============================================================================

def extract_vitals_features():
    """Extract features from vital signs time series."""
    print("\n" + "="*70)
    print("6. Vital Signs Time Series Feature Extraction")
    print("="*70)
    
    from medtabx.data.io_vitals import load_vital_signs
    
    vitals_path = Path(__file__).parent / "data/vitals/vitals_timeseries.csv"
    df = load_vital_signs(str(vitals_path))
    
    print(f"\nVital signs data shape: {df.shape}")
    
    # Extract features for each vital sign
    vital_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'spo2']
    
    features = {}
    for col in vital_cols:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
    
    print(f"\nExtracted {len(features)} features")
    print(f"Sample features:")
    for i, (name, value) in enumerate(list(features.items())[:8]):
        print(f"  {name:25s}: {value:.2f}")
    
    feature_vector = np.array(list(features.values()))
    print(f"\nFeature vector shape: {feature_vector.shape}")
    
    return feature_vector



# ============================================================================
# Main
# ============================================================================

def main():
    """Run all feature extraction examples."""
    
    # Extract features from all modalities
    tabular_features = extract_tabular_features()
    nifti_features = extract_nifti_features()
    dicom_features = extract_dicom_features()
    eeg_features = extract_eeg_features()
    fmri_features = extract_fmri_features()
    vitals_features = extract_vitals_features()
    
    # Summary
    print("\n" + "="*70)
    print("Feature Extraction Summary")
    print("="*70)
    
    print(f"\nTabular features:  {tabular_features.shape}")
    print(f"NIfTI features:    {nifti_features.shape}")
    print(f"DICOM features:    {dicom_features.shape}")
    print(f"EEG features:      {eeg_features.shape}")
    print(f"fMRI features:     {fmri_features.shape}")
    print(f"Vitals features:   {vitals_features.shape}")
    
    print("\n✓ All feature extractions completed successfully!")

if __name__ == "__main__":
    main()
