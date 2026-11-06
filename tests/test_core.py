"""Tests for core modules."""
import pytest
import torch
from medtabx.core import stype, DatasetSchema, encoder_registry


def test_stype_properties():
    """Test stype property methods."""
    # Test PyTorch Frame stypes
    assert stype.numerical.name == 'numerical'
    assert stype.categorical.name == 'categorical'

    # Test medical imaging stypes
    assert stype.image_3d.is_medical_imaging
    assert stype.mask_3d.is_medical_imaging

    # Test medical time series stypes
    assert stype.eeg.is_medical_timeseries
    assert stype.fmri.is_medical_timeseries
    assert stype.vital_sign.is_medical_timeseries

    # Test medical code stypes
    assert stype.medical_code.is_medical_code
    assert stype.lab_value.is_medical_code
    assert stype.medication.is_medical_code


def test_dataset_schema():
    """Test DatasetSchema creation and validation."""
    schema = DatasetSchema(
        col_to_stype={
            'age': stype.numerical,
            'gender': stype.categorical,
            'ct_scan': stype.image_3d,
            'label': stype.categorical,
        },
        target_cols=['label']
    )

    assert len(schema.feature_cols) == 3
    assert 'label' not in schema.feature_cols
    assert 'ct_scan' in schema.path_cols
    assert schema.num_cols == 3
