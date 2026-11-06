"""
MedTabX Encoders

Encoders transform TensorFrame data into column-wise embeddings.
"""

# Base encoder classes
from .encoder import FeatureEncoder
from .stype_encoder import StypeEncoder

# StypeWise encoder
from .stypewise_encoder import StypeWiseFeatureEncoder

# Specific stype encoders from PyTorch Frame
from .stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StackEncoder,
    LinearBucketEncoder,
    LinearPeriodicEncoder,
    ExcelFormerEncoder,
    LinearEmbeddingEncoder,
    MultiCategoricalEmbeddingEncoder,
    TimestampEncoder,
)
from .categorical import CategoricalEncoder
from .lab_value import LabValueEncoder
from .medical_code import MedicalCodeEncoder
from .medication import MedicationEncoder
from .ecg import ECGEncoder
from .eeg import EEGEncoder
from .fmri import FMRIEncoder
from .imaging3d import Imaging3DEncoder
from .mask3d import Mask3DEncoder
from .numeric import NumericEncoder
from .text import TextEmbeddedEncoder
from .vital_sign import VitalSignEncoder

# Default encoder selection
from .defaults import (
    DEFAULT_MEDICAL_ENCODER_CLS,
    DEFAULT_TABULAR_ENCODER_CLS,
    build_stype_encoder_dict,
    default_tabular_stype_encoder_dict,
    infer_encoder_kwargs,
)

from .image_encoders import (ImageEncoderSpec, list_image_encoders,
                             load_image_encoder, register_image_encoder)

__all__ = [
    'FeatureEncoder',
    'StypeEncoder',
    'StypeWiseFeatureEncoder',
    'EmbeddingEncoder',
    'LinearEncoder',
    'StackEncoder',
    'LinearBucketEncoder',
    'LinearPeriodicEncoder',
    'ExcelFormerEncoder',
    'LinearEmbeddingEncoder',
    'MultiCategoricalEmbeddingEncoder',
    'TimestampEncoder',
    'NumericEncoder',
    'CategoricalEncoder',
    'TextEmbeddedEncoder',
    'Imaging3DEncoder',
    'Mask3DEncoder',
    'FMRIEncoder',
    'ECGEncoder',
    'EEGEncoder',
    'VitalSignEncoder',
    'LabValueEncoder',
    'MedicalCodeEncoder',
    'MedicationEncoder',
    # Default encoder selection
    'default_tabular_stype_encoder_dict',
    'build_stype_encoder_dict',
    'infer_encoder_kwargs',
    'DEFAULT_TABULAR_ENCODER_CLS',
    'DEFAULT_MEDICAL_ENCODER_CLS',
    # Frozen 3D image encoders
    'ImageEncoderSpec',
    'list_image_encoders',
    'load_image_encoder',
    'register_image_encoder',
]
