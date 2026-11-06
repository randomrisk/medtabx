"""
Stype (Semantic Type) definitions for MedTabX.

Merges PyTorch Frame's stype with medical-specific extensions.
"""
from enum import Enum


class stype(Enum):
    r"""The semantic type of a column.

    Extends PyTorch Frame's stype with medical imaging and time series types.

    Attributes:
        # PyTorch Frame stypes:
        numerical: Numerical columns.
        categorical: Categorical columns.
        text_embedded: Pre-computed embeddings of text columns.
        text_tokenized: Tokenized text columns for finetuning.
        multicategorical: Multicategorical columns.
        sequence_numerical: Sequence of numerical values.
        embedding: Embedding columns.
        timestamp: Timestamp columns.
        image_embedded: Pre-computed embeddings of image columns.

        # Medical extensions:
        image_3d: 3D medical images (CT/MRI volumes).
        mask_3d: 3D segmentation masks.
        timeseries_4d: Generic 4D medical time series.
        report_text: Raw clinical report text with default embedding path.
        tabular_derived: Derived scalar biomarkers/features.
        eeg: EEG time series data.
        ecg: ECG time series data.
        fmri: fMRI time series data.
        vital_sign: Vital sign time series.
        medical_code: Medical codes (ICD-10, CPT) with hierarchy.
        lab_value: Lab values with reference ranges.
        medication: Medication with dosage/frequency.
    """
    # PyTorch Frame stypes
    numerical = 'numerical'
    categorical = 'categorical'
    text_embedded = 'text_embedded'
    text_tokenized = 'text_tokenized'
    multicategorical = 'multicategorical'
    sequence_numerical = 'sequence_numerical'
    timestamp = 'timestamp'
    image_embedded = 'image_embedded'
    embedding = 'embedding'

    # Medical imaging stypes
    image_3d = 'image_3d'
    mask_3d = 'mask_3d'
    timeseries_4d = 'timeseries_4d'
    report_text = 'report_text'
    tabular_derived = 'tabular_derived'

    # Medical time series stypes
    eeg = 'eeg'
    ecg = 'ecg'
    fmri = 'fmri'
    vital_sign = 'vital_sign'

    # Medical code stypes
    medical_code = 'medical_code'
    lab_value = 'lab_value'
    medication = 'medication'

    @property
    def is_text_stype(self) -> bool:
        """Check if this is a text stype."""
        return self in [stype.text_embedded, stype.text_tokenized, stype.report_text]

    @property
    def is_image_stype(self) -> bool:
        """Check if this is an image stype (2D embedded or 3D raw)."""
        return self in [
            stype.image_embedded,
            stype.image_3d,
            stype.mask_3d,
            stype.timeseries_4d,
        ]

    @property
    def is_medical_imaging(self) -> bool:
        """Check if this is a medical imaging stype."""
        return self in [stype.image_3d, stype.mask_3d, stype.timeseries_4d]

    @property
    def is_medical_timeseries(self) -> bool:
        """Check if this is a medical time series stype."""
        return self in [stype.eeg, stype.ecg, stype.fmri, stype.vital_sign]

    @property
    def is_medical_code(self) -> bool:
        """Check if this is a medical code stype."""
        return self in [stype.medical_code, stype.lab_value, stype.medication]

    @property
    def use_multi_nested_tensor(self) -> bool:
        r"""Check if data is stored in MultiNestedTensor."""
        return self in [stype.multicategorical, stype.sequence_numerical]

    @property
    def use_multi_embedding_tensor(self) -> bool:
        r"""Check if data is stored in MultiEmbeddingTensor."""
        return self in [
            stype.text_embedded, stype.image_embedded, stype.embedding
        ]

    @property
    def use_dict_multi_nested_tensor(self) -> bool:
        r"""Check if data is stored in dict of MultiNestedTensor."""
        return self in [stype.text_tokenized]

    @property
    def use_multi_tensor(self) -> bool:
        r"""Check if data is stored in any MultiTensor variant."""
        return self.use_multi_nested_tensor or self.use_multi_embedding_tensor

    @property
    def parent(self):
        r"""Get parent stype for grouping in TensorFrame."""
        if self == stype.text_embedded:
            return stype.embedding
        elif self == stype.image_embedded:
            return stype.embedding
        elif self == stype.report_text:
            return stype.embedding
        elif self == stype.tabular_derived:
            return stype.numerical
        else:
            return self

    def __str__(self) -> str:
        return f'{self.name}'


# Convenience aliases (PyTorch Frame style)
numerical = stype('numerical')
categorical = stype('categorical')
text_embedded = stype('text_embedded')
text_tokenized = stype('text_tokenized')
multicategorical = stype('multicategorical')
sequence_numerical = stype('sequence_numerical')
timestamp = stype('timestamp')
image_embedded = stype('image_embedded')
embedding = stype('embedding')

# Medical aliases
image_3d = stype('image_3d')
mask_3d = stype('mask_3d')
timeseries_4d = stype('timeseries_4d')
report_text = stype('report_text')
tabular_derived = stype('tabular_derived')
eeg = stype('eeg')
ecg = stype('ecg')
fmri = stype('fmri')
vital_sign = stype('vital_sign')
medical_code = stype('medical_code')
lab_value = stype('lab_value')
medication = stype('medication')
