"""Tests for medical encoders."""
import pytest
import torch
from medtabx.encoders.imaging3d import Imaging3DEncoder
from medtabx.encoders.eeg import EEGEncoder
from medtabx.encoders.fmri import FMRIEncoder
from medtabx.encoders.mask3d import Mask3DEncoder
from medtabx.encoders.vital_sign import VitalSignEncoder


def test_imaging3d_encoder():
    """Test Imaging3DEncoder forward pass."""
    encoder = Imaging3DEncoder(embedding_dim=64, in_channels=1)
    # [batch_size=2, num_cols=1, channels=1, D=32, H=32, W=32]
    x = torch.randn(2, 1, 1, 32, 32, 32)

    output = encoder(x)
    assert output.shape == (2, 1, 64)


def test_eeg_encoder():
    """Test EEGEncoder forward pass."""
    encoder = EEGEncoder(embedding_dim=64, num_channels=64, seq_len=1000)
    # [batch_size=2, num_cols=1, num_channels=64, seq_len=1000]
    x = torch.randn(2, 1, 64, 1000)

    output = encoder(x)
    assert output.shape == (2, 1, 64)


def test_fmri_encoder():
    """Test FMRIEncoder forward pass."""
    encoder = FMRIEncoder(embedding_dim=64, num_rois=100, seq_len=200)
    # [batch_size=2, num_cols=1, num_rois=100, seq_len=200]
    x = torch.randn(2, 1, 100, 200)

    output = encoder(x)
    assert output.shape == (2, 1, 64)
