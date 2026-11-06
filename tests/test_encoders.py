"""Tests for encoder modules."""
import pytest
import torch
from medtabx.encoders.numeric import NumericEncoder
from medtabx.encoders.categorical import CategoricalEncoder


def test_numeric_encoder():
    """Test NumericEncoder forward pass."""
    encoder = NumericEncoder(embedding_dim=64, num_cols=3)
    x = torch.randn(8, 3)  # [batch_size=8, num_cols=3]

    output = encoder(x)

    assert output.shape == (8, 3, 64)


def test_categorical_encoder():
    """Test CategoricalEncoder forward pass."""
    encoder = CategoricalEncoder(embedding_dim=64, num_categories=10)
    x = torch.randint(0, 10, (8, 2))  # [batch_size=8, num_cols=2]

    output = encoder(x)

    assert output.shape == (8, 2, 64)
