"""Tests for the registry of frozen 3D image encoders."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from medtabx.encoders.foundation import FrozenImageTokenizer
from medtabx.encoders.image_encoders import (ImageEncoderSpec,
                                             get_image_encoder_spec,
                                             list_image_encoders,
                                             load_image_encoder,
                                             register_image_encoder)


def test_builtin_encoders_are_registered():
    names = set(list_image_encoders())
    assert {"resenc_unet_b", "medicalnet_resnet10", "medicalnet_resnet50",
            "swinunetr_ssl", "brainiac_vit_b"} <= names


def test_resenc_by_name_gives_320_wide_tokens():
    tok = FrozenImageTokenizer("resenc_unet_b", grid=(2, 2, 2),
                               checkpoint="random")
    out = tok(torch.randn(1, 1, 64, 64, 64))
    assert out.shape == (1, 8, 320)
    assert tok.token_dim == 320
    assert not any(p.requires_grad for p in tok.parameters())


def test_resenc_needs_a_checkpoint():
    with pytest.raises(ValueError, match="needs a checkpoint"):
        load_image_encoder("resenc_unet_b")


def test_unknown_encoder_lists_the_registry():
    with pytest.raises(KeyError, match="Registered"):
        get_image_encoder_spec("no_such_encoder")


def test_custom_encoder_can_be_registered():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 16, 3, stride=4, padding=1)

        def forward(self, x):
            return self.conv(x)

    spec = register_image_encoder(ImageEncoderSpec(
        name="tiny_test_encoder", out_channels=16,
        build=lambda checkpoint=None: Tiny(), size_multiple=4),
        overwrite=True)
    tok = FrozenImageTokenizer("tiny_test_encoder", grid=(1, 1, 1))
    assert tok(torch.randn(2, 1, 16, 16, 16)).shape == (2, 1, 16)
    assert tok.spec is spec
    with pytest.raises(KeyError, match="already registered"):
        register_image_encoder(spec)


@pytest.mark.parametrize("name, size, width", [
    ("medicalnet_resnet10", 64, 512),
    ("swinunetr_ssl", 64, 768),
    ("brainiac_vit_b", 96, 768),
])
def test_monai_encoders_shapes_with_random_weights(name, size, width):
    pytest.importorskip("monai")
    tok = FrozenImageTokenizer(name, grid=(2, 2, 2), checkpoint="random")
    out = tok(torch.randn(1, 1, size, size, size))
    assert out.shape == (1, 8, width)
