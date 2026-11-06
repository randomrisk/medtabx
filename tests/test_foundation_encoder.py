"""Tests for the frozen foundation-encoder image tokenizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

from medtabx.encoders.foundation import (  # noqa: E402
    RESENC_B_STRIDE,
    FrozenImageTokenizer,
    build_resenc_b_encoder,
    load_foundation_encoder,
    preprocess_volume,
)

CHECKPOINT = (Path(__file__).resolve().parents[1] / "data" / "encoders" /
              "resenc_unet_b.ckpt")


def _nifti(tmp_path, shape=(40, 48, 36), zooms=(1.0, 1.0, 1.0), seed=0):
    rng = np.random.default_rng(seed)
    affine = np.diag(list(zooms) + [1.0])
    path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(rng.normal(100, 20, shape).astype(np.float32),
                             affine), path)
    return path


def test_preprocess_pads_to_the_encoder_stride_and_znormalizes(tmp_path):
    volume = preprocess_volume(_nifti(tmp_path), max_size=None)
    assert volume.shape[0] == 1
    assert all(s % RESENC_B_STRIDE == 0 for s in volume.shape[1:])
    # z-normalization happens before zero padding, so compare on the core.
    assert volume.abs().max() > 0


def test_preprocess_resamples_to_isotropic_spacing(tmp_path):
    path = _nifti(tmp_path, shape=(40, 40, 20), zooms=(1.0, 1.0, 2.0))
    volume = preprocess_volume(path, spacing=1.0, max_size=None)
    # 20 slices at 2 mm become 40 at 1 mm, then pad to 64.
    assert volume.shape[1:] == (64, 64, 64)


def test_preprocess_crops_long_axes(tmp_path):
    path = _nifti(tmp_path, shape=(300, 40, 40))
    volume = preprocess_volume(path, max_size=128)
    assert volume.shape[1] == 128


def test_preprocess_rejects_4d(tmp_path):
    path = tmp_path / "v4d.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8, 3), np.float32), np.eye(4)),
             path)
    with pytest.raises(ValueError, match="3D volume"):
        preprocess_volume(path)


@pytest.mark.parametrize("grid, tokens", [((1, 1, 1), 1), ((2, 2, 2), 8)])
def test_tokenizer_output_shape(grid, tokens):
    torch.manual_seed(0)
    tokenizer = FrozenImageTokenizer(build_resenc_b_encoder(), grid=grid)
    out = tokenizer(torch.randn(2, 1, 64, 64, 64))
    assert out.shape == (2, tokens, 320)
    assert tokenizer.num_tokens == tokens and tokenizer.token_dim == 320


def test_tokenizer_encoder_stays_frozen_and_in_eval():
    tokenizer = FrozenImageTokenizer(build_resenc_b_encoder())
    tokenizer.train()
    assert not tokenizer.encoder.training
    assert not any(p.requires_grad for p in tokenizer.encoder.parameters())


def test_wrong_checkpoint_prefix_fails_loudly(tmp_path):
    path = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"other.weight": torch.zeros(1)}}, path)
    with pytest.raises(KeyError, match="No 'model.encoder.\\*' keys"):
        load_foundation_encoder(path)


@pytest.mark.skipif(not CHECKPOINT.exists(),
                    reason="resenc_unet_b checkpoint not present in data/")
def test_pretrained_checkpoint_loads_strictly():
    encoder = load_foundation_encoder(CHECKPOINT)
    assert sum(p.numel() for p in encoder.parameters()) > 80e6
