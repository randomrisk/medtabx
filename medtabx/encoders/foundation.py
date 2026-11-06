"""Frozen 3D image encoders used as image tokenizers.

MedTabX does not train image encoders. It takes a pretrained network, freezes
it, and turns its deepest feature map into a small set of image tokens that
sit in the same sequence as the clinical and derived columns
(volume -> encoder -> pool -> tokens [K, C]).

The encoder is chosen by name from :mod:`medtabx.encoders.image_encoders`
(``resenc_unet_b``, ``medicalnet_resnet10``, ``swinunetr_ssl``,
``brainiac_vit_b``, or a user-registered encoder). The default
``resenc_unet_b`` is the residual U-Net encoder (features
32-64-128-256-320-320) of the FOMO challenge baseline; it loads any
checkpoint of that architecture. Its preprocessing follows the pretraining
setup: RAS orientation, 1 mm isotropic voxels and volume-wise
z-normalization. Inputs are assumed skull-stripped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .third_party.resenc import ResidualBlock, ResidualUNetEncoder

__all__ = [
    "RESENC_B_FEATURES",
    "build_resenc_b_encoder",
    "load_foundation_encoder",
    "preprocess_volume",
    "FrozenImageTokenizer",
]

#: Channels per stage of ``resenc_unet_b``; the deepest stage gives the token
#: width.
RESENC_B_FEATURES = (32, 64, 128, 256, 320, 320)
RESENC_B_BLOCKS = (1, 3, 4, 6, 6, 6)

#: Total downsampling of the encoder (five stride-2 stages). Spatial sizes are
#: padded to a multiple of this so the residual additions line up.
RESENC_B_STRIDE = 2**(len(RESENC_B_FEATURES) - 1)


def build_resenc_b_encoder(input_channels: int = 1) -> ResidualUNetEncoder:
    """The ``resenc_unet_b`` encoder, configured as in Asparagus."""
    return ResidualUNetEncoder(
        input_channels=input_channels,
        features_per_stage=RESENC_B_FEATURES,
        conv_op=nn.Conv3d,
        pool_op=nn.MaxPool3d,
        kernel_size=3,
        stride=2,
        n_blocks_per_stage=RESENC_B_BLOCKS,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        block=ResidualBlock,
    )


def load_foundation_encoder(
    checkpoint_path: str | Path,
    prefix: str = "model.encoder.",
) -> ResidualUNetEncoder:
    r"""Build the encoder and load pretrained weights from a Lightning ckpt.

    Loading is strict: every encoder parameter must be present in the
    checkpoint and nothing extra may match, so a checkpoint for a different
    architecture fails loudly instead of leaving layers randomly initialized.

    Args:
        checkpoint_path: A ``last.ckpt`` from an Asparagus pretraining run.
        prefix (str): Key prefix of the encoder inside ``state_dict``.
            (default: :obj:`"model.encoder."`)
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu",
                            weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        k[len(prefix):]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }
    if not encoder_state:
        found = sorted({".".join(k.split(".")[:2]) for k in state_dict})[:10]
        raise KeyError(
            f"No '{prefix}*' keys in {checkpoint_path}. Top-level prefixes "
            f"found: {found}.")

    encoder = build_resenc_b_encoder()
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder


def _resample_to_spacing(volume: np.ndarray, zooms: Sequence[float],
                         spacing: float) -> np.ndarray:
    target = [max(1, int(round(s * z / spacing)))
              for s, z in zip(volume.shape, zooms)]
    if list(volume.shape) == target:
        return volume
    tensor = torch.from_numpy(volume)[None, None]
    tensor = F.interpolate(tensor, size=target, mode="trilinear",
                           align_corners=False)
    return tensor[0, 0].numpy()


def preprocess_volume(
    source: str | Path | nib.Nifti1Image,
    spacing: float | None = 1.0,
    max_size: int | None = 256,
    size_multiple: int = RESENC_B_STRIDE,
    fixed_size: tuple[int, int, int] | None = None,
    normalization: str = "zscore",
) -> Tensor:
    r"""Load a 3D volume the way the foundation encoder expects it.

    Steps: closest-canonical (RAS) orientation, optional resampling to
    isotropic ``spacing`` mm, centre crop of any axis longer than
    ``max_size`` voxels, volume-wise z-normalization, and zero padding of
    each axis to a multiple of the encoder stride.

    Args:
        source: A NIfTI path or image. 4D inputs must be reduced to 3D first.
        spacing (float, optional): Target isotropic voxel size in mm.
            :obj:`None` keeps the native grid. (default: :obj:`1.0`)
        max_size (int, optional): Longest allowed axis after resampling, to
            bound GPU memory. (default: :obj:`256`)
        size_multiple (int): Each axis is zero-padded to a multiple of this.
            (default: the stride of ``resenc_unet_b``, 32)
        fixed_size (tuple, optional): Resize to this shape (encoders with a
            fixed input size). (default: :obj:`None`)
        normalization (str): ``"zscore"`` over the whole volume or
            ``"zscore_nonzero"`` over nonzero voxels. (default: ``"zscore"``)

    Returns:
        Tensor: Shape ``[1, X, Y, Z]`` (a channel axis, no batch axis).
    """
    image = source if isinstance(source, nib.Nifti1Image) else nib.load(
        str(source))
    image = nib.as_closest_canonical(image)
    volume = np.asarray(image.get_fdata(), dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {volume.shape}. Select a single "
            f"frame of 4D inputs before tokenizing.")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    if spacing is not None:
        zooms = [float(z) for z in image.header.get_zooms()[:3]]
        volume = _resample_to_spacing(volume, zooms, spacing)

    if max_size is not None:
        slices = []
        for size in volume.shape:
            if size > max_size:
                start = (size - max_size) // 2
                slices.append(slice(start, start + max_size))
            else:
                slices.append(slice(None))
        volume = volume[tuple(slices)]

    tensor = torch.from_numpy(np.ascontiguousarray(volume))
    if fixed_size is not None and tuple(tensor.shape) != tuple(fixed_size):
        tensor = F.interpolate(tensor[None, None], size=tuple(fixed_size),
                               mode="trilinear", align_corners=False)[0, 0]
    if normalization == "zscore":
        # Volume-wise z-normalization, as in pretraining of resenc_unet_b.
        tensor = (tensor - tensor.mean()) / tensor.std(
            unbiased=False).clamp_min(1e-8)
    elif normalization == "zscore_nonzero":
        mask = tensor != 0
        values = tensor[mask] if mask.any() else tensor.flatten()
        tensor = torch.where(mask, (tensor - values.mean()) / values.std(
            unbiased=False).clamp_min(1e-8), torch.zeros_like(tensor))
    else:
        raise ValueError(f"Unknown normalization '{normalization}'.")

    pad: list[int] = []
    for size in reversed(tensor.shape):  # F.pad wants the last axis first
        target = -(-size // size_multiple) * size_multiple
        diff = target - size
        pad.extend([diff // 2, diff - diff // 2])
    tensor = F.pad(tensor, pad)
    return tensor.unsqueeze(0)


class FrozenImageTokenizer(nn.Module):
    r"""Turn a 3D volume into ``K`` image tokens of width ``C``.

    The deepest encoder feature map is average-pooled onto a ``grid`` of
    cells; each cell becomes one token. ``grid=(1, 1, 1)`` gives the single
    global embedding used for linear probing, larger grids keep coarse
    spatial layout (``(2, 2, 2)`` gives 8 tokens, one per octant).

    The encoder is frozen and kept in eval mode: MedTabX only trains what
    sits on top of these tokens.

    Args:
        encoder: A registered encoder name (see
            :func:`~medtabx.encoders.image_encoders.list_image_encoders`), a
            ``resenc_unet_b`` encoder module, or a module that returns a
            feature map ``[B, C, x, y, z]`` (then give ``spec``).
        grid (tuple): Token grid. (default: :obj:`(1, 1, 1)`)
        checkpoint: Weights for a named encoder (see
            :func:`~medtabx.encoders.image_encoders.load_image_encoder`).
        spec: :class:`~medtabx.encoders.image_encoders.ImageEncoderSpec` of a
            custom module (token width and preprocessing).
    """

    def __init__(self, encoder: nn.Module | str,
                 grid: tuple[int, int, int] = (1, 1, 1),
                 checkpoint: str | Path | None = None,
                 spec=None) -> None:
        from .image_encoders import (_DeepestStage, get_image_encoder_spec,
                                     load_image_encoder)
        super().__init__()
        if isinstance(encoder, str):
            encoder, spec = load_image_encoder(encoder, checkpoint)
        elif isinstance(encoder, ResidualUNetEncoder):
            encoder = _DeepestStage(encoder)
            spec = get_image_encoder_spec("resenc_unet_b")
        elif spec is None:
            raise ValueError("Give `spec` for a custom encoder module.")
        self.spec = spec
        self.encoder = encoder
        self.grid = tuple(grid)
        self.pool = nn.AdaptiveAvgPool3d(self.grid)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def token_dim(self) -> int:
        return self.spec.out_channels

    @property
    def num_tokens(self) -> int:
        return int(np.prod(self.grid))

    def train(self, mode: bool = True) -> FrozenImageTokenizer:
        # The encoder stays in eval mode whatever the caller does.
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, volume: Tensor) -> Tensor:
        """``[B, 1, X, Y, Z]`` -> ``[B, K, C]``."""
        deepest = self.encoder(volume)
        pooled = self.pool(deepest)  # [B, C, gx, gy, gz]
        return pooled.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def tokenize_paths(self, paths: Iterable[str | Path],
                       device: str | torch.device = "cuda",
                       autocast: bool = True, **preprocess_kwargs) -> Tensor:
        """Tokenize volumes one at a time; returns ``[N, K, C]`` on CPU.

        Preprocessing follows the encoder spec; keyword arguments override it.
        """
        preprocess_kwargs = {**self.spec.preprocess_kwargs,
                             **preprocess_kwargs}
        tokens = []
        use_amp = autocast and torch.device(device).type == "cuda"
        for path in paths:
            volume = preprocess_volume(path, **preprocess_kwargs)
            volume = volume.unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                out = self(volume)
            tokens.append(out.float().cpu())
        return torch.cat(tokens, dim=0)
