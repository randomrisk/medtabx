"""Registry of frozen 3D image encoders.

An image encoder turns one preprocessed volume ``[B, 1, X, Y, Z]`` into its
deepest feature map ``[B, C, x, y, z]``. :class:`~medtabx.encoders.foundation.
FrozenImageTokenizer` pools that map onto a token grid, so every encoder in
this registry gives image tokens of the same form ``[B, K, C]``.

Built-in encoders:

==========================  ======  ==========================================
Name                        Width   Weights
==========================  ======  ==========================================
``resenc_unet_b``           320     Any checkpoint of the residual U-Net-B
                                    encoder (FOMO challenge baseline
                                    architecture), for example weights from
                                    self-supervised pretraining on FOMO300K.
``medicalnet_resnet10``     512     MedicalNet (Med3D), downloaded by MONAI.
``medicalnet_resnet50``     2048    MedicalNet (Med3D), downloaded by MONAI.
``swinunetr_ssl``           768     Swin UNETR self-supervised weights
                                    (Tang et al., CVPR 2022).
``brainiac_vit_b``          768     BrainIAC ViT-B/16 (Tak et al., 2026);
                                    the checkpoint file is given by the user.
==========================  ======  ==========================================

The MONAI-based encoders need ``pip install monai``. Each set of weights keeps
the license of its source; check it before use.

Register your own encoder with :func:`register_image_encoder`::

    register_image_encoder(ImageEncoderSpec(
        name="my_encoder", out_channels=256, build=my_build_function))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

__all__ = [
    "ImageEncoderSpec",
    "register_image_encoder",
    "list_image_encoders",
    "get_image_encoder_spec",
    "load_image_encoder",
]


@dataclass(frozen=True)
class ImageEncoderSpec:
    r"""Description of one frozen image encoder.

    Args:
        name: Registry key.
        out_channels: Channels of the returned feature map (the token width).
        build: ``build(checkpoint, **kwargs) -> nn.Module``. The module maps
            ``[B, 1, X, Y, Z]`` to ``[B, C, x, y, z]``.
        description: One line for :func:`list_image_encoders`.
        size_multiple: Each spatial axis is zero-padded to a multiple of this.
        fixed_size: If set, volumes are resized to this shape (for encoders
            with a fixed input size, such as vision transformers).
        spacing: Isotropic voxel size in mm used during pretraining;
            :obj:`None` keeps the native grid.
        normalization: ``"zscore"`` (whole volume) or ``"zscore_nonzero"``
            (statistics over nonzero voxels).
        requires: Python packages that :attr:`build` imports.
        source: Where the weights come from, and their license.
    """

    name: str
    out_channels: int
    build: Callable[..., nn.Module]
    description: str = ""
    size_multiple: int = 32
    fixed_size: tuple[int, int, int] | None = None
    spacing: float | None = 1.0
    normalization: str = "zscore"
    requires: tuple[str, ...] = ()
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def preprocess_kwargs(self) -> dict:
        return {"spacing": self.spacing, "size_multiple": self.size_multiple,
                "fixed_size": self.fixed_size,
                "normalization": self.normalization}


_REGISTRY: dict[str, ImageEncoderSpec] = {}


def register_image_encoder(spec: ImageEncoderSpec,
                           overwrite: bool = False) -> ImageEncoderSpec:
    """Add an encoder to the registry."""
    if spec.name in _REGISTRY and not overwrite:
        raise KeyError(f"Image encoder '{spec.name}' is already registered.")
    _REGISTRY[spec.name] = spec
    return spec


def list_image_encoders() -> dict[str, str]:
    """``{name: description}`` of every registered encoder."""
    return {name: spec.description for name, spec in _REGISTRY.items()}


def get_image_encoder_spec(name: str) -> ImageEncoderSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown image encoder '{name}'. Registered: "
                       f"{sorted(_REGISTRY)}") from None


def load_image_encoder(name: str, checkpoint: str | Path | None = None,
                       **kwargs) -> tuple[nn.Module, ImageEncoderSpec]:
    r"""Build a registered encoder and load its weights.

    Args:
        name: Registry key, see :func:`list_image_encoders`.
        checkpoint: Path to the weights. :obj:`None` downloads the default
            weights where the encoder has them. ``"random"`` keeps random
            initialization (for controls and tests).

    Returns:
        ``(module, spec)``; the module returns ``[B, C, x, y, z]``.
    """
    spec = get_image_encoder_spec(name)
    for package in spec.requires:
        try:
            __import__(package)
        except ImportError as err:
            raise ImportError(
                f"Image encoder '{name}' needs the package '{package}' "
                f"(pip install {package}).") from err
    module = spec.build(checkpoint, **kwargs)
    return module, spec


def _is_random(checkpoint) -> bool:
    return isinstance(checkpoint, str) and checkpoint == "random"


def _load_state_dict(path: str | Path) -> dict[str, Tensor]:
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 -- e.g. Lightning checkpoints
        warnings.warn(f"{path} is not a plain tensor file; loading it with "
                      f"weights_only=False. Load only files you trust.")
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(state, dict) and key in state and isinstance(
                state[key], dict):
            state = state[key]
            break
    return state


def _strip_prefix(state: dict[str, Tensor],
                  prefixes: tuple[str, ...]) -> dict[str, Tensor]:
    for prefix in prefixes:
        if prefix and any(k.startswith(prefix) for k in state):
            return {k[len(prefix):]: v for k, v in state.items()
                    if k.startswith(prefix)}
    return state


# Residual U-Net-B ##########################################################


class _DeepestStage(nn.Module):
    """Wrap an encoder that returns a list of stage outputs."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)[-1]


def _build_resenc_unet_b(checkpoint=None, prefix: str = "model.encoder."):
    from .foundation import build_resenc_b_encoder, load_foundation_encoder
    if checkpoint is None:
        raise ValueError(
            "resenc_unet_b needs a checkpoint path, for example a FOMO "
            "challenge baseline checkpoint or a self-supervised checkpoint; "
            "use checkpoint='random' for random initialization.")
    encoder = (build_resenc_b_encoder() if _is_random(checkpoint) else
               load_foundation_encoder(checkpoint, prefix=prefix))
    return _DeepestStage(encoder)


register_image_encoder(ImageEncoderSpec(
    name="resenc_unet_b",
    out_channels=320,
    build=_build_resenc_unet_b,
    description="Residual U-Net-B encoder (FOMO baseline architecture); "
                "user checkpoint",
    size_multiple=32,
    spacing=1.0,
    normalization="zscore",
    source="Architecture: gardening_tools / asparagus (Apache-2.0). Weights: "
           "the checkpoint given by the user.",
))


# MedicalNet ResNet (MONAI) #################################################


class _ResNetFeatures(nn.Module):
    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.resnet = resnet

    def forward(self, x: Tensor) -> Tensor:
        r = self.resnet
        x = r.act(r.bn1(r.conv1(x)))
        if not r.no_max_pool:
            x = r.maxpool(x)
        return r.layer4(r.layer3(r.layer2(r.layer1(x))))


def _medicalnet_builder(depth: int):
    def build(checkpoint=None):
        from monai.networks.nets import resnet as monai_resnet
        bias_downsample, shortcut = (
            monai_resnet.get_medicalnet_pretrained_resnet_args(depth))
        factory = getattr(monai_resnet, f"resnet{depth}")
        kwargs = dict(spatial_dims=3, n_input_channels=1, feed_forward=False,
                      shortcut_type=shortcut,
                      bias_downsample=bias_downsample)
        if _is_random(checkpoint):
            model = factory(pretrained=False, **kwargs)
        elif checkpoint is None:
            model = factory(pretrained=True, **kwargs)
        else:
            model = factory(pretrained=False, **kwargs)
            state = _strip_prefix(_load_state_dict(checkpoint),
                                  ("module.", "model."))
            model.load_state_dict(state, strict=False)
        return _ResNetFeatures(model)
    return build


for _depth, _width in ((10, 512), (50, 2048)):
    register_image_encoder(ImageEncoderSpec(
        name=f"medicalnet_resnet{_depth}",
        out_channels=_width,
        build=_medicalnet_builder(_depth),
        description=f"MedicalNet 3D ResNet-{_depth} (Med3D, 23 datasets); "
                    f"downloaded by MONAI",
        size_multiple=16,
        spacing=1.0,
        normalization="zscore_nonzero",
        requires=("monai",),
        source="Chen et al., Med3D (2019); "
               "https://huggingface.co/TencentMedicalNet; MIT License.",
    ))


# Swin UNETR self-supervised encoder (MONAI) ################################

SWINUNETR_SSL_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/"
                     "releases/download/0.8.1/model_swinvit.pt")


class _SwinFeatures(nn.Module):
    def __init__(self, swin_unetr: nn.Module) -> None:
        super().__init__()
        self.swin_vit = swin_unetr.swinViT

    def forward(self, x: Tensor) -> Tensor:
        return self.swin_vit(x, True)[-1]


def _build_swinunetr_ssl(checkpoint=None):
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(in_channels=1, out_channels=1, feature_size=48,
                      spatial_dims=3)
    if not _is_random(checkpoint):
        if checkpoint is None:
            weights = torch.hub.load_state_dict_from_url(
                SWINUNETR_SSL_URL, map_location="cpu", weights_only=True)
        else:
            weights = {"state_dict": _load_state_dict(checkpoint)}
        model.load_from(weights)
    return _SwinFeatures(model)


register_image_encoder(ImageEncoderSpec(
    name="swinunetr_ssl",
    out_channels=768,
    build=_build_swinunetr_ssl,
    description="Swin UNETR encoder, self-supervised on 5,050 CT volumes",
    size_multiple=32,
    spacing=1.0,
    normalization="zscore_nonzero",
    requires=("monai",),
    source="Tang et al., CVPR 2022; MONAI release 0.8.1 (Apache-2.0).",
))


# BrainIAC ViT-B/16 (MONAI) #################################################


class _ViTFeatures(nn.Module):
    def __init__(self, vit: nn.Module, grid: tuple[int, int, int]) -> None:
        super().__init__()
        self.vit = vit
        self.grid = grid

    def forward(self, x: Tensor) -> Tensor:
        tokens, _ = self.vit(x)
        if tokens.shape[1] == int(torch.tensor(self.grid).prod()) + 1:
            tokens = tokens[:, 1:]  # drop a class token
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, *self.grid)


def _build_brainiac_vit_b(checkpoint=None):
    from monai.networks.nets import ViT
    vit = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
              hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
              proj_type="conv", classification=False, spatial_dims=3)
    if checkpoint is None:
        raise ValueError(
            "brainiac_vit_b needs the BrainIAC backbone checkpoint "
            "(see https://github.com/AIM-KannLab/BrainIAC); use "
            "checkpoint='random' for random initialization.")
    if not _is_random(checkpoint):
        state = _strip_prefix(_load_state_dict(checkpoint),
                              ("backbone.", "model.backbone.", "module.",
                               "model.", "encoder."))
        missing, unexpected = vit.load_state_dict(state, strict=False)
        if missing:
            raise KeyError(f"BrainIAC checkpoint is missing {len(missing)} "
                           f"ViT parameters, for example {missing[:3]}.")
        if unexpected:
            warnings.warn(f"Ignored {len(unexpected)} checkpoint keys, for "
                          f"example {unexpected[:3]}.")
    return _ViTFeatures(vit, (6, 6, 6))


register_image_encoder(ImageEncoderSpec(
    name="brainiac_vit_b",
    out_channels=768,
    build=_build_brainiac_vit_b,
    description="BrainIAC ViT-B/16, SimCLR on structural brain MRI; "
                "user checkpoint",
    size_multiple=16,
    fixed_size=(96, 96, 96),
    spacing=None,
    normalization="zscore_nonzero",
    requires=("monai",),
    source="Tak et al., Nature Neuroscience 2026; "
           "https://github.com/AIM-KannLab/BrainIAC; non-commercial license.",
))
