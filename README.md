# MedTabX

MedTabX is a Python package for multimodal tabular learning on clinical data.
It represents clinical tables, imaging-derived measurements and frozen image
encodings in one typed data contract, and it runs classical learners,
gradient-boosted trees, deep tabular networks and tabular foundation models
on that contract with the same data splits and the same missing-modality
protocol.

## Main components

- **Typed data contract.** Every column keeps its value type (numerical,
  categorical, embedding, ...), its modality and its observation state. The
  contract gives a flat view `X [B, F]` for feature-vector learners and a
  token view `H [B, N, C]` for attention models.
- **Shared testbed.** Concatenation, late, subset and gated fusion for any
  supported learner, evaluated on fixed outer folds.
- **Diagnostics.** Modality imbalance, prediction alignment, width-matched
  shuffled controls and controlled missing-modality tests.
- **Failure-aware recommendation.** In-fold modality selection and a
  budgeted search over learner and fusion configurations (`AutoRecipe`).
- **Medical inputs.** Loaders and encoders for NIfTI and DICOM images, a
  frozen 3D U-Net encoder for image tokens, atlas-based lesion features,
  perfusion features, EEG, ECG, vital signs and clinical text.

## Installation

```bash
git clone https://github.com/randomrisk/medtabx.git
cd medtabx
pip install -e .                 # core
pip install -e ".[all]"          # imaging, signals, text and all learners
```

Python 3.10 or later and PyTorch 2.0 or later are required. The core install
includes nibabel and SimpleITK for image input. The optional
learners (XGBoost, LightGBM, CatBoost, TabPFN) are used when they are
installed.

## Quick start

```python
from medtabx.auto import AutoRecipe, build_frame, compare_models

modalities = {"demographics": ["age", "sex"],
              "presentation": ["nihss"],
              "labs": ["glucose", "inr"]}
data, builder = build_frame(df_train, "poor_outcome", modalities,
                            images={"dwi": dwi_tokens_train})

board = compare_models(data, ["lr", "lr@late", "rf@late", "et@subset"])
auto = AutoRecipe("balanced", budget=40).fit(data)
print(auto.explain())

new = builder.transform(df_test, images={"dwi": dwi_tokens_test})
p = auto.predict(new)
```

A complete version on synthetic data is in `examples/auto_recipe_demo.py`.

## Image encoders

Images enter the contract as tokens from a frozen 3D encoder. The encoder is
chosen by name; all encoders give tokens of the same form `[B, K, C]`.

```python
from medtabx.encoders import list_image_encoders
from medtabx.auto.frame import encode_images

print(list_image_encoders())
tokens = encode_images({"dwi": dwi_paths}, encoder="medicalnet_resnet10")
```

| Name | Token width | Weights | License of the weights |
|---|---|---|---|
| `resenc_unet_b` | 320 | Your checkpoint of the residual U-Net-B encoder (FOMO challenge baseline architecture), for example FOMO baseline weights or self-supervised weights such as LeJEPA pretraining on FOMO300K | Depends on the checkpoint |
| `medicalnet_resnet10`, `medicalnet_resnet50` | 512, 2048 | MedicalNet (Med3D), downloaded by MONAI | MIT |
| `swinunetr_ssl` | 768 | Swin UNETR self-supervised weights (Tang et al., CVPR 2022), downloaded from the MONAI release | Apache-2.0 |
| `brainiac_vit_b` | 768 | BrainIAC ViT-B/16 (Tak et al., 2026); download the checkpoint from the BrainIAC project | Non-commercial research |

The MONAI-based encoders need `pip install -e ".[encoders]"`. To add another
encoder, register it with `medtabx.encoders.register_image_encoder`.

## Repository layout

| Path | Content |
|---|---|
| `medtabx/` | The package |
| `examples/` | Short, runnable examples on synthetic or public sample data |
| `experiments/` | Scripts for the experiments of the accompanying paper |
| `tests/` | Unit tests (`pytest`) |

The graphical interface starts with `python -m medtabx.app`.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"
```

Tests that need files that are not in this repository are skipped.


## License

MIT License (see `LICENSE`). The package includes third-party code under its
own licenses; see `THIRD_PARTY_NOTICES.md`.
