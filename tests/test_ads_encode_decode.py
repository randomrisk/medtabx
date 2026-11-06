"""End-to-end multimodal encode/decode smoke test on the ADS fixture."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def _resize_3d_volume(array, out_shape=(32, 32, 16)):
    tensor = torch.as_tensor(array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=out_shape, mode="trilinear", align_corners=False)
    return resized.squeeze(0)


def _resize_3d_mask(array, out_shape=(32, 32, 16)):
    tensor = torch.as_tensor(array, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=out_shape, mode="nearest")
    return resized.squeeze(0)


def _simple_text_embedding(text: str, dim: int = 16) -> torch.Tensor:
    values = torch.zeros(dim, dtype=torch.float32)
    values[0] = len(text)
    values[1] = text.count("stroke")
    values[2] = text.count("perfusion")
    values[3] = text.count("left")
    values[4] = text.count("right")
    values[5] = sum(c.isdigit() for c in text)
    values[6] = text.count("\n")
    values[7] = sum(c in "aeiouAEIOU" for c in text)
    values[8] = sum(c.isupper() for c in text)
    values[9] = sum(c.islower() for c in text)
    values[10] = sum(c == "." for c in text)
    values[11] = sum(c == "," for c in text)
    values[12] = sum(c == ":" for c in text)
    values[13] = sum(c == "-" for c in text)
    values[14] = len(text.split())
    values[15] = len(set(text.split()))
    if values.abs().max() > 0:
        values = values / values.abs().max()
    return values


def test_ads_multimodal_encode_decode():
    from medtabx.encoders.fmri import FMRIEncoder
    from medtabx.encoders.imaging3d import Imaging3DEncoder
    from medtabx.encoders.mask3d import Mask3DEncoder
    from medtabx.encoders.numeric import NumericEncoder
    from medtabx.encoders.text import TextEmbeddedEncoder
    from medtabx.decoders.classification import ClassificationDecoder
    from medtabx.decoders.regression import RegressionDecoder
    from medtabx.features.ads_features import build_ads_subject_feature_row
    from medtabx.features.perfusion import extract_roi_time_series
    from medtabx.interaction.pooling_fusion import MeanPoolingFusion

    subject_root = Path(__file__).parent.parent / "examples" / "data" / "ads_case" / "sub-02e8eb42"
    row = build_ads_subject_feature_row(subject_root)

    dwi = nib.load(row["dwi_path"]).get_fdata()
    stroke_mask = nib.load(row["stroke_mask_path"]).get_fdata()
    hp_mask = nib.load(row["hp_mask_path"]).get_fdata()
    pwi = nib.load(row["pwi_path"]).get_fdata()

    # Numerical derived features
    numeric_cols = [
        "pwi_repetition_time",
        "pwi_echo_time",
        "pwi_num_timepoints",
        "pwi_peak_timepoint_index",
        "brain_ttp_mean",
        "brain_cbf_mean",
        "stroke_mask_voxels",
        "hp_mask_voxels",
    ]
    numeric_x = torch.tensor([[float(row[col]) for col in numeric_cols]], dtype=torch.float32)

    # Imaging
    dwi_small = _resize_3d_volume(dwi)
    image_x = dwi_small.unsqueeze(0).unsqueeze(0)  # [B, num_cols, C, D, H, W]

    # Masks
    stroke_small = _resize_3d_mask(stroke_mask)
    hp_small = _resize_3d_mask(hp_mask)
    mask_x = torch.stack([stroke_small, hp_small], dim=0).unsqueeze(0)  # [B, num_cols, C, D, H, W]

    # PWI ROI time series: split left/right hemisphere in PWI space
    atlas = torch.zeros(pwi.shape[:3], dtype=torch.int64)
    mid = atlas.shape[0] // 2
    atlas[:mid, :, :] = 1
    atlas[mid:, :, :] = 2
    roi_series = extract_roi_time_series(pwi, atlas.numpy())
    roi_array = np.stack([roi_series[1], roi_series[2]], axis=0)
    roi_x = torch.from_numpy(roi_array).to(torch.float32).unsqueeze(0)
    roi_x = roi_x.view(1, 1, 2, -1)

    # Text embeddings from reports
    text_x = torch.stack(
        [
            _simple_text_embedding(row["dwi_report_text"]),
            _simple_text_embedding(row["pwi_report_text"]),
        ],
        dim=0,
    ).unsqueeze(0)

    emb_dim = 32
    numeric_encoder = NumericEncoder(embedding_dim=emb_dim, num_cols=numeric_x.shape[1])
    image_encoder = Imaging3DEncoder(embedding_dim=emb_dim, in_channels=1)
    mask_encoder = Mask3DEncoder(embedding_dim=emb_dim, num_classes=1)
    fmri_encoder = FMRIEncoder(embedding_dim=emb_dim, num_rois=2, seq_len=roi_x.shape[-1])
    text_encoder = TextEmbeddedEncoder(embedding_dim=emb_dim, input_dim=text_x.shape[-1])

    numeric_emb = numeric_encoder(numeric_x)
    image_emb = image_encoder(image_x)
    mask_emb = mask_encoder(mask_x)
    fmri_emb = fmri_encoder(roi_x)
    text_emb = text_encoder(text_x)

    modalities = {
        "numeric": numeric_emb,
        "image": image_emb,
        "mask": mask_emb,
        "fmri": fmri_emb,
        "text": text_emb,
    }
    fusion = MeanPoolingFusion(channels=emb_dim)
    fused = fusion(modalities).unsqueeze(1)

    class_decoder = ClassificationDecoder(embedding_dim=emb_dim, num_classes=2)
    reg_decoder = RegressionDecoder(embedding_dim=emb_dim, num_targets=1)

    class_logits = class_decoder(fused)
    reg_pred = reg_decoder(fused)

    assert class_logits.shape == (1, 2)
    assert reg_pred.shape == (1, 1)

    class_target = torch.tensor([1], dtype=torch.long)
    reg_target = torch.tensor([[float(row["brain_ttp_mean"])]], dtype=torch.float32)
    loss = F.cross_entropy(class_logits, class_target) + F.mse_loss(reg_pred, reg_target)
    loss.backward()
