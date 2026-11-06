"""Multimodal patient table -> the MedTabX typed contract.

The contract is :class:`~medtabx.benchmarks.typed_bench.TypedData`: every
column carries a value type (numeric / categorical / image tokens), a
modality and a missing mask. Two views are derived from it, so any downstream
model can consume the same data:

* **flat view** ``X [B, F]`` with a feature -> modality map, for scikit-learn
  style learners (LR, RF, GBDT, TabPFN, ...);
* **token view** ``H [B, N, C]`` from the typed token embedding (value +
  type + modality + column + missing), for attention models.

The token view is a shared interface, not a claim that modalities are
aligned: cross-modal interaction is still learned by the downstream model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

import medtabx.core.stype as st
from medtabx.benchmarks.typed_bench import TypedData

#: Semantic types that become numeric columns in the contract.
NUMERIC_STYPES = {st.numerical, st.lab_value, st.vital_sign,
                  st.tabular_derived}
#: Semantic types that become categorical columns.
CATEGORICAL_STYPES = {st.categorical, st.medical_code, st.medication}
#: Semantic types that arrive as image tokens (precomputed or encoded).
IMAGE_STYPES = {st.image_3d, st.mask_3d, st.image_embedded, st.embedding}


@dataclass
class FrameBuilder:
    """Learns the column layout on training data and applies it to new data.

    Args:
        target: label column (binary 0/1).
        modalities: ``{modality: [columns]}``. Columns not listed are
            ignored (IDs, leakage-prone variables).
        stypes: optional ``{column: medtabx stype}``; otherwise inferred
            (object / bool / <= ``max_levels`` distinct integers ->
            categorical, else numeric).
        group: optional column with a site / center label (for shift
            evaluation), kept as ``TypedData.groups``.
        zero_is_missing: numeric columns where 0 means "not measured".
    """

    target: str
    modalities: dict[str, list[str]]
    stypes: dict[str, Any] = field(default_factory=dict)
    group: str | None = None
    zero_is_missing: Sequence[str] = ()
    max_levels: int = 10

    def _kind(self, df: pd.DataFrame, col: str) -> str:
        s = self.stypes.get(col)
        if s is not None:
            if s in NUMERIC_STYPES:
                return "num"
            if s in CATEGORICAL_STYPES:
                return "cat"
            raise ValueError(f"{col}: stype {s} is not a table column; pass "
                             "image modalities through `images=`")
        v = df[col]
        if v.dtype == object or v.dtype == bool or str(v.dtype) == "category":
            return "cat"
        u = v.dropna().unique()
        if (np.issubdtype(v.dtype, np.integer) or np.all(np.mod(u, 1) == 0)) \
                and len(u) <= self.max_levels:
            return "cat"
        return "num"

    def fit(self, df: pd.DataFrame) -> "FrameBuilder":
        self.num_cols, self.num_mod = [], []
        self.cat_cols, self.cat_mod, self.levels = [], [], {}
        for mod, cols in self.modalities.items():
            for c in cols:
                if c not in df:
                    raise KeyError(f"column {c!r} (modality {mod}) not found")
                if self._kind(df, c) == "num":
                    self.num_cols.append(c)
                    self.num_mod.append(mod)
                else:
                    self.cat_cols.append(c)
                    self.cat_mod.append(mod)
                    self.levels[c] = sorted(df[c].dropna().astype(str)
                                            .unique().tolist())
        return self

    def transform(self, df: pd.DataFrame,
                  images: dict[str, np.ndarray] | None = None,
                  image_present: dict[str, np.ndarray] | None = None,
                  ) -> TypedData:
        """Build the contract. ``images[mod]`` is ``[B, K, D]`` (or
        ``[B, D]``) frozen-encoder tokens in the row order of ``df``;
        ``image_present[mod]`` marks rows that have the image (default: rows
        whose tokens are not all zero / NaN)."""
        n = len(df)
        num = None
        if self.num_cols:
            num = df[self.num_cols].apply(pd.to_numeric, errors="coerce") \
                .to_numpy(dtype=float)
            for j, c in enumerate(self.num_cols):
                if c in self.zero_is_missing:
                    num[num[:, j] == 0, j] = np.nan
        cat = None
        if self.cat_cols:
            cat = np.full((n, len(self.cat_cols)), -1, dtype=int)
            for j, c in enumerate(self.cat_cols):
                code = {v: i for i, v in enumerate(self.levels[c])}
                cat[:, j] = df[c].astype(str).where(df[c].notna()) \
                    .map(code).fillna(-1).astype(int).to_numpy()
        img, present, img_mod = None, None, []
        if images:
            arrs, pres = [], []
            for mod, a in images.items():
                a = np.asarray(a, dtype=np.float32)
                if a.ndim == 2:
                    a = a[:, None, :]
                if len(a) != n:
                    raise ValueError(f"images[{mod!r}] has {len(a)} rows, "
                                     f"table has {n}")
                p = (np.asarray(image_present[mod], bool)
                     if image_present and mod in image_present else
                     ~(np.isnan(a).all(axis=(1, 2))
                       | (np.nan_to_num(a) == 0).all(axis=(1, 2))))
                arrs.append(np.nan_to_num(a))
                pres.append(p)
                img_mod.append(mod)
            k = {a.shape[1:] for a in arrs}
            if len(k) != 1:
                raise ValueError(f"image token shapes differ: {k}")
            img = np.stack(arrs, axis=1).astype(np.float16)
            present = np.stack(pres, axis=1)
        y = df[self.target].to_numpy()
        if not set(np.unique(y[~pd.isna(y)])) <= {0, 1}:
            raise ValueError("target must be binary 0/1")
        return TypedData(
            y=y.astype(int),
            num=num, num_cols=list(self.num_cols),
            num_modality=list(self.num_mod),
            cat=cat, cat_cols=list(self.cat_cols),
            cat_modality=list(self.cat_mod),
            cat_cardinality=[max(2, len(self.levels[c]))
                             for c in self.cat_cols],
            img=img, img_present=present, img_modality=img_mod,
            groups=df[self.group].to_numpy() if self.group else None)


def build_frame(df: pd.DataFrame, target: str,
                modalities: dict[str, list[str]],
                images: dict[str, np.ndarray] | None = None,
                image_present: dict[str, np.ndarray] | None = None, **kw
                ) -> tuple[TypedData, FrameBuilder]:
    """One-call version: returns the contract and the fitted builder (use
    ``builder.transform`` on new patients)."""
    b = FrameBuilder(target=target, modalities=modalities, **kw).fit(df)
    return b.transform(df, images=images, image_present=image_present), b


def encode_images(paths: dict[str, Sequence[str | None]],
                  checkpoint: str | None = None, grid: int = 2,
                  device: str = "cuda",
                  encoder: str = "resenc_unet_b") -> dict[str, np.ndarray]:
    """Frozen image-encoder tokens for image paths (``None`` = absent).

    ``encoder`` names a registered image encoder
    (:func:`medtabx.encoders.list_image_encoders`) and ``checkpoint`` its
    weights. Returns ``{modality: [B, grid**3, D]}`` for
    :meth:`FrameBuilder.transform`.
    """
    import torch

    from medtabx.encoders.foundation import FrozenImageTokenizer
    tok = FrozenImageTokenizer(encoder, grid=(grid,) * 3,
                               checkpoint=checkpoint).to(device)
    out = {}
    for mod, ps in paths.items():
        rows = [i for i, p in enumerate(ps) if p]
        z = tok.tokenize_paths([ps[i] for i in rows], torch.device(device))
        z = np.asarray(z.float().cpu() if hasattr(z, "cpu") else z)
        full = np.full((len(ps),) + z.shape[1:], np.nan, np.float32)
        full[rows] = z
        out[mod] = full
    return out


# Views ####################################################################


def describe(d: TypedData) -> dict[str, Any]:
    """Dataset meta-features used by the recommender (and logged with every
    run so a learned recommender can be trained later)."""
    mods = d.modalities
    miss = {}
    for m in mods:
        cols = []
        if d.num is not None:
            cols += [np.isnan(d.num[:, j]) for j, mm in
                     enumerate(d.num_modality) if mm == m]
        if d.cat is not None:
            cols += [d.cat[:, j] < 0 for j, mm in
                     enumerate(d.cat_modality) if mm == m]
        if d.img is not None and m in d.img_modality:
            cols += [~d.img_present[:, d.img_modality.index(m)]]
        miss[m] = float(np.all(cols, axis=0).mean()) if cols else 0.0
    n_num = 0 if d.num is None else d.num.shape[1]
    n_cat = 0 if d.cat is None else d.cat.shape[1]
    n_img = len(d.img_modality)
    cell_missing = []
    if d.num is not None:
        cell_missing.append(np.isnan(d.num).mean())
    if d.cat is not None:
        cell_missing.append((d.cat < 0).mean())
    return {
        "n": int(len(d)), "pos_rate": float(np.mean(d.y)),
        "n_modalities": len(mods), "modalities": mods,
        "n_numeric": n_num, "n_categorical": n_cat,
        "n_image_modalities": n_img,
        "image_tokens": None if d.img is None else list(d.img.shape[2:]),
        "flat_features": n_num + n_cat + n_img * 32,
        "modality_missing": miss,
        "max_modality_missing": max(miss.values()) if miss else 0.0,
        "cell_missing": float(np.mean(cell_missing)) if cell_missing else 0.0,
        "n_groups": (None if d.groups is None
                     else int(pd.Series(d.groups).nunique())),
        "value_types": [t for t, k in (("num", n_num), ("cat", n_cat),
                                       ("img", n_img)) if k],
    }


def flat_view(d: TypedData, img_components: int = 32):
    """``X [B, F]`` (standardized, NaN = missing) and the feature ->
    modality list, as fed to scikit-learn style learners. Fit-on-self; use
    the zoo learners for train/test-consistent statistics."""
    from medtabx.benchmarks.typed_bench import _ModalityBlocks
    blocks = _ModalityBlocks(img_components).fit(d)
    b = blocks.transform(d)
    x = np.concatenate([b[m] for m in blocks.mods], axis=1)
    groups = [m for m in blocks.mods for _ in range(b[m].shape[1])]
    return x, groups


def token_view(d: TypedData, d_model: int = 64, seed: int = 0):
    """``H [B, N, C]`` and the missing mask ``[B, N]`` from an (untrained)
    typed token embedding -- the tensor contract attention models consume."""
    import torch

    from medtabx.models.typed_token import (TypedTokenConfig,
                                            TypedTokenTransformer)
    torch.manual_seed(seed)
    model = TypedTokenTransformer(d.schema(),
                                  TypedTokenConfig(d_model=d_model)).eval()
    num = None
    if d.num is not None:
        mu, sd = np.nanmean(d.num, 0), np.nanstd(d.num, 0)
        sd[~(sd > 0)] = 1.0
        num = torch.tensor((d.num - np.nan_to_num(mu)) / sd,
                           dtype=torch.float32)
    cat = None if d.cat is None else torch.tensor(d.cat)
    img = None if d.img is None else torch.tensor(d.img.astype(np.float32))
    pres = None if d.img is None else torch.tensor(d.img_present)
    with torch.no_grad():
        h, missing = model.tokens(num, cat, img, pres)
    return h, missing
