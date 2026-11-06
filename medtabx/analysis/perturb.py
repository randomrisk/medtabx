"""Controlled test-time corruption of one modality.

A failure-aware model should trust a modality less for the patients whose
copy of it is bad. To test that, :func:`corrupt` damages one modality for a
chosen fraction of patients and leaves everything else bit-for-bit
unchanged, so the corrupted and clean patients can be compared inside the
same test fold.

Two kinds of damage:

* ``"swap"`` -- the patient receives another patient's values for that
  modality (a mislabeled or mismatched acquisition). Values stay realistic;
  only the link to this patient is lost. This is the permutation control of
  :mod:`medtabx.analysis.info`, applied to selected patients.
* ``"noise"`` -- Gaussian noise of ``sigma`` column standard deviations is
  added (a degraded acquisition). Categorical columns, which have no scale,
  are swapped instead.

Only patients who have the modality can be corrupted; a missing modality
stays missing.
"""

from __future__ import annotations

import copy

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData


def _present(d: TypedData, modality: str) -> np.ndarray:
    seen = np.zeros(len(d), bool)
    if d.num is not None:
        cols = [j for j, m in enumerate(d.num_modality) if m == modality]
        if cols:
            seen |= ~np.isnan(d.num[:, cols]).all(axis=1)
    if d.cat is not None:
        cols = [j for j, m in enumerate(d.cat_modality) if m == modality]
        if cols:
            seen |= (d.cat[:, cols] >= 0).any(axis=1)
    if d.img is not None and modality in d.img_modality:
        seen |= d.img_present[:, d.img_modality.index(modality)]
    return seen


def corrupt(d: TypedData, modality: str, frac: float, kind: str = "swap",
            sigma: float = 1.0, seed: int = 0
            ) -> tuple[TypedData, np.ndarray]:
    """Copy of ``d`` with ``modality`` corrupted for a fraction of patients.

    Returns the corrupted copy and the ``[N]`` bool mask of corrupted rows.
    """
    if kind not in ("swap", "noise"):
        raise ValueError(f"kind must be 'swap' or 'noise', not {kind!r}")
    if modality not in d.modalities:
        raise KeyError(f"unknown modality {modality!r}")
    rng = np.random.default_rng(seed)
    eligible = np.nonzero(_present(d, modality))[0]
    k = int(round(frac * len(eligible)))
    mask = np.zeros(len(d), bool)
    out = copy.copy(d)
    if k == 0 or len(eligible) < 2:
        return out, mask
    rows = np.sort(rng.choice(eligible, k, replace=False))
    mask[rows] = True
    # a donor for every corrupted row: another patient with the modality
    donors = np.array([rng.choice(eligible[eligible != i]) for i in rows])

    if d.num is not None:
        cols = [j for j, m in enumerate(d.num_modality) if m == modality]
        if cols:
            out.num = d.num.copy()
            if kind == "swap":
                out.num[np.ix_(rows, cols)] = d.num[np.ix_(donors, cols)]
            else:
                sd = np.nanstd(d.num[:, cols], axis=0)
                sd[~(sd > 0)] = 1.0
                block = out.num[np.ix_(rows, cols)]
                noisy = block + rng.normal(size=block.shape) * sigma * sd
                out.num[np.ix_(rows, cols)] = np.where(np.isnan(block),
                                                       np.nan, noisy)
    if d.cat is not None:
        cols = [j for j, m in enumerate(d.cat_modality) if m == modality]
        if cols:
            out.cat = d.cat.copy()
            out.cat[np.ix_(rows, cols)] = d.cat[np.ix_(donors, cols)]
    if d.img is not None and modality in d.img_modality:
        j = d.img_modality.index(modality)
        out.img = d.img.copy()
        if kind == "swap":
            out.img[rows, j] = d.img[donors, j]
        else:
            sd = float(d.img[d.img_present[:, j], j].std()) or 1.0
            out.img[rows, j] = (d.img[rows, j]
                                + rng.normal(size=d.img[rows, j].shape)
                                * sigma * sd)
    return out, mask
