"""One model per modality, plus its cross-fitted predictions.

Every failure-aware method in this package starts from the same object: a
learner per modality, fitted on the training rows where that modality is
present, and the same learner's out-of-fold predictions on the training set.
The out-of-fold matrix is what a selector or a router may learn from without
seeing its own training labels twice.

``P[i, j]`` is P(y=1) for patient *i* from the model of modality *j*, and
NaN when patient *i* does not have modality *j*. Uniform late fusion is the
row mean over the observed entries, which is how
:class:`~medtabx.benchmarks.typed_bench.ModalitySubsetEnsemble` with
``singletons_only=True`` scores a patient.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData, _ModalityBlocks


def make_learner(base: str | Callable[[int], object], seed: int,
                 tabpfn_ckpt: str | None = None):
    """``"lr"``, ``"tabpfn"`` or a factory ``seed -> estimator``."""
    if callable(base):
        return base(seed)
    if base == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=0.05, max_iter=5000, random_state=seed)
    if base == "tabpfn":
        from tabpfn import TabPFNClassifier
        from medtabx.benchmarks.typed_bench import _device, _tabpfn_ckpt
        kw = {"device": str(_device()), "random_state": seed}
        # the repo-local weights, as in the zoo; without them TabPFN tries
        # to download and needs an interactive license acceptance
        ckpt = tabpfn_ckpt or _tabpfn_ckpt()
        if ckpt:
            kw["model_path"] = ckpt
        return TabPFNClassifier(**kw)
    from medtabx.auto.zoo import BASES
    if base in BASES and BASES[base].factory is not None:
        return BASES[base].factory(seed)
    raise KeyError(f"unknown base learner {base!r}")


def late_mean(p: np.ndarray, fallback: float = 0.5) -> np.ndarray:
    """Uniform late fusion of a ``[N, M]`` matrix with NaN for absent."""
    seen = ~np.isnan(p)
    count = seen.sum(axis=1)
    total = np.where(seen, p, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), fallback)


def stratified_splits(y: np.ndarray, n_splits: int, seed: int):
    """Stratified folds, with fewer folds when a class is rare."""
    from sklearn.model_selection import StratifiedKFold
    k = min(n_splits, int(np.bincount(y.astype(int)).min()))
    if k < 2:
        return []
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


class ModalityExperts:
    """Per-modality learners and their out-of-fold predictions.

    Args:
        base: per-modality learner (see :func:`make_learner`).
        img_components: PCA width per image modality.
        impute: fill NaN with 0 (the training mean) before the learner;
            default only for ``"lr"``.
        inner_splits, inner_repeats: cross-fitting for the out-of-fold
            matrix; repeats are averaged.
        min_rows: a modality observed in fewer training rows is skipped.
    """

    def __init__(self, base="lr", img_components: int = 32,
                 impute: bool | None = None, inner_splits: int = 5,
                 inner_repeats: int = 1, min_rows: int = 10,
                 tabpfn_ckpt: str | None = None) -> None:
        self.base = base
        self.img_components = img_components
        self.impute = (base == "lr") if impute is None else impute
        self.inner_splits = inner_splits
        self.inner_repeats = inner_repeats
        self.min_rows = min_rows
        self.tabpfn_ckpt = tabpfn_ckpt

    def _design(self, block: np.ndarray, rows) -> np.ndarray:
        x = block[rows]
        return np.nan_to_num(x) if self.impute else x

    def _fit_one(self, block, y, rows, seed):
        return make_learner(self.base, seed, self.tabpfn_ckpt).fit(
            self._design(block, rows), y[rows])

    def _usable(self, y, rows) -> bool:
        return rows.sum() >= self.min_rows and len(np.unique(y[rows])) == 2

    def fit(self, d: TypedData, seed: int = 0, oof: bool = True,
            shuffle: dict[str, np.ndarray] | None = None):
        """Fit one learner per modality on all training rows.

        With ``oof``, also cross-fit the ``[N, M]`` matrix :attr:`oof_`.
        ``shuffle`` maps a modality to a row permutation applied to that
        modality's block before anything is fitted: the permutation control
        of :mod:`medtabx.analysis.info`, at the level of one modality.
        """
        self.blocks = _ModalityBlocks(self.img_components).fit(d)
        b = self.blocks.transform(d)
        if shuffle:
            b = {m: (v[shuffle[m]] if m in shuffle else v)
                 for m, v in b.items()}
        y = d.y.astype(int)
        self.mods: list[str] = []
        self.models: dict[str, object] = {}
        for m in self.blocks.mods:
            rows = ~np.isnan(b[m]).all(axis=1)
            if self._usable(y, rows):
                self.mods.append(m)
                self.models[m] = self._fit_one(b[m], y, rows, seed)
        if not self.mods:
            raise ValueError("no modality had enough observed rows to fit")
        if oof:
            self.oof_ = self._cross_fit(b, y, seed)
        return self

    def _cross_fit(self, b: dict[str, np.ndarray], y: np.ndarray,
                   seed: int) -> np.ndarray:
        n = len(y)
        total = np.zeros((n, len(self.mods)))
        count = np.zeros((n, len(self.mods)))
        for r in range(self.inner_repeats):
            for tr, te in stratified_splits(y, self.inner_splits,
                                            seed + 1000 * r):
                train = np.zeros(n, bool)
                train[tr] = True
                for j, m in enumerate(self.mods):
                    seen = ~np.isnan(b[m]).all(axis=1)
                    rows = train & seen
                    test = te[seen[te]]
                    if not self._usable(y, rows) or len(test) == 0:
                        continue
                    model = self._fit_one(b[m], y, rows, seed + r)
                    total[test, j] += model.predict_proba(
                        self._design(b[m], test))[:, 1]
                    count[test, j] += 1
        out = np.full((n, len(self.mods)), np.nan)
        ok = count > 0
        out[ok] = total[ok] / count[ok]
        return out

    def present(self, d: TypedData) -> np.ndarray:
        """``[N, M]`` bool: which fitted modalities each patient has."""
        b = self.blocks.transform(d)
        return np.column_stack([~np.isnan(b[m]).all(axis=1)
                                for m in self.mods])

    def predict_matrix(self, d: TypedData) -> np.ndarray:
        """``[N, M]`` P(y=1) per modality, NaN where the modality is absent.
        """
        b = self.blocks.transform(d)
        out = np.full((len(d), len(self.mods)), np.nan)
        for j, m in enumerate(self.mods):
            rows = ~np.isnan(b[m]).all(axis=1)
            if rows.any():
                out[rows, j] = self.models[m].predict_proba(
                    self._design(b[m], rows))[:, 1]
        return out
