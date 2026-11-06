"""Acquisition-aware gating: decide what to trust from how it was acquired.

The fusions in :mod:`medtabx.auto.zoo` combine modalities either uniformly
(``late``) or through a gate learned from the task label (the typed token
model's ``gated_experts`` head). With a few hundred labelled patients the
second can be hard to fit.

HACA3 (Zuo et al., 2023) solves the analogous problem in MR harmonization
differently: its attention weights come from the *contrast and artifact*
representations of the images, never from the downstream objective, so the
model asks "which of these acquisitions should I believe" and not "which
one predicted well". That question needs no labels and can therefore be
answered per patient.

This module ports that idea to the typed contract:

    alpha_m(x)  propto  present_m(x) * reliability_m ** w_r * typicality_m(x) ** w_q

* ``reliability_m`` is how well modality *m* predicts on its own, estimated
  once per training fold by inner cross-validation. It is a per-modality
  scalar, not a per-patient quantity, so it stays stable at small n.
* ``typicality_m(x)`` is label-free: how ordinary this patient's modality-*m*
  features look next to the training distribution of that modality. An
  outlying scan (motion, a different protocol, a failed registration) gets a
  low value and is down-weighted for that patient only.

Both exponents are exposed, so the same class covers uniform late fusion
(``w_r = w_q = 0``), reliability weighting alone, quality weighting alone,
and the combination. Nothing here imputes a missing modality: an absent
block simply gets weight zero and the rest are renormalized, which is also
how HACA3 handles a missing contrast.
"""

from __future__ import annotations

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData, _ModalityBlocks


class AcquisitionGatedFusion:
    """One model per modality, combined by an acquisition-aware gate.

    Args:
        base: ``"lr"``, ``"tabpfn"``, or a factory ``seed -> estimator`` with
            scikit-learn ``fit`` / ``predict_proba``.
        w_reliability: exponent on the per-modality reliability term; 0
            ignores how well a modality predicts.
        w_quality: exponent on the per-patient typicality term; 0 ignores
            how unusual this patient's acquisition looks.
        temperature: softens (``>1``) or sharpens (``<1``) the weights.
        inner_splits: folds used to estimate reliability inside the training
            fold.
    """

    def __init__(self, name: str, base="lr", w_reliability: float = 1.0,
                 w_quality: float = 1.0, temperature: float = 1.0,
                 inner_splits: int = 5, img_components: int = 32,
                 impute: bool | None = None,
                 tabpfn_ckpt: str | None = None) -> None:
        self.name = name
        self.base = base
        self.w_reliability = w_reliability
        self.w_quality = w_quality
        self.temperature = max(temperature, 1e-6)
        self.inner_splits = inner_splits
        self.img_components = img_components
        self.impute = (base == "lr") if impute is None else impute
        self.tabpfn_ckpt = tabpfn_ckpt
        self._reliability_y = None

    def set_reliability_target(self, y) -> None:
        """Measure reliability against ``y`` instead of the fitted label.

        The ordinal harness uses this to separate two questions that are
        otherwise confounded: whether gating helps at all, and whether it
        helps *because* a modality's reliability is allowed to differ between
        cut-points. Passing the shared dichotomy here freezes reliability
        across thresholds; leaving it unset lets each threshold estimate its
        own, which is the behaviour the cumulative decomposition gives for
        free.
        """
        self._reliability_y = None if y is None else np.asarray(y)

    # -- the per-modality learner ------------------------------------------

    def _learner(self, seed: int):
        if callable(self.base):
            return self.base(seed)
        if self.base == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.05, max_iter=5000,
                                      random_state=seed)
        from tabpfn import TabPFNClassifier
        from medtabx.benchmarks.typed_bench import _device
        kwargs = {"device": str(_device()), "random_state": seed}
        if self.tabpfn_ckpt:
            kwargs["model_path"] = self.tabpfn_ckpt
        return TabPFNClassifier(**kwargs)

    def _design(self, block: np.ndarray, rows=None) -> np.ndarray:
        x = block if rows is None else block[rows]
        return np.nan_to_num(x) if self.impute else x

    # -- fitting -----------------------------------------------------------

    def fit(self, d: TypedData, seed: int = 0):
        self.blocks = _ModalityBlocks(self.img_components).fit(d)
        b = self.blocks.transform(d)
        y_rel = d.y if self._reliability_y is None else self._reliability_y
        if len(y_rel) != len(d):
            raise ValueError(
                f"reliability target has {len(y_rel)} rows, data has {len(d)}")
        self.mods = [m for m in self.blocks.mods]
        self.models: dict[str, object] = {}
        self.reliability: dict[str, float] = {}
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for m in self.mods:
            block = b[m]
            observed = ~np.isnan(block).all(axis=1)
            if observed.sum() < 10 or len(np.unique(d.y[observed])) < 2:
                continue
            # how ordinary a patient's features are, in this modality
            seen = np.nan_to_num(block[observed])
            mu = seen.mean(axis=0)
            sd = seen.std(axis=0)
            sd[~(sd > 0)] = 1.0
            self.stats[m] = (mu, sd)
            self.models[m] = self._learner(seed).fit(
                self._design(block, observed), d.y[observed])
            self.reliability[m] = self._estimate_reliability(
                block, y_rel, observed, seed)

        if not self.models:
            raise ValueError("no modality had enough observed rows to fit")
        # scale reliabilities to a mean of one so the exponent is the only
        # thing that controls how much they matter
        values = np.array(list(self.reliability.values()), dtype=float)
        scale = values.mean() if values.mean() > 0 else 1.0
        self.reliability = {k: v / scale for k, v in self.reliability.items()}
        return self

    def _estimate_reliability(self, block: np.ndarray, y: np.ndarray,
                              observed: np.ndarray, seed: int) -> float:
        """Cross-fitted skill of this modality alone, as AUROC above chance.

        Uses labels, but only to produce one number per modality, which is
        far easier to estimate at these sample sizes than a per-patient gate.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        idx = np.nonzero(observed)[0]
        y_obs = y[idx]
        if len(idx) < 20 or len(np.unique(y_obs)) < 2:
            return 0.5
        p = np.full(len(idx), np.nan)
        splits = min(self.inner_splits, int(np.bincount(y_obs).min()), 5)
        if splits < 2:
            return 0.5
        skf = StratifiedKFold(splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros(len(idx)), y_obs):
            if len(np.unique(y_obs[tr])) < 2:
                continue
            model = self._learner(seed).fit(
                self._design(block, idx[tr]), y_obs[tr])
            p[te] = model.predict_proba(self._design(block, idx[te]))[:, 1]
        ok = ~np.isnan(p)
        if ok.sum() < 10 or len(np.unique(y_obs[ok])) < 2:
            return 0.5
        auc = float(roc_auc_score(y_obs[ok], p[ok]))
        return max(auc - 0.5, 1e-3)      # skill above chance, never negative

    # -- prediction --------------------------------------------------------

    def _typicality(self, m: str, block: np.ndarray) -> np.ndarray:
        """Label-free score in (0, 1]: 1 for an ordinary acquisition, lower
        for one that sits far from the training distribution."""
        mu, sd = self.stats[m]
        z = (np.nan_to_num(block) - mu) / sd
        d2 = np.mean(z ** 2, axis=1)          # mean squared z over features
        return np.exp(-0.5 * np.clip(d2 - 1.0, 0.0, None))

    def weights(self, d: TypedData) -> tuple[np.ndarray, list[str]]:
        """The gate itself: ``[N, M]`` weights over the fitted modalities.

        Exposed so that users and the interface can see what the model
        trusts, per patient.
        """
        b = self.blocks.transform(d)
        mods = list(self.models)
        w = np.zeros((len(d), len(mods)))
        for j, m in enumerate(mods):
            block = b[m]
            present = ~np.isnan(block).all(axis=1)
            term = np.where(present, 1.0, 0.0)
            if self.w_reliability:
                term = term * self.reliability[m] ** self.w_reliability
            if self.w_quality:
                term = term * self._typicality(m, block) ** self.w_quality
            w[:, j] = term
        w = w ** (1.0 / self.temperature)
        total = w.sum(axis=1, keepdims=True)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        return w, mods

    def predict(self, d: TypedData) -> np.ndarray:
        b = self.blocks.transform(d)
        w, mods = self.weights(d)
        out = np.zeros(len(d))
        for j, m in enumerate(mods):
            block = b[m]
            present = ~np.isnan(block).all(axis=1)
            if not present.any():
                continue
            p = np.full(len(d), 0.0)
            p[present] = self.models[m].predict_proba(
                self._design(block, present))[:, 1]
            out += w[:, j] * p
        # a patient with nothing observed falls back to the base rate
        none = w.sum(axis=1) <= 0
        out[none] = 0.5
        return out
