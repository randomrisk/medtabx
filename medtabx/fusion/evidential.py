"""Evidential fusion: combine modality opinions by Dempster's rule.

Trusted Multi-View Classification (Han et al., ICLR 2021; TPAMI 2022) gives
each view a subjective opinion -- per-class belief masses ``b_k`` and an
uncertainty mass ``u`` with ``sum_k b_k + u = 1`` -- derived from Dirichlet
evidence, and combines the views with the reduced Dempster-Shafer rule:

    C   = sum_{i != j} b_i^1 b_j^2                     (conflict)
    b_k = (b_k^1 b_k^2 + b_k^1 u^2 + b_k^2 u^1) / (1 - C)
    u   = u^1 u^2 / (1 - C)

A view with no evidence (``u = 1``) leaves the other unchanged, and a view
that disagrees with confident evidence is discounted by the conflict term.

TMC learns the evidence with a neural head. Our per-modality learners
(TabPFN, GBDT, LR) output only a probability, so this is a decision-level
adaptation, implemented here from the equations above (no code copied):

    evidence_m(x) = strength * reliability_m ** w_r * typicality_m(x) ** w_q
    e_1 = evidence * p_m(x),   e_0 = evidence * (1 - p_m(x))

``reliability_m`` is the cross-fitted AUROC above chance of modality *m* and
``typicality_m(x)`` the label-free acquisition score of
:mod:`medtabx.auto.gating`, so a modality is trusted less when it predicts
poorly in the training fold or when this patient's acquisition looks unusual.
An absent modality has zero evidence. ``strength="auto"`` picks the scale
that maximizes the cross-fitted log-likelihood of the combined prediction.

Unlike :class:`~medtabx.auto.gating.AcquisitionGatedFusion` this is not an
exact generalization of uniform late fusion: as ``strength -> 0`` the
prediction tends to 0.5 while its ranking tends to that of the
evidence-weighted mean of the modality probabilities.
"""

from __future__ import annotations

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData
from medtabx.fusion.experts import ModalityExperts

K = 2  # binary outcome; the rule below is written for two classes


def opinion(p: np.ndarray, evidence: np.ndarray):
    """Binary subjective opinion from P(y=1) and total evidence."""
    s = evidence + K
    return evidence * (1 - p) / s, evidence * p / s, K / s


def combine(o1, o2):
    """Reduced Dempster-Shafer combination of two binary opinions."""
    b0a, b1a, ua = o1
    b0b, b1b, ub = o2
    conflict = b0a * b1b + b1a * b0b
    scale = 1.0 / np.maximum(1.0 - conflict, 1e-12)
    b0 = (b0a * b0b + b0a * ub + b0b * ua) * scale
    b1 = (b1a * b1b + b1a * ub + b1b * ua) * scale
    u = ua * ub * scale
    return b0, b1, u


def probability(o) -> np.ndarray:
    """Expected P(y=1) of the Dirichlet behind an opinion."""
    _, b1, u = o
    s = K / np.maximum(u, 1e-12)
    return (b1 * s + 1.0) / s


class EvidentialFusion:
    """Per-patient evidential fusion of modality experts.

    Args:
        name: label for result tables.
        base: per-modality learner.
        strength: evidence scale, or ``"auto"`` to choose it in-fold from
            ``strength_grid``.
        w_reliability, w_quality: exponents as in the acquisition gate.
    """

    def __init__(self, name: str = "evidential fusion", base="lr",
                 strength: float | str = "auto",
                 strength_grid=(1.0, 3.0, 10.0, 30.0, 100.0),
                 w_reliability: float = 1.0, w_quality: float = 1.0,
                 inner_splits: int = 5, img_components: int = 32,
                 impute: bool | None = None) -> None:
        self.name = name
        self.base = base
        self.strength = strength
        self.strength_grid = tuple(strength_grid)
        self.w_reliability = w_reliability
        self.w_quality = w_quality
        self.inner_splits = inner_splits
        self.img_components = img_components
        self.impute = impute

    def fit(self, d: TypedData, seed: int = 0):
        from sklearn.metrics import roc_auc_score
        y = d.y.astype(int)
        self.experts = ModalityExperts(
            self.base, img_components=self.img_components,
            impute=self.impute, inner_splits=self.inner_splits
        ).fit(d, seed=seed)
        b = self.experts.blocks.transform(d)
        oof = self.experts.oof_
        self.reliability: dict[str, float] = {}
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for j, m in enumerate(self.experts.mods):
            ok = ~np.isnan(oof[:, j])
            auc = (roc_auc_score(y[ok], oof[ok, j])
                   if ok.sum() >= 10 and len(np.unique(y[ok])) == 2 else 0.5)
            self.reliability[m] = max(auc - 0.5, 1e-3)
            seen = np.nan_to_num(b[m][~np.isnan(b[m]).all(axis=1)])
            sd = seen.std(axis=0)
            sd[~(sd > 0)] = 1.0
            self.stats[m] = (seen.mean(axis=0), sd)
        scale = np.mean(list(self.reliability.values()))
        self.reliability = {m: v / scale for m, v in self.reliability.items()}

        if self.strength == "auto":
            unit = self._unit_evidence(d)
            ll = [self._loglik(y, self._fuse(oof, s * unit))
                  for s in self.strength_grid]
            self.strength_ = self.strength_grid[int(np.argmax(ll))]
        else:
            self.strength_ = float(self.strength)
        return self

    @staticmethod
    def _loglik(y, p) -> float:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _typicality(self, m: str, block: np.ndarray) -> np.ndarray:
        # same score as AcquisitionGatedFusion._typicality
        mu, sd = self.stats[m]
        z = (np.nan_to_num(block) - mu) / sd
        return np.exp(-0.5 * np.clip(np.mean(z ** 2, axis=1) - 1.0, 0, None))

    def _unit_evidence(self, d: TypedData) -> np.ndarray:
        """``[N, M]`` evidence at strength 1; zero where absent."""
        b = self.experts.blocks.transform(d)
        out = np.zeros((len(d), len(self.experts.mods)))
        for j, m in enumerate(self.experts.mods):
            present = ~np.isnan(b[m]).all(axis=1)
            term = present.astype(float)
            if self.w_reliability:
                term = term * self.reliability[m] ** self.w_reliability
            if self.w_quality:
                term = term * self._typicality(m, b[m]) ** self.w_quality
            out[:, j] = term
        return out

    @staticmethod
    def _fuse(p: np.ndarray, evidence: np.ndarray) -> np.ndarray:
        n = len(p)
        fused = (np.zeros(n), np.zeros(n), np.ones(n))      # vacuous
        for j in range(p.shape[1]):
            ev = np.where(np.isnan(p[:, j]), 0.0, evidence[:, j])
            fused = combine(fused, opinion(np.nan_to_num(p[:, j], nan=0.5),
                                           ev))
        return probability(fused)

    def weights(self, d: TypedData) -> tuple[np.ndarray, list[str]]:
        """Normalized evidence per patient and modality, for reporting."""
        ev = self._unit_evidence(d)
        total = ev.sum(axis=1, keepdims=True)
        w = np.divide(ev, total, out=np.zeros_like(ev), where=total > 0)
        return w, list(self.experts.mods)

    def predict(self, d: TypedData) -> np.ndarray:
        p = self.experts.predict_matrix(d)
        return self._fuse(p, self.strength_ * self._unit_evidence(d))
