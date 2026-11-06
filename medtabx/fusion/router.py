"""A per-patient router over modality experts, learned by stacking.

The experts are the per-modality models of
:class:`~medtabx.fusion.experts.ModalityExperts` and, optionally, one model
on all columns (the ``concat`` expert). A small meta-learner reads each
patient's expert predictions together with which modalities that patient
has, and outputs the final probability. It is trained on the cross-fitted
predictions only, so it never sees an expert's in-sample output.

This is the "choose between the unimodal experts f1..fM and the fused
expert f12" design discussed in the failure-aware literature, in the form
that can be fitted on a few hundred patients: a regularized logistic
regression over expert logits and absence flags.

``blend`` mixes the router with uniform late fusion of the same experts:

    p = (1 - blend) * late(x) + blend * router(x)

``blend = 0`` is exactly uniform late fusion, so the router is a strict
generalization of it and can be ablated to it on identical models.
"""

from __future__ import annotations

import numpy as np

from medtabx.benchmarks.typed_bench import TabPFNConcat, TypedData
from medtabx.fusion.experts import (
    ModalityExperts,
    late_mean,
    make_learner,
    stratified_splits,
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class StackedRouter:
    """Stacked per-patient router over modality experts.

    Args:
        name: label for result tables.
        base: learner for every expert.
        include_concat: add one expert on all columns.
        blend: weight of the router against uniform late fusion (0..1).
        C: inverse L2 strength of the meta logistic regression.
    """

    def __init__(self, name: str = "stacked router", base="lr",
                 include_concat: bool = True, blend: float = 1.0,
                 C: float = 0.1, inner_splits: int = 5,
                 img_components: int = 32, impute: bool | None = None
                 ) -> None:
        self.name = name
        self.base = base
        self.include_concat = include_concat
        self.blend = float(blend)
        self.C = C
        self.inner_splits = inner_splits
        self.img_components = img_components
        self.impute = impute

    def _concat(self):
        experts = self.experts
        return TabPFNConcat("concat expert",
                            base=lambda s: make_learner(self.base, s),
                            impute=experts.impute, indicators=True,
                            img_components=self.img_components)

    def _concat_oof(self, d: TypedData, seed: int) -> np.ndarray:
        out = np.full(len(d), np.nan)
        for tr, te in stratified_splits(d.y.astype(int), self.inner_splits,
                                        seed):
            m = self._concat().fit(d.subset(tr), seed)
            out[te] = m.predict(d.subset(te))
        return out

    def _features(self, p: np.ndarray, p_cat: np.ndarray | None
                  ) -> np.ndarray:
        absent = np.isnan(p)
        cols = [np.where(absent, 0.0, _logit(np.nan_to_num(p, nan=0.5))),
                absent.astype(float)]
        if p_cat is not None:
            cols.append(_logit(np.nan_to_num(p_cat, nan=0.5))[:, None])
        return np.concatenate(cols, axis=1)

    def fit(self, d: TypedData, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        self.experts = ModalityExperts(
            self.base, img_components=self.img_components,
            impute=self.impute, inner_splits=self.inner_splits
        ).fit(d, seed=seed)
        p_cat_oof = None
        if self.include_concat:
            p_cat_oof = self._concat_oof(d, seed)
            self.concat_model = self._concat().fit(d, seed)
        x = self._features(self.experts.oof_, p_cat_oof)
        self.meta = LogisticRegression(C=self.C, max_iter=5000).fit(
            x, d.y.astype(int))
        return self

    def predict(self, d: TypedData) -> np.ndarray:
        p = self.experts.predict_matrix(d)
        late = late_mean(p)
        if self.blend == 0.0:
            return late
        p_cat = self.concat_model.predict(d) if self.include_concat else None
        routed = self.meta.predict_proba(self._features(p, p_cat))[:, 1]
        return (1.0 - self.blend) * late + self.blend * routed
