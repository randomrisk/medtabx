"""Missing-modality robustness: is it a property of the model, or of the
training distribution?

In the earlier experiments only the typed token models were trained with
modality dropout, while the GBDTs, TabPFN and the seven PyTorch Frame models
never saw an absent modality during training -- but were tested with one
removed. Their collapse may therefore be a train/test mismatch of the
protocol rather than a property of those models.

:class:`DropoutAugmented` removes that confound: it wraps *any* learner on
the typed contract and augments the training set with copies of each patient
whose modalities are randomly dropped, exactly the augmentation the typed
model gets internally. Because the contract knows which columns belong to
which modality, this works for models that have no notion of modality.
"""

from __future__ import annotations

import copy

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData


def drop_per_row(d: TypedData, keep: dict[str, np.ndarray]) -> TypedData:
    """Copy of ``d`` where modality ``m`` is missing for rows ``~keep[m]``.

    Numeric columns become NaN, categorical -1 and image modalities absent --
    the same state a genuinely missing modality has, so nothing is imputed.
    """
    out = copy.copy(d)
    if d.num is not None:
        out.num = d.num.copy()
        for j, m in enumerate(d.num_modality):
            if m in keep:
                out.num[~keep[m], j] = np.nan
    if d.cat is not None:
        out.cat = d.cat.copy()
        for j, m in enumerate(d.cat_modality):
            if m in keep:
                out.cat[~keep[m], j] = -1
    if d.img is not None:
        out.img_present = d.img_present.copy()
        for j, m in enumerate(d.img_modality):
            if m in keep:
                out.img_present[~keep[m], j] = False
    return out


def concat(parts: list[TypedData]) -> TypedData:
    """Stack several contracts with the same schema (row-wise)."""
    out = copy.copy(parts[0])
    out.y = np.concatenate([p.y for p in parts])
    for name in ("num", "cat", "img", "img_present", "groups"):
        vals = [getattr(p, name) for p in parts]
        if vals[0] is not None:
            setattr(out, name, np.concatenate(vals, axis=0))
    return out


def random_modality_dropout(d: TypedData, p: float, rng, keep_one: bool = True
                            ) -> TypedData:
    """One augmented copy: each modality is dropped with probability ``p``,
    independently per patient; ``keep_one`` never drops every observed
    modality of a patient."""
    mods = d.modalities
    keep = {m: rng.random(len(d)) >= p for m in mods}
    if keep_one:
        observed = observed_modalities(d)
        none = ~np.any([keep[m] & observed[m] for m in mods], axis=0)
        for i in np.nonzero(none)[0]:
            have = [m for m in mods if observed[m][i]]
            if have:
                keep[have[rng.integers(len(have))]][i] = True
    return drop_per_row(d, keep)


def observed_modalities(d: TypedData) -> dict[str, np.ndarray]:
    """Which patients actually have each modality (before any dropout)."""
    out = {}
    for m in d.modalities:
        cols = []
        if d.num is not None:
            cols += [~np.isnan(d.num[:, j]) for j, mm in
                     enumerate(d.num_modality) if mm == m]
        if d.cat is not None:
            cols += [d.cat[:, j] >= 0 for j, mm in
                     enumerate(d.cat_modality) if mm == m]
        if d.img is not None and m in d.img_modality:
            cols += [d.img_present[:, d.img_modality.index(m)]]
        out[m] = (np.any(cols, axis=0) if cols
                  else np.zeros(len(d), bool))
    return out


class DropoutAugmented:
    """Any learner, trained on modality-dropout augmented data.

    Args:
        model: a model id for :func:`medtabx.auto.zoo.create_model`, or a
            factory returning a learner with ``fit(TypedData, seed)`` /
            ``predict(TypedData)``.
        p: probability of dropping each modality of each augmented copy.
        n_aug: augmented copies of the training set (0 = plain training).
        keep_one: never drop all of a patient's observed modalities.
    """

    def __init__(self, model, p: float = 0.3, n_aug: int = 2,
                 keep_one: bool = True, name: str | None = None) -> None:
        self.spec = model
        self.p, self.n_aug, self.keep_one = p, n_aug, keep_one
        self._name = name

    def _build(self):
        if callable(self.spec):
            return self.spec()
        from medtabx.auto.zoo import create_model
        return create_model(self.spec)

    @property
    def name(self) -> str:
        if self._name:
            return self._name
        inner = getattr(self.model, "name", None) if hasattr(self, "model") \
            else (self.spec if isinstance(self.spec, str) else "model")
        return f"{inner} + modality dropout"

    def fit(self, d: TypedData, seed: int = 0):
        rng = np.random.default_rng(seed)
        parts = [d] + [random_modality_dropout(d, self.p, rng, self.keep_one)
                       for _ in range(self.n_aug)]
        self.model = self._build()
        self.model.fit(concat(parts) if len(parts) > 1 else d, seed)
        self._name = self._name or f"{self.model.name} + modality dropout"
        return self

    def predict(self, d: TypedData) -> np.ndarray:
        return self.model.predict(d)


def reliance(model, d: TypedData, modality: str) -> float:
    """Mean absolute change in predicted probability when one modality is
    removed at test time: how much the model leans on it."""
    p0 = np.asarray(model.predict(d), float)
    p1 = np.asarray(model.predict(d.drop([modality])), float)
    return float(np.mean(np.abs(p0 - p1)))


def reliance_profile(model, d: TypedData) -> dict[str, float]:
    return {m: reliance(model, d, m) for m in d.modalities}
