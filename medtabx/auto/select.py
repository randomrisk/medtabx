"""Choose which modalities to use, from the training data alone.

Adding a modality does not guarantee a better model, and the best single
modality is not known in advance. :class:`GreedyModalitySelector` chooses
*which subset* of modalities to use, inside the training fold. Use it when a
dataset has several modalities and you want the fitted model to use only
those that add information.

Forward selection, as in greedy modality selection (Cheng et al., UAI 2022;
He et al., JMLR 2024): start from no modality, add the one that raises the
cross-fitted AUROC of uniform late fusion the most, stop when the gain falls
below ``eps``. For late fusion the score of any subset is the row mean of
the out-of-fold per-modality predictions, so each step costs no refit.

With ``permutation_filter`` a modality must also beat its own shuffled copy:
the subset with modality *m* is compared with the same subset where *m*'s
rows are permuted across patients (same width, no information). This uses
the permutation control of :mod:`medtabx.analysis.info` as a decision rule;
a modality that only adds width is not taken.

Patients who have none of the chosen modalities are scored at 0.5 during
selection, so a subset that covers fewer patients pays for it. The whole
population is scored, including patients with missing modalities.
"""

from __future__ import annotations

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData
from medtabx.fusion.experts import ModalityExperts, late_mean


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5


class ModalitySelected:
    """Any learner restricted to a fixed set of modalities."""

    def __init__(self, inner, modalities: list[str], name: str | None = None
                 ) -> None:
        self.inner = inner
        self.selected = list(modalities)
        self.name = name or f"{getattr(inner, 'name', 'model')} " \
                            f"[{'+'.join(self.selected)}]"

    def fit(self, d: TypedData, seed: int = 0):
        self.inner.fit(d.select(self.selected), seed=seed)
        return self

    def predict(self, d: TypedData) -> np.ndarray:
        return self.inner.predict(d.select(self.selected))


class GreedyModalitySelector:
    """Forward modality selection inside the training data, then refit.

    Args:
        name: label for result tables.
        final: zoo model id fitted on the chosen modalities, e.g.
            ``"tabpfn@late"``.
        select_base: per-modality learner used to score subsets. A fast
            learner (``"lr"``) keeps nested evaluation affordable; it need
            not equal the base of ``final``.
        eps: minimum AUROC gain for adding a modality.
        permutation_filter: also require the gain over the same subset with
            the candidate modality shuffled to exceed ``eps``.
        inner_splits, inner_repeats: cross-fitting for the subset scores.
        max_modalities: stop after this many. ``1`` chooses the best single
            modality inside the training fold. This gives a single-modality
            baseline that does not use the test data.
    """

    def __init__(self, name: str = "greedy selection",
                 final: str = "lr@late", select_base="lr",
                 eps: float = 0.005, permutation_filter: bool = False,
                 inner_splits: int = 5, inner_repeats: int = 2,
                 img_components: int = 32, create_kw: dict | None = None,
                 max_modalities: int | None = None) -> None:
        self.name = name
        self.final = final
        self.select_base = select_base
        self.eps = eps
        self.permutation_filter = permutation_filter
        self.max_modalities = max_modalities
        self.inner_splits = inner_splits
        self.inner_repeats = inner_repeats
        self.img_components = img_components
        self.create_kw = create_kw or {}

    def _experts(self) -> ModalityExperts:
        return ModalityExperts(self.select_base,
                               img_components=self.img_components,
                               inner_splits=self.inner_splits,
                               inner_repeats=self.inner_repeats)

    def select(self, d: TypedData, seed: int = 0) -> list[str]:
        """Run the selection only; returns the chosen modalities."""
        y = d.y.astype(int)
        real = self._experts().fit(d, seed=seed)
        mods, p = real.mods, real.oof_
        p_shuf = None
        if self.permutation_filter:
            rng = np.random.default_rng(seed + 7)
            perm = {m: rng.permutation(len(d)) for m in mods}
            shuf = self._experts().fit(d, seed=seed, shuffle=perm)
            # the shuffled fit can skip a modality the real fit kept
            p_shuf = np.full_like(p, np.nan)
            for j, m in enumerate(mods):
                if m in shuf.mods:
                    p_shuf[:, j] = shuf.oof_[:, shuf.mods.index(m)]

        def score(cols_real, col_shuf=None):
            mat = p[:, cols_real]
            if col_shuf is not None:
                mat = np.column_stack([mat, p_shuf[:, col_shuf]])
            return _auroc(y, late_mean(mat))

        chosen: list[int] = []
        current = 0.5
        self.trace_: list[dict] = []
        limit = min(len(mods), self.max_modalities or len(mods))
        while len(chosen) < limit:
            best = None
            for j in range(len(mods)):
                if j in chosen:
                    continue
                a = score(chosen + [j])
                row = {"step": len(chosen) + 1, "modality": mods[j],
                       "auroc": a, "gain": a - current,
                       "auroc_shuffled": np.nan, "gain_over_shuffled": np.nan,
                       "taken": False}
                ok = row["gain"] >= self.eps
                if self.permutation_filter:
                    a_sh = score(chosen, j) if np.isfinite(
                        p_shuf[:, j]).any() else 0.5
                    row["auroc_shuffled"] = a_sh
                    row["gain_over_shuffled"] = a - a_sh
                    ok = ok and a - a_sh >= self.eps
                row["eligible"] = bool(ok)
                self.trace_.append(row)
                if ok and (best is None or a > best[1]):
                    best = (j, a)
            if best is None:
                break
            chosen.append(best[0])
            current = best[1]
            for row in self.trace_:
                if row["step"] == len(chosen) and \
                        row["modality"] == mods[best[0]]:
                    row["taken"] = True

        if not chosen:
            # nothing beats chance in-fold: keep the single best modality so
            # the model still predicts, and record that it was a fallback
            aucs = [score([j]) for j in range(len(mods))]
            chosen = [int(np.argmax(aucs))]
            self.fallback_ = True
        else:
            self.fallback_ = False
        self.inner_auroc_ = score(chosen)
        self.selected = [mods[j] for j in chosen]
        return self.selected

    def fit(self, d: TypedData, seed: int = 0):
        from medtabx.auto.zoo import create_model
        self.select(d, seed=seed)
        self.model = ModalitySelected(
            create_model(self.final, **self.create_kw), self.selected
        ).fit(d, seed=seed)
        return self

    def predict(self, d: TypedData) -> np.ndarray:
        return self.model.predict(d)
