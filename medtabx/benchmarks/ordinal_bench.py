"""Ordinal harness: predict the whole mRS scale, not one dichotomy.

The binary harness in :mod:`medtabx.benchmarks.typed_bench` is left exactly
as it is. This module sits on top of it and asks the same cohort ``K-1``
cumulative questions instead of one, so every learner, fusion, augmentation
and gate already registered in ``METHODS`` carries over unchanged.

Two strategies are available:

``cumulative``
    One binary model per threshold ``P(y > k)``, all fitted
    on the same training rows of the same fold, then assembled into a level
    distribution with the coherence repair of
    :mod:`medtabx.metrics.ordinal`. Respects the ordering, and recovers the
    conventional mRS 0--2 against 3--6 result exactly at ``k = 2``.

``multinomial``
    The control. One ``K``-way model that treats the levels as unordered
    categories. If the ordering carries no usable information, this should do
    as well. Compare the two strategies to measure what the ordering adds.

Folds are drawn once per repeat and shared by every threshold, strategy and
method, so all of them are paired at the fold level, as in the binary
comparison. Levels too rare to stratify on are merged with their neighbour
for the purpose of splitting only; the labels themselves are never collapsed.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from medtabx.benchmarks.typed_bench import (
    METHODS,
    TypedData,
    _Flattener,
    _ModalityBlocks,
)
from medtabx.metrics.ordinal import (
    N_LEVELS_MRS,
    distribution,
    monotone_repair,
    ordinal_metrics,
)


def stratification_label(y: np.ndarray, n_splits: int) -> np.ndarray:
    """A coarsened copy of ``y`` in which every value occurs at least
    ``n_splits`` times, so stratified splitting is well defined.

    The rarest level is merged into its nearest neighbour, preferring the
    more severe one when the distance ties, and the process repeats until
    every group is large enough. For mRS this keeps the clinically usual
    grouping: level 5 (severe disability, 3 patients here) joins level 6
    (dead) rather than being separated from it. Only the splitting is
    affected; the labels the models are scored against are never collapsed.
    """
    out = np.asarray(y, int).copy()
    while True:
        levels, counts = np.unique(out, return_counts=True)
        if len(levels) < 2 or counts.min() >= n_splits:
            return out
        rare = levels[counts.argmin()]
        others = levels[levels != rare]
        distance = np.abs(others - rare)
        # nearest neighbour; on a tie take the more severe (higher) level
        candidates = others[distance == distance.min()]
        out[out == rare] = candidates.max()


# -- the two strategies ----------------------------------------------------


class CumulativeLink:
    """``K-1`` binary models from ``METHODS``, one per threshold.

    ``shared_reliability`` only affects methods that gate their modalities by
    a measured reliability (the ``acq`` fusion). Left False, each threshold
    estimates its own per-modality reliability, which is what the
    decomposition gives for free and what the modality-by-cut-point matrix
    says it should do. Set True, every threshold is forced to use the
    reliability measured against the conventional dichotomy, which is the
    control that isolates *threshold-varying* gating from gating as such.
    """

    strategy = "cumulative"

    def __init__(self, method: str, n_levels: int = N_LEVELS_MRS,
                 shared_reliability: bool = False,
                 dichotomy_at: int = 2) -> None:
        self.method = method
        self.n_levels = n_levels
        self.shared_reliability = shared_reliability
        self.dichotomy_at = dichotomy_at
        self.name = f"{method} [cumulative]" + (
            " shared-r" if shared_reliability else "")

    def fit(self, train: TypedData, y_ord: np.ndarray, seed: int = 0):
        self.models = []
        shared = (y_ord > self.dichotomy_at).astype(int)
        for k in range(self.n_levels - 1):
            target = (y_ord > k).astype(int)
            if len(np.unique(target)) < 2:
                self.models.append(float(target.mean()))   # degenerate fold
                continue
            model = METHODS[self.method]()
            if self.shared_reliability:
                if not hasattr(model, "set_reliability_target"):
                    raise TypeError(
                        f"{self.method} has no gate whose reliability could "
                        f"be shared across thresholds")
                model.set_reliability_target(shared)
            self.models.append(model.fit(train.with_y(target), seed=seed))
        return self

    def predict_cumulative(self, test: TypedData) -> np.ndarray:
        cols = []
        for m in self.models:
            if isinstance(m, float):
                cols.append(np.full(len(test), m))
            else:
                cols.append(np.asarray(m.predict(test), dtype=float))
        return np.stack(cols, axis=1)


class Multinomial:
    """One unordered ``K``-way model on the flat view, as the control."""

    strategy = "multinomial"

    def __init__(self, base: str = "lr", n_levels: int = N_LEVELS_MRS) -> None:
        self.base = base
        self.n_levels = n_levels
        self.name = f"{base} [multinomial]"

    def _learner(self, seed: int):
        if self.base == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.05, max_iter=5000,
                                      random_state=seed)
        if self.base == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=seed)
        if self.base == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(400, random_state=seed, n_jobs=-1)
        raise KeyError(f"no multinomial control for base {self.base!r}")

    def fit(self, train: TypedData, y_ord: np.ndarray, seed: int = 0):
        self.flat = _Flattener(impute=True).fit(train)
        self.classes_seen = np.unique(y_ord)
        self.model = self._learner(seed).fit(self.flat.transform(train), y_ord)
        return self

    def predict_cumulative(self, test: TypedData) -> np.ndarray:
        """Expressed as cumulative probabilities so both strategies are
        scored by exactly the same code."""
        proba = self.model.predict_proba(self.flat.transform(test))
        p = np.zeros((len(test), self.n_levels))
        for j, c in enumerate(self.model.classes_):
            p[:, int(c)] = proba[:, j]
        # P(y > k) = sum_{j > k} P(y = j)
        tail = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
        return tail[:, 1:]


# -- the harness -----------------------------------------------------------


def evaluate_ordinal_scenarios(model, test: TypedData, y_test: np.ndarray,
                               scenarios: dict[str, list[str]],
                               n_levels: int) -> list[dict]:
    rows = []
    for scen, dropped in scenarios.items():
        t = test.drop(dropped) if dropped else test
        cum = np.clip(model.predict_cumulative(t), 0.0, 1.0)
        rows.append({"scenario": scen,
                     **ordinal_metrics(y_test, cum, n_levels)})
    return rows


def run_ordinal_cv(data: TypedData, y_ord: np.ndarray,
                   methods: list[str],
                   scenarios: dict[str, list[str]],
                   multinomial_bases: tuple[str, ...] = ("lr", "hgb"),
                   shared_reliability_methods: tuple[str, ...] = (),
                   n_levels: int = N_LEVELS_MRS, n_splits: int = 5,
                   repeats: int = 5, seed: int = 0,
                   log: Callable[[str], None] = print) -> list[dict]:
    """Repeated stratified CV over the full ordinal scale.

    ``data.y`` is ignored; ``y_ord`` holds the levels ``0 .. n_levels-1``.
    Returns one row per (repeat, fold, model, scenario).
    """
    from sklearn.model_selection import StratifiedKFold

    y_ord = np.asarray(y_ord, int)
    if len(y_ord) != len(data):
        raise ValueError(f"y_ord has {len(y_ord)} rows, data has {len(data)}")
    strat = stratification_label(y_ord, n_splits)
    if not np.array_equal(strat, y_ord):
        merged = {int(a): int(b) for a, b in zip(y_ord, strat) if a != b}
        log(f"  stratifying on merged levels {merged} "
            f"(labels themselves unchanged)")

    builders = [(m, lambda m=m: CumulativeLink(m, n_levels)) for m in methods]
    builders += [(f"{m}+sharedr",
                  lambda m=m: CumulativeLink(m, n_levels,
                                             shared_reliability=True))
                 for m in shared_reliability_methods]
    builders += [(f"{b}@multinomial", lambda b=b: Multinomial(b, n_levels))
                 for b in multinomial_bases]

    rows = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(data)), strat)):
            train, test = data.subset(tr), data.subset(te)
            for name, build in builders:
                t0 = time.time()
                model = build().fit(train, y_ord[tr], seed=seed + 10 * r + f)
                fit_s = time.time() - t0
                for row in evaluate_ordinal_scenarios(
                        model, test, y_ord[te], scenarios, n_levels):
                    rows.append({"repeat": r, "fold": f, "method": name,
                                 "label": model.name,
                                 "strategy": model.strategy, "split": "cv",
                                 "train_n": len(tr), "fit_seconds": fit_s,
                                 **row})
                log(f"  r{r} f{f} {name:24s} {fit_s:6.1f}s")
    return rows


class StackedOrdinal:
    """One model over all thresholds at once, via the stacked design.

    Fitting the thresholds independently (:class:`CumulativeLink`) lets every
    modality act differently at every cut-point. In small datasets it can
    also produce incoherent probabilities for many patients. The standard deep
    ordinal remedy goes to the opposite extreme: share one coefficient
    vector and give each threshold only its own intercept. This forces
    proportional odds, so it *cannot represent* a modality effect that
    drifts across cut-points.

    This class is the middle ground. Each patient is replicated once per
    threshold with target ``y > k``; the threshold dummies act as the
    intercepts ``alpha_k``; the feature coefficients are shared. Modalities
    named in ``varying`` additionally get their own coefficients per
    threshold, which is exactly the partial proportional odds model of
    Peterson and Harrell (1990). So:

    * ``varying = ()``        proportional odds (the shared-weight baseline)
    * ``varying = ("history", "presentation")``  partial proportional odds
    * ``varying = "all"``     every modality free, but still with the ordered
                              intercepts that independent fitting throws away
    * ``varying = "auto"``    the set is chosen *inside the training fold* by
                              the same drift test used in the analysis, so no
                              test-fold information enters the choice

    Because the intercepts are shared and ordered by the data, monotonicity
    violations should be far rarer here than under independent fitting --
    which is a prediction this class exists to test, not an assumption.
    """

    def __init__(self, varying=(), n_levels: int = N_LEVELS_MRS,
                 C: float = 0.05, img_components: int = 16,
                 name: str | None = None) -> None:
        self.varying = varying
        self.n_levels = n_levels
        self.C = C
        self.img_components = img_components
        tag = ("proportional" if varying == () else
               "free" if varying == "all" else
               "partial:auto" if varying == "auto" else
               "partial:" + "+".join(varying))
        self.name = name or f"stacked [{tag}]"
        self.strategy = tag.split(":")[0]
        self.selected = varying

    # -- design ------------------------------------------------------------

    def _features(self, d: TypedData) -> tuple[np.ndarray, np.ndarray]:
        """``[N, F]`` imputed features plus the modality label of each column.

        A per-modality presence indicator is appended so that absence stays
        visible to the model, as everywhere else in the framework.
        """
        b = self.blocks.transform(d)
        cols, groups = [], []
        for m in self.blocks.mods:
            block = b[m]
            present = ~np.isnan(block).all(axis=1)
            cols.append(np.nan_to_num(block))
            groups += [m] * block.shape[1]
            cols.append(present.astype(float)[:, None])
            groups.append(m)
        return np.concatenate(cols, axis=1), np.array(groups)

    def _stack(self, x: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Replicate rows once per threshold and add the threshold terms."""
        n, km1 = len(x), self.n_levels - 1
        rows = np.tile(x, (km1, 1))                     # shared coefficients
        dummies = np.repeat(np.eye(km1), n, axis=0)     # the intercepts alpha_k
        parts = [rows, dummies]
        if self.selected == "all":
            free = np.ones(len(groups), bool)
        else:
            free = np.isin(groups, list(self.selected))
        if free.any():
            # one copy of the free columns per threshold: the interaction that
            # lets those modalities act differently at different cut-points
            inter = np.zeros((n * km1, int(free.sum()) * km1))
            for k in range(km1):
                lo, hi = k * n, (k + 1) * n
                c0 = k * int(free.sum())
                inter[lo:hi, c0:c0 + int(free.sum())] = x[:, free]
            parts.append(inter)
        return np.concatenate(parts, axis=1)

    # -- fit / predict -----------------------------------------------------

    def _select_varying(self, train: TypedData, y_ord: np.ndarray,
                        seed: int, inner_splits: int = 3,
                        min_abs_slope: float = 0.01) -> tuple[str, ...]:
        """Which modalities drift enough to deserve their own coefficients.

        Decided on the training fold alone, by the same statistic the analysis
        uses: the slope of single-modality AUROC against the cut-point index.
        Selecting on all the data would leak test information, so the choice
        is nested inside the training fold.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        km1 = self.n_levels - 1
        strat = stratification_label(y_ord, inner_splits)
        per_mod = {m: [] for m in train.modalities}
        skf = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
        for itr, ite in skf.split(np.zeros(len(y_ord)), strat):
            blocks = _ModalityBlocks(self.img_components).fit(
                train.subset(itr))
            b_tr = blocks.transform(train.subset(itr))
            b_te = blocks.transform(train.subset(ite))
            for m in train.modalities:
                x_tr = np.nan_to_num(b_tr[m])
                x_te = np.nan_to_num(b_te[m])
                prof = []
                for k in range(km1):
                    a = (y_ord[itr] > k).astype(int)
                    c = (y_ord[ite] > k).astype(int)
                    if len(np.unique(a)) < 2 or len(np.unique(c)) < 2:
                        prof.append(np.nan)
                        continue
                    lr = LogisticRegression(C=0.05, max_iter=5000,
                                            random_state=seed)
                    p = lr.fit(x_tr, a).predict_proba(x_te)[:, 1]
                    prof.append(roc_auc_score(c, p))
                per_mod[m].append(prof)

        chosen = []
        for m, profs in per_mod.items():
            mean = np.nanmean(np.array(profs, dtype=float), axis=0)
            ok = ~np.isnan(mean)
            if ok.sum() < 3:
                continue
            slope = np.polyfit(np.arange(km1)[ok], mean[ok], 1)[0]
            if abs(slope) >= min_abs_slope:
                chosen.append(m)
        return tuple(chosen)

    def fit(self, train: TypedData, y_ord: np.ndarray, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if self.varying == "auto":
            self.selected = self._select_varying(train, y_ord, seed)
        else:
            self.selected = self.varying
        self.blocks = _ModalityBlocks(self.img_components).fit(train)
        x, self.groups = self._features(train)
        self.scaler = StandardScaler().fit(x)
        design = self._stack(self.scaler.transform(x), self.groups)
        target = np.concatenate([(np.asarray(y_ord) > k).astype(int)
                                 for k in range(self.n_levels - 1)])
        self.model = LogisticRegression(
            C=self.C, max_iter=10000, fit_intercept=False,
            random_state=seed).fit(design, target)
        return self

    def predict_cumulative(self, test: TypedData) -> np.ndarray:
        x, _ = self._features(test)
        design = self._stack(self.scaler.transform(x), self.groups)
        p = self.model.predict_proba(design)[:, 1]
        # the stack is threshold-major: reshape back to [N, K-1]
        return p.reshape(self.n_levels - 1, len(test)).T
