"""How much does a modality group actually contribute?

Prediction-based information decomposition on the typed contract. For two
groups of modalities A and B (e.g. clinical vs image) and label Y we
cross-fit a probe and read information off its log loss:

    I(Y;A) ~= H(Y) - CE(A)          information in A about Y
    U(B|A) ~= CE(A) - CE(A,B)       information B adds on top of A (unique)
    R      ~= I(Y;A) + I(Y;B) - I(Y;A,B)      redundancy (co-information)
    S      ~= -R when negative                 synergy

These are estimator-dependent lower bounds, not exact mutual information:
they say what a given probe can extract at this sample size, which is the
quantity that matters for "does adding this modality help a model".

Every estimate comes with the matched **permutation control**: the same
group with its rows shuffled carries zero information but costs exactly the
same number of parameters, so the gap between real and shuffled tells
information cost from dimensionality cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData, _ModalityBlocks


def _blocks(d: TypedData, img_components: int, seed: int = 0):
    b = _ModalityBlocks(img_components).fit(d)
    return b, b.transform(d)


def group_matrix(blocks: dict[str, np.ndarray], mods: list[str],
                 impute: bool = True) -> np.ndarray:
    x = np.concatenate([blocks[m] for m in mods], axis=1)
    return np.nan_to_num(x) if impute else x


def shuffled(x: np.ndarray, seed: int) -> np.ndarray:
    """Same matrix, rows permuted: zero information, same dimensionality."""
    return x[np.random.default_rng(seed).permutation(len(x))]


def cross_fit_ce(x: np.ndarray, y: np.ndarray, n_splits: int = 5,
                 seed: int = 0, probe: str = "lr",
                 C: float = 0.05) -> tuple[float, float]:
    """Cross-fitted cross-entropy (nats) and AUROC of a probe on ``x``.

    ``x`` with zero columns gives the marginal baseline H(Y).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    p = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        if x.shape[1] == 0:
            p[te] = np.clip(y[tr].mean(), 1e-6, 1 - 1e-6)
            continue
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd[~(sd > 0)] = 1.0
        xt, xe = (x[tr] - mu) / sd, (x[te] - mu) / sd
        if probe == "lr":
            m = LogisticRegression(C=C, max_iter=5000, random_state=seed)
        elif probe == "gbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=3,
                l2_regularization=1.0, random_state=seed)
        else:
            raise ValueError(probe)
        m.fit(xt, y[tr])
        p[te] = np.clip(m.predict_proba(xe)[:, 1], 1e-6, 1 - 1e-6)
    ce = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5
    return ce, auroc


@dataclass
class Decomposition:
    """Information (nats) a probe extracts from two modality groups."""

    a: list[str]
    b: list[str]
    dim_a: int
    dim_b: int
    h_y: float
    ce_a: float
    ce_b: float
    ce_ab: float
    ce_ab_shuffled_b: float
    auroc_a: float = float("nan")
    auroc_b: float = float("nan")
    auroc_ab: float = float("nan")
    auroc_ab_shuffled_b: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def i_a(self) -> float:
        return self.h_y - self.ce_a

    @property
    def i_b(self) -> float:
        return self.h_y - self.ce_b

    @property
    def i_ab(self) -> float:
        return self.h_y - self.ce_ab

    @property
    def unique_b(self) -> float:
        """Information B adds on top of A (negative = adding B hurts)."""
        return self.ce_a - self.ce_ab

    @property
    def redundancy(self) -> float:
        return self.i_a + self.i_b - self.i_ab

    @property
    def synergy(self) -> float:
        return max(0.0, -self.redundancy)

    @property
    def dimensional_cost(self) -> float:
        """What A loses by concatenating *uninformative* columns of B's
        width: CE(A, shuffled B) - CE(A). Positive = pure capacity cost."""
        return self.ce_ab_shuffled_b - self.ce_a

    @property
    def information_gain_net(self) -> float:
        """Real minus shuffled: how much of B's effect is information."""
        return self.ce_ab_shuffled_b - self.ce_ab

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(i_a=self.i_a, i_b=self.i_b, i_ab=self.i_ab,
                 unique_b=self.unique_b, redundancy=self.redundancy,
                 synergy=self.synergy,
                 dimensional_cost=self.dimensional_cost,
                 information_gain_net=self.information_gain_net,
                 **self.extra)
        return d


def decompose(data: TypedData, group_a: list[str], group_b: list[str],
              img_components: int = 32, probe: str = "lr",
              n_splits: int = 5, seed: int = 0) -> Decomposition:
    """Decompose what groups A and B contribute, with the shuffled-B control.
    """
    _, blocks = _blocks(data, img_components)
    xa = group_matrix(blocks, group_a)
    xb = group_matrix(blocks, group_b)
    y = data.y
    h_y, _ = cross_fit_ce(np.zeros((len(y), 0)), y, n_splits, seed, probe)
    ce_a, au_a = cross_fit_ce(xa, y, n_splits, seed, probe)
    ce_b, au_b = cross_fit_ce(xb, y, n_splits, seed, probe)
    ce_ab, au_ab = cross_fit_ce(np.concatenate([xa, xb], 1), y, n_splits,
                                seed, probe)
    ce_sh, au_sh = cross_fit_ce(
        np.concatenate([xa, shuffled(xb, seed + 1)], 1), y, n_splits, seed,
        probe)
    return Decomposition(
        a=list(group_a), b=list(group_b), dim_a=xa.shape[1],
        dim_b=xb.shape[1], h_y=h_y, ce_a=ce_a, ce_b=ce_b, ce_ab=ce_ab,
        ce_ab_shuffled_b=ce_sh, auroc_a=au_a, auroc_b=au_b, auroc_ab=au_ab,
        auroc_ab_shuffled_b=au_sh,
        extra={"probe": probe, "seed": seed, "n": len(y),
               "img_components": img_components})


def capacity_sweep(data: TypedData, group_a: list[str], group_b: list[str],
                   dims: list[int], probe: str = "lr", n_splits: int = 5,
                   seeds: int = 5) -> list[dict]:
    """Vary how many dimensions group B contributes (PCA per image modality
    / random projection for tabular groups) and record performance of A,
    A+B and A+shuffled(B).

    The prediction under "fusion hurts because of capacity, not content" is
    an inverted U in ``dim_b`` whose peak grows with ``n`` and with the
    unique information of B, and a real-vs-shuffled gap that closes as
    ``dim_b`` grows.
    """
    rows = []
    for k in dims:
        for s in range(seeds):
            d = decompose(data, group_a, group_b, img_components=k,
                          probe=probe, n_splits=n_splits, seed=s)
            row = d.as_dict()
            row.update(requested_dim=k, seed=s)
            rows.append(row)
    return rows
