"""Modality alignment and imbalance diagnostics on the typed contract.

Replicates, for heterogeneous medical data with more than two modalities,
the measurements of Yang et al., *Understanding Multimodal Learning from
Modality Fusion and Alignment Perspectives* (TPAMI 2026):

* per-modality ("unimodal branch") performance and the **gap** between the
  strongest and weakest modality (their Fig. 1a);
* the **Jensen-Shannon divergence** between the branch predictive
  distributions of the same patient, i.e. how *aligned* the modalities are
  (their Fig. 1e);
* whether the fused model beats the best single modality at all (the dagger
  in their Table 3).

Two ways to get the branches:

* :func:`probe_branches` -- model-free: one cross-fitted probe per modality
  on the typed contract. Says whether the *data* are aligned.
* :func:`expert_branches` -- read the per-modality experts out of a fitted
  :class:`~medtabx.models.typed_token.TypedTokenTransformer`, the exact
  analogue of splitting the fused classifier weights into W^a, W^v.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medtabx.benchmarks.typed_bench import TypedData, _ModalityBlocks

EPS = 1e-6


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-sample JS divergence (nats) between two Bernoulli predictions."""
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    m = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    return 0.5 * (kl(p, m) + kl(q, m))


def pairwise_js(probs: dict[str, np.ndarray],
                present: dict[str, np.ndarray] | None = None
                ) -> tuple[float, dict[str, float]]:
    """Mean JS over modality pairs observed for the same patient."""
    mods = list(probs)
    per_pair, total, count = {}, 0.0, 0
    for i, a in enumerate(mods):
        for b in mods[i + 1:]:
            ok = np.ones(len(probs[a]), bool)
            if present:
                ok = present[a] & present[b]
            if not ok.any():
                continue
            v = float(js_divergence(probs[a][ok], probs[b][ok]).mean())
            per_pair[f"{a}|{b}"] = v
            total += v * ok.sum()
            count += int(ok.sum())
    return (total / count if count else float("nan")), per_pair


def probe_branches(data: TypedData, n_splits: int = 5, seed: int = 0,
                   img_components: int = 32, probe: str = "lr"
                   ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Cross-fitted P(y=1 | one modality) per modality, plus presence masks.

    Absent modalities give no prediction (masked out), never an imputed one.
    """
    from medtabx.analysis.info import cross_fit_ce  # noqa: F401 (shared probe)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    blocks = _ModalityBlocks(img_components).fit(data)
    b = blocks.transform(data)
    probs, present = {}, {}
    for m in blocks.mods:
        x = b[m]
        ok = ~np.isnan(x).all(axis=1)
        p = np.full(len(data), np.nan)
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros(len(data)), data.y):
            tr = tr[ok[tr]]
            if len(np.unique(data.y[tr])) < 2:
                continue
            mu, sd = np.nanmean(x[tr], 0), np.nanstd(x[tr], 0)
            sd[~(sd > 0)] = 1.0
            fit_x = np.nan_to_num((x[tr] - mu) / sd)
            if probe == "lr":
                est = LogisticRegression(C=0.05, max_iter=5000,
                                         random_state=seed)
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                est = HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, max_depth=3,
                    random_state=seed)
            est.fit(fit_x, data.y[tr])
            te_ok = te[ok[te]]
            if len(te_ok):
                p[te_ok] = est.predict_proba(
                    np.nan_to_num((x[te_ok] - mu) / sd))[:, 1]
        probs[m] = p
        present[m] = ok & ~np.isnan(p)
    return probs, present


def expert_branches(model, data: TypedData, method) -> tuple[
        dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-modality expert predictions of a fitted typed token transformer.

    ``method`` is the :class:`TypedTransformerMethod` that built the tensors
    (it owns the fold's standardization), so this is the fused model's own
    decomposition, not a separate probe.
    """
    import torch

    b = method._tensors(data)
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        num = None if b.get("num") is None else b["num"].to(dev)
        cat = None if b.get("cat") is None else b["cat"].to(dev)
        img = None if b.get("img") is None else b["img"].to(dev)
        pres = None if b.get("present") is None else b["present"].to(dev)
        _, missing = model.tokens(num, cat, img, pres)
        logits, observed = model._experts(num, cat, img, missing)
        p = torch.sigmoid(logits[..., 0]).cpu().numpy()
        obs = observed.cpu().numpy()
    mods = model.schema.modalities
    return ({m: p[:, j] for j, m in enumerate(mods)},
            {m: obs[:, j] for j, m in enumerate(mods)})


@dataclass
class AlignmentReport:
    """Branch performance, imbalance and alignment for one dataset/model."""

    auroc: dict[str, float]
    present_rate: dict[str, float]
    js_mean: float
    js_pairs: dict[str, float] = field(default_factory=dict)
    fused_auroc: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def best_modality(self) -> str:
        return max(self.auroc, key=lambda m: self.auroc[m])

    @property
    def gap(self) -> float:
        """Strongest minus weakest modality: their modality imbalance."""
        v = [x for x in self.auroc.values() if not np.isnan(x)]
        return float(max(v) - min(v)) if v else float("nan")

    @property
    def fusion_gain(self) -> float:
        """Fused minus best single modality; negative = the dagger case."""
        return self.fused_auroc - max(self.auroc.values())

    def as_dict(self) -> dict:
        d = {f"auroc_{m}": v for m, v in self.auroc.items()}
        d.update({f"present_{m}": v for m, v in self.present_rate.items()})
        d.update({f"js_{k}": v for k, v in self.js_pairs.items()})
        d.update(js_mean=self.js_mean, gap=self.gap,
                 best_modality=self.best_modality,
                 fused_auroc=self.fused_auroc, fusion_gain=self.fusion_gain,
                 **self.extra)
        return d


def branch_report(probs: dict[str, np.ndarray],
                  present: dict[str, np.ndarray], y: np.ndarray,
                  fused: np.ndarray | None = None, **extra
                  ) -> AlignmentReport:
    from sklearn.metrics import roc_auc_score

    auroc, rate = {}, {}
    for m, p in probs.items():
        ok = present[m] & ~np.isnan(p)
        rate[m] = float(ok.mean())
        auroc[m] = (float(roc_auc_score(y[ok], p[ok]))
                    if ok.sum() > 10 and len(np.unique(y[ok])) == 2
                    else float("nan"))
    js, pairs = pairwise_js(probs, present)
    fa = float("nan")
    if fused is not None and len(np.unique(y)) == 2:
        fa = float(roc_auc_score(y, fused))
    return AlignmentReport(auroc=auroc, present_rate=rate, js_mean=js,
                           js_pairs=pairs, fused_auroc=fa, extra=extra)
