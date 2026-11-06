"""Ordinal outcomes: cumulative-link targets, coherence repair and metrics.

The modified Rankin Scale is an ordered seven-level scale (0 = no symptoms,
6 = dead), and the usual analysis collapses it to one dichotomy, mRS 0--2
against 3--6. That throws away the ordering and makes the result depend on a
cut-point that was chosen by convention rather than by the data.

Rather than replace the binary machinery, we decompose the ordinal problem
into the ``K-1`` cumulative questions

    P(y > 0),  P(y > 1),  ...,  P(y > K-2),

following Frank and Hall (2001). Each is an ordinary binary problem, so every
learner, fusion and augmentation in the framework applies unchanged, and the
conventional dichotomy is recovered exactly at ``k = 2``. Each patient now
carries ``K-1`` binary labels at once, which is what makes this a multi-label
formulation of an ordinal task rather than a seven-way one-of-K problem; the
labels are nested, and that nesting is the structure the metrics below
exploit.

Fitting the thresholds independently can produce an incoherent answer,
``P(y > 2) > P(y > 1)``, which no distribution admits.
:func:`monotone_repair` restores the ordering and reports how often it had to
intervene --- a diagnostic worth reporting, since it measures how far the
independent-binary shortcut is from a genuine ordinal model.
"""

from __future__ import annotations

import numpy as np

# mRS 0..6
N_LEVELS_MRS = 7


def cumulative_targets(y: np.ndarray, n_levels: int) -> np.ndarray:
    """``[N, n_levels-1]`` binary matrix whose column ``k`` is ``y > k``."""
    y = np.asarray(y)
    return np.stack([(y > k).astype(int) for k in range(n_levels - 1)],
                    axis=1)


def monotone_repair(cum: np.ndarray) -> tuple[np.ndarray, float]:
    """Force ``P(y > k)`` to be non-increasing in ``k``, row by row.

    Takes ``[N, K-1]`` independent threshold probabilities and returns the
    repaired array together with the fraction of rows that violated the
    ordering. The repair is a running minimum, the isotonic projection under
    the constraint that matters here, and it leaves already-coherent rows
    untouched.
    """
    cum = np.asarray(cum, dtype=float)
    fixed = np.minimum.accumulate(cum, axis=1)
    violated = float((np.abs(fixed - cum) > 1e-12).any(axis=1).mean())
    return fixed, violated


def distribution(cum: np.ndarray) -> np.ndarray:
    """``[N, K]`` level probabilities from coherent cumulative probabilities.

    ``P(y = 0) = 1 - P(y > 0)``, ``P(y = k) = P(y > k-1) - P(y > k)``, and
    ``P(y = K-1) = P(y > K-2)``.
    """
    cum = np.clip(np.asarray(cum, dtype=float), 0.0, 1.0)
    n, km1 = cum.shape
    padded = np.concatenate(
        [np.ones((n, 1)), cum, np.zeros((n, 1))], axis=1)
    p = padded[:, :-1] - padded[:, 1:]
    p = np.clip(p, 0.0, None)
    total = p.sum(axis=1, keepdims=True)
    return np.divide(p, total, out=np.full_like(p, 1.0 / (km1 + 1)),
                     where=total > 0)


def expected_level(p: np.ndarray) -> np.ndarray:
    """The predicted score itself: ``sum_k k * P(y = k)``."""
    levels = np.arange(p.shape[1])
    return p @ levels


# -- metrics ---------------------------------------------------------------


def quadratic_weighted_kappa(y: np.ndarray, pred: np.ndarray,
                             n_levels: int) -> float:
    """Agreement between true and predicted levels, penalizing a
    disagreement by the square of its distance. 1 is perfect, 0 is chance."""
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    o = np.zeros((n_levels, n_levels))
    for a, b in zip(y, pred):
        o[a, b] += 1
    w = (np.arange(n_levels)[:, None] - np.arange(n_levels)[None, :]) ** 2
    hist_y = np.bincount(y, minlength=n_levels)
    hist_p = np.bincount(pred, minlength=n_levels)
    e = np.outer(hist_y, hist_p) / max(len(y), 1)
    denom = (w * e).sum()
    return float(1.0 - (w * o).sum() / denom) if denom > 0 else 0.0


def ordinal_c_index(y: np.ndarray, score: np.ndarray) -> float:
    """Generalization of AUROC to more than two ordered levels.

    Over all patient pairs whose true levels differ, the fraction whose
    predicted scores are ordered the same way, with ties counting a half.
    Equals the binary AUROC when there are two levels.
    """
    y, score = np.asarray(y, float), np.asarray(score, float)
    dy = y[:, None] - y[None, :]
    comparable = dy < 0                      # i strictly below j
    if not comparable.any():
        return float("nan")
    ds = score[:, None] - score[None, :]
    concordant = (ds < 0)[comparable].sum()
    ties = (ds == 0)[comparable].sum()
    return float((concordant + 0.5 * ties) / comparable.sum())


def threshold_auroc(y: np.ndarray, cum: np.ndarray) -> dict[str, float]:
    """AUROC at each cut-point, so a model that works only at the
    conventional dichotomy can be told from one that works everywhere."""
    from sklearn.metrics import roc_auc_score

    out = {}
    for k in range(cum.shape[1]):
        target = (np.asarray(y) > k).astype(int)
        out[f"auroc_gt{k}"] = (
            float(roc_auc_score(target, cum[:, k]))
            if len(np.unique(target)) > 1 else float("nan"))
    return out


def ordinal_metrics(y: np.ndarray, cum: np.ndarray,
                    n_levels: int = N_LEVELS_MRS) -> dict[str, float]:
    """Everything we report for one ordinal prediction.

    ``cum`` holds the raw ``[N, K-1]`` threshold probabilities; the repair and
    the assembly into a distribution happen here, so callers cannot forget
    them.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import log_loss

    y = np.asarray(y, int)
    fixed, violated = monotone_repair(cum)
    p = distribution(fixed)
    score = expected_level(p)
    argmax = p.argmax(axis=1)

    rho = spearmanr(y, score).statistic if len(np.unique(y)) > 1 else np.nan
    return {
        "qwk": quadratic_weighted_kappa(y, argmax, n_levels),
        "qwk_expected": quadratic_weighted_kappa(
            y, np.rint(score).astype(int).clip(0, n_levels - 1), n_levels),
        "c_index": ordinal_c_index(y, score),
        "spearman": float(rho),
        "mae_levels": float(np.abs(score - y).mean()),
        "acc": float((argmax == y).mean()),
        "acc_within1": float((np.abs(argmax - y) <= 1).mean()),
        "nll": float(log_loss(y, p, labels=list(range(n_levels)))),
        "monotonicity_violated": violated,
        # the conventional dichotomy, for comparison with the binary results
        "auroc_dichotomy": threshold_auroc(y, cum).get("auroc_gt2", np.nan),
        **threshold_auroc(y, cum),
    }
