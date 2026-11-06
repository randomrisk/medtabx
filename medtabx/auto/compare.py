"""``compare_models``: run many downstream models on one typed contract and
return a leaderboard (in the spirit of PyCaret's ``compare_models``).

Every model is scored on the same stratified folds (identical to
``typed_bench.run_cv`` for the same seed) and, per fold, under test-time
scenarios where one modality is removed. The leaderboard reports
discrimination (AUROC, AUPRC), calibration (Brier, ECE), robustness to a
missing modality and fit time. A model that fails is recorded with its error
instead of stopping the run.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from medtabx.auto import zoo
from medtabx.benchmarks.typed_bench import TypedData
from medtabx.metrics.functional import (
    binary_auroc,
    binary_average_precision,
    brier_score,
    expected_calibration_error,
)


def default_scenarios(d: TypedData) -> dict[str, list[str]]:
    """``full`` plus one scenario per modality removed at test time."""
    scen = {"full": []}
    if len(d.modalities) > 1:
        scen.update({f"no {m}": [m] for m in d.modalities})
    return scen


def splits(y: np.ndarray, n_splits: int = 5, repeats: int = 1,
           seed: int = 0) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Repeated stratified K-fold, same seeds as ``typed_bench.run_cv``."""
    from sklearn.model_selection import StratifiedKFold
    out = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            out.append((r, f, tr, te))
    return out


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y_t = torch.tensor(y)
    p_t = torch.tensor(np.clip(p, 0, 1), dtype=torch.float64)
    return {"auroc": binary_auroc(p_t, y_t),
            "auprc": binary_average_precision(p_t, y_t),
            "brier": brier_score(p_t, y_t),
            "ece": expected_calibration_error(p_t, y_t, num_bins=10)}


def evaluate(model_id: str, data: TypedData, split, scenarios,
             seed: int = 0, **zoo_kw) -> list[dict]:
    """Fit one model on one split; one row per scenario (or one error row).
    """
    r, f, tr, te = split
    train, test = data.subset(tr), data.subset(te)
    t0 = time.time()
    try:
        model = zoo.create_model(model_id, **zoo_kw).fit(
            train, seed=seed + 10 * r + f)
        fit_s = time.time() - t0
        rows = []
        for scen, dropped in scenarios.items():
            p = model.predict(test.drop(dropped) if dropped else test)
            rows.append({"model": model_id, "repeat": r, "fold": f,
                         "scenario": scen, "fit_seconds": fit_s,
                         **score(test.y, np.asarray(p, float))})
        return rows
    except Exception as e:  # noqa: BLE001 -- keep the leaderboard going
        return [{"model": model_id, "repeat": r, "fold": f,
                 "scenario": "full", "error": f"{type(e).__name__}: {e}",
                 "trace": traceback.format_exc(limit=3)}]


def leaderboard(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold rows: one line per model, best AUROC first."""
    ok = rows[rows.get("error", pd.Series(index=rows.index)).isna()] \
        if "error" in rows else rows
    full = ok[ok.scenario == "full"]
    agg = (full.groupby("model")
           .agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                auprc=("auprc", "mean"), brier=("brier", "mean"),
                ece=("ece", "mean"), fit_s=("fit_seconds", "median"),
                folds=("auroc", "count")))
    miss = ok[ok.scenario != "full"]
    if len(miss):
        agg = agg.join(miss.groupby("model").agg(
            auroc_missing=("auroc", "mean")))
        agg["missing_drop"] = agg["auroc"] - agg["auroc_missing"]
    if "error" in rows:
        err = rows[rows["error"].notna()].groupby("model")["error"].first()
        agg = agg.join(err, how="outer")
    agg.insert(0, "label", [zoo.label(m) for m in agg.index])
    agg.insert(1, "family", [zoo.parse(m)[0].family for m in agg.index])
    return agg.sort_values("auroc", ascending=False)


@dataclass
class Comparison:
    rows: pd.DataFrame
    table: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.table = leaderboard(self.rows)

    @property
    def best(self) -> str:
        return self.table["auroc"].idxmax()

    def __repr__(self) -> str:
        cols = [c for c in ("label", "auroc", "auroc_sd", "auprc", "brier",
                            "ece", "auroc_missing", "missing_drop", "fit_s",
                            "error") if c in self.table]
        return self.table[cols].round(3).to_string()


def compare_models(data: TypedData, models: str | Iterable[str] = "all",
                   n_splits: int = 5, repeats: int = 1,
                   scenarios: dict[str, list[str]] | None = None,
                   seed: int = 0, log: Callable[[str], None] | None = print,
                   **zoo_kw) -> Comparison:
    """Cross-validate many models on the same contract.

    Args:
        models: ``"all"`` (every feasible model id here), a family name
            (``"gbdt"``, ``"deep"``, ``"foundation"``, ``"tree"``,
            ``"linear"``, ``"typed_token"``), or a list of model ids such as
            ``["cat", "tabpfn@late", "rf@subset", "pf_ft"]``.
        scenarios: test-time missing-modality scenarios; default removes
            each modality in turn.
    """
    from medtabx.auto.frame import describe
    meta = describe(data)
    if isinstance(models, str):
        families = {b.family for b in zoo.BASES.values()}
        ids = zoo.list_models(data)
        if models in families:
            ids = [m for m in ids if zoo.parse(m)[0].family == models]
        elif models != "all":
            ids = [models]
    else:
        ids = list(models)
    for m in ids:
        ok, why = zoo.feasible(m, meta)
        if not ok:
            raise ValueError(f"{m}: {why}")
    scenarios = scenarios or default_scenarios(data)
    rows = []
    for sp in splits(data.y, n_splits, repeats, seed):
        for m in ids:
            out = evaluate(m, data, sp, scenarios, seed=seed, **zoo_kw)
            rows += out
            if log:
                r0 = out[0]
                msg = (f"error {r0['error']}" if "error" in r0 else
                       f"auroc {r0['auroc']:.3f}  {r0['fit_seconds']:.1f}s")
                log(f"  r{sp[0]} f{sp[1]} {m:18s} {msg}")
    return Comparison(pd.DataFrame(rows))
