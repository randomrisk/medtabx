"""AutoRecipe: recommend a downstream model and fusion for a typed dataset.

One scheme, two steps (budgeted algorithm selection):

1. **Rule prior.** Dataset meta-features (:func:`medtabx.auto.frame.describe`)
   and the objective filter infeasible models and give every feasible
   ``base@fusion`` a prior score from explicit, cited rules. Each rule that
   fires is kept as a human-readable reason.
2. **Successive halving.** The top-``k`` candidates by prior (at least one
   per model family, plus a linear reference) are cross-validated on a
   growing number of folds; after each rung only the best ``1/eta`` by
   utility survive. Folds are cached, so later rungs only add new folds.

Utility (higher is better):

    U = AUROC - w_cal * ECE - w_time * log10(1 + fit_s)
        - w_missing * max(0, AUROC_full - AUROC_missing)

A learned recommender (meta-learning across datasets, evaluated
leave-one-dataset-out) needs many more datasets than exist here; every run
therefore appends its meta-features, candidates and scores to a JSONL
``history`` file that can train one later.

``AutoRecipe`` itself follows the learner protocol (``fit(data, seed)`` /
``predict(data)``), so it can be evaluated with nested CV like any model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from medtabx.auto import zoo
from medtabx.auto.compare import default_scenarios, evaluate, splits
from medtabx.auto.frame import describe
from medtabx.benchmarks.typed_bench import TypedData


@dataclass
class Objective:
    """Weights of the utility. Presets: ``accuracy``, ``balanced``,
    ``robust`` (penalizes loss under a missing modality), ``fast``."""

    w_cal: float = 0.1
    w_time: float = 0.005
    w_missing: float = 0.25
    max_cost: float | None = None   # drop bases above this relative cost

    @classmethod
    def preset(cls, name: str) -> "Objective":
        return {
            "accuracy": cls(0.0, 0.0, 0.0),
            "balanced": cls(),
            "robust": cls(w_cal=0.1, w_time=0.005, w_missing=0.75),
            "fast": cls(w_cal=0.1, w_time=0.05, w_missing=0.25,
                        max_cost=1.0),
        }[name]


# Rule prior ###############################################################


@dataclass
class Rule:
    name: str
    applies: Callable[[dict, "zoo.Base", str, Objective], bool]
    delta: float
    reason: str


RULES: list[Rule] = [
    Rule("tabpfn_small_n",
         lambda m, b, f, o: b.family == "foundation" and m["n"] <= 3000,
         +2.0, "in-context tabular foundation model; strongest reported "
               "regime is <= 10k rows (Hollmann et al., Nature 2025)"),
    Rule("gbdt_default",
         lambda m, b, f, o: b.family == "gbdt",
         +1.5, "GBDTs are the reference to beat on tabular data "
               "(Grinsztajn et al. 2022; PyTorch Frame 2024)"),
    Rule("forest_small_n",
         lambda m, b, f, o: b.family == "tree" and m["n"] < 1000,
         +0.5, "bagged trees are low-variance on small cohorts"),
    Rule("deep_needs_data",
         lambda m, b, f, o: b.family in ("deep", "typed_token")
         and m["n"] < 2000,
         -1.5, "deep tabular models rarely beat GBDT below a few thousand "
               "rows (PyTorch Frame 2024)"),
    Rule("deep_large_n",
         lambda m, b, f, o: b.family in ("deep", "typed_token")
         and m["n"] >= 10_000,
         +1.0, "enough rows for representation learning"),
    Rule("linear_tiny_n",
         lambda m, b, f, o: b.family == "linear" and m["n"] < 300,
         +0.5, "strongly regularized linear models are hard to beat at "
               "very small n"),
    Rule("structured_fusion_multimodal",
         lambda m, b, f, o: f in ("late", "subset")
         and m["n_modalities"] >= 2 and len(m["value_types"]) >= 2,
         +1.0, "heterogeneous modalities: per-modality models avoid one "
               "block dominating and never impute a whole modality"),
    Rule("structured_fusion_missing",
         lambda m, b, f, o: f in ("late", "subset")
         and (m["max_modality_missing"] > 0.05 or o.w_missing > 0.5),
         +1.0, "modalities are missing (or robustness is weighted): only "
               "members whose inputs are present score a patient"),
    Rule("wide_images",
         lambda m, b, f, o: f == "late"
         and m["flat_features"] > m["n"] and m["n_image_modalities"] > 0,
         +0.5, "more features than patients: per-modality blocks keep "
               "each model low-dimensional"),
    Rule("concat_deep_fragile",
         lambda m, b, f, o: b.family in ("deep",) and o.w_missing > 0
         and m["n_modalities"] >= 2,
         -1.0, "whole-input deep tabular models degrade sharply when a "
               "modality is missing at test time"),
    Rule("subset_cost",
         lambda m, b, f, o: f == "subset" and b.cost >= 3,
         -0.5, "subset ensembles multiply an already expensive learner"),
]


@dataclass
class Candidate:
    model: str
    label: str
    family: str
    prior: float
    reasons: list[str] = field(default_factory=list)
    rung: int = -1
    folds: int = 0
    auroc: float = float("nan")
    ece: float = float("nan")
    fit_s: float = float("nan")
    missing_drop: float = float("nan")
    utility: float = float("nan")
    error: str | None = None


def prior(meta: dict, objective: Objective,
          models: list[str] | None = None) -> list[Candidate]:
    """Feasible candidates with rule-based prior scores (highest first)."""
    out = []
    for mid in models or zoo.list_models(include_unavailable=True):
        ok, why = zoo.feasible(mid, meta)
        b, fusion = zoo.parse(mid)
        if not ok:
            continue
        if objective.max_cost is not None and b.cost > objective.max_cost:
            continue
        c = Candidate(mid, zoo.label(mid), b.family, 0.0)
        for r in RULES:
            if r.applies(meta, b, fusion, objective):
                c.prior += r.delta
                c.reasons.append(f"{r.delta:+.1f} {r.name}: {r.reason}")
        c.prior -= 0.1 * math.log10(1 + b.cost)  # tie-break: cheaper first
        out.append(c)
    return sorted(out, key=lambda c: -c.prior)


def portfolio(cands: list[Candidate], k: int) -> list[Candidate]:
    """``k`` candidates: first the best of each family (in prior order),
    then the next best by prior, so the search can overrule the rules
    without exceeding the budget."""
    chosen, seen = [], set()
    for c in cands:
        if len(chosen) < k and c.family not in seen:
            chosen.append(c)
            seen.add(c.family)
    for c in cands:
        if len(chosen) >= k:
            break
        if c not in chosen:
            chosen.append(c)
    return sorted(chosen, key=lambda c: -c.prior)


def sh_schedule(n_cand: int, max_folds: int, eta: int = 3,
                min_folds: int = 2) -> list[tuple[int, int]]:
    """(candidates, cumulative folds) per rung."""
    rungs, n, f = [], n_cand, min_folds
    while True:
        rungs.append((n, min(f, max_folds)))
        if f >= max_folds or n <= 1:
            break
        n, f = max(1, math.ceil(n / eta)), f * eta
    return rungs


def sh_cost(n_cand: int, max_folds: int, eta: int = 3) -> int:
    cost, prev = 0, 0
    for n, f in sh_schedule(n_cand, max_folds, eta):
        cost += n * (f - prev)
        prev = f
    return cost


# AutoRecipe ###############################################################


class AutoRecipe:
    """Recommend and fit a downstream model for a typed dataset.

    Args:
        objective: an :class:`Objective` or preset name.
        budget: maximum number of model fits spent on the search.
        n_splits, repeats: inner CV for the search (folds are the fidelity).
        eta: successive-halving reduction factor.
        models: restrict the search space (model ids); default = all.
        history: JSONL file to append the run record to (meta-dataset for a
            future learned recommender).
        select_modalities: before the search, choose the modalities inside
            the training data (forward selection with the permutation
            filter, :mod:`medtabx.auto.select`) and search only on those.
            Off by default, so existing runs are unchanged.
    """

    name = "MedTabX AutoRecipe"

    def __init__(self, objective: str | Objective = "balanced",
                 budget: int = 60, n_splits: int = 5, repeats: int = 2,
                 eta: int = 3, models: list[str] | None = None,
                 history: str | Path | None = None,
                 log: Callable[[str], None] | None = None,
                 select_modalities: bool = False) -> None:
        self.objective = (Objective.preset(objective)
                          if isinstance(objective, str) else objective)
        self.budget = budget
        self.n_splits, self.repeats, self.eta = n_splits, repeats, eta
        self.models = models
        self.history = history
        self.log = log or (lambda s: None)
        self.select_modalities = select_modalities
        self.selected: list[str] | None = None

    def _utility(self, c: Candidate) -> float:
        o = self.objective
        drop = 0.0 if np.isnan(c.missing_drop) else max(0.0, c.missing_drop)
        ece = 0.0 if np.isnan(c.ece) else c.ece
        return (c.auroc - o.w_cal * ece
                - o.w_time * math.log10(1 + c.fit_s) - o.w_missing * drop)

    def search(self, data: TypedData, seed: int = 0) -> pd.DataFrame:
        """Run the recommendation; returns the candidate table."""
        t0 = time.time()
        self.meta = describe(data)
        cands = prior(self.meta, self.objective, self.models)
        if not cands:
            raise RuntimeError("no feasible model for this data")
        max_folds = self.n_splits * self.repeats
        k = len(cands)
        while k > 1 and sh_cost(k, max_folds, self.eta) > self.budget:
            k -= 1
        pool = portfolio(cands, k)
        self.log(f"AutoRecipe: {len(cands)} feasible, searching {len(pool)} "
                 f"with <= {sh_cost(len(pool), max_folds, self.eta)} fits")
        all_splits = splits(data.y, self.n_splits, self.repeats, seed)
        order = np.random.default_rng(seed).permutation(len(all_splits))
        all_splits = [all_splits[i] for i in order]
        scen = default_scenarios(data)
        if self.objective.w_missing == 0:
            scen = {"full": []}
        rows: dict[str, list[dict]] = {c.model: [] for c in pool}
        alive, used = list(pool), 0
        for rung, (n_keep, n_folds) in enumerate(
                sh_schedule(len(pool), max_folds, self.eta)):
            alive = sorted(alive, key=lambda c: -c.utility
                           if not np.isnan(c.utility) else np.inf)[:n_keep]
            for c in alive:
                for sp in all_splits[used:n_folds]:
                    rows[c.model] += evaluate(c.model, data, sp, scen,
                                              seed=seed)
                self._update(c, rows[c.model], rung)
            used = n_folds
            alive = [c for c in alive if c.error is None]
            self.log(f"  rung {rung}: {n_folds} folds, "
                     + ", ".join(f"{c.model}={c.utility:.3f}" for c in
                                 sorted(alive, key=lambda c: -c.utility)))
            if not alive:
                raise RuntimeError("every candidate failed")
        # Winner: the highest utility among those that reached the last rung.
        top = max(c.rung for c in pool)
        finalists = [c for c in pool if c.rung == top and c.error is None]
        self.best = max(finalists, key=lambda c: c.utility)
        self.candidates = pool
        self.seconds = time.time() - t0
        self.table = pd.DataFrame([asdict(c) for c in pool]).sort_values(
            ["rung", "utility"], ascending=False).reset_index(drop=True)
        if self.history:
            self._write_history()
        return self.table

    def _update(self, c: Candidate, rows: list[dict], rung: int) -> None:
        c.rung = rung
        err = [r for r in rows if "error" in r]
        if err:
            c.error = err[0]["error"]
            c.utility = -np.inf
            return
        df = pd.DataFrame(rows)
        full = df[df.scenario == "full"]
        c.folds = len(full)
        c.auroc = full.auroc.mean()
        c.ece = full.ece.mean()
        c.fit_s = full.fit_seconds.median()
        miss = df[df.scenario != "full"]
        c.missing_drop = (c.auroc - miss.auroc.mean()) if len(miss) else np.nan
        c.utility = self._utility(c)

    def explain(self) -> str:
        b = self.best
        lines = [f"Recommended: {b.model}  ({b.label})",
                 f"  CV AUROC {b.auroc:.3f} over {b.folds} folds, ECE "
                 f"{b.ece:.3f}, drop with a missing modality "
                 f"{b.missing_drop:+.3f}, fit {b.fit_s:.1f}s, utility "
                 f"{b.utility:.3f}",
                 "  prior reasons:"] + [f"    {r}" for r in b.reasons]
        lines.append("  runners-up: " + ", ".join(
            f"{c.model} ({c.utility:.3f})" for c in sorted(
                [c for c in self.candidates if c is not b
                 and not np.isnan(c.utility)],
                key=lambda c: -c.utility)[:3]))
        return "\n".join(lines)

    def _write_history(self) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "meta": self.meta, "objective": asdict(self.objective),
               "budget": self.budget, "seconds": self.seconds,
               "best": self.best.model,
               "candidates": [asdict(c) for c in self.candidates]}
        p = Path(self.history)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")

    # Learner protocol ------------------------------------------------------

    def fit(self, data: TypedData, seed: int = 0) -> "AutoRecipe":
        """Search on ``data`` and refit the recommended model on all of it.
        """
        self.selected = None
        if self.select_modalities and len(data.modalities) > 1:
            from medtabx.auto.select import (
                GreedyModalitySelector,
                ModalitySelected,
            )
            self.selected = GreedyModalitySelector(
                permutation_filter=True).select(data, seed=seed)
            self.log(f"AutoRecipe: modalities kept in-fold: {self.selected}")
            data = data.select(self.selected)
        self.search(data, seed=seed)
        model = zoo.create_model(self.best.model)
        if self.selected is not None:
            model = ModalitySelected(model, self.selected)
        self.model = model.fit(data, seed=seed)
        return self

    def predict(self, data: TypedData) -> np.ndarray:
        return self.model.predict(data)
