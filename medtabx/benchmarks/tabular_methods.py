"""Tabular-model baselines for the typed benchmark, compared the way the
PyTorch Frame paper does (arXiv:2404.00776, Sec. 5.1).

* The seven deep tabular models shipped in ``medtabx.models`` (MLP, ResNet,
  FT-Transformer, TabNet, TabTransformer, Trompt, ExcelFormer) run through one
  adapter that builds a :class:`TensorFrame` per fold.
* GBDT references: LightGBM, XGBoost, CatBoost.
* All of them get the *same* per-fold features: numeric and categorical
  columns, one missing indicator per modality, and each image modality
  reduced to ``img_components`` PCA components (fit on the training fold).
  So a GBDT sees the same image embeddings as the deep models.
* Hyperparameters: Optuna TPE, ``n_trials`` trials (paper: 20) scored by
  AUROC on an inner stratified hold-out of the training fold only; the best
  configuration is then refit on the whole training fold. ``n_trials=0``
  uses the default configuration.

Methods plug into :data:`medtabx.benchmarks.typed_bench.METHODS` and so run on
exactly the folds and test-time scenarios of the other experiments.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from medtabx.benchmarks.typed_bench import (
    METHODS,
    TypedData,
    _device,
    _TorchBase,
)

N_TRIALS = int(os.environ.get("MEDTABX_N_TRIALS", "20"))
IMG_COMPONENTS = 16


# Features shared by every tabular model ###################################


class _FrameTable:
    """TypedData -> DataFrame with named, typed columns. Statistics (PCA,
    imputation means, category levels) come from the training fold only."""

    def __init__(self, img_components: int = IMG_COMPONENTS) -> None:
        self.k = img_components

    def fit(self, d: TypedData) -> "_FrameTable":
        self.num_cols = list(d.num_cols) or [
            f"num{j}" for j in range(0 if d.num is None else d.num.shape[1])]
        self.cat_cols = list(d.cat_cols) or [
            f"cat{j}" for j in range(0 if d.cat is None else d.cat.shape[1])]
        self.mods = list(dict.fromkeys(
            list(d.num_modality) + list(d.cat_modality)
            + list(d.img_modality)))
        self.pca = {}
        if d.img is not None:
            from sklearn.decomposition import PCA
            g = d.img.astype(np.float32).mean(axis=2)
            for j, m in enumerate(d.img_modality):
                ok = d.img_present[:, j]
                k = int(min(self.k, ok.sum() - 1, g.shape[2]))
                if k >= 1:
                    self.pca[m] = PCA(k, random_state=0).fit(g[ok, j])
        raw = self._raw(d)
        self.numeric = [c for c in raw.columns if c not in self.cat_cols]
        self.means = raw[self.numeric].mean().fillna(0.0)
        return self

    def _raw(self, d: TypedData) -> pd.DataFrame:
        cols: dict[str, Any] = {}
        for j, c in enumerate(self.num_cols):
            cols[c] = d.num[:, j].astype(float)
        for j, c in enumerate(self.cat_cols):
            v = d.cat[:, j]
            cols[c] = np.where(v >= 0, np.char.add("c", v.astype(str)), "NA")
        if d.img is not None:
            g = d.img.astype(np.float32).mean(axis=2)
            for j, m in enumerate(d.img_modality):
                if m not in self.pca:
                    continue
                z = self.pca[m].transform(g[:, j]).astype(float)
                z[~d.img_present[:, j]] = np.nan
                for i in range(z.shape[1]):
                    cols[f"{m}_pc{i}"] = z[:, i]
        df = pd.DataFrame(cols)
        for m in self.mods:
            own = [c for c in df.columns if self._mod_of(c, d) == m]
            if own:
                miss = np.ones(len(df), dtype=bool)
                for c in own:
                    miss &= (df[c].isna().to_numpy() if c not in self.cat_cols
                             else (df[c] == "NA").to_numpy())
                df[f"missing_{m}"] = miss.astype(float)
        return df

    def _mod_of(self, col: str, d: TypedData) -> str | None:
        if col in self.num_cols:
            return d.num_modality[self.num_cols.index(col)]
        if col in self.cat_cols:
            return d.cat_modality[self.cat_cols.index(col)]
        for m in self.pca:
            if col.startswith(f"{m}_pc"):
                return m
        return None

    def frame(self, d: TypedData, impute: bool) -> pd.DataFrame:
        df = self._raw(d)
        for c in self.numeric:
            if c not in df:
                df[c] = np.nan
        df = df[self.numeric + self.cat_cols]
        if impute:
            df[self.numeric] = df[self.numeric].fillna(self.means)
        return df


# GBDT references ##########################################################


def _stratified_holdout(y: np.ndarray, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    val = []
    for c in (0, 1):
        cls = np.nonzero(y == c)[0]
        val.extend(rng.choice(cls, max(1, int(round(len(cls) * frac))),
                              replace=False).tolist())
    mask = np.ones(len(y), dtype=bool)
    mask[val] = False
    return np.nonzero(mask)[0], np.array(sorted(val))


def _auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else 0.5


def _tune(space, fit_score, n_trials: int, seed: int) -> dict:
    """Optuna TPE over ``space(trial) -> params``; ``fit_score(params)``
    returns the inner hold-out AUROC. Returns the best params (defaults when
    ``n_trials == 0``)."""
    if n_trials <= 0:
        return space(None)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: fit_score(space(t)), n_trials=n_trials)
    return space(optuna.trial.FixedTrial(study.best_params))


def _pick(trial, name, choices, default):
    return default if trial is None else trial.suggest_categorical(
        name, choices)


class GBDTLibMethod:
    """LightGBM / XGBoost / CatBoost on the shared per-fold features, with
    native missing-value and categorical handling."""

    def __init__(self, kind: str, n_trials: int | None = None) -> None:
        self.kind = kind
        self.n_trials = N_TRIALS if n_trials is None else n_trials
        self.name = {"lgbm": "LightGBM", "xgb": "XGBoost",
                     "cat": "CatBoost"}[kind]

    def _space(self, t):
        return {
            "n_estimators": _pick(t, "n_estimators", [100, 200, 400], 200),
            "learning_rate": _pick(t, "learning_rate", [0.01, 0.03, 0.1],
                                   0.03),
            "depth": _pick(t, "depth", [2, 3, 4, 6], 3),
            "l2": _pick(t, "l2", [0.1, 1.0, 10.0], 1.0),
            "subsample": _pick(t, "subsample", [0.7, 1.0], 0.8),
        }

    def _x(self, d):
        df = self.table.frame(d, impute=False)
        if self.kind == "cat":
            return df
        for c in self.table.cat_cols:
            df[c] = pd.Categorical(df[c], categories=self.levels[c])
        return df

    def _model(self, p, seed):
        if self.kind == "lgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                num_leaves=2 ** p["depth"], max_depth=p["depth"],
                reg_lambda=p["l2"], subsample=p["subsample"],
                subsample_freq=1, min_child_samples=5, random_state=seed,
                n_jobs=4, verbose=-1)
        if self.kind == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"], max_depth=p["depth"],
                reg_lambda=p["l2"], subsample=p["subsample"],
                enable_categorical=True, tree_method="hist",
                random_state=seed, n_jobs=4, verbosity=0)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=p["n_estimators"], learning_rate=p["learning_rate"],
            depth=p["depth"], l2_leaf_reg=p["l2"],
            subsample=p["subsample"], bootstrap_type="Bernoulli",
            cat_features=list(self.table.cat_cols), random_seed=seed,
            thread_count=4, verbose=False, allow_writing_files=False)

    def fit(self, d, seed):
        self.table = _FrameTable().fit(d)
        raw = self.table.frame(d, impute=False)
        self.levels = {c: sorted(set(raw[c].tolist()) | {"NA"})
                       for c in self.table.cat_cols}
        x, y = self._x(d), d.y
        tr, va = _stratified_holdout(y, 0.2, seed)

        def score(p):
            m = self._model(p, seed).fit(x.iloc[tr], y[tr])
            return _auroc(y[va], m.predict_proba(x.iloc[va])[:, 1])

        self.params = _tune(self._space, score, self.n_trials, seed)
        self.m = self._model(self.params, seed).fit(x, y)
        return self

    def predict(self, d):
        return self.m.predict_proba(self._x(d))[:, 1]


# PyTorch Frame model adapter ##############################################


def _materialize(df: pd.DataFrame, y: np.ndarray, numeric: list[str],
                 cat: list[str], col_stats=None):
    import medtabx.core.stype as stype
    from medtabx.data import Dataset
    frame = df.copy()
    frame["__y__"] = y.astype(int)
    col_to_stype = {c: stype.numerical for c in numeric}
    col_to_stype.update({c: stype.categorical for c in cat})
    col_to_stype["__y__"] = stype.categorical
    ds = Dataset(df=frame, col_to_stype=col_to_stype, target_col="__y__")
    if col_stats is not None:
        col_stats = {**col_stats,
                     "__y__": ds_target_stats(frame["__y__"])}
    ds.materialize(col_stats=col_stats)
    return ds


def ds_target_stats(ser: pd.Series):
    from medtabx.data.stats import StatType
    return {StatType.COUNT: ([0, 1], [int((ser == 0).sum()),
                                      int((ser == 1).sum())])}


class _FrameNet(nn.Module):
    """Wraps a PyTorch Frame model; Trompt's per-layer outputs are averaged
    in training (as in the reference implementation) and the last layer is
    used at inference."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, tf):
        out = self.net(tf)
        if out.dim() == 3:
            out = out.mean(dim=1) if self.training else out[:, -1]
        return out


class FrameModelMethod(_TorchBase):
    """One of the seven PyTorch Frame models in ``medtabx.models``."""

    LABELS = {"mlp": "MLP", "resnet": "ResNet", "ft": "FT-Transformer",
              "tabnet": "TabNet", "tabtf": "TabTransformer",
              "trompt": "Trompt", "excel": "ExcelFormer"}
    #: Tuning trials run shorter; the refit uses the full schedule.
    tune_epochs = 60
    tune_patience = 10

    def __init__(self, arch: str, n_trials: int | None = None) -> None:
        self.arch = arch
        self.n_trials = N_TRIALS if n_trials is None else n_trials
        self.name = self.LABELS[arch]

    def _space(self, t) -> dict:
        a = self.arch
        p: dict[str, Any] = {"lr": _pick(t, "lr", [1e-4, 3e-4, 1e-3], 1e-3)}
        if a in ("mlp", "resnet"):
            p |= {"channels": _pick(t, "channels", [64, 128, 256], 128),
                  "num_layers": _pick(t, "num_layers", [1, 2, 4], 2),
                  "dropout": _pick(t, "dropout", [0.0, 0.2, 0.4], 0.2)}
        elif a == "ft":
            p |= {"channels": _pick(t, "channels", [32, 64, 128], 64),
                  "num_layers": _pick(t, "num_layers", [1, 2, 4], 2)}
        elif a == "tabnet":
            p |= {"split": _pick(t, "split", [32, 64, 128], 64),
                  "num_layers": _pick(t, "num_layers", [3, 4, 6], 4),
                  "gamma": _pick(t, "gamma", [1.2, 1.5, 1.8], 1.5)}
        elif a == "tabtf":
            p |= {"channels": _pick(t, "channels", [16, 32, 64], 32),
                  "num_layers": _pick(t, "num_layers", [2, 4, 6], 4),
                  "num_heads": _pick(t, "num_heads", [2, 4, 8], 4),
                  "dropout": _pick(t, "dropout", [0.0, 0.2], 0.2)}
        elif a == "trompt":
            p |= {"channels": _pick(t, "channels", [32, 64, 128], 64),
                  "num_prompts": _pick(t, "num_prompts", [32, 64, 128], 64),
                  "num_layers": _pick(t, "num_layers", [2, 4, 6], 4)}
        elif a == "excel":
            p |= {"channels": _pick(t, "channels", [16, 32, 64], 32),
                  "num_layers": _pick(t, "num_layers", [2, 4, 5], 4),
                  "num_heads": _pick(t, "num_heads", [2, 4], 4),
                  "dropout": _pick(t, "dropout", [0.0, 0.3], 0.3)}
        return p

    def _net(self, p, col_stats, col_names_dict, num_cols: int) -> nn.Module:
        from medtabx import models as M
        common = dict(col_stats=col_stats, col_names_dict=col_names_dict)
        a = self.arch
        if a == "mlp":
            net = M.MLP(channels=p["channels"], out_channels=1,
                        num_layers=p["num_layers"],
                        dropout_prob=p["dropout"], **common)
        elif a == "resnet":
            net = M.ResNet(channels=p["channels"], out_channels=1,
                           num_layers=p["num_layers"],
                           dropout_prob=p["dropout"], **common)
        elif a == "ft":
            net = M.FTTransformer(channels=p["channels"], out_channels=1,
                                  num_layers=p["num_layers"], **common)
        elif a == "tabnet":
            net = M.TabNet(out_channels=1, num_layers=p["num_layers"],
                           split_feat_channels=p["split"],
                           split_attn_channels=p["split"],
                           gamma=p["gamma"], **common)
        elif a == "tabtf":
            net = M.TabTransformer(
                channels=p["channels"], out_channels=1,
                num_layers=p["num_layers"], num_heads=p["num_heads"],
                encoder_pad_size=2, attn_dropout=p["dropout"],
                ffn_dropout=p["dropout"], **common)
        elif a == "trompt":
            net = M.Trompt(channels=p["channels"], out_channels=1,
                           num_prompts=p["num_prompts"],
                           num_layers=p["num_layers"], **common)
        else:
            net = M.ExcelFormer(
                in_channels=p["channels"], out_channels=1, num_cols=num_cols,
                num_layers=p["num_layers"], num_heads=p["num_heads"],
                residual_dropout=p["dropout"], **common)
        return _FrameNet(net)

    # ExcelFormer is numeric-only: its categoricals are one-hot encoded.
    def _df(self, d: TypedData) -> pd.DataFrame:
        df = self.table.frame(d, impute=True)
        if self.arch != "excel" or not self.table.cat_cols:
            return df
        oh = pd.get_dummies(df[self.table.cat_cols].astype(
            pd.CategoricalDtype(self.levels_flat)), dtype=float)
        oh = oh.reindex(columns=self.onehot_cols, fill_value=0.0)
        return pd.concat([df[self.table.numeric], oh], axis=1)

    def _tf(self, d: TypedData):
        df = self._df(d)
        ds = _materialize(df, d.y, self.numeric, self.cat,
                          col_stats=self.col_stats)
        return ds.tensor_frame

    def _forward(self, model, b, idx, dev):
        return model(b["tf"][idx].to(dev))

    def _fit_once(self, p, tf, y, seed, short: bool):
        torch.manual_seed(seed)
        model = self._net(p, self.col_stats, tf.col_names_dict,
                          num_cols=len(self.numeric) + len(self.cat))
        saved = (self.lr, self.max_epochs, self.patience)
        self.lr = p["lr"]
        if short:
            self.max_epochs, self.patience = self.tune_epochs, \
                self.tune_patience
        try:
            self._train(model, {"tf": tf, "y": torch.tensor(
                y, dtype=torch.float32)}, seed)
        finally:
            self.lr, self.max_epochs, self.patience = saved
        return model

    @torch.no_grad()
    def _prob(self, model, tf) -> np.ndarray:
        model.eval()
        out = model(tf.to(_device()))
        return torch.sigmoid(out.squeeze(-1)).float().cpu().numpy()

    def fit(self, d, seed):
        self.table = _FrameTable().fit(d)
        if self.arch == "excel" and self.table.cat_cols:
            raw = self.table.frame(d, impute=True)
            self.levels_flat = sorted(set(np.ravel(
                raw[self.table.cat_cols].to_numpy()).tolist()) | {"NA"})
            oh = pd.get_dummies(raw[self.table.cat_cols].astype(
                pd.CategoricalDtype(self.levels_flat)), dtype=float)
            self.onehot_cols = [c for c in oh.columns if oh[c].any()]
            self.numeric = self.table.numeric + self.onehot_cols
            self.cat = []
        else:
            self.numeric, self.cat = self.table.numeric, self.table.cat_cols
        self.col_stats = None
        train_ds = _materialize(self._df(d), d.y, self.numeric, self.cat)
        self.col_stats = {k: v for k, v in train_ds.col_stats.items()
                          if k != "__y__"}
        tf = train_ds.tensor_frame
        tr, va = _stratified_holdout(d.y, 0.2, seed)
        tf_tr, tf_va = tf[torch.tensor(tr)], tf[torch.tensor(va)]

        def score(p):
            try:
                m = self._fit_once(p, tf_tr, d.y[tr], seed, short=True)
                return _auroc(d.y[va], self._prob(m, tf_va))
            except (RuntimeError, ValueError):  # e.g. an invalid head split
                return 0.0

        self.params = _tune(self._space, score, self.n_trials, seed)
        self.model = self._fit_once(self.params, tf, d.y, seed, short=False)
        return self

    def predict(self, d):
        return self._prob(self.model, self._tf(d))


def register(n_trials: int | None = None) -> None:
    """Add the tabular methods to ``typed_bench.METHODS`` (idempotent)."""
    for kind in ("lgbm", "xgb", "cat"):
        METHODS[kind] = (lambda k=kind: GBDTLibMethod(k, n_trials))
    for arch in FrameModelMethod.LABELS:
        METHODS[f"pf_{arch}"] = (lambda a=arch: FrameModelMethod(a, n_trials))


TABULAR_METHODS = ["lgbm", "xgb", "cat"] + [
    f"pf_{a}" for a in FrameModelMethod.LABELS]

register()

__all__ = ["GBDTLibMethod", "FrameModelMethod", "TABULAR_METHODS",
           "register"]
