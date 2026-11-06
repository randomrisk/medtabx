"""Controlled comparison harness for typed multimodal prediction.

Every method sees the same data, the same folds and the same test-time
missing-modality scenarios; only the fusion strategy differs. Use it to
test whether a fusion method adds value over simpler baselines.

Data are held in :class:`TypedData`: standardized-at-fit-time numerical
columns (``NaN`` = missing), categorical codes (``-1`` = missing), frozen
image tokens ``[N, M, K, D]`` with a presence mask, and a binary label.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from medtabx.metrics.functional import (
    binary_auroc,
    binary_average_precision,
    brier_score,
)
from medtabx.models.typed_token import (
    TypedTokenConfig,
    TypedTokenSchema,
    TypedTokenTransformer,
)

__all__ = ["TypedData", "METHODS", "run_cv", "run_group_shift",
           "evaluate_scenarios"]


@dataclass
class TypedData:
    y: np.ndarray
    num: np.ndarray | None = None          # [N, Cn] float, NaN missing
    num_cols: list[str] = field(default_factory=list)
    num_modality: list[str] = field(default_factory=list)
    cat: np.ndarray | None = None          # [N, Cc] int, -1 missing
    cat_cols: list[str] = field(default_factory=list)
    cat_modality: list[str] = field(default_factory=list)
    cat_cardinality: list[int] = field(default_factory=list)
    img: np.ndarray | None = None          # [N, M, K, D]
    img_present: np.ndarray | None = None  # [N, M] bool
    img_modality: list[str] = field(default_factory=list)
    groups: np.ndarray | None = None       # e.g. center, for shift splits

    def __len__(self) -> int:
        return len(self.y)

    @property
    def modalities(self) -> list[str]:
        return self.schema().modalities

    def schema(self) -> TypedTokenSchema:
        return TypedTokenSchema(
            num_modality=list(self.num_modality),
            cat_modality=list(self.cat_modality),
            cat_cardinality=list(self.cat_cardinality),
            img_modality=list(self.img_modality),
            img_tokens=1 if self.img is None else self.img.shape[2],
            img_dim=320 if self.img is None else self.img.shape[3])

    def subset(self, idx: np.ndarray) -> "TypedData":
        out = copy.copy(self)
        out.y = self.y[idx]
        for name in ("num", "cat", "img", "img_present", "groups"):
            value = getattr(self, name)
            if value is not None:
                setattr(out, name, value[idx])
        return out

    def with_y(self, y: np.ndarray) -> "TypedData":
        """Same inputs, a different label. Used by the ordinal harness, which
        asks one cumulative question per threshold of the same cohort."""
        out = copy.copy(self)
        out.y = np.asarray(y)
        return out

    def select(self, modalities: list[str]) -> "TypedData":
        """Copy restricted to the columns of the given modalities, for
        training on a subset of sources (source ablation)."""
        out = copy.copy(self)
        keep = set(modalities)
        nj = [j for j, m in enumerate(self.num_modality) if m in keep]
        cj = [j for j, m in enumerate(self.cat_modality) if m in keep]
        ij = [j for j, m in enumerate(self.img_modality) if m in keep]
        out.num = self.num[:, nj] if self.num is not None and nj else None
        out.num_cols = [self.num_cols[j] for j in nj] if self.num_cols else []
        out.num_modality = [self.num_modality[j] for j in nj]
        out.cat = self.cat[:, cj] if self.cat is not None and cj else None
        out.cat_cols = [self.cat_cols[j] for j in cj] if self.cat_cols else []
        out.cat_modality = [self.cat_modality[j] for j in cj]
        out.cat_cardinality = [self.cat_cardinality[j] for j in cj]
        if self.img is not None and ij:
            out.img = self.img[:, ij]
            out.img_present = self.img_present[:, ij]
            out.img_modality = [self.img_modality[j] for j in ij]
        else:
            out.img, out.img_present, out.img_modality = None, None, []
        return out

    def drop(self, modalities: list[str]) -> "TypedData":
        """Copy with the given modalities made missing for every subject."""
        out = copy.copy(self)
        if self.num is not None:
            out.num = self.num.copy()
            for j, m in enumerate(self.num_modality):
                if m in modalities:
                    out.num[:, j] = np.nan
        if self.cat is not None:
            out.cat = self.cat.copy()
            for j, m in enumerate(self.cat_modality):
                if m in modalities:
                    out.cat[:, j] = -1
        if self.img is not None:
            out.img_present = self.img_present.copy()
            for j, m in enumerate(self.img_modality):
                if m in modalities:
                    out.img_present[:, j] = False
        return out


# Feature flattening for classical baselines ###############################


class _Flattener:
    """Concat features: numeric, one-hot categorical, one global image
    vector per modality. Missing values are either left as NaN (for GBDT) or
    mean-imputed with a missing indicator per column / modality."""

    def __init__(self, impute: bool = True) -> None:
        self.impute = impute

    def _raw(self, d: TypedData):
        blocks, groups = [], []
        if d.num is not None and d.num.shape[1]:
            blocks.append(d.num.astype(float))
            groups += list(d.num_modality)
        if d.cat is not None and d.cat.shape[1]:
            for j, card in enumerate(d.cat_cardinality):
                oh = np.zeros((len(d), card))
                ok = d.cat[:, j] >= 0
                oh[ok, d.cat[ok, j]] = 1
                oh[~ok] = np.nan
                blocks.append(oh)
                groups += [d.cat_modality[j]] * card
        if d.img is not None:
            g = d.img.astype(np.float32).mean(axis=2)  # [N, M, D]
            for j, m in enumerate(d.img_modality):
                v = g[:, j].astype(float)
                v[~d.img_present[:, j]] = np.nan
                blocks.append(v)
                groups += [m] * v.shape[1]
        return np.concatenate(blocks, axis=1), np.array(groups)

    def fit(self, d: TypedData):
        x, self.groups = self._raw(d)
        self.mean = np.nanmean(x, axis=0)
        self.mean = np.where(np.isnan(self.mean), 0.0, self.mean)
        self.std = np.nanstd(x, axis=0)
        self.std = np.where((self.std == 0) | np.isnan(self.std), 1.0,
                            self.std)
        self.mods = list(dict.fromkeys(self.groups))
        return self

    def transform(self, d: TypedData) -> np.ndarray:
        x, _ = self._raw(d)
        x = (x - self.mean) / self.std
        if not self.impute:
            return x
        indicators = np.stack(
            [np.isnan(x[:, self.groups == m]).all(axis=1) for m in self.mods],
            axis=1).astype(float)
        x = np.nan_to_num(x, nan=0.0)  # mean after standardization
        return np.concatenate([x, indicators], axis=1)


def _fit_lr(x, y, seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.05, max_iter=5000,
                              random_state=seed).fit(x, y)


class LogisticBaseline:
    name = "Concat + LR"

    def fit(self, d, seed):
        self.f = _Flattener().fit(d)
        self.m = _fit_lr(self.f.transform(d), d.y, seed)
        return self

    def predict(self, d):
        return self.m.predict_proba(self.f.transform(d))[:, 1]


class GBDTBaseline:
    name = "Concat + GBDT"

    def fit(self, d, seed):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.f = _Flattener(impute=False).fit(d)
        self.m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=seed).fit(
                self.f.transform(d), d.y)
        return self

    def predict(self, d):
        return self.m.predict_proba(self.f.transform(d))[:, 1]


class LateFusionBaseline:
    """One LR per modality; average the probabilities that are available."""
    name = "Late fusion (LR)"

    def fit(self, d, seed):
        self.f = _Flattener(impute=False).fit(d)
        x = self.f.transform(d)
        self.models = {}
        for m in self.f.mods:
            cols = self.f.groups == m
            xm = x[:, cols]
            ok = ~np.isnan(xm).all(axis=1)
            if ok.sum() > 5 and len(np.unique(d.y[ok])) == 2:
                self.models[m] = _fit_lr(np.nan_to_num(xm[ok]), d.y[ok], seed)
        return self

    def predict(self, d):
        x = self.f.transform(d)
        probs = np.full((len(d), len(self.models)), np.nan)
        for i, (m, model) in enumerate(self.models.items()):
            xm = x[:, self.f.groups == m]
            ok = ~np.isnan(xm).all(axis=1)
            if ok.any():
                probs[ok, i] = model.predict_proba(np.nan_to_num(xm[ok]))[:, 1]
        out = np.nanmean(probs, axis=1)
        return np.where(np.isnan(out), 0.5, out)


# Torch models ##############################################################


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchBase:
    """Shared training loop: AdamW, early stopping on an inner stratified
    validation split, restore the best epoch. Identical for every neural
    method so none of them gets a tuning advantage."""

    max_epochs = 200
    patience = 20
    val_fraction = 0.15
    lr = 1e-3
    weight_decay = 1e-2
    batch_size = 32
    #: Record per-epoch validation metrics (and, for typed models, the
    #: per-modality branch AUROCs and their JS divergence) in ``history``.
    track_epochs = False

    def _align_alpha(self, model, epoch: int) -> float | None:
        """Alignment weight for this epoch (None = no alignment)."""
        cfg = getattr(model, "config", None)
        w = getattr(cfg, "align_weight", 0.0) if cfg is not None else 0.0
        if w <= 0:
            return None
        sched = getattr(cfg, "align_schedule", "const")
        if sched == "const":
            return w
        if sched == "decay":        # Yang et al. Eq. 13: align first, fuse later
            return w * (1.0 - math.exp(-1.0 / epoch))
        if sched == "stepwise":
            return w if epoch <= self.max_epochs // 2 else 0.0
        raise ValueError(f"unknown align_schedule {sched!r}")

    def _unlabeled_loss(self, model, dev, generator):
        """Label-free loss on unlabeled rows (alignment + optional entropy);
        None unless a semi-supervised set was given."""
        return None

    def _epoch_metrics(self, model, batches, val_idx, dev) -> dict:
        return {}

    def _train(self, model, batches, seed):
        dev = _device()
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        g = torch.Generator().manual_seed(seed)
        y = batches["y"]
        rng = np.random.default_rng(seed)
        val_idx = []
        for c in (0, 1):
            cls = np.nonzero(y.numpy() == c)[0]
            k = max(1, int(round(len(cls) * self.val_fraction)))
            val_idx.extend(rng.choice(cls, k, replace=False).tolist())
        val_idx = torch.tensor(sorted(val_idx))
        mask = torch.ones(len(y), dtype=torch.bool)
        mask[val_idx] = False
        train_idx = torch.nonzero(mask).squeeze(1)

        best, best_state, bad = float("inf"), None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            alpha = self._align_alpha(model, epoch)
            model.train()
            perm = train_idx[torch.randperm(len(train_idx), generator=g)]
            for s in range(0, len(perm), self.batch_size):
                idx = perm[s:s + self.batch_size]
                if len(idx) < 2:
                    continue
                out = self._forward(model, batches, idx, dev)
                loss = loss_fn(out.squeeze(1), y[idx].to(dev))
                aux = (model.auxiliary_loss(y[idx].to(dev))
                       if hasattr(model, "auxiliary_loss") else None)
                if aux is not None:
                    loss = loss + model.config.aux_weight * aux
                align = (model.alignment_loss()
                         if hasattr(model, "alignment_loss") else None)
                extra = self._unlabeled_loss(model, dev, g)
                if extra is not None:
                    align = extra if align is None else align + extra
                if align is not None and alpha:
                    if model.config.align_integration == "convex":
                        a = min(float(alpha), 1.0)
                        loss = (1 - a) * loss + a * align
                    else:
                        loss = loss + alpha * align
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out = self._forward(model, batches, val_idx, dev)
                val = loss_fn(out.squeeze(1), y[val_idx].to(dev)).item()
            if self.track_epochs:
                self.history.append(
                    {"epoch": epoch, "val_loss": val, "alpha": alpha or 0.0,
                     **self._epoch_metrics(model, batches, val_idx, dev)})
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model


class MLPBaseline(_TorchBase):
    name = "Concat + MLP"

    def fit(self, d, seed):
        torch.manual_seed(seed)
        self.f = _Flattener().fit(d)
        x = torch.tensor(self.f.transform(d), dtype=torch.float32)
        self.model = nn.Sequential(
            nn.Linear(x.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        self._train(self.model, {"x": x, "y": torch.tensor(d.y, dtype=torch.float32)}, seed)
        return self

    def _forward(self, model, b, idx, dev):
        return model(b["x"][idx].to(dev))

    @torch.no_grad()
    def predict(self, d):
        x = torch.tensor(self.f.transform(d), dtype=torch.float32).to(_device())
        return torch.sigmoid(self.model(x)).squeeze(1).cpu().numpy()


class TypedTransformerMethod(_TorchBase):
    """TypedTokenTransformer with a given embedding configuration.

    Image-token handling is shared by every transformer variant, so ablations
    differ only in the embedding switches:

    - ``img_pool``  ``"global"`` averages the K spatial tokens of a modality
      into one (the linear-probing representation); ``"tokens"`` keeps all K.
    - ``img_pca``   project each modality's tokens onto this many principal
      components fitted on the training fold (whitened); ``None`` keeps the
      full encoder width. A small-cohort regularizer.
    """

    #: Defaults for small cohorts; see the class docstring.
    img_pool = "tokens"
    img_pca: int | None = None
    #: Optional unlabeled patients (same schema). The alignment loss needs no
    #: labels, so they can still constrain the model (Yang et al., Eq. 17-18).
    unlabeled: "TypedData | None" = None

    def __init__(self, name: str, img_pool: str | None = None,
                 img_pca: int | None | str = "default", **config) -> None:
        self.name = name
        # Wide & deep by default for every transformer variant, so the
        # ablations differ only in their embedding switches.
        config.setdefault("wide", True)
        self.config = config
        if img_pool is not None:
            self.img_pool = img_pool
        if img_pca != "default":
            self.img_pca = img_pca

    def _img(self, d):
        x = d.img.astype(np.float32)
        if self.img_pool == "global":
            x = x.mean(axis=2, keepdims=True)
        return x

    def _tensors(self, d):
        num = None
        if d.num is not None and d.num.shape[1]:
            num = torch.tensor((d.num - self.mu) / self.sd, dtype=torch.float32)
        cat = (torch.tensor(d.cat, dtype=torch.long)
               if d.cat is not None and d.cat.shape[1] else None)
        img = present = None
        if d.img is not None:
            x = (self._img(d) - self.img_mu) / self.img_sd
            if self.pca is not None:
                x = np.einsum("nmkd,mdc->nmkc", x, self.pca)
            x[~d.img_present] = 0.0
            img = torch.tensor(x.astype(np.float32))
            present = torch.tensor(d.img_present)
        return {"num": num, "cat": cat, "img": img, "present": present,
                "y": torch.tensor(d.y, dtype=torch.float32)}

    def _forward(self, model, b, idx, dev):
        get = lambda k: None if b[k] is None else b[k][idx].to(dev)  # noqa
        return model(num=get("num"), cat=get("cat"), img=get("img"),
                     img_present=get("present"))

    def fit(self, d, seed):
        torch.manual_seed(seed)
        if d.num is not None and d.num.shape[1]:
            self.mu = np.nanmean(d.num, axis=0)
            self.mu = np.where(np.isnan(self.mu), 0, self.mu)
            self.sd = np.nanstd(d.num, axis=0)
            self.sd = np.where((self.sd == 0) | np.isnan(self.sd), 1, self.sd)
        self.pca = None
        schema = d.schema()
        if d.img is not None:
            # Per modality, per dimension, over the tokens that are present.
            x = self._img(d)
            m, dim = x.shape[1], x.shape[3]
            self.img_mu = np.zeros((1, m, 1, dim), np.float32)
            self.img_sd = np.ones((1, m, 1, dim), np.float32)
            for j in range(m):
                ok = d.img_present[:, j]
                if ok.any():
                    v = x[ok, j].reshape(-1, dim)
                    self.img_mu[0, j, 0] = v.mean(0)
                    self.img_sd[0, j, 0] = v.std(0) + 1e-6
            if self.img_pca:
                # Whitened PCA per modality, fitted on training tokens only.
                c = min(self.img_pca, dim)
                self.pca = np.zeros((m, dim, c), np.float32)
                z = (x - self.img_mu) / self.img_sd
                for j in range(m):
                    ok = d.img_present[:, j]
                    v = z[ok, j].reshape(-1, dim)
                    if len(v) > 1:
                        _, sv, vt = np.linalg.svd(v - v.mean(0),
                                                  full_matrices=False)
                        k = min(c, vt.shape[0])
                        scale = sv[:k] / np.sqrt(max(1, len(v) - 1)) + 1e-6
                        self.pca[j, :, :k] = (vt[:k] / scale[:, None]).T
                schema.img_dim = c
            schema.img_tokens = x.shape[2]
        self.model = TypedTokenTransformer(schema,
                                           TypedTokenConfig(**self.config))
        self._ssl = (self._tensors(self.unlabeled)
                     if self.unlabeled is not None else None)
        self._train(self.model, self._tensors(d), seed)
        return self

    def _unlabeled_loss(self, model, dev, generator):
        if self._ssl is None or model.config.align_weight <= 0:
            return None
        n = len(self._ssl["y"])
        k = min(self.batch_size, n)
        idx = torch.randperm(n, generator=generator)[:k]
        out = self._forward(model, self._ssl, idx, dev)
        loss = model.alignment_loss()
        if model.config.ssl_entropy > 0:
            p = torch.sigmoid(out.squeeze(1)).clamp(1e-6, 1 - 1e-6)
            ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
            loss = ent * model.config.ssl_entropy if loss is None else \
                loss + model.config.ssl_entropy * ent
        return loss

    def _only_modality(self, batches, idx, mod, schema, dev):
        """The validation rows with every modality except ``mod`` made
        missing, in tensor space (NaN / -1 / absent)."""
        num = cat = img = pres = None
        observed = torch.zeros(len(idx), dtype=torch.bool)
        if batches["num"] is not None:
            num = batches["num"][idx].clone()
            keep = torch.tensor([m == mod for m in schema.num_modality])
            num[:, ~keep] = float("nan")
            if keep.any():
                observed |= ~torch.isnan(num[:, keep]).all(dim=1)
            num = num.to(dev)
        if batches["cat"] is not None:
            cat = batches["cat"][idx].clone()
            keep = torch.tensor([m == mod for m in schema.cat_modality])
            cat[:, ~keep] = -1
            if keep.any():
                observed |= (cat[:, keep] >= 0).any(dim=1)
            cat = cat.to(dev)
        if batches["img"] is not None:
            img = batches["img"][idx].to(dev)
            pres = batches["present"][idx].clone()
            keep = torch.tensor([m == mod for m in schema.img_modality])
            pres[:, ~keep] = False
            if keep.any():
                observed |= pres[:, keep].any(dim=1)
            pres = pres.to(dev)
        return {"num": num, "cat": cat, "img": img, "present": pres}, observed

    @torch.no_grad()
    def _epoch_metrics(self, model, batches, val_idx, dev) -> dict:
        """Per-modality branch AUROC and their JS divergence on the inner
        validation rows -- the training curves of Yang et al., Fig. 1.

        A branch is what the fused model predicts when only that modality is
        present. They split the fused linear classifier into per-modality
        weights; masking is the equivalent that works for any head, and it
        needs no expert modules.
        """
        from medtabx.analysis.alignment import branch_report
        schema = model.schema
        fused = torch.sigmoid(
            self._forward(model, batches, val_idx, dev).squeeze(1)) \
            .cpu().numpy()
        probs, present = {}, {}
        for m in schema.modalities:
            sub, observed = self._only_modality(batches, val_idx, m, schema,
                                                dev)
            out = model(num=sub["num"], cat=sub["cat"], img=sub["img"],
                        img_present=sub["present"])
            p = torch.sigmoid(out.squeeze(1)).cpu().numpy()
            probs[m] = np.where(observed.numpy(), p, np.nan)
            present[m] = observed.numpy()
        return branch_report(probs, present,
                             batches["y"][val_idx].numpy(),
                             fused=fused).as_dict()


    @torch.no_grad()
    def predict(self, d):
        b = self._tensors(d)
        idx = torch.arange(len(d))
        out = self._forward(self.model, b, idx, _device())
        return torch.sigmoid(out).squeeze(1).cpu().numpy()


class TypedStackingMethod(TypedTransformerMethod):
    """Typed stacking: late-fusion experts + a typed, missingness-aware gate.

    1. One L2 logistic regression per modality (exactly the late-fusion
       experts). Their training-set logits are **cross-fitted** (5-fold
       out-of-fold) so the gate never learns from in-sample predictions.
    2. A typed token transformer reads the patient's full typed input and
       outputs a softmax gate over the modalities that are present; the
       prediction is the gated sum of expert logits.

    With a uniform gate this is late fusion, so any difference is the value
    of learning *which modality to trust for this patient*.
    """

    def __init__(self, name: str, n_inner: int = 5, **config) -> None:
        config.setdefault("head", "stacked")
        config.setdefault("wide", False)
        super().__init__(name, **config)
        self.n_inner = n_inner

    def _expert_features(self, d):
        x = self.flat.transform(d)
        return {m: x[:, self.flat.groups == m] for m in self.modalities}

    def _expert_logits(self, models, feats, n):
        out = np.full((n, len(self.modalities)), np.nan, np.float32)
        for j, m in enumerate(self.modalities):
            if m not in models:
                continue
            xm = feats[m]
            ok = ~np.isnan(xm).all(axis=1)
            if ok.any():
                out[ok, j] = models[m].decision_function(np.nan_to_num(xm[ok]))
        return out

    def _fit_experts(self, d, idx, seed):
        feats = self._expert_features(d)
        models = {}
        for m in self.modalities:
            xm = feats[m][idx]
            ok = ~np.isnan(xm).all(axis=1)
            y = d.y[idx][ok]
            if ok.sum() > 5 and len(np.unique(y)) == 2:
                models[m] = _fit_lr(np.nan_to_num(xm[ok]), y, seed)
        return models

    def fit(self, d, seed):
        from sklearn.model_selection import StratifiedKFold

        self.flat = _Flattener(impute=False).fit(d)
        self.modalities = d.schema().modalities
        feats = self._expert_features(d)
        oof = np.full((len(d), len(self.modalities)), np.nan, np.float32)
        skf = StratifiedKFold(self.n_inner, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros(len(d)), d.y):
            models = self._fit_experts(d, tr, seed)
            sub = {m: v[te] for m, v in feats.items()}
            oof[te] = self._expert_logits(models, sub, len(te))
        self.experts = self._fit_experts(d, np.arange(len(d)), seed)
        self._train_experts = oof
        return super().fit(d, seed)

    def _tensors(self, d):
        b = super()._tensors(d)
        if getattr(self, "_train_experts", None) is not None and \
                len(d) == len(self._train_experts):
            ex = self._train_experts
        else:
            ex = self._expert_logits(self.experts, self._expert_features(d),
                                     len(d))
        b["experts"] = torch.tensor(ex)
        return b

    def _forward(self, model, b, idx, dev):
        get = lambda k: None if b[k] is None else b[k][idx].to(dev)  # noqa
        return model(num=get("num"), cat=get("cat"), img=get("img"),
                     img_present=get("present"),
                     expert_logits=get("experts"))

    @torch.no_grad()
    def predict(self, d):
        self._train_experts = None  # test time: use the full-data experts
        return super().predict(d)


_NONE = dict(use_type=False, use_modality=False, use_column=False,
             use_missing=False)

#: name -> factory. Order is the table order.
METHODS: dict[str, Callable[[], object]] = {
    "lr": LogisticBaseline,
    "gbdt": GBDTBaseline,
    "late": LateFusionBaseline,
    "mlp": MLPBaseline,
    "plain_tf": lambda: TypedTransformerMethod(
        "Plain transformer", **_NONE),
    "plain_tf_md": lambda: TypedTransformerMethod(
        "Plain transformer + mod. dropout", **_NONE, modality_dropout=0.3),
    "ft_tf": lambda: TypedTransformerMethod(
        "FT-Transformer (column id only)",
        **{**_NONE, "use_column": True}),
    "medtabx": lambda: TypedTransformerMethod(
        "MedTabX (typed + missing)", modality_dropout=0.3),
    "medtabx_nodrop": lambda: TypedTransformerMethod(
        "MedTabX, no mod. dropout"),
    "abl_no_type": lambda: TypedTransformerMethod(
        "MedTabX - Type", use_type=False, modality_dropout=0.3),
    "abl_no_modality": lambda: TypedTransformerMethod(
        "MedTabX - Modality", use_modality=False, modality_dropout=0.3),
    "abl_no_column": lambda: TypedTransformerMethod(
        "MedTabX - Column", use_column=False, modality_dropout=0.3),
    "abl_no_missing": lambda: TypedTransformerMethod(
        "MedTabX - Missing (masked instead)", use_missing=False,
        modality_dropout=0.3),
    # Typed stacking. The default variant uses no modality dropout.
    "stack_typed": lambda: TypedStackingMethod("MedTabX typed stacking"),
    "stack_plain": lambda: TypedStackingMethod(
        "Stacking, plain gate (no identity emb.)", **_NONE),
    "stack_typed_md": lambda: TypedStackingMethod(
        "MedTabX typed stacking + mod. dropout", modality_dropout=0.3),
    # Modality alignment: same model, plus an explicit penalty on
    # disagreement between the per-modality experts of the same patient.
    "medtabx_align_js": lambda: TypedTransformerMethod(
        "MedTabX + JS alignment", modality_dropout=0.3, align_weight=0.5,
        align="js"),
    "medtabx_align_rep": lambda: TypedTransformerMethod(
        "MedTabX + representation alignment", modality_dropout=0.3,
        align_weight=0.5, align="rep"),
    "medtabx_align_hybrid": lambda: TypedTransformerMethod(
        "MedTabX + hybrid alignment", modality_dropout=0.3,
        align_weight=0.5, align="hybrid"),
    "medtabx_aux": lambda: TypedTransformerMethod(
        "MedTabX + per-modality aux loss (hard alignment)",
        modality_dropout=0.3, head="gated_experts"),
}


def _metrics(y, p) -> dict[str, float]:
    y_t, p_t = torch.tensor(y), torch.tensor(p, dtype=torch.float64)
    return {"auroc": binary_auroc(p_t, y_t),
            "auprc": binary_average_precision(p_t, y_t),
            "brier": brier_score(p_t, y_t)}


def evaluate_scenarios(model, test: TypedData,
                       scenarios: dict[str, list[str]]) -> list[dict]:
    rows = []
    for scen, dropped in scenarios.items():
        p = model.predict(test.drop(dropped) if dropped else test)
        rows.append({"scenario": scen, **_metrics(test.y, p), "_pred": p})
    return rows


def run_cv(data: TypedData, methods: list[str],
           scenarios: dict[str, list[str]], n_splits: int = 5,
           repeats: int = 3, train_fraction: float = 1.0,
           external: TypedData | None = None,
           external_scenarios: dict[str, list[str]] | None = None,
           seed: int = 0, log: Callable[[str], None] = print) -> list[dict]:
    """Repeated stratified CV. Returns one row per (repeat, fold, method,
    scenario). With ``external``, each fold's model is also scored there."""
    from sklearn.model_selection import StratifiedKFold

    rows = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(data)), data.y)):
            if train_fraction < 1.0:
                rng = np.random.default_rng(seed + 100 * r + f)
                keep = []
                for c in np.unique(data.y[tr]):
                    cls = tr[data.y[tr] == c]
                    k = max(2, int(round(len(cls) * train_fraction)))
                    keep.append(rng.choice(cls, k, replace=False))
                tr = np.sort(np.concatenate(keep))
            train, test = data.subset(tr), data.subset(te)
            for name in methods:
                t0 = time.time()
                model = METHODS[name]().fit(train, seed=seed + 10 * r + f)
                fit_s = time.time() - t0
                for row in evaluate_scenarios(model, test, scenarios):
                    rows.append({"repeat": r, "fold": f, "method": name,
                                 "label": model.name, "split": "cv",
                                 "train_n": len(tr), "fit_seconds": fit_s,
                                 **row})
                if external is not None:
                    for row in evaluate_scenarios(
                            model, external, external_scenarios or {"full": []}):
                        rows.append({"repeat": r, "fold": f, "method": name,
                                     "label": model.name, "split": "external",
                                     "train_n": len(tr), "fit_seconds": fit_s,
                                     **row})
                log(f"  r{r} f{f} {name:16s} {time.time() - t0:5.1f}s")
    return rows


def run_group_shift(data: TypedData, methods: list[str],
                    scenarios: dict[str, list[str]], train_group,
                    test_group, seeds: int = 5,
                    log: Callable[[str], None] = print) -> list[dict]:
    """Train on one group (e.g. center) and test on another. Repeats over
    model seeds, since the split itself is fixed."""
    if data.groups is None:
        raise ValueError("TypedData.groups is required for a group shift.")
    tr = np.nonzero(data.groups == train_group)[0]
    te = np.nonzero(data.groups == test_group)[0]
    train, test = data.subset(tr), data.subset(te)
    rows = []
    for s in range(seeds):
        for name in methods:
            t0 = time.time()
            model = METHODS[name]().fit(train, seed=1000 + s)
            fit_s = time.time() - t0
            for row in evaluate_scenarios(model, test, scenarios):
                rows.append({"repeat": s, "fold": 0, "method": name,
                             "label": model.name,
                             "split": f"{train_group}->{test_group}",
                             "train_n": len(tr), "fit_seconds": fit_s,
                             **row})
            log(f"  seed{s} {name:16s} {time.time() - t0:5.1f}s")
    return rows


# Typed modality-subset ensembles ##########################################


class _ModalityBlocks:
    """Per-modality feature blocks built from the typed schema.

    Numeric and categorical columns stay as they are (standardized, NaN kept);
    each image modality's global token is reduced by a PCA fitted on the
    training rows where that modality is present. A row whose modality is
    absent gets an all-NaN block, which is how absence is detected later.
    """

    def __init__(self, img_components: int = 32) -> None:
        self.k = img_components

    def fit(self, d: TypedData):
        from sklearn.decomposition import PCA

        self.flat = _Flattener(impute=False).fit(d)
        x = self.flat.transform(d)
        self.mods = list(self.flat.mods)
        self.img_mods = set(d.img_modality)
        self.pca = {}
        for m in self.mods:
            block = x[:, self.flat.groups == m]
            if m in self.img_mods and block.shape[1] > self.k:
                ok = ~np.isnan(block).all(axis=1)
                k = min(self.k, int(ok.sum()) - 1)
                self.pca[m] = PCA(k, random_state=0).fit(
                    np.nan_to_num(block[ok]))
        return self

    def transform(self, d: TypedData) -> dict[str, np.ndarray]:
        x = self.flat.transform(d)
        out = {}
        for m in self.mods:
            block = x[:, self.flat.groups == m]
            absent = np.isnan(block).all(axis=1)
            if m in self.pca:
                z = self.pca[m].transform(np.nan_to_num(block))
                z[absent] = np.nan
                block = z
            out[m] = block
        return out


class ModalitySubsetEnsemble:
    """Typed modality-subset ensemble.

    Members are trained on subsets of the schema's modalities: every single
    modality (late fusion), the full set (concat), and random subsets in
    between. A test subject is scored only by members whose modalities are
    all present for that subject, so absent inputs are never imputed. With
    only the singletons this is late fusion; with only the full set it is
    concatenation.

    Args:
        base: ``"lr"``, ``"tabpfn"``, or a factory ``seed -> estimator``
            with scikit-learn ``fit`` / ``predict_proba``.
        n_members: total members (singletons + full set + random subsets).
        impute: fill NaN with 0 (the training mean after standardization)
            before the base learner; default only for ``"lr"``. Set it for
            any base that cannot take NaN.
        indicators: with ``impute``, append one absent-flag per modality of
            a multi-modality member so the base still sees missingness.
    """

    def __init__(self, name: str, base: str | Callable[[int], object] = "tabpfn",
                 n_members: int = 32, img_components: int = 32,
                 tabpfn_ckpt: str | None = None,
                 singletons_only: bool = False,
                 impute: bool | None = None,
                 indicators: bool = False) -> None:
        self.name = name
        self.singletons_only = singletons_only
        self.base = base
        self.n_members = n_members
        self.img_components = img_components
        self.tabpfn_ckpt = tabpfn_ckpt
        self.impute = (base == "lr") if impute is None else impute
        self.indicators = indicators

    def _learner(self, seed):
        if callable(self.base):
            return self.base(seed)
        if self.base == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(C=0.05, max_iter=5000,
                                      random_state=seed)
        from tabpfn import TabPFNClassifier
        kwargs = {"device": str(_device()), "random_state": seed}
        if self.tabpfn_ckpt:
            kwargs["model_path"] = self.tabpfn_ckpt
        return TabPFNClassifier(**kwargs)

    def _design(self, blocks, subset, rows=None):
        x = np.concatenate([blocks[m] for m in subset], axis=1)
        if self.impute and self.indicators and len(subset) > 1:
            absent = [np.isnan(blocks[m]).all(axis=1, keepdims=True)
                      for m in subset]
            x = np.concatenate([x] + [a.astype(float) for a in absent],
                               axis=1)
        if rows is not None:
            x = x[rows]
        if self.impute:
            x = np.nan_to_num(x)  # standardized: 0 is the training mean
        return x

    def fit(self, d, seed):
        self.blocks = _ModalityBlocks(self.img_components).fit(d)
        b = self.blocks.transform(d)
        mods = self.blocks.mods
        rng = np.random.default_rng(seed)
        subsets = [(m, ) for m in mods]
        if not self.singletons_only:
            subsets.append(tuple(mods))
        seen = set(subsets)
        tries = 0
        while (not self.singletons_only and len(subsets) < self.n_members
               and tries < 1000):
            tries += 1
            size = int(rng.integers(2, max(3, len(mods))))
            s = tuple(sorted(rng.choice(mods, min(size, len(mods)),
                                        replace=False).tolist(),
                             key=mods.index))
            if s not in seen:
                seen.add(s)
                subsets.append(s)
        self.members = []
        for i, s in enumerate(subsets):
            # Train each member on the rows that have all of its modalities.
            rows = np.ones(len(d), bool)
            for m in s:
                rows &= ~np.isnan(b[m]).all(axis=1)
            y = d.y[rows]
            if rows.sum() < 10 or len(np.unique(y)) < 2:
                continue
            model = self._learner(seed + i).fit(self._design(b, s, rows), y)
            self.members.append((s, model))
        return self

    def predict(self, d):
        b = self.blocks.transform(d)
        present = {m: ~np.isnan(v).all(axis=1) for m, v in b.items()}
        total = np.zeros(len(d))
        count = np.zeros(len(d))
        for s, model in self.members:
            ok = np.ones(len(d), bool)
            for m in s:
                ok &= present[m]
            if ok.any():
                total[ok] += model.predict_proba(self._design(b, s, ok))[:, 1]
                count[ok] += 1
        return np.where(count > 0, total / np.maximum(count, 1), 0.5)


class TabPFNConcat(ModalitySubsetEnsemble):
    """Plain TabPFN on the concatenated features (NaN-aware), no typing."""

    def fit(self, d, seed):
        self.blocks = _ModalityBlocks(self.img_components).fit(d)
        b = self.blocks.transform(d)
        self.members = [(tuple(self.blocks.mods), self._learner(seed).fit(
            self._design(b, self.blocks.mods), d.y))]
        return self

    def predict(self, d):
        # Concat has no way to drop a member: absent modalities stay NaN.
        b = self.blocks.transform(d)
        s, model = self.members[0]
        return model.predict_proba(self._design(b, s))[:, 1]


def _tabpfn_ckpt():
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2] / "data/tabpfn/"
         "tabpfn-v3-classifier-v3_default.ckpt")
    return str(p) if p.exists() else None


METHODS.update({
    "subset_lr": lambda: ModalitySubsetEnsemble(
        "Typed subset ensemble (LR)", base="lr"),
    "pfn_concat": lambda: TabPFNConcat(
        "TabPFN concat", base="tabpfn", tabpfn_ckpt=_tabpfn_ckpt()),
    "pfn_late": lambda: ModalitySubsetEnsemble(
        "TabPFN late fusion", base="tabpfn", singletons_only=True,
        tabpfn_ckpt=_tabpfn_ckpt()),
    "subset_pfn": lambda: ModalitySubsetEnsemble(
        "MedTabX-PFN typed subset ensemble", base="tabpfn",
        tabpfn_ckpt=_tabpfn_ckpt()),
})
