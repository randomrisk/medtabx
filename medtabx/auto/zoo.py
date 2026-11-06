"""Model zoo: any downstream learner on the same typed contract.

A model id is ``"<base>"`` or ``"<base>@<fusion>"``:

* **base** -- a learner. Flat learners (``lr``, ``rf``, ``et``, ``hgb``,
  ``lgbm``, ``xgb``, ``cat``, ``tabpfn``, ``tabicl``, or anything registered
  with :func:`register_base`) take the flat view; deep tabular models
  (``pf_mlp`` ... ``pf_excel``, the PyTorch Frame family) and the MedTabX
  typed token transformer (``medtabx``) take all inputs at once.
* **fusion** -- how the schema's modalities are combined (flat learners
  only): ``concat`` (default; one model, missing flags appended), ``late``
  (one model per modality, average of the models whose modality is present),
  ``subset`` (members on random modality subsets, scored only when all
  their modalities are present; nothing is imputed at test time) or
  ``acq`` (one model per modality as in ``late``, but weighted per patient by
  acquisition condition in place of a uniform average; see
  :mod:`medtabx.auto.gating`).

Every model exposes ``fit(TypedData, seed)`` and ``predict(TypedData) ->
P(y=1)``, so all of them run through the same CV / scenario harness.

Failure-aware extensions, all opt-in and absent from
:func:`list_models`, so the default search space is unchanged:

* ``<id>[m1+m2]`` -- the model restricted to the named modalities.
* ``sel:<id>`` / ``selp:<id>`` -- the modalities are chosen inside the
  training data by forward selection (``selp`` adds the permutation filter),
  then ``<id>`` is fitted on them (:mod:`medtabx.auto.select`).
* fusions ``stack`` (stacked per-patient router) and ``ds`` (evidential
  Dempster-Shafer fusion) for any flat learner (:mod:`medtabx.fusion`).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

from medtabx.benchmarks.typed_bench import (
    METHODS,
    ModalitySubsetEnsemble,
    TabPFNConcat,
    TypedData,
    TypedTransformerMethod,
    _device,
    _tabpfn_ckpt,
)

FUSIONS = ("concat", "late", "subset", "acq")


@dataclass
class Base:
    """A downstream learner and what it needs."""

    name: str
    label: str
    family: str                      # linear | tree | gbdt | foundation | deep | typed_token | custom
    factory: Callable[[int], Any] | None = None   # flat learners: seed -> estimator
    builder: Callable[[], Any] | None = None      # whole-input learners
    handles_nan: bool = False
    requires: tuple[str, ...] = ()
    max_rows: int | None = None
    max_features: int | None = None
    gpu: bool = False
    cost: float = 1.0                # relative fit cost, for budgets
    notes: str = ""
    fusions: tuple[str, ...] = field(default=FUSIONS)

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(p) is not None
                   for p in self.requires)


BASES: dict[str, Base] = {}


def register_base(name: str, factory: Callable[[int], Any], *,
                  label: str | None = None, family: str = "custom",
                  handles_nan: bool = False, requires: tuple[str, ...] = (),
                  max_rows: int | None = None,
                  max_features: int | None = None, cost: float = 1.0,
                  gpu: bool = False) -> Base:
    """Plug in any scikit-learn style binary classifier.

    ``factory(seed)`` must return an object with ``fit(X, y)`` and
    ``predict_proba(X)``. It is then available with every fusion, e.g.
    ``"<name>"``, ``"<name>@late"``, ``"<name>@subset"``.
    """
    b = Base(name, label or name, family, factory=factory,
             handles_nan=handles_nan, requires=requires, max_rows=max_rows,
             max_features=max_features, cost=cost, gpu=gpu)
    BASES[name] = b
    return b


# Built-in learners ########################################################


def _lr(seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.05, max_iter=5000, random_state=seed)


def _rf(seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                  max_features="sqrt", n_jobs=4,
                                  random_state=seed)


def _et(seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                n_jobs=4, random_state=seed)


def _hgb(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
        random_state=seed)


def _lgbm(seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=3,
                          num_leaves=8, subsample=0.8, subsample_freq=1,
                          reg_lambda=1.0, min_child_samples=5,
                          random_state=seed, n_jobs=4, verbose=-1)


def _xgb(seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=3,
                         subsample=0.8, reg_lambda=1.0, tree_method="hist",
                         random_state=seed, n_jobs=4, verbosity=0)


def _cat(seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                              random_seed=seed, thread_count=4,
                              verbose=False, allow_writing_files=False)


def _tabpfn(seed):
    from tabpfn import TabPFNClassifier
    kw = {"device": str(_device()), "random_state": seed}
    ckpt = _tabpfn_ckpt()
    if ckpt:
        kw["model_path"] = ckpt
    return TabPFNClassifier(**kw)


def _tabicl(seed):
    from tabicl import TabICLClassifier
    return TabICLClassifier(random_state=seed, device=str(_device()))


for _b in [
    Base("lr", "Logistic regression", "linear", _lr, cost=0.1),
    Base("rf", "Random forest", "tree", _rf, handles_nan=True, cost=1),
    Base("et", "Extra trees", "tree", _et, cost=1),
    Base("hgb", "HistGradientBoosting", "gbdt", _hgb, handles_nan=True,
         cost=0.5),
    Base("lgbm", "LightGBM", "gbdt", _lgbm, handles_nan=True,
         requires=("lightgbm",), cost=0.5),
    Base("xgb", "XGBoost", "gbdt", _xgb, handles_nan=True,
         requires=("xgboost",), cost=1),
    Base("cat", "CatBoost", "gbdt", _cat, handles_nan=True,
         requires=("catboost",), cost=2),
    Base("tabpfn", "TabPFN", "foundation", _tabpfn, handles_nan=True,
         requires=("tabpfn",), max_rows=10_000, max_features=2_000,
         gpu=True, cost=3, notes="in-context; weights under data/tabpfn"),
    Base("tabicl", "TabICL", "foundation", _tabicl, handles_nan=True,
         requires=("tabicl",), max_rows=100_000, gpu=True, cost=3),
]:
    BASES[_b.name] = _b


def _frame_model(arch):
    def build():
        from medtabx.benchmarks.tabular_methods import FrameModelMethod
        return FrameModelMethod(arch, n_trials=0)
    return build


for _arch, _label in [("mlp", "MLP"), ("resnet", "ResNet"),
                      ("ft", "FT-Transformer"), ("tabnet", "TabNet"),
                      ("tabtf", "TabTransformer"), ("trompt", "Trompt"),
                      ("excel", "ExcelFormer")]:
    BASES[f"pf_{_arch}"] = Base(
        f"pf_{_arch}", f"{_label} (PyTorch Frame)", "deep",
        builder=_frame_model(_arch), gpu=True, cost=4, fusions=("concat",),
        notes="whole-input neural model on the typed columns")

BASES["medtabx"] = Base(
    "medtabx", "MedTabX typed token transformer", "typed_token",
    builder=lambda: TypedTransformerMethod(
        "MedTabX typed token transformer", modality_dropout=0.3),
    handles_nan=True, gpu=True, cost=2, fusions=("concat",),
    notes="consumes the token view H[B,N,C] directly")


# Model ids ################################################################


#: Prefix that trains a model on modality-dropout augmented data
#: (:class:`medtabx.analysis.robustness.DropoutAugmented`), so any learner
#: can be given the augmentation the typed model has internally.
AUG_PREFIX = "aug:"

#: Prefixes that choose the modalities inside the training data
#: (:class:`medtabx.auto.select.GreedyModalitySelector`); the value says
#: whether the permutation filter is on.
SELECT_PREFIXES = {"sel:": False, "selp:": True}

#: Fusions every flat learner accepts but that stay out of the default
#: search space (:mod:`medtabx.fusion`).
OPT_IN_FUSIONS = ("stack", "ds")


def _prefixes(model_id: str) -> tuple[list[str], str]:
    """Leading ``aug:`` / ``sel:`` / ``selp:`` prefixes and the rest."""
    found = []
    while True:
        for pre in (AUG_PREFIX, *SELECT_PREFIXES):
            if model_id.startswith(pre):
                found.append(pre)
                model_id = model_id[len(pre):]
                break
        else:
            return found, model_id


def _strip_prefixes(model_id: str) -> str:
    return _prefixes(model_id)[1]


def split_subset(model_id: str) -> tuple[str, list[str] | None]:
    """``"lr@late[a+b]"`` -> ``("lr@late", ["a", "b"])``."""
    if model_id.endswith("]") and "[" in model_id:
        core, _, rest = model_id.partition("[")
        mods = [m for m in rest[:-1].split("+") if m]
        if not mods:
            raise ValueError(f"empty modality subset in {model_id!r}")
        return core, mods
    return model_id, None


def parse(model_id: str) -> tuple[Base, str]:
    model_id, _ = split_subset(_strip_prefixes(model_id))
    base, _, fusion = model_id.partition("@")
    fusion = fusion or "concat"
    if base not in BASES:
        raise KeyError(f"unknown base {base!r}; known: {sorted(BASES)}")
    b = BASES[base]
    opt_in = b.factory is not None and fusion in OPT_IN_FUSIONS
    if fusion not in b.fusions and not opt_in:
        raise ValueError(f"{base} supports fusion {b.fusions}, not {fusion}")
    return b, fusion


def label(model_id: str) -> str:
    b, fusion = parse(model_id)
    name = b.label if fusion == "concat" else f"{b.label} [{fusion}]"
    prefixes, core = _prefixes(model_id)
    _, mods = split_subset(core)
    if mods:
        name = f"{name} {{{'+'.join(mods)}}}"
    for pre in prefixes:
        if pre in SELECT_PREFIXES:
            name += ", in-fold modality selection" + (
                " + permutation filter" if SELECT_PREFIXES[pre] else "")
    return f"{name} + mod. dropout" if AUG_PREFIX in prefixes else name


def create_model(model_id: str, n_members: int = 16,
                 img_components: int = 32, aug_p: float = 0.3,
                 n_aug: int = 2, gate_reliability: float = 1.0,
                 gate_quality: float = 1.0, gate_temperature: float = 1.0):
    """Instantiate a model id as a learner with ``fit(TypedData, seed)`` /
    ``predict(TypedData)``."""
    if model_id.startswith(AUG_PREFIX):
        from medtabx.analysis.robustness import DropoutAugmented
        inner = model_id[len(AUG_PREFIX):]
        return DropoutAugmented(
            lambda: create_model(inner, n_members=n_members,
                                 img_components=img_components,
                                 gate_reliability=gate_reliability,
                                 gate_quality=gate_quality,
                                 gate_temperature=gate_temperature),
            p=aug_p, n_aug=n_aug, name=label(model_id))
    kw_inner = dict(n_members=n_members, img_components=img_components,
                    gate_reliability=gate_reliability,
                    gate_quality=gate_quality,
                    gate_temperature=gate_temperature)
    for pre, perm in SELECT_PREFIXES.items():
        if model_id.startswith(pre):
            from medtabx.auto.select import GreedyModalitySelector
            return GreedyModalitySelector(
                label(model_id), final=model_id[len(pre):],
                permutation_filter=perm, img_components=img_components,
                create_kw=kw_inner)
    core, mods = split_subset(model_id)
    if mods:
        from medtabx.auto.select import ModalitySelected
        return ModalitySelected(create_model(core, **kw_inner), mods,
                                name=label(model_id))
    b, fusion = parse(model_id)
    if not b.available:
        raise ImportError(f"{model_id} needs {b.requires}")
    if fusion == "stack":
        from medtabx.fusion import StackedRouter
        return StackedRouter(label(model_id), base=b.factory,
                             impute=not b.handles_nan,
                             img_components=img_components)
    if fusion == "ds":
        from medtabx.fusion import EvidentialFusion
        return EvidentialFusion(label(model_id), base=b.factory,
                                impute=not b.handles_nan,
                                img_components=img_components)
    if b.builder is not None:
        m = b.builder()
        m.name = label(model_id)
        return m
    kw = dict(base=b.factory, impute=not b.handles_nan,
              img_components=img_components)
    if fusion == "concat":
        return TabPFNConcat(label(model_id), indicators=True, **kw)
    if fusion == "late":
        return ModalitySubsetEnsemble(label(model_id), singletons_only=True,
                                      **kw)
    if fusion == "acq":
        from medtabx.auto.gating import AcquisitionGatedFusion
        return AcquisitionGatedFusion(
            label(model_id), w_reliability=gate_reliability,
            w_quality=gate_quality, temperature=gate_temperature, **kw)
    return ModalitySubsetEnsemble(label(model_id), n_members=n_members,
                                  indicators=True, **kw)


def feasible(model_id: str, meta: dict | None = None) -> tuple[bool, str]:
    """Whether a model can run here, on this data. Returns (ok, reason)."""
    try:
        b, fusion = parse(model_id)
    except (KeyError, ValueError) as e:
        return False, str(e)
    if not b.available:
        return False, f"missing package {', '.join(b.requires)}"
    if meta:
        if fusion != "concat" and meta["n_modalities"] < 2:
            return False, "one modality: late/subset/acq fusion equals concat"
        if b.max_rows and meta["n"] > b.max_rows:
            return False, f"n={meta['n']} > {b.max_rows} rows supported"
        if b.max_features and meta["flat_features"] > b.max_features:
            return False, f"{meta['flat_features']} features > {b.max_features}"
    return True, ""


def list_models(data: TypedData | None = None, include_unavailable=False
                ) -> list[str]:
    """All model ids (base × supported fusion), feasible for ``data``."""
    from medtabx.auto.frame import describe
    meta = describe(data) if data is not None else None
    ids = []
    for b in BASES.values():
        for f in b.fusions:
            mid = b.name if f == "concat" else f"{b.name}@{f}"
            if include_unavailable or feasible(mid, meta)[0]:
                ids.append(mid)
    return ids


def ensure_registered(model_ids: list[str], **kw) -> None:
    """Expose model ids to the CV harness (``typed_bench.run_cv``)."""
    for mid in model_ids:
        METHODS[mid] = (lambda m=mid: create_model(m, **kw))
