from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from medtabx.core.registry import tree_model_registry


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    elif hasattr(data, "to_numpy"):
        data = data.to_numpy()
    return np.asarray(data)


class BaseTreeModel:
    """Common wrapper interface for optional tree-model backends."""

    backend_name: str

    def fit(self, x: Any, y: Any):
        raise NotImplementedError

    def predict(self, x: Any):
        raise NotImplementedError

    def predict_proba(self, x: Any):
        raise NotImplementedError

    def save_model(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load_model(cls, path: str):
        raise NotImplementedError


@tree_model_registry.register("hist_gbdt_classifier")
class HistGradientBoostingClassifierModel(BaseTreeModel):
    backend_name = "hist_gbdt_classifier"

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, x: Any, y: Any):
        self.model.fit(_to_numpy(x), _to_numpy(y))
        return self

    def predict(self, x: Any):
        return self.model.predict(_to_numpy(x))

    def predict_proba(self, x: Any):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("This backend does not expose predict_proba.")
        return self.model.predict_proba(_to_numpy(x))

    def save_model(self, path: str) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load_model(cls, path: str):
        import joblib

        inst = cls()
        inst.model = joblib.load(path)
        return inst


@tree_model_registry.register("hist_gbdt_regressor")
class HistGradientBoostingRegressorModel(BaseTreeModel):
    backend_name = "hist_gbdt_regressor"

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model = HistGradientBoostingRegressor(**kwargs)

    def fit(self, x: Any, y: Any):
        self.model.fit(_to_numpy(x), _to_numpy(y))
        return self

    def predict(self, x: Any):
        return self.model.predict(_to_numpy(x))

    def predict_proba(self, x: Any):
        raise AttributeError("Regression backends do not expose predict_proba.")

    def save_model(self, path: str) -> None:
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load_model(cls, path: str):
        import joblib

        inst = cls()
        inst.model = joblib.load(path)
        return inst


def _register_optional_backend(registry_name: str, import_path: str, class_name: str) -> None:
    module_name, symbol_name = import_path.rsplit(".", 1)

    @tree_model_registry.register(registry_name)
    class OptionalTreeModel(BaseTreeModel):
        backend_name = registry_name

        def __init__(self, **kwargs):
            module = __import__(module_name, fromlist=[symbol_name])
            backend_cls = getattr(module, symbol_name)
            self.model = backend_cls(**kwargs)

        def fit(self, x: Any, y: Any):
            self.model.fit(_to_numpy(x), _to_numpy(y))
            return self

        def predict(self, x: Any):
            return self.model.predict(_to_numpy(x))

        def predict_proba(self, x: Any):
            if not hasattr(self.model, "predict_proba"):
                raise AttributeError(
                    f"{class_name} does not expose predict_proba for this task."
                )
            return self.model.predict_proba(_to_numpy(x))

        def save_model(self, path: str) -> None:
            if hasattr(self.model, "save_model"):
                self.model.save_model(path)
                return
            if hasattr(self.model, "save_model"):
                self.model.save_model(path)
                return
            import joblib

            joblib.dump(self.model, path)

        @classmethod
        def load_model(cls, path: str):
            inst = cls()
            if hasattr(inst.model, "load_model"):
                inst.model.load_model(path)
                return inst
            import joblib

            inst.model = joblib.load(path)
            return inst


for _name, _path, _class_name in [
    ("xgboost_classifier", "xgboost.XGBClassifier", "XGBClassifier"),
    ("xgboost_regressor", "xgboost.XGBRegressor", "XGBRegressor"),
    ("lightgbm_classifier", "lightgbm.LGBMClassifier", "LGBMClassifier"),
    ("lightgbm_regressor", "lightgbm.LGBMRegressor", "LGBMRegressor"),
    ("catboost_classifier", "catboost.CatBoostClassifier", "CatBoostClassifier"),
    ("catboost_regressor", "catboost.CatBoostRegressor", "CatBoostRegressor"),
]:
    _register_optional_backend(_name, _path, _class_name)


@dataclass(slots=True)
class TreeModelAdapter:
    """High-level adapter mirroring the benchmark-facing tree API."""

    model: BaseTreeModel

    def fit(self, x: Any, y: Any):
        self.model.fit(x, y)
        return self

    def predict(self, x: Any):
        return self.model.predict(x)

    def predict_proba(self, x: Any):
        return self.model.predict_proba(x)

    def save(self, path: str) -> None:
        self.model.save_model(path)


def get_tree_model_class(name: str):
    return tree_model_registry.get(name)


def create_tree_model(name: str, **kwargs) -> TreeModelAdapter:
    cls = get_tree_model_class(name)
    return TreeModelAdapter(model=cls(**kwargs))


def available_tree_backends() -> dict[str, bool]:
    availability: dict[str, bool] = {}
    for name in tree_model_registry.list():
        cls = tree_model_registry.get(name)
        try:
            cls()
        except Exception:
            availability[name] = False
        else:
            availability[name] = True
    return availability
