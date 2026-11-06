from pathlib import Path

import pandas as pd
import pytest

from medtabx.io import inspect, load
from medtabx.io.base_loader import BaseLoader
from medtabx.io.contracts import MedSample
from medtabx.io.detector import detect_format, detect_modality
from medtabx.io.errors import BackendNotAvailableError, LoadFailedError
from medtabx.io.registry import LoaderRegistry
from medtabx.io.resolver import LoaderResolver


class DummyFailLoader(BaseLoader):
    name = "dummy_fail"
    supported_modalities = ["dummy"]
    supported_formats = ["dummy"]
    priority = 10

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def can_load(cls, path: str) -> bool:
        return Path(path).suffix == ".dummy"

    @classmethod
    def probe(cls, path: str):
        return {"path": path}

    @classmethod
    def load(cls, path: str, options=None):
        raise RuntimeError("forced failure")


class DummyNoLoadLoader(BaseLoader):
    name = "dummy_noload"
    supported_modalities = ["dummy"]
    supported_formats = ["dummy"]
    priority = 20

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def can_load(cls, path: str) -> bool:
        return False

    @classmethod
    def probe(cls, path: str):
        return {"path": path}

    @classmethod
    def load(cls, path: str, options=None):
        return MedSample(modality="dummy", data=None)


def test_detector_csv():
    assert detect_modality("x.csv") == "tabular"
    assert detect_format("x.csv") == "csv"


def test_registry_duplicate_rejected():
    registry = LoaderRegistry()
    registry.register(DummyFailLoader)
    with pytest.raises(ValueError):
        registry.register(DummyFailLoader)


def test_resolver_strict_backend_errors_for_mismatch(tmp_path):
    file_path = tmp_path / "sample.dummy"
    file_path.write_text("x", encoding="utf-8")

    registry = LoaderRegistry()
    registry.register(DummyNoLoadLoader)

    resolver = LoaderResolver(registry)
    with pytest.raises(BackendNotAvailableError):
        resolver.resolve(str(file_path), modality="dummy", backend="dummy_noload")


def test_resolver_reports_attempts(tmp_path):
    file_path = tmp_path / "sample.dummy"
    file_path.write_text("x", encoding="utf-8")

    registry = LoaderRegistry()
    registry.register(DummyNoLoadLoader)

    resolver = LoaderResolver(registry)
    with pytest.raises(LoadFailedError) as exc:
        resolver.resolve(str(file_path), modality="dummy")

    assert "can_load returned False" in str(exc.value)


def test_load_csv_returns_medsample(tmp_path):
    csv_path = tmp_path / "patients.csv"
    df = pd.DataFrame({"age": [21, 34], "gender": ["f", "m"]})
    df.to_csv(csv_path, index=False)

    sample = load(str(csv_path))

    assert isinstance(sample, MedSample)
    assert sample.modality == "tabular"
    assert sample.meta["loader_backend"] == "pandas"
    assert sample.meta["detected_format"] == "csv"
    assert list(sample.data.columns) == ["age", "gender"]


def test_inspect_csv(tmp_path):
    csv_path = tmp_path / "patients.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    metadata = inspect(str(csv_path))

    assert metadata["detected_modality"] == "tabular"
    assert metadata["loader_backend"] == "pandas"
