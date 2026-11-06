"""Capability and dependency utilities for IO backends."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


def package_available(module_name: str) -> bool:
    try:
        import_module(module_name)
        return True
    except Exception:
        return False


def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"
    except Exception:
        return "unknown"
