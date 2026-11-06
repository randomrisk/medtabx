"""Typed errors for the MedTabX IO subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


class IOErrorBase(Exception):
    """Base class for MedTabX IO exceptions."""


class UnsupportedModalityError(IOErrorBase):
    """Raised when modality cannot be detected or supported."""


class BackendNotAvailableError(IOErrorBase):
    """Raised when requested backend is missing/unusable."""


@dataclass
class Attempt:
    backend: str
    reason: str


class LoadFailedError(IOErrorBase):
    """Raised when all candidate backends fail loading."""

    def __init__(self, message: str, attempts: Optional[List[Attempt]] = None):
        self.attempts = attempts or []
        detail = "\n".join(f"- {a.backend}: {a.reason}" for a in self.attempts)
        suffix = f"\nAttempts:\n{detail}" if detail else ""
        super().__init__(f"{message}{suffix}")
