"""Failure-aware fusion on the typed contract.

* :mod:`~medtabx.fusion.experts` -- one model per modality and its
  cross-fitted predictions, shared by everything below.
* :mod:`~medtabx.fusion.router` -- a stacked per-patient router over the
  modality experts (and a concat expert); ``blend=0`` is late fusion.
* :mod:`~medtabx.fusion.evidential` -- decision-level adaptation of the
  Dempster-Shafer combination of Trusted Multi-View Classification.

Modality *selection* (which modalities to use at all) lives in
:mod:`medtabx.auto.select`; per-patient acquisition gating in
:mod:`medtabx.auto.gating`.
"""

from medtabx.fusion.evidential import EvidentialFusion
from medtabx.fusion.experts import ModalityExperts, late_mean
from medtabx.fusion.router import StackedRouter

__all__ = ["EvidentialFusion", "ModalityExperts", "StackedRouter",
           "late_mean"]
