"""Mechanistic analyses on the typed contract.

* :mod:`~medtabx.analysis.info` -- what a modality group contributes
  (information vs dimensional cost), with permutation controls.
* :mod:`~medtabx.analysis.alignment` -- per-modality branches, modality
  imbalance and Jensen-Shannon alignment.
* :mod:`~medtabx.analysis.robustness` -- modality-dropout augmentation for
  any learner, so missing-modality robustness can be compared fairly.
"""

from medtabx.analysis.alignment import (
    AlignmentReport,
    branch_report,
    expert_branches,
    js_divergence,
    pairwise_js,
    probe_branches,
)
from medtabx.analysis.info import (
    Decomposition,
    capacity_sweep,
    cross_fit_ce,
    decompose,
)

__all__ = ["AlignmentReport", "branch_report", "expert_branches",
           "js_divergence", "pairwise_js", "probe_branches",
           "Decomposition", "capacity_sweep", "cross_fit_ce", "decompose"]
