"""Graph Tsetlin Machine wrapper for hiring classification.

This module wraps the CAIR ``GraphTsetlinMachine.MultiClassGraphTsetlinMachine``
(or ``tmu`` equivalent), exposing:

* :class:`HireTM`, single-seed classifier.
* :class:`HireTMEnsemble`, K-seed ensemble; majority vote at
 inference, with per-class vote sums preserved for the SAT receipt
 and the fairness audit.
* :class:`FiringClauseSpec`, the on-the-wire description of a clause
 that the explainer / recourse / SAT modules consume.
* :class:`BackendUnavailableError`, raised when neither GraphTsetlinMachine
 nor a pure-Python fallback can be loaded.

A pure-Python toy fallback (:class:`MajorityVoteFallback`) lets the
full pipeline (explain, recourse, verify, fairness) be exercised by
the test suite without a real TM kernel installed.
"""

from .firing import FiringClauseSpec, ClassScoreSummary
from .hire_tm import HireTM, BackendUnavailableError
from .ensemble import HireTMEnsemble
from .fallback import MajorityVoteFallback

__all__ = [
  "FiringClauseSpec",
  "ClassScoreSummary",
  "HireTM",
  "HireTMEnsemble",
  "BackendUnavailableError",
  "MajorityVoteFallback",
]
