"""Fairness audit + clause-deletion intervention.

The audit module computes group-conditional metrics
(demographic parity, equal opportunity, equalized odds) and per-clause
firing-rate divergence. The intervention module then runs a held-out
eval pass with flagged clauses masked out, reporting the
accuracy / fairness trade-off.

Why this is unique to TMs: a TM clause is a discrete unit you can
surgically delete, unlike a feature implicit across millions of
parameters in a neural model.
"""

from .metrics import (
  FairnessMetrics,
  compute_fairness_metrics,
  group_predictions,
)
from .audit import ClauseFairnessAudit, audit_clauses
from .intervention import (
  DeleteClausesResult,
  apply_clause_deletion,
)

__all__ = [
  "FairnessMetrics",
  "compute_fairness_metrics",
  "group_predictions",
  "ClauseFairnessAudit",
  "audit_clauses",
  "DeleteClausesResult",
  "apply_clause_deletion",
]
