"""Per-clause fairness audit.

For every trained clause, count how often it fires on each
protected-group subpopulation. Clauses whose firing rate diverges
across groups by more than a configurable threshold are flagged as
**unfair-suspect**; the deployer can then run
:func:`apply_clause_deletion` (in ``intervention.py``) and measure
the trade-off between fairness improvement and accuracy loss.

This is the unique-to-TM intervention. SHAP/LIME on a neural model
can only TELL you the feature was salient; you cannot edit the
model to remove the saliency without retraining. With a TM the
clause is a discrete unit you can drop.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..data.dataset import ProtectedAttribute
from ..graphs.built_graphs import BuiltGraphs
from ..tm.firing import FiringClauseSpec


@dataclass
class ClauseFairnessAudit:
  """Per-clause group-conditional firing statistics."""

  attribute: ProtectedAttribute
  n_total: int
  per_clause: dict[int, dict[str, float]] = field(default_factory=dict)
  suspect_clause_ids: list[int] = field(default_factory=list)
  suspect_threshold: float = 0.1

  def to_dict(self) -> dict[str, Any]:
    return {
      "attribute": self.attribute.value,
      "n_total": self.n_total,
      "suspect_threshold": float(self.suspect_threshold),
      "suspect_clause_ids": list(self.suspect_clause_ids),
      "per_clause": {
        str(cid): {k: float(v) for k, v in stats.items()}
        for cid, stats in self.per_clause.items()
      },
    }


def _firing_clauses_match(
  clause: FiringClauseSpec, literal_vec: list[int],
) -> bool:
  if any(literal_vec[l] != 1 for l in clause.positive_literal_ids
      if l < len(literal_vec)):
    return False
  if any(literal_vec[l] != 0 for l in clause.negative_literal_ids
      if l < len(literal_vec)):
    return False
  return True


def audit_clauses(
  *,
  clauses: list[FiringClauseSpec],
  built: BuiltGraphs,
  protected: list[dict[ProtectedAttribute, str]],
  attribute: ProtectedAttribute,
  suspect_threshold: float = 0.1,
) -> ClauseFairnessAudit:
  """Score every clause's firing-rate divergence across groups.

  Args:
    clauses: full clause bank to audit (typically the trained
      ensemble's union of clauses).
    built: BuiltGraphs covering the samples used for the audit.
    protected: per-sample protected attribute dict.
    attribute: which protected attribute to group by.
    suspect_threshold: minimum cross-group firing-rate gap to
      flag a clause as suspect. Default 0.1 (10 percentage points)
      matches the FAccT-community informal threshold.
  """
  n = built.n_samples
  per_group_count = defaultdict(int)
  per_group_clause_fires: dict[int, dict[str, int]] = defaultdict(
    lambda: defaultdict(int)
  )

  for i in range(n):
    prot = protected[i] if i < len(protected) else {}
    if attribute not in prot:
      continue
    g = str(prot[attribute])
    per_group_count[g] += 1
    lit_vec = built.get_active_literal_vector(i)
    for clause in clauses:
      if _firing_clauses_match(clause, lit_vec):
        per_group_clause_fires[clause.clause_id][g] += 1

  audit = ClauseFairnessAudit(
    attribute=attribute,
    n_total=sum(per_group_count.values()),
    suspect_threshold=suspect_threshold,
  )
  suspect: list[tuple[float, int]] = []
  for cid, fires in per_group_clause_fires.items():
    rates: dict[str, float] = {}
    for g, c in per_group_count.items():
      rates[g] = (fires.get(g, 0) / c) if c > 0 else 0.0
    if rates:
      gap = max(rates.values()) - min(rates.values())
    else:
      gap = 0.0
    audit.per_clause[cid] = {**rates, "gap": gap}
    if gap >= suspect_threshold:
      suspect.append((gap, cid))
  audit.suspect_clause_ids = [cid for _, cid in sorted(suspect, reverse=True)]
  return audit
