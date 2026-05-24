"""Clause-deletion fairness intervention.

Given a list of clause ids flagged as fairness-suspect by the audit,
produce a *replayed* prediction set on a held-out evaluation slice
that excludes those clauses from the vote arithmetic. The intervention
reports:

* accuracy delta (clause-deletion almost always trades some accuracy),
* fairness-gap delta on the same slice,
* the deleted clauses themselves (so an auditor can review).

The intervention is operationally invertible: nothing is destroyed
on the trained model. Clauses are masked from the per-sample summary
arithmetic only at eval time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..data.dataset import ProtectedAttribute
from ..tm.firing import ClassScoreSummary, FiringClauseSpec
from .metrics import FairnessMetrics, compute_fairness_metrics


@dataclass
class DeleteClausesResult:
  """Outcome of running an eval slice through the masked model."""

  deleted_clause_ids: list[int] = field(default_factory=list)
  before_predictions: list[int] = field(default_factory=list)
  after_predictions: list[int] = field(default_factory=list)
  accuracy_before: float = 0.0
  accuracy_after: float = 0.0
  fairness_before: dict[str, float] | None = None
  fairness_after: dict[str, float] | None = None
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "deleted_clause_ids": list(self.deleted_clause_ids),
      "accuracy_before": float(self.accuracy_before),
      "accuracy_after": float(self.accuracy_after),
      "fairness_before": self.fairness_before,
      "fairness_after": self.fairness_after,
      "n_eval": len(self.before_predictions),
      "notes": list(self.notes),
    }


def _re_aggregate(
  summary: ClassScoreSummary, deleted: set[int], n_classes: int,
) -> int:
  """Recompute the predicted class from ``summary.firing_clauses`` with deletions."""
  scores = [0] * n_classes
  for fc in summary.firing_clauses:
    if fc.clause_id in deleted:
      continue
    if 0 <= fc.class_id < n_classes:
      scores[fc.class_id] += fc.polarity * fc.weight
  return max(range(n_classes), key=lambda c: scores[c])


def apply_clause_deletion(
  *,
  summaries: list[ClassScoreSummary],
  labels: list[int],
  protected: list[dict[ProtectedAttribute, str]],
  attribute: ProtectedAttribute,
  deleted_clause_ids: list[int],
  favourable_class_id: int = 1,
  n_classes: int | None = None,
) -> DeleteClausesResult:
  """Run the masked-model eval pass."""
  if n_classes is None:
    if not summaries:
      raise ValueError("Empty summaries list.")
    n_classes = max(len(s.scores) for s in summaries)
  deleted = set(deleted_clause_ids)

  before_preds: list[int] = []
  after_preds: list[int] = []
  for summary in summaries:
    before_preds.append(int(summary.argmax_label))
    after_preds.append(_re_aggregate(summary, deleted, n_classes))

  def _acc(preds: list[int]) -> float:
    if not preds:
      return 0.0
    return sum(int(p == y) for p, y in zip(preds, labels)) / len(preds)

  before_acc = _acc(before_preds)
  after_acc = _acc(after_preds)
  fb = compute_fairness_metrics(
    before_preds, labels, protected, attribute,
    favourable_class_id=favourable_class_id,
  ).to_dict()
  fa = compute_fairness_metrics(
    after_preds, labels, protected, attribute,
    favourable_class_id=favourable_class_id,
  ).to_dict()
  return DeleteClausesResult(
    deleted_clause_ids=sorted(deleted),
    before_predictions=before_preds,
    after_predictions=after_preds,
    accuracy_before=before_acc,
    accuracy_after=after_acc,
    fairness_before=fb,
    fairness_after=fa,
  )
