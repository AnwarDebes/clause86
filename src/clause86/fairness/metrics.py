"""Group-conditional fairness metrics.

Implements the three metrics most commonly demanded by EU AI Act
Annex III section 4 deployers + the FAccT literature:

* **Demographic parity**: P(Y_hat=1 | A=a) equal across groups.
* **Equal opportunity**: P(Y_hat=1 | Y=1, A=a) equal across groups.
* **Equalized odds**: both TPR and FPR equal across groups.

We report per-group rates AND pairwise gaps; the gap is the
auditor-facing scalar.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..data.dataset import ProtectedAttribute


@dataclass
class FairnessMetrics:
  """Result container for a single protected-attribute audit."""

  attribute: ProtectedAttribute
  favourable_class_id: int
  n_total: int
  per_group: dict[str, dict[str, float]] = field(default_factory=dict)
  demographic_parity_gap: float = 0.0
  equal_opportunity_gap: float = 0.0
  equalized_odds_gap: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "attribute": self.attribute.value,
      "favourable_class_id": self.favourable_class_id,
      "n_total": self.n_total,
      "per_group": {
        g: {k: float(v) for k, v in stats.items()}
        for g, stats in self.per_group.items()
      },
      "demographic_parity_gap": float(self.demographic_parity_gap),
      "equal_opportunity_gap": float(self.equal_opportunity_gap),
      "equalized_odds_gap": float(self.equalized_odds_gap),
    }


def group_predictions(
  predictions: list[int],
  labels: list[int],
  protected: list[dict[ProtectedAttribute, str]],
  attribute: ProtectedAttribute,
) -> dict[str, dict[str, list[int]]]:
  """Group (pred, label) pairs by protected-attribute value."""
  out: dict[str, dict[str, list[int]]] = defaultdict(
    lambda: {"pred": [], "label": []}
  )
  for p, y, prot in zip(predictions, labels, protected):
    if attribute not in prot:
      continue
    g = str(prot[attribute])
    out[g]["pred"].append(int(p))
    out[g]["label"].append(int(y))
  return dict(out)


def compute_fairness_metrics(
  predictions: list[int],
  labels: list[int],
  protected: list[dict[ProtectedAttribute, str]],
  attribute: ProtectedAttribute,
  favourable_class_id: int = 1,
) -> FairnessMetrics:
  """Compute per-group rates + the three gap metrics."""
  grouped = group_predictions(predictions, labels, protected, attribute)
  per_group: dict[str, dict[str, float]] = {}
  for group, gd in grouped.items():
    preds = gd["pred"]
    ys = gd["label"]
    n = len(preds)
    if n == 0:
      continue
    n_fav = sum(1 for p in preds if p == favourable_class_id)
    pos = [(p, y) for p, y in zip(preds, ys) if y == favourable_class_id]
    neg = [(p, y) for p, y in zip(preds, ys) if y != favourable_class_id]
    tpr = (
      sum(1 for p, _ in pos if p == favourable_class_id) / len(pos)
      if pos else 0.0
    )
    fpr = (
      sum(1 for p, _ in neg if p == favourable_class_id) / len(neg)
      if neg else 0.0
    )
    per_group[group] = {
      "n": float(n),
      "fav_rate": n_fav / n,
      "tpr": tpr,
      "fpr": fpr,
    }
  fav_rates = [stats["fav_rate"] for stats in per_group.values()]
  tprs = [stats["tpr"] for stats in per_group.values()]
  fprs = [stats["fpr"] for stats in per_group.values()]
  dp_gap = (max(fav_rates) - min(fav_rates)) if fav_rates else 0.0
  eo_gap = (max(tprs) - min(tprs)) if tprs else 0.0
  eodds_gap = max(
    (max(tprs) - min(tprs)) if tprs else 0.0,
    (max(fprs) - min(fprs)) if fprs else 0.0,
  )
  return FairnessMetrics(
    attribute=attribute,
    favourable_class_id=favourable_class_id,
    n_total=len(predictions),
    per_group=per_group,
    demographic_parity_gap=dp_gap,
    equal_opportunity_gap=eo_gap,
    equalized_odds_gap=eodds_gap,
  )
