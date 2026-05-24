"""HireTMEnsemble: K-seed ensemble of HireTM classifiers.

The ensemble is the canonical reporting unit in this project, matching
the 5-seed protocol used in Anwar's portfolio (Papers A/B/C +
graphtm-cbr + clauseguard). Aggregation:

* Hard-vote across members for the predicted label.
* Sum per-class score across members for the SAT receipt.
* Union of firing clauses across members for the Article 86 explanation
 (each clause keeps its ``member_id``, so the receipt can be replayed
 member-by-member).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..graphs.built_graphs import BuiltGraphs
from .firing import ClassScoreSummary, FiringClauseSpec
from .hire_tm import HireTM


@dataclass
class HireTMEnsemble:
  """K-seed ensemble of :class:`HireTM`."""

  n_clauses: int = 400
  T: int = 200
  s: float = 10.0
  depth: int = 2
  n_classes: int = 0
  seeds: tuple[int, ...] = (42, 123, 456, 789, 1337)
  allow_fallback: bool = True

  members: list[HireTM] = field(default_factory=list, init=False)

  # ------------------------------------------------------------------
  def fit(
    self,
    train_built: BuiltGraphs,
    train_labels: list[int],
    *,
    n_epochs: int = 25,
    n_classes: int | None = None,
  ) -> "HireTMEnsemble":
    if n_classes is None:
      n_classes = max(train_labels) + 1
    self.n_classes = n_classes
    self.members = []
    for seed in self.seeds:
      tm = HireTM(
        n_clauses=self.n_clauses, T=self.T, s=self.s, depth=self.depth,
        seed=seed, allow_fallback=self.allow_fallback,
      )
      tm.fit(train_built, train_labels, n_epochs=n_epochs, n_classes=n_classes)
      self.members.append(tm)
    return self

  # ------------------------------------------------------------------
  def predict(self, built: BuiltGraphs) -> list[int]:
    if not self.members:
      raise RuntimeError("HireTMEnsemble.fit must be called before predict.")
    per_member = [m.predict(built) for m in self.members]
    out: list[int] = []
    for i in range(built.n_samples):
      counts = Counter(pm[i] for pm in per_member)
      out.append(counts.most_common(1)[0][0])
    return out

  # ------------------------------------------------------------------
  def explain(self, built: BuiltGraphs, sample_idx: int) -> ClassScoreSummary:
    if not self.members:
      raise RuntimeError("HireTMEnsemble.fit must be called before explain.")
    per_class = [0] * self.n_classes
    all_firing: list[FiringClauseSpec] = []
    for mid, m in enumerate(self.members):
      summary = m.explain(built, sample_idx)
      for c in range(self.n_classes):
        if c < len(summary.scores):
          per_class[c] += int(summary.scores[c])
      for fc in summary.firing_clauses:
        all_firing.append(FiringClauseSpec(
          clause_id=fc.clause_id,
          class_id=fc.class_id,
          polarity=fc.polarity,
          weight=fc.weight,
          positive_literal_ids=fc.positive_literal_ids,
          negative_literal_ids=fc.negative_literal_ids,
          member_id=mid,
        ))
    argmax = max(range(self.n_classes), key=lambda c: per_class[c])
    others = [per_class[c] for c in range(self.n_classes) if c != argmax]
    margin = per_class[argmax] - (max(others) if others else 0)
    return ClassScoreSummary(
      scores=tuple(per_class),
      argmax_label=argmax,
      margin=margin,
      firing_clauses=tuple(all_firing),
    )

  def num_clauses_total(self) -> int:
    return sum(m.num_clauses_total() for m in self.members)
