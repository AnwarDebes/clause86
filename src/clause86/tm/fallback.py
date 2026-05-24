"""MajorityVoteFallback: a stand-in classifier that is NOT a TM.

This exists ONLY so the downstream modules (explain, recourse, verify,
fairness, eval) can be exercised by the test suite without a real TM
backend installed. It learns a per-class literal-presence prior and
predicts the arg-max of the dot product. Its "clauses" are then
synthesised from the top-k positive literals per class so that the
explain / recourse code paths run.

Any benchmarking script that reaches for this fallback should refuse
to log results; the ``backend_kind`` field on the per-sample log will
carry ``"fallback"`` so this is detectable downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..graphs.built_graphs import BuiltGraphs
from .firing import ClassScoreSummary, FiringClauseSpec


@dataclass
class MajorityVoteFallback:
  """Toy classifier producing synthetic firing clauses."""

  n_classes: int
  seed: int = 42
  top_k_literals_per_clause: int = 3
  n_clauses_per_class: int = 8

  _class_prior: np.ndarray = field(default=None, init=False)
  _class_lit_score: np.ndarray = field(default=None, init=False)
  _synth_clauses: list[FiringClauseSpec] = field(default_factory=list, init=False)

  # ------------------------------------------------------------------
  def fit(self, built: BuiltGraphs, labels: list[int]) -> None:
    n_lit = built.n_literals
    n_cls = self.n_classes
    class_count = np.zeros(n_cls, dtype=np.float64)
    lit_count = np.zeros((n_cls, n_lit), dtype=np.float64)
    for i, lab in enumerate(labels):
      if not (0 <= lab < n_cls):
        continue
      class_count[lab] += 1.0
      for lid in built.per_sample_active_literals[i]:
        if 0 <= lid < n_lit:
          lit_count[lab, lid] += 1.0
    class_prior = class_count / max(class_count.sum(), 1.0)
    # Pointwise mutual-information-ish score.
    lit_prior = lit_count.sum(axis=0)
    lit_prior = np.where(lit_prior > 0, lit_prior, 1.0)
    lit_given_class = lit_count / (class_count[:, None] + 1.0)
    lit_marginal = lit_prior / (class_count.sum() + 1.0)
    lit_marginal = np.where(lit_marginal > 0, lit_marginal, 1e-6)
    self._class_lit_score = np.log(
      (lit_given_class + 1e-6) / lit_marginal[None, :]
    )
    self._class_prior = class_prior
    self._synth_clauses = self._synthesise_clauses(built, lit_count, class_count)

  def _synthesise_clauses(
    self,
    built: BuiltGraphs,
    lit_count: np.ndarray,
    class_count: np.ndarray,
  ) -> list[FiringClauseSpec]:
    out: list[FiringClauseSpec] = []
    n_cls = self.n_classes
    k = self.top_k_literals_per_clause
    n_per_class = self.n_clauses_per_class
    for c in range(n_cls):
      order = np.argsort(-self._class_lit_score[c])
      for j in range(n_per_class):
        start = j * k
        stop = start + k
        if start >= len(order):
          break
        lit_ids = tuple(int(x) for x in order[start:stop])
        out.append(FiringClauseSpec(
          clause_id=c * n_per_class + j,
          class_id=c,
          polarity=1,
          weight=1,
          positive_literal_ids=lit_ids,
          negative_literal_ids=(),
          member_id=None,
        ))
    return out

  # ------------------------------------------------------------------
  def predict(self, built: BuiltGraphs) -> list[int]:
    if self._class_lit_score is None:
      raise RuntimeError("MajorityVoteFallback.fit must be called first.")
    n = built.n_samples
    preds: list[int] = []
    for i in range(n):
      scores = np.array(self._class_prior, dtype=np.float64).copy()
      for lid in built.per_sample_active_literals[i]:
        if 0 <= lid < self._class_lit_score.shape[1]:
          scores += self._class_lit_score[:, lid]
      preds.append(int(np.argmax(scores)))
    return preds

  def explain(self, built: BuiltGraphs, sample_idx: int) -> ClassScoreSummary:
    scores = np.array(self._class_prior, dtype=np.float64).copy()
    for lid in built.per_sample_active_literals[sample_idx]:
      if 0 <= lid < self._class_lit_score.shape[1]:
        scores += self._class_lit_score[:, lid]
    argmax = int(np.argmax(scores))

    # Firing clauses for this sample: every synthesised clause
    # whose positive literal set is fully satisfied by the sample.
    active = set(built.per_sample_active_literals[sample_idx])
    firing: list[FiringClauseSpec] = []
    per_class_score = [0] * self.n_classes
    for clause in self._synth_clauses:
      if set(clause.positive_literal_ids).issubset(active):
        firing.append(clause)
        per_class_score[clause.class_id] += clause.polarity * clause.weight
    if not firing:
      # Emit a single zero-literal "default" clause for the
      # predicted class so downstream code always sees >=1 firing.
      firing.append(FiringClauseSpec(
        clause_id=10_000 + argmax,
        class_id=argmax,
        polarity=1,
        weight=1,
        positive_literal_ids=(),
        negative_literal_ids=(),
        member_id=None,
      ))
      per_class_score[argmax] += 1
    argmax_label = int(np.argmax(per_class_score))
    others = [v for c, v in enumerate(per_class_score) if c != argmax_label]
    margin = per_class_score[argmax_label] - (max(others) if others else 0)
    return ClassScoreSummary(
      scores=tuple(per_class_score),
      argmax_label=argmax_label,
      margin=margin,
      firing_clauses=tuple(firing),
    )

  def num_clauses_total(self) -> int:
    return len(self._synth_clauses)
