"""On-the-wire description of a firing clause + per-class vote summary.

Everything downstream of the TM (explainer, recourse, SAT receipt,
fairness audit, faithfulness eval) consumes this structure. Keeping
the format here decouples downstream modules from the specific TM
backend (CAIR ``MultiClassGraphTsetlinMachine``, ``tmu``, or the
in-tree :class:`MajorityVoteFallback`).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FiringClauseSpec:
  """One clause that fired for a particular sample.

  Attributes:
    clause_id: stable, globally-unique clause id within a single
      trained ``HireTM`` (or ensemble member id : clause id
      joined for ensembles).
    class_id: which output class this clause votes for.
    polarity: ``+1`` for positive-polarity clause (votes FOR
      ``class_id``); ``-1`` for negative-polarity (votes AGAINST
      ``class_id``).
    weight: integer vote weight; typically 1 in CAIR TMs, may be
      larger if the backend supports weighted clauses.
    positive_literal_ids: literal ids that must be 1 for this
      clause to fire.
    negative_literal_ids: literal ids that must be 0 for this
      clause to fire.
    member_id: ensemble-member index if part of an ensemble,
      ``None`` otherwise.
  """

  clause_id: int
  class_id: int
  polarity: int
  weight: int
  positive_literal_ids: tuple[int, ...] = field(default_factory=tuple)
  negative_literal_ids: tuple[int, ...] = field(default_factory=tuple)
  member_id: int | None = None

  def as_dict(self) -> dict:
    return {
      "clause_id": self.clause_id,
      "class_id": self.class_id,
      "polarity": self.polarity,
      "weight": self.weight,
      "positive_literal_ids": list(self.positive_literal_ids),
      "negative_literal_ids": list(self.negative_literal_ids),
      "member_id": self.member_id,
    }


@dataclass(frozen=True)
class ClassScoreSummary:
  """Per-sample summary of the multi-class TM vote arithmetic.

  Attributes:
    scores: vote score per class.
    argmax_label: arg-max class id (the recorded prediction).
    margin: ``scores[argmax_label] - max_{c != argmax} scores[c]``.
    firing_clauses: list of firing clauses that produced this
      vote (across all classes, both polarities).
  """

  scores: tuple[int, ...]
  argmax_label: int
  margin: int
  firing_clauses: tuple[FiringClauseSpec, ...]

  def as_dict(self) -> dict:
    return {
      "scores": list(self.scores),
      "argmax_label": self.argmax_label,
      "margin": self.margin,
      "firing_clauses": [c.as_dict() for c in self.firing_clauses],
    }
