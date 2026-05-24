"""Generate candidate recourse edits from the firing clauses of an
adverse decision.

Strategy:

1. For every firing clause of the predicted (adverse) class with
  positive polarity (so it's a clause that *voted for* the adverse
  class), propose flipping each of its positive literals to 0.
2. For every firing clause of the predicted (adverse) class with
  negative polarity (so it's voting AGAINST a different class that
  we want to flip TO), propose flipping each of its NEGATED
  literals to 1.
3. For every NON-firing clause of the desired target class,
  propose flipping ALL of its positive literals to 1 simultaneously
  so the clause newly fires.
4. Filter through :class:`FeasibilityChecker`: drop edits that touch
  immutable literals, drop edits whose semantics is "lose years of
  experience" or "remove a degree" (we model only positive change).

The candidate generator returns ``RecourseEdit`` items ordered by an
*a-priori* utility score, so the greedy search can prune early.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graphs.built_graphs import BuiltGraphs, LiteralSpec
from ..tm.firing import FiringClauseSpec
from .feasibility import FeasibilityChecker


@dataclass(frozen=True)
class RecourseEdit:
  """One proposed flip of a literal.

  Attributes:
    literal_id: position in the literal vector.
    from_value: current value (0 or 1).
    to_value: proposed value (0 or 1).
    rationale: short string describing why the edit was proposed
      (which clause + premise).
    utility_prior: a heuristic non-negative score used to prune
      candidate ordering before running the TM. Higher means
      more likely to flip the prediction.
  """

  literal_id: int
  from_value: int
  to_value: int
  rationale: str = ""
  utility_prior: float = 0.0

  def __post_init__(self) -> None:
    if self.from_value not in (0, 1):
      raise ValueError(f"from_value must be 0 or 1, got {self.from_value}")
    if self.to_value not in (0, 1):
      raise ValueError(f"to_value must be 0 or 1, got {self.to_value}")
    if self.from_value == self.to_value:
      raise ValueError(
        f"Edit is a no-op: literal {self.literal_id} {self.from_value}->{self.to_value}"
      )


def generate_candidates(
  *,
  sample_idx: int,
  built: BuiltGraphs,
  firing_clauses: list[FiringClauseSpec],
  adverse_class_id: int,
  target_class_id: int,
  feasibility: FeasibilityChecker | None = None,
  all_clauses_for_target: list[FiringClauseSpec] | None = None,
) -> list[RecourseEdit]:
  """Build the candidate-edit list for one sample's recourse step."""
  literal_specs = built.literal_specs
  literal_vec = built.get_active_literal_vector(sample_idx)
  feasibility = feasibility or FeasibilityChecker(literal_specs=literal_specs)
  out: list[RecourseEdit] = []

  # (1) + (2): break adverse-class clauses.
  for clause in firing_clauses:
    if clause.class_id != adverse_class_id:
      continue
    if clause.polarity > 0:
      for lid in clause.positive_literal_ids:
        if lid >= len(literal_vec):
          continue
        if literal_vec[lid] != 1:
          continue
        if not feasibility.may_flip(lid, from_v=1, to_v=0):
          continue
        out.append(RecourseEdit(
          literal_id=lid,
          from_value=1,
          to_value=0,
          rationale=f"break adverse-class clause {clause.clause_id} "
               f"(positive premise {lid})",
          utility_prior=float(abs(clause.polarity * clause.weight)),
        ))
      for lid in clause.negative_literal_ids:
        if lid >= len(literal_vec):
          continue
        if literal_vec[lid] != 0:
          continue
        if not feasibility.may_flip(lid, from_v=0, to_v=1):
          continue
        out.append(RecourseEdit(
          literal_id=lid,
          from_value=0,
          to_value=1,
          rationale=f"break adverse-class clause {clause.clause_id} "
               f"(satisfy negated premise {lid})",
          utility_prior=float(abs(clause.polarity * clause.weight)),
        ))

  # (3): newly fire target-class clauses.
  if all_clauses_for_target:
    for clause in all_clauses_for_target:
      if clause.class_id != target_class_id:
        continue
      if clause.polarity != 1:
        continue
      # If we already fire this clause we don't need to flip anything.
      pos_ok = all(literal_vec[l] == 1 for l in clause.positive_literal_ids
             if l < len(literal_vec))
      neg_ok = all(literal_vec[l] == 0 for l in clause.negative_literal_ids
             if l < len(literal_vec))
      if pos_ok and neg_ok:
        continue
      for lid in clause.positive_literal_ids:
        if lid >= len(literal_vec):
          continue
        if literal_vec[lid] == 1:
          continue
        if not feasibility.may_flip(lid, from_v=0, to_v=1):
          continue
        out.append(RecourseEdit(
          literal_id=lid,
          from_value=0,
          to_value=1,
          rationale=f"newly fire target-class clause {clause.clause_id} "
               f"(satisfy positive premise {lid})",
          utility_prior=0.5 * float(clause.weight),
        ))

  # De-duplicate by (literal_id, to_value); keep the highest-utility one.
  dedup: dict[tuple[int, int], RecourseEdit] = {}
  for e in out:
    key = (e.literal_id, e.to_value)
    if key not in dedup or dedup[key].utility_prior < e.utility_prior:
      dedup[key] = e
  return sorted(dedup.values(), key=lambda x: -x.utility_prior)
