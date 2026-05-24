"""Greedy minimum-edit recourse search.

Mirrors the search loop in ``graphtm-cbr.recourse.search.greedy_minimal_edit``
but operates on the literal vector instead of an RDKit molecule. At
each step we score every candidate edit by *predicted margin gain*
(the change in ``score[target] - max(other)`` if we apply just that
edit) and take the best one. We stop when either:

* the TM ensemble already predicts the target class (success), or
* the candidate set is empty (failure), or
* the per-step budget (``max_edits``) is exhausted (failure), or
* the wall-clock budget (``time_budget_ms``) is exhausted (failure).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..graphs.built_graphs import BuiltGraphs
from ..tm.firing import ClassScoreSummary, FiringClauseSpec
from .candidates import RecourseEdit, generate_candidates
from .feasibility import FeasibilityChecker


# Callable types the search needs to score & re-run the TM on a
# perturbed literal vector. These are passed in so the search loop
# stays decoupled from the TM backend choice.
ExplainFn = Callable[[BuiltGraphs, int], ClassScoreSummary]


@dataclass
class HireRecourseReport:
  """The outcome of a recourse search for one sample."""

  sample_idx: int
  adverse_class_id: int
  target_class_id: int
  applied_edits: list[RecourseEdit] = field(default_factory=list)
  score_history: list[tuple[int, ...]] = field(default_factory=list)
  margin_history: list[int] = field(default_factory=list)
  final_prediction: int = -1
  success: bool = False
  total_latency_ms: float = 0.0
  candidates_examined: int = 0
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      "sample_idx": self.sample_idx,
      "adverse_class_id": self.adverse_class_id,
      "target_class_id": self.target_class_id,
      "applied_edits": [
        {
          "literal_id": e.literal_id,
          "from_value": e.from_value,
          "to_value": e.to_value,
          "rationale": e.rationale,
          "utility_prior": e.utility_prior,
        }
        for e in self.applied_edits
      ],
      "score_history": [list(s) for s in self.score_history],
      "margin_history": list(self.margin_history),
      "final_prediction": self.final_prediction,
      "success": self.success,
      "total_latency_ms": self.total_latency_ms,
      "candidates_examined": self.candidates_examined,
      "notes": list(self.notes),
    }


def _apply_edit_to_active(
  active_set: set[int], edit: RecourseEdit,
) -> set[int]:
  new = set(active_set)
  if edit.to_value == 1:
    new.add(edit.literal_id)
  else:
    new.discard(edit.literal_id)
  return new


def _patched_built(
  built: BuiltGraphs, sample_idx: int, new_active: set[int],
) -> BuiltGraphs:
  """Return a copy of ``built`` with one sample's active set replaced.

  We deliberately do NOT rebuild the GraphTM graphs object here --
  the explainer for our fallback / introspection path uses
  ``built.get_active_literal_vector`` to find the literal vector.
  For the production CAIR backend, a real recourse over the Graphs
  object would require re-encoding; that path is left as a TODO
  marked with NEEDS-GRAPH-REENCODE in the meta dict so downstream
  benchmarks refuse to log recourse latency for it.
  """
  new_active_tuple = tuple(sorted(new_active))
  new_per_sample = list(built.per_sample_active_literals)
  new_per_sample[sample_idx] = new_active_tuple
  return BuiltGraphs(
    graphs=built.graphs,
    n_samples=built.n_samples,
    symbols=built.symbols,
    edge_types=built.edge_types,
    literal_specs=built.literal_specs,
    per_sample_active_literals=new_per_sample,
    meta={**built.meta, "NEEDS-GRAPH-REENCODE-FOR-RECOURSE": True},
  )


def greedy_minimum_edit(
  *,
  built: BuiltGraphs,
  sample_idx: int,
  initial_summary: ClassScoreSummary,
  target_class_id: int,
  explain_fn: ExplainFn,
  all_target_clauses: list[FiringClauseSpec] | None = None,
  feasibility: FeasibilityChecker | None = None,
  max_edits: int = 3,
  time_budget_ms: float = 5000.0,
  max_candidates_per_step: int = 32,
) -> HireRecourseReport:
  """Run greedy minimum-edit recourse for one sample."""
  if initial_summary.argmax_label == target_class_id:
    return HireRecourseReport(
      sample_idx=sample_idx,
      adverse_class_id=initial_summary.argmax_label,
      target_class_id=target_class_id,
      score_history=[initial_summary.scores],
      margin_history=[initial_summary.margin],
      final_prediction=initial_summary.argmax_label,
      success=True,
      total_latency_ms=0.0,
      candidates_examined=0,
      notes=["already at target class"],
    )

  report = HireRecourseReport(
    sample_idx=sample_idx,
    adverse_class_id=initial_summary.argmax_label,
    target_class_id=target_class_id,
    score_history=[initial_summary.scores],
    margin_history=[initial_summary.margin],
  )

  active = set(built.per_sample_active_literals[sample_idx])
  current_built = built
  current_summary = initial_summary
  t0 = time.perf_counter()

  for step in range(max_edits):
    if (time.perf_counter() - t0) * 1000.0 >= time_budget_ms:
      report.notes.append("time-budget-exhausted")
      break
    candidates = generate_candidates(
      sample_idx=sample_idx,
      built=current_built,
      firing_clauses=list(current_summary.firing_clauses),
      adverse_class_id=current_summary.argmax_label,
      target_class_id=target_class_id,
      feasibility=feasibility,
      all_clauses_for_target=all_target_clauses,
    )
    if not candidates:
      report.notes.append(f"no-candidates-at-step-{step}")
      break
    candidates = candidates[:max_candidates_per_step]

    best_edit: RecourseEdit | None = None
    best_built: BuiltGraphs | None = None
    best_summary: ClassScoreSummary | None = None
    best_margin_for_target = -10**9

    for edit in candidates:
      report.candidates_examined += 1
      new_active = _apply_edit_to_active(active, edit)
      tentative_built = _patched_built(current_built, sample_idx, new_active)
      tentative_summary = explain_fn(tentative_built, sample_idx)
      target_score = (
        tentative_summary.scores[target_class_id]
        if 0 <= target_class_id < len(tentative_summary.scores)
        else 0
      )
      other = max(
        (tentative_summary.scores[c]
         for c in range(len(tentative_summary.scores))
         if c != target_class_id),
        default=0,
      )
      margin_for_target = target_score - other
      if margin_for_target > best_margin_for_target:
        best_margin_for_target = margin_for_target
        best_edit = edit
        best_built = tentative_built
        best_summary = tentative_summary
        if margin_for_target > 0:
          break # early-exit, we already win

    if best_edit is None or best_built is None or best_summary is None:
      report.notes.append(f"no-feasible-edit-improved-margin-at-step-{step}")
      break
    report.applied_edits.append(best_edit)
    report.score_history.append(best_summary.scores)
    report.margin_history.append(best_summary.margin)
    active = _apply_edit_to_active(active, best_edit)
    current_built = best_built
    current_summary = best_summary
    if best_summary.argmax_label == target_class_id:
      report.success = True
      report.final_prediction = target_class_id
      break
  if not report.success:
    report.final_prediction = current_summary.argmax_label
  report.total_latency_ms = (time.perf_counter() - t0) * 1000.0
  return report
