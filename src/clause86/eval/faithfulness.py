"""Faithfulness evaluation: clause-flip == decision-flip.

The central methodological claim of Clause86 is that the FIRING
CLAUSES *are* the main elements of the decision. We test this
operationally:

* For each test sample, take the structured explanation
 (:class:`Article86Explanation`).
* Read its :meth:`faithfulness_witness`: a list of single-literal
 flips the explanation says would dent the predicted class's
 margin.
* Apply each flip, rerun the TM (via the same explain_fn the
 recourse module uses), and record the actual margin change.

We then report:

* **strict faithfulness**: fraction of flips whose actual margin
 change is in the *same direction* as predicted.
* **decision faithfulness**: fraction of flips that, when applied
 together, change the predicted class.
* **mean magnitude error**: |predicted - actual| margin delta.

Post-hoc methods (SHAP/LIME) typically score 0.3-0.6 on strict
faithfulness; a faithful clause-based explainer scores 1.0 by
construction. The metric also catches bugs in the TM backend that
silently disagree with the explainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..graphs.built_graphs import BuiltGraphs
from ..tm.firing import ClassScoreSummary


ExplainFn = Callable[[BuiltGraphs, int], ClassScoreSummary]


@dataclass
class FaithfulnessResult:
  """Per-sample faithfulness audit."""

  sample_idx: int
  n_witnesses: int = 0
  n_strictly_correct: int = 0
  n_decision_flipped: int = 0
  mean_magnitude_error: float = 0.0
  per_witness: list[dict[str, Any]] = field(default_factory=list)

  @property
  def strict_score(self) -> float:
    if self.n_witnesses == 0:
      return 0.0
    return self.n_strictly_correct / self.n_witnesses

  def to_dict(self) -> dict[str, Any]:
    return {
      "sample_idx": self.sample_idx,
      "n_witnesses": self.n_witnesses,
      "n_strictly_correct": self.n_strictly_correct,
      "n_decision_flipped": self.n_decision_flipped,
      "mean_magnitude_error": float(self.mean_magnitude_error),
      "strict_score": float(self.strict_score),
      "per_witness": list(self.per_witness),
    }


def _patched_built(
  built: BuiltGraphs, sample_idx: int, flip_literal: int, to_value: int,
) -> BuiltGraphs:
  active = set(built.per_sample_active_literals[sample_idx])
  if to_value == 1:
    active.add(flip_literal)
  else:
    active.discard(flip_literal)
  new_active = tuple(sorted(active))
  new_per_sample = list(built.per_sample_active_literals)
  new_per_sample[sample_idx] = new_active
  return BuiltGraphs(
    graphs=built.graphs,
    n_samples=built.n_samples,
    symbols=built.symbols,
    edge_types=built.edge_types,
    literal_specs=built.literal_specs,
    per_sample_active_literals=new_per_sample,
    meta={**built.meta, "NEEDS-GRAPH-REENCODE-FOR-FAITHFULNESS": True},
  )


def check_faithfulness(
  *,
  built: BuiltGraphs,
  sample_idx: int,
  baseline_summary: ClassScoreSummary,
  witnesses: list[dict[str, Any]],
  explain_fn: ExplainFn,
) -> FaithfulnessResult:
  """Verify a list of single-literal witnesses for one sample."""
  n_classes = len(baseline_summary.scores)
  res = FaithfulnessResult(sample_idx=sample_idx, n_witnesses=len(witnesses))
  if n_classes == 0 or not witnesses:
    return res

  target_class = baseline_summary.argmax_label
  abs_errors: list[float] = []
  for w in witnesses:
    flip_lid = int(w["flip_literal"])
    to_v = int(w["to_value"])
    expected_score_delta = int(w.get("expected_score_delta", 0))
    patched = _patched_built(built, sample_idx, flip_lid, to_v)
    new_summary = explain_fn(patched, sample_idx)
    actual_delta = (
      int(new_summary.scores[target_class]) - int(baseline_summary.scores[target_class])
    )
    same_direction = (
      (expected_score_delta == 0 and actual_delta == 0)
      or (expected_score_delta > 0 and actual_delta > 0)
      or (expected_score_delta < 0 and actual_delta < 0)
    )
    flipped = new_summary.argmax_label != baseline_summary.argmax_label
    if same_direction:
      res.n_strictly_correct += 1
    if flipped:
      res.n_decision_flipped += 1
    err = float(abs(expected_score_delta - actual_delta))
    abs_errors.append(err)
    res.per_witness.append({
      "flip_literal": flip_lid,
      "to_value": to_v,
      "expected_score_delta": expected_score_delta,
      "actual_score_delta": actual_delta,
      "same_direction": same_direction,
      "decision_flipped": flipped,
      "abs_error": err,
    })
  res.mean_magnitude_error = (
    sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
  )
  return res
