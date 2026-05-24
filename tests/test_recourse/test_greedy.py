from clause86.graphs.built_graphs import BuiltGraphs, LiteralSpec
from clause86.recourse import FeasibilityChecker, greedy_minimum_edit
from clause86.tm.firing import ClassScoreSummary, FiringClauseSpec


def _toy_built() -> BuiltGraphs:
  specs = [
    LiteralSpec(literal_id=0, feature_name="years_experience",
          bucket_or_category=">=3", feature_kind="ordinal",
          nl_template="x"),
    LiteralSpec(literal_id=1, feature_name="has_degree",
          bucket_or_category="1", feature_kind="binary",
          nl_template="x"),
    LiteralSpec(literal_id=2, feature_name="empty_resume",
          bucket_or_category="1", feature_kind="binary",
          nl_template="x"),
    LiteralSpec(literal_id=3, feature_name="gender",
          bucket_or_category="F", feature_kind="immutable",
          nl_template="x", immutable=True),
  ]
  return BuiltGraphs(
    graphs=None, n_samples=1, symbols=[], edge_types=[],
    literal_specs=specs,
    per_sample_active_literals=[(2, 3)], # empty resume, female
  )


def _explain_fn(b: BuiltGraphs, idx: int) -> ClassScoreSummary:
  """Toy oracle: HIRE (class 1) iff (years_exp>=3 AND has_degree) AND NOT empty_resume."""
  active = set(b.per_sample_active_literals[idx])
  years = 0 in active
  degree = 1 in active
  empty = 2 in active
  target = 1 if (years and degree and not empty) else 0
  if target == 1:
    scores = (0, 2)
    firing = (
      FiringClauseSpec(clause_id=0, class_id=1, polarity=1, weight=1,
               positive_literal_ids=(0, 1),
               negative_literal_ids=(2,)),
    )
  else:
    scores = (2, 0)
    firing = (
      FiringClauseSpec(clause_id=1, class_id=0, polarity=1, weight=1,
               positive_literal_ids=(2,)),
      FiringClauseSpec(clause_id=2, class_id=0, polarity=1, weight=1,
               negative_literal_ids=(1,)),
    )
  return ClassScoreSummary(
    scores=scores, argmax_label=target,
    margin=abs(scores[1] - scores[0]),
    firing_clauses=firing,
  )


def test_greedy_recourse_flips_decision():
  built = _toy_built()
  initial = _explain_fn(built, 0)
  assert initial.argmax_label == 0
  target_clauses = [
    FiringClauseSpec(clause_id=0, class_id=1, polarity=1, weight=1,
             positive_literal_ids=(0, 1),
             negative_literal_ids=(2,)),
  ]
  report = greedy_minimum_edit(
    built=built, sample_idx=0, initial_summary=initial, target_class_id=1,
    explain_fn=_explain_fn,
    all_target_clauses=target_clauses,
    feasibility=FeasibilityChecker(literal_specs=built.literal_specs),
    max_edits=3,
  )
  assert report.success is True
  # The applied edits must avoid the immutable gender literal.
  for edit in report.applied_edits:
    assert built.literal_specs[edit.literal_id].immutable is False


def test_recourse_already_target_short_circuits():
  built = _toy_built()
  initial = ClassScoreSummary(
    scores=(0, 5), argmax_label=1, margin=5, firing_clauses=tuple(),
  )
  report = greedy_minimum_edit(
    built=built, sample_idx=0, initial_summary=initial, target_class_id=1,
    explain_fn=_explain_fn, max_edits=3,
  )
  assert report.success is True
  assert report.applied_edits == []
