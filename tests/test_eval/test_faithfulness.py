from clause86.eval import check_faithfulness
from clause86.graphs.built_graphs import BuiltGraphs, LiteralSpec
from clause86.tm.firing import ClassScoreSummary, FiringClauseSpec


def _toy_built() -> BuiltGraphs:
  specs = [
    LiteralSpec(literal_id=0, feature_name="A", bucket_or_category="v",
          feature_kind="binary", nl_template="x"),
    LiteralSpec(literal_id=1, feature_name="B", bucket_or_category="v",
          feature_kind="binary", nl_template="x"),
  ]
  return BuiltGraphs(
    graphs=None, n_samples=1, symbols=[], edge_types=[],
    literal_specs=specs, per_sample_active_literals=[(0,)],
  )


def _explain_fn(b: BuiltGraphs, idx: int) -> ClassScoreSummary:
  active = set(b.per_sample_active_literals[idx])
  if 0 in active:
    return ClassScoreSummary(
      scores=(0, 2), argmax_label=1, margin=2,
      firing_clauses=(
        FiringClauseSpec(clause_id=0, class_id=1, polarity=1, weight=2,
                 positive_literal_ids=(0,)),
      ),
    )
  return ClassScoreSummary(
    scores=(2, 0), argmax_label=0, margin=2,
    firing_clauses=(
      FiringClauseSpec(clause_id=1, class_id=0, polarity=1, weight=2),
    ),
  )


def test_faithful_clause_flips_margin_as_predicted():
  built = _toy_built()
  base = _explain_fn(built, 0)
  witnesses = [
    {"flip_literal": 0, "to_value": 0, "expected_score_delta": -2},
  ]
  res = check_faithfulness(
    built=built, sample_idx=0, baseline_summary=base,
    witnesses=witnesses, explain_fn=_explain_fn,
  )
  assert res.n_witnesses == 1
  assert res.n_strictly_correct == 1
  assert res.n_decision_flipped == 1
  assert res.mean_magnitude_error == 0.0


def test_decision_does_not_flip_for_unrelated_literal():
  built = _toy_built()
  base = _explain_fn(built, 0)
  witnesses = [
    {"flip_literal": 1, "to_value": 1, "expected_score_delta": 0},
  ]
  res = check_faithfulness(
    built=built, sample_idx=0, baseline_summary=base,
    witnesses=witnesses, explain_fn=_explain_fn,
  )
  assert res.n_decision_flipped == 0
