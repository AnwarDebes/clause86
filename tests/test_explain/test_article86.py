from clause86.explain import Article86Explainer
from clause86.tm.firing import ClassScoreSummary, FiringClauseSpec


def _stub_summary(
  n_classes: int = 2, predicted: int = 1, scores: tuple[int, ...] = (1, 3),
) -> ClassScoreSummary:
  firing = (
    FiringClauseSpec(
      clause_id=0, class_id=predicted, polarity=1, weight=1,
      positive_literal_ids=(0, 2),
    ),
    FiringClauseSpec(
      clause_id=1, class_id=predicted, polarity=1, weight=1,
      positive_literal_ids=(3,),
    ),
    FiringClauseSpec(
      clause_id=2, class_id=1 - predicted, polarity=1, weight=1,
      positive_literal_ids=(4,),
    ),
  )
  return ClassScoreSummary(
    scores=scores, argmax_label=predicted, margin=scores[predicted] - scores[1 - predicted],
    firing_clauses=firing,
  )


def _stub_built(literal_specs):
  from clause86.graphs.built_graphs import BuiltGraphs

  return BuiltGraphs(
    graphs=None, n_samples=1, symbols=[], edge_types=[],
    literal_specs=literal_specs,
    per_sample_active_literals=[tuple(range(len(literal_specs)))],
  )


def test_main_elements_pick_positive_winning_clauses():
  from clause86.graphs.built_graphs import LiteralSpec

  specs = [
    LiteralSpec(literal_id=i, feature_name=f"f{i}", bucket_or_category=f"v{i}",
          feature_kind="binary", nl_template=f"f{i} = 1")
    for i in range(5)
  ]
  built = _stub_built(specs)
  explainer = Article86Explainer(class_names=["reject", "hire"], deployer_name="X")
  exp = explainer.explain(_stub_summary(), built, 0)
  assert exp.predicted_class == 1
  assert exp.predicted_class_name == "hire"
  # Two positive-vote clauses for the winning class -> two main elements.
  assert len(exp.main_elements) == 2


def test_nl_render_contains_disclosure():
  from clause86.graphs.built_graphs import LiteralSpec

  specs = [
    LiteralSpec(literal_id=i, feature_name=f"f{i}", bucket_or_category=f"v{i}",
          feature_kind="binary", nl_template=f"f{i} = 1")
    for i in range(5)
  ]
  built = _stub_built(specs)
  explainer = Article86Explainer(class_names=["reject", "hire"], deployer_name="Acme")
  exp = explainer.explain(_stub_summary(), built, 0)
  nl = exp.render_nl(max_elements=3)
  assert "Acme" in nl
  assert "main elements" in nl.lower()


def test_faithfulness_witness_returns_one_per_main_element():
  from clause86.graphs.built_graphs import LiteralSpec

  specs = [
    LiteralSpec(literal_id=i, feature_name=f"f{i}", bucket_or_category=f"v{i}",
          feature_kind="binary", nl_template=f"f{i} = 1")
    for i in range(5)
  ]
  built = _stub_built(specs)
  explainer = Article86Explainer(class_names=["reject", "hire"], deployer_name="X")
  exp = explainer.explain(_stub_summary(), built, 0)
  lit_vec = [1] * 5
  wits = exp.faithfulness_witness(specs, lit_vec)
  assert len(wits) == len(exp.main_elements)
  for w in wits:
    assert "flip_literal" in w
    assert w["from_value"] in (0, 1)


def test_serialise_to_dict_round_trip():
  from clause86.graphs.built_graphs import LiteralSpec

  specs = [
    LiteralSpec(literal_id=i, feature_name=f"f{i}", bucket_or_category=f"v{i}",
          feature_kind="binary", nl_template=f"f{i} = 1")
    for i in range(5)
  ]
  built = _stub_built(specs)
  explainer = Article86Explainer(class_names=["reject", "hire"], deployer_name="X")
  exp = explainer.explain(_stub_summary(), built, 0)
  d = exp.to_dict()
  assert d["predicted_class"] == 1
  assert d["main_elements"]
  assert d["disclosure"]
