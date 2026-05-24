from clause86.data.dataset import ProtectedAttribute
from clause86.fairness import audit_clauses
from clause86.graphs.built_graphs import BuiltGraphs, LiteralSpec
from clause86.tm.firing import FiringClauseSpec


def test_audit_flags_group_skewed_clause():
  specs = [
    LiteralSpec(literal_id=0, feature_name="years_experience",
          bucket_or_category=">=3", feature_kind="ordinal",
          nl_template="x"),
    LiteralSpec(literal_id=1, feature_name="gender",
          bucket_or_category="F", feature_kind="immutable",
          nl_template="x", immutable=True),
  ]
  # 4 samples: 2 female with literal_0; 2 male without literal_0.
  per_sample_active = [(0, 1), (0, 1), (1,), (1,)]
  built = BuiltGraphs(
    graphs=None, n_samples=4, symbols=[], edge_types=[],
    literal_specs=specs,
    per_sample_active_literals=[
      tuple(sorted(a)) for a in [(0, 1), (0, 1), (), ()]
    ],
  )
  protected = [
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "M"},
    {ProtectedAttribute.GENDER: "M"},
  ]
  clauses = [
    FiringClauseSpec(clause_id=11, class_id=1, polarity=1, weight=1,
             positive_literal_ids=(0,)),
  ]
  audit = audit_clauses(
    clauses=clauses, built=built, protected=protected,
    attribute=ProtectedAttribute.GENDER, suspect_threshold=0.1,
  )
  assert 11 in audit.suspect_clause_ids
  assert audit.per_clause[11]["gap"] == 1.0


def test_audit_no_suspect_for_balanced_clause():
  specs = [
    LiteralSpec(literal_id=0, feature_name="years_experience",
          bucket_or_category=">=3", feature_kind="ordinal",
          nl_template="x"),
  ]
  built = BuiltGraphs(
    graphs=None, n_samples=4, symbols=[], edge_types=[],
    literal_specs=specs,
    per_sample_active_literals=[(0,), (0,), (0,), (0,)],
  )
  protected = [
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "M"},
    {ProtectedAttribute.GENDER: "M"},
  ]
  clauses = [
    FiringClauseSpec(clause_id=22, class_id=1, polarity=1, weight=1,
             positive_literal_ids=(0,)),
  ]
  audit = audit_clauses(
    clauses=clauses, built=built, protected=protected,
    attribute=ProtectedAttribute.GENDER, suspect_threshold=0.1,
  )
  assert audit.suspect_clause_ids == []
