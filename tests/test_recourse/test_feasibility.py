from clause86.graphs.built_graphs import LiteralSpec
from clause86.recourse import FeasibilityChecker


def _spec(lid: int, kind: str, immutable: bool = False, name: str = "f") -> LiteralSpec:
  return LiteralSpec(
    literal_id=lid, feature_name=name, bucket_or_category="v",
    feature_kind=kind, nl_template="x", immutable=immutable,
  )


def test_immutable_blocks_every_flip():
  specs = [_spec(0, "categorical", immutable=True)]
  fc = FeasibilityChecker(literal_specs=specs)
  assert fc.may_flip(0, from_v=0, to_v=1) is False
  assert fc.may_flip(0, from_v=1, to_v=0) is False


def test_ordinal_only_flips_upward():
  specs = [_spec(0, "ordinal")]
  fc = FeasibilityChecker(literal_specs=specs)
  assert fc.may_flip(0, from_v=0, to_v=1) is True
  assert fc.may_flip(0, from_v=1, to_v=0) is False


def test_binary_default_allows_both_directions():
  specs = [_spec(0, "binary", name="has_degree")]
  fc = FeasibilityChecker(literal_specs=specs)
  assert fc.may_flip(0, from_v=0, to_v=1) is True


def test_negative_binary_features_only_remove():
  specs = [_spec(0, "binary", name="empty_resume")]
  fc = FeasibilityChecker(literal_specs=specs)
  assert fc.may_flip(0, from_v=1, to_v=0) is True
  assert fc.may_flip(0, from_v=0, to_v=1) is False


def test_text_literal_flips_upward_only():
  specs = [_spec(0, "text")]
  fc = FeasibilityChecker(literal_specs=specs)
  assert fc.may_flip(0, from_v=0, to_v=1) is True
  assert fc.may_flip(0, from_v=1, to_v=0) is False


def test_out_of_range_literal_rejected():
  specs = [_spec(0, "binary")]
  fc = FeasibilityChecker(literal_specs=specs)
  assert fc.may_flip(99, from_v=0, to_v=1) is False
