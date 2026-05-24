from clause86.data import load_synthetic_hiring
from clause86.data.dataset import FeatureKind
from clause86.graphs import TabularBipartiteGraphBuilder


def _tab_schema_only(schema):
  return [s for s in schema if s.kind != FeatureKind.TEXT]


def test_tabular_builder_fit_creates_literals():
  ds = load_synthetic_hiring(n_train=30, n_dev=0, n_test=0, seed=42)
  builder = TabularBipartiteGraphBuilder(schema=_tab_schema_only(ds.schema))
  builder.fit(ds.records)
  built = builder.transform(ds.records, graph_backend="stub")
  assert built.n_samples == 30
  assert built.n_literals > 0
  # Every literal has a feature_name.
  for spec in built.literal_specs:
    assert spec.feature_name


def test_tabular_immutables_are_marked():
  ds = load_synthetic_hiring(n_train=30, n_dev=0, n_test=0, seed=42)
  builder = TabularBipartiteGraphBuilder(schema=_tab_schema_only(ds.schema))
  builder.fit(ds.records)
  built = builder.transform(ds.records, graph_backend="stub")
  immutable_literals = [s for s in built.literal_specs if s.immutable]
  assert immutable_literals
  # gender + age_group are the two immutables in the synthetic schema.
  immutable_feature_names = {s.feature_name for s in immutable_literals}
  assert immutable_feature_names == {"gender", "age_group"}


def test_tabular_per_sample_active_literals_in_bounds():
  ds = load_synthetic_hiring(n_train=20, n_dev=0, n_test=0, seed=42)
  builder = TabularBipartiteGraphBuilder(schema=_tab_schema_only(ds.schema))
  builder.fit(ds.records)
  built = builder.transform(ds.records, graph_backend="stub")
  for active in built.per_sample_active_literals:
    for lid in active:
      assert 0 <= lid < built.n_literals


def test_ordinal_thresholds_are_monotonic_and_unique():
  ds = load_synthetic_hiring(n_train=80, n_dev=0, n_test=0, seed=42)
  builder = TabularBipartiteGraphBuilder(schema=_tab_schema_only(ds.schema))
  builder.fit(ds.records)
  ts = builder.ordinal_thresholds.get("years_experience", [])
  assert ts == sorted(set(ts))


def test_get_active_literal_vector_round_trip():
  ds = load_synthetic_hiring(n_train=10, n_dev=0, n_test=0, seed=42)
  builder = TabularBipartiteGraphBuilder(schema=_tab_schema_only(ds.schema))
  builder.fit(ds.records)
  built = builder.transform(ds.records, graph_backend="stub")
  vec = built.get_active_literal_vector(0)
  assert sum(vec) == len(built.per_sample_active_literals[0])
