from clause86.data import load_synthetic_hiring
from clause86.graphs import CompositeGraphBuilder


def test_composite_with_text_off_falls_back_to_tabular():
  ds = load_synthetic_hiring(n_train=15, n_dev=0, n_test=0, seed=42)
  builder = CompositeGraphBuilder(
    schema=ds.schema,
    tokenizer_name=None,
    spacy_model=None,
    use_dep_edges=False,
    use_sequential_edges=False,
  )
  builder.fit(ds)
  built = builder.transform(ds, graph_backend="stub")
  # Whitespace tokenizer adds text literals; tabular adds the rest.
  assert built.n_samples == len(ds)
  assert built.n_literals > 0
  assert built.meta.get("builder") in ("CompositeGraphBuilder", "SubwordDepGraphBuilder")


def test_composite_per_sample_indices_within_bounds():
  ds = load_synthetic_hiring(n_train=15, n_dev=0, n_test=0, seed=42)
  builder = CompositeGraphBuilder(
    schema=ds.schema,
    tokenizer_name=None,
    spacy_model=None,
    use_dep_edges=False,
    use_sequential_edges=False,
  )
  builder.fit(ds)
  built = builder.transform(ds, graph_backend="stub")
  for active in built.per_sample_active_literals:
    for lid in active:
      assert 0 <= lid < built.n_literals
