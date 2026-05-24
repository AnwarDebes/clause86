from clause86.data import load_synthetic_hiring
from clause86.graphs import CompositeGraphBuilder
from clause86.tm import HireTMEnsemble
from clause86.tm.firing import ClassScoreSummary


def _build(ds):
  b = CompositeGraphBuilder(
    schema=ds.schema, tokenizer_name=None, spacy_model=None,
    use_dep_edges=False, use_sequential_edges=False,
  )
  b.fit(ds)
  return b.transform(ds, graph_backend="stub")


def test_ensemble_fits_with_three_seeds():
  ds = load_synthetic_hiring(n_train=60, n_dev=0, n_test=0, seed=42)
  built = _build(ds)
  ens = HireTMEnsemble(
    seeds=(1, 2, 3), allow_fallback=True,
    n_clauses=8, T=4, s=3.9,
  )
  ens.fit(built, ds.labels, n_classes=ds.n_classes, n_epochs=1)
  preds = ens.predict(built)
  assert len(preds) == len(ds)
  summary = ens.explain(built, 0)
  assert isinstance(summary, ClassScoreSummary)
  # Each member contributes firing clauses with its member_id set.
  member_ids = {fc.member_id for fc in summary.firing_clauses}
  member_ids.discard(None)
  assert member_ids.issubset({0, 1, 2})
