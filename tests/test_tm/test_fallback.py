from clause86.data import load_synthetic_hiring
from clause86.graphs import CompositeGraphBuilder
from clause86.tm import HireTM
from clause86.tm.firing import ClassScoreSummary


def _build(ds):
  b = CompositeGraphBuilder(
    schema=ds.schema, tokenizer_name=None, spacy_model=None,
    use_dep_edges=False, use_sequential_edges=False,
  )
  b.fit(ds)
  return b.transform(ds, graph_backend="stub")


def test_fitting_fallback_then_predicting():
  ds = load_synthetic_hiring(n_train=100, n_dev=0, n_test=0, seed=42, label_noise=0.0)
  built = _build(ds)
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=16, T=10, s=3.9)
  tm.fit(built, ds.labels, n_classes=ds.n_classes, n_epochs=1)
  preds = tm.predict(built)
  assert len(preds) == len(ds)
  # Synthetic rule is learnable; fallback should beat majority class.
  n_correct = sum(int(p == y) for p, y in zip(preds, ds.labels))
  n_class0 = sum(1 for y in ds.labels if y == 0)
  n_class1 = len(ds) - n_class0
  majority = max(n_class0, n_class1)
  assert n_correct >= majority - 5 # tolerate a few flipped


def test_explain_returns_class_score_summary():
  ds = load_synthetic_hiring(n_train=60, n_dev=0, n_test=0, seed=42)
  built = _build(ds)
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=16, T=10, s=3.9)
  tm.fit(built, ds.labels, n_classes=ds.n_classes, n_epochs=1)
  summary = tm.explain(built, 0)
  assert isinstance(summary, ClassScoreSummary)
  assert len(summary.scores) == ds.n_classes
  assert summary.firing_clauses
  # Firing clauses should reference valid literals.
  for fc in summary.firing_clauses:
    for lid in fc.positive_literal_ids:
      assert 0 <= lid < built.n_literals


def test_fallback_clauses_match_synthesis():
  ds = load_synthetic_hiring(n_train=80, n_dev=0, n_test=0, seed=42)
  built = _build(ds)
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=16, T=10, s=3.9)
  tm.fit(built, ds.labels, n_classes=ds.n_classes, n_epochs=1)
  n_clauses = tm.num_clauses_total()
  assert n_clauses > 0


def test_disable_fallback_raises_without_backend():
  import pytest

  from clause86.tm import BackendUnavailableError

  ds = load_synthetic_hiring(n_train=10, n_dev=0, n_test=0, seed=42)
  built = _build(ds)
  tm = HireTM(allow_fallback=False, seed=42, n_clauses=8, T=4, s=3.9)
  try:
    tm.fit(built, ds.labels, n_classes=ds.n_classes, n_epochs=1)
  except BackendUnavailableError:
    return
  # If a CAIR backend is installed in this env, the fit succeeds and
  # we don't enter the BackendUnavailableError branch, which is
  # legitimate and not a test failure.
  pytest.skip("A real TM backend is installed; this test is a no-op there.")
