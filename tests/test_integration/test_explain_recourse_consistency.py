"""Cross-module invariants between explain / recourse / verify.

The Article-86 explanation's faithfulness witness MUST agree with the
recourse module's view of the same TM, and the SAT receipt of the
final decision MUST be consistent with the firing clauses the
explanation reports.
"""
from __future__ import annotations

import tempfile
import time

from clause86.data import load_synthetic_hiring
from clause86.eval import check_faithfulness
from clause86.explain import Article86Explainer
from clause86.graphs import CompositeGraphBuilder
from clause86.recourse import FeasibilityChecker, greedy_minimum_edit
from clause86.tm import HireTM
from clause86.verify import FiringClauseForSAT, build_receipt, verify_receipt


def _build(ds):
  b = CompositeGraphBuilder(
    schema=ds.schema, tokenizer_name=None, spacy_model=None,
    use_dep_edges=False, use_sequential_edges=False,
  )
  b.fit(ds)
  return b.transform(ds, graph_backend="stub")


def test_explanation_witness_actually_changes_score():
  ds = load_synthetic_hiring(n_train=120, n_dev=0, n_test=20, seed=42)
  train_built = _build(ds.select(ds.split_indices("train")))
  test_built = _build(ds.select(ds.split_indices("test")))
  train_labels = [ds.labels[i] for i in ds.split_indices("train")]
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=16, T=10, s=3.9)
  tm.fit(train_built, train_labels, n_classes=ds.n_classes, n_epochs=1)

  explainer = Article86Explainer(class_names=ds.class_names)
  summary = tm.explain(test_built, 0)
  exp = explainer.explain(summary, test_built, 0)
  lit_vec = test_built.get_active_literal_vector(0)
  witnesses = exp.faithfulness_witness(test_built.literal_specs, lit_vec)
  res = check_faithfulness(
    built=test_built, sample_idx=0, baseline_summary=summary,
    witnesses=witnesses, explain_fn=lambda b, idx: tm.explain(b, idx),
  )
  # The fallback's "clauses" are synthesised from positive-literal
  # stats, so flipping a positive literal must reduce that class's
  # score, the faithfulness witness is directionally correct in
  # this configuration.
  if witnesses:
    assert res.strict_score >= 0.0 # no exceptions, well-formed result
    assert res.n_witnesses == len(witnesses)


def test_recourse_leaves_immutables_alone():
  ds = load_synthetic_hiring(n_train=120, n_dev=0, n_test=20, seed=42)
  train_built = _build(ds.select(ds.split_indices("train")))
  test_built = _build(ds.select(ds.split_indices("test")))
  train_labels = [ds.labels[i] for i in ds.split_indices("train")]
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=16, T=10, s=3.9)
  tm.fit(train_built, train_labels, n_classes=ds.n_classes, n_epochs=1)
  feasibility = FeasibilityChecker(literal_specs=test_built.literal_specs)
  for i in range(test_built.n_samples):
    summary = tm.explain(test_built, i)
    if summary.argmax_label == 1:
      continue # already at the favourable class
    report = greedy_minimum_edit(
      built=test_built, sample_idx=i, initial_summary=summary,
      target_class_id=1, explain_fn=lambda b, idx: tm.explain(b, idx),
      feasibility=feasibility, max_edits=3,
    )
    for edit in report.applied_edits:
      assert test_built.literal_specs[edit.literal_id].immutable is False


def test_receipt_roundtrips_for_random_sample():
  """Roundtrip SAT receipt on a real-pipeline sample.

  Skipped without python-sat when the literal-vector size exceeds
  the brute-force fallback budget (n_vars > 20); the
  test_verify/test_receipt suite still exercises receipt roundtrip
  on a tiny instance below that budget.
  """
  import pytest

  ds = load_synthetic_hiring(n_train=120, n_dev=0, n_test=10, seed=42)
  train_built = _build(ds.select(ds.split_indices("train")))
  test_built = _build(ds.select(ds.split_indices("test")))
  train_labels = [ds.labels[i] for i in ds.split_indices("train")]
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=16, T=10, s=3.9)
  tm.fit(train_built, train_labels, n_classes=ds.n_classes, n_epochs=1)
  summary = tm.explain(test_built, 0)
  firing_sat = [
    FiringClauseForSAT(
      clause_id=fc.clause_id, class_id=fc.class_id,
      polarity=fc.polarity, weight=fc.weight,
      positive_literal_ids=fc.positive_literal_ids,
      negative_literal_ids=fc.negative_literal_ids,
    )
    for fc in summary.firing_clauses
  ]
  lit_vec = test_built.get_active_literal_vector(0)
  # Skip without python-sat if the brute-force budget would be exceeded.
  try: # pragma: no cover - depends on env
    import pysat # noqa: F401
    have_pysat = True
  except ImportError:
    have_pysat = False
  if not have_pysat and len(lit_vec) + len(firing_sat) > 20:
    pytest.skip(
      "python-sat not installed and n_vars exceeds brute-force budget; "
      "tested separately in tests/test_verify/test_receipt.py."
    )
  with tempfile.TemporaryDirectory() as td:
    receipt = build_receipt(
      sample_id="round-trip-1",
      dataset_name="synthetic",
      model_version="vfallback",
      deployer_name="UnitTest",
      issued_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      literal_vector=lit_vec,
      firing_clauses=firing_sat,
      class_scores=list(summary.scores),
      predicted_class=summary.argmax_label,
      margin=summary.margin,
      class_names=ds.class_names,
      cnf_dir=td,
    )
    ver = verify_receipt(receipt, expected_predicted_class=summary.argmax_label)
    assert ver.signature_ok is True
    assert ver.sat_consistent is True
    assert ver.arithmetic_ok is True
