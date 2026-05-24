"""End-to-end Clause86 pipeline on a single dataset / single seed.

Glues every module together: load -> build graphs -> fit GraphTM
-> predict -> Article 86 explain -> recourse -> SAT receipt -> fairness
audit -> JSONL log. Per-dataset builder choice:

* ``synthetic``, tabular bipartite over the structured features.
* ``adult_income``, tabular bipartite.
* ``hr_analytics``, tabular bipartite.
* ``bias_in_bios``, subword + dependency text multigraph
 (no structured features other than gender, which is held out as
 the protected attribute).

Run::

  python experiments/full_pipeline.py --dataset synthetic --seed 42
  python experiments/full_pipeline.py --dataset adult_income --seed 42
  python experiments/full_pipeline.py --dataset bias_in_bios --seed 42 --max-train 10000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from clause86.data import (
  HiringDataset,
  load_adult_income,
  load_bias_in_bios,
  load_hr_analytics,
  load_synthetic_hiring,
)
from clause86.data.dataset import FeatureKind, ProtectedAttribute
from clause86.eval import (
  JsonlLogger,
  accuracy_score,
  classification_report,
  check_faithfulness,
  macro_f1_score,
)
from clause86.explain import Article86Explainer
from clause86.fairness import (
  apply_clause_deletion,
  audit_clauses,
  compute_fairness_metrics,
)
from clause86.graphs import (
  SubwordDepGraphBuilder,
  TabularBipartiteGraphBuilder,
)
from clause86.graphs.built_graphs import BuiltGraphs
from clause86.recourse import FeasibilityChecker, greedy_minimum_edit
from clause86.tm import HireTM, HireTMEnsemble
from clause86.tm.firing import FiringClauseSpec
from clause86.utils import seed_everything
from clause86.verify import FiringClauseForSAT, build_receipt


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_dataset(name: str, max_train: int | None) -> HiringDataset:
  if name == "synthetic":
    return load_synthetic_hiring(
      n_train=max_train or 400, n_dev=80, n_test=80, seed=42,
    )
  if name == "bias_in_bios":
    return load_bias_in_bios(max_samples_per_split=max_train)
  if name == "adult_income":
    return load_adult_income()
  if name == "hr_analytics":
    ds = load_hr_analytics()
    n = len(ds)
    cut = int(n * 0.8)
    ds.split[:cut] = ["train"] * cut
    ds.split[cut:] = ["test"] * (n - cut)
    return ds
  raise ValueError(f"Unknown dataset: {name!r}")


def _favourable_class(name: str) -> int:
  return {
    "synthetic": 1,
    "bias_in_bios": 0,
    "adult_income": 1,
    "hr_analytics": 0,
  }.get(name, 1)


def _primary_attribute(name: str) -> ProtectedAttribute:
  return {
    "synthetic": ProtectedAttribute.GENDER,
    "bias_in_bios": ProtectedAttribute.GENDER,
    "adult_income": ProtectedAttribute.GENDER,
    "hr_analytics": ProtectedAttribute.GENDER,
  }.get(name, ProtectedAttribute.GENDER)


def _build_for_dataset(
  *,
  dataset: str,
  ds_train: HiringDataset,
  ds_test: HiringDataset,
  graph_backend: str,
) -> tuple[BuiltGraphs, BuiltGraphs, str]:
  """Pick a single-mode graph builder per dataset and run it."""
  if dataset == "bias_in_bios":
    builder = SubwordDepGraphBuilder(
      tokenizer_name="bert-base-uncased",
      spacy_model=None, # dep edges added in phase 1b
      max_subwords=64,
      use_sequential_edges=True,
      use_dep_edges=False,
      min_token_freq=3,
      max_vocab=2000,
    )
    train_texts = [r.get("bio", "") for r in ds_train.records]
    test_texts = [r.get("bio", "") for r in ds_test.records]
    builder.fit(train_texts)
    train_built = builder.transform(train_texts, graph_backend=graph_backend)
    test_built = builder.transform(
      test_texts, graph_backend=graph_backend, init_with=train_built.graphs,
    )
    return train_built, test_built, "subword_dep"

  # Tabular path for synthetic / adult / hr_analytics.
  schema = [s for s in ds_train.schema if s.kind != FeatureKind.TEXT]
  builder = TabularBipartiteGraphBuilder(schema=schema)
  builder.fit(ds_train.records)
  train_built = builder.transform(ds_train.records, graph_backend=graph_backend)
  test_built = builder.transform(
    ds_test.records, graph_backend=graph_backend, init_with=train_built.graphs,
  )
  return train_built, test_built, "tabular_bipartite"


def _safe_split(ds: HiringDataset) -> tuple[HiringDataset, HiringDataset]:
  train_idx = ds.split_indices("train")
  test_idx = ds.split_indices("test") or ds.split_indices("dev")
  if not test_idx:
    n = len(ds)
    cut = int(n * 0.8)
    train_idx, test_idx = list(range(cut)), list(range(cut, n))
  return ds.select(train_idx), ds.select(test_idx)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run(
  *,
  dataset: str,
  seed: int,
  out_dir: str | Path,
  max_train: int | None = None,
  n_epochs: int = 10,
  n_clauses: int = 200,
  T: int = 100,
  s: float = 5.0,
  depth: int = 2,
  ensemble: bool = False,
  n_seeds: int = 5,
  max_recourse: int = 3,
  max_recourse_samples: int = 50,
  max_explain_samples: int = 50,
  allow_fallback: bool = True,
  graph_backend: str = "auto",
) -> dict:
  seed_everything(seed)
  out_dir = Path(out_dir) / f"seed_{seed}"
  out_dir.mkdir(parents=True, exist_ok=True)
  log_path = out_dir / "log.jsonl"
  receipts_dir = out_dir / "receipts"
  receipts_dir.mkdir(exist_ok=True)
  logger = JsonlLogger(log_path, seed=seed, run_id=f"{dataset}_seed{seed}")

  t0 = time.perf_counter()
  ds = _load_dataset(dataset, max_train=max_train)
  ds_train, ds_test = _safe_split(ds)

  logger.log("data", {
    "name": ds.name,
    "n_train": len(ds_train),
    "n_test": len(ds_test),
    "n_classes": ds.n_classes,
    "meta": ds.meta,
  })

  train_built, test_built, builder_name = _build_for_dataset(
    dataset=dataset, ds_train=ds_train, ds_test=ds_test,
    graph_backend=graph_backend,
  )
  logger.log("graphs", {
    "builder": builder_name,
    "n_literals": train_built.n_literals,
    "n_symbols": len(train_built.symbols),
    "n_edge_types": len(train_built.edge_types),
    "n_train_samples": train_built.n_samples,
    "n_test_samples": test_built.n_samples,
  })

  if ensemble:
    model: HireTMEnsemble | HireTM = HireTMEnsemble(
      n_clauses=n_clauses, T=T, s=s, depth=depth,
      seeds=tuple(seed + i for i in range(n_seeds)),
      allow_fallback=allow_fallback,
    )
  else:
    model = HireTM(
      n_clauses=n_clauses, T=T, s=s, depth=depth, seed=seed,
      allow_fallback=allow_fallback,
    )
  fit_start = time.perf_counter()
  model.fit(train_built, ds_train.labels, n_epochs=n_epochs, n_classes=ds.n_classes)
  fit_seconds = time.perf_counter() - fit_start

  backend_kind = getattr(
    getattr(model, "_backend_kind", "ensemble"),
    "__str__", lambda: ""
  )()
  if isinstance(model, HireTMEnsemble) and model.members:
    backend_kind = model.members[0]._backend_kind
  else:
    backend_kind = getattr(model, "_backend_kind", "unknown")

  test_preds = model.predict(test_built)
  report = classification_report(ds_test.labels, test_preds, n_classes=ds.n_classes)
  logger.log("eval.classification_report", {
    "backend_kind": backend_kind,
    "fit_seconds": fit_seconds,
    **report.to_dict(),
  })

  explainer = Article86Explainer(
    class_names=ds.class_names,
    deployer_name="Demo Deployer",
  )
  feasibility = FeasibilityChecker(literal_specs=test_built.literal_specs)
  fav = _favourable_class(dataset)
  n_test = test_built.n_samples
  recourse_success = 0
  recourse_attempts = 0
  recourse_latencies: list[float] = []
  faithfulness_hits = 0
  faithfulness_total = 0

  # The CAIR-backed explain() is a pure-Python pass over all clauses.
  # On a 16k-sample test set with 200 clauses that's ~50 seconds at
  # best, we sub-sample to the audit budget below to keep the
  # 5-seed protocol tractable. Accuracy + macro-F1 above use the
  # full test set via fast CUDA predict().
  audit_budget = max(max_explain_samples, max_recourse_samples)
  audit_indices = list(range(min(audit_budget, n_test)))

  all_summaries = []
  for i in audit_indices:
    summary = model.explain(test_built, i)
    all_summaries.append(summary)
    if i < max_explain_samples:
      exp = explainer.explain(summary, test_built, i)
      logger.log("explain.article86", {
        "sample_idx": i,
        "label": ds_test.labels[i],
        "predicted": summary.argmax_label,
        "explanation": exp.to_dict(),
      })

      lit_vec = test_built.get_active_literal_vector(i)
      witnesses = exp.faithfulness_witness(test_built.literal_specs, lit_vec)
      fres = check_faithfulness(
        built=test_built,
        sample_idx=i,
        baseline_summary=summary,
        witnesses=witnesses,
        explain_fn=lambda b, idx: model.explain(b, idx),
      )
      logger.log("eval.faithfulness", fres.to_dict())
      faithfulness_hits += fres.n_strictly_correct
      faithfulness_total += fres.n_witnesses

      firing_sat = [
        FiringClauseForSAT(
          clause_id=fc.clause_id,
          class_id=fc.class_id,
          polarity=fc.polarity,
          weight=fc.weight,
          positive_literal_ids=fc.positive_literal_ids,
          negative_literal_ids=fc.negative_literal_ids,
        )
        for fc in summary.firing_clauses
      ]
      receipt = build_receipt(
        sample_id=f"{dataset}_seed{seed}_sample{i}",
        dataset_name=dataset,
        model_version=f"clause86-v0.1.0-seed{seed}",
        deployer_name="Demo Deployer",
        issued_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        literal_vector=lit_vec,
        firing_clauses=firing_sat,
        class_scores=list(summary.scores),
        predicted_class=summary.argmax_label,
        margin=summary.margin,
        class_names=ds.class_names,
        cnf_dir=receipts_dir,
      )
      (receipts_dir / f"{receipt.sample_id}.receipt.json").write_text(
        json.dumps(receipt.__dict__, ensure_ascii=False, default=str),
        encoding="utf-8",
      )

    if i < max_recourse_samples and summary.argmax_label != fav:
      recourse_attempts += 1
      r_start = time.perf_counter()
      rep = greedy_minimum_edit(
        built=test_built,
        sample_idx=i,
        initial_summary=summary,
        target_class_id=fav,
        explain_fn=lambda b, idx: model.explain(b, idx),
        feasibility=feasibility,
        max_edits=max_recourse,
      )
      recourse_latencies.append((time.perf_counter() - r_start) * 1000)
      if rep.success:
        recourse_success += 1
      logger.log("recourse.greedy", rep.to_dict())

  primary_attr = _primary_attribute(dataset)
  fairness = compute_fairness_metrics(
    test_preds, ds_test.labels, ds_test.protected,
    primary_attr, favourable_class_id=fav,
  )
  logger.log("fairness.metrics", fairness.to_dict())

  # Sub-sample protected + labels for clause-level audit + intervention,
  # matching audit_indices.
  audit_labels = [ds_test.labels[i] for i in audit_indices]
  audit_protected = [ds_test.protected[i] for i in audit_indices]
  # Restricted BuiltGraphs view: same backend graphs, restricted
  # per-sample literals for the audit subset only.
  audit_built_active = [test_built.per_sample_active_literals[i] for i in audit_indices]

  union_clauses: list[FiringClauseSpec] = []
  seen: set[int] = set()
  for s in all_summaries:
    for fc in s.firing_clauses:
      if fc.clause_id not in seen:
        seen.add(fc.clause_id)
        union_clauses.append(fc)
  audit_built = BuiltGraphs(
    graphs=test_built.graphs,
    n_samples=len(audit_indices),
    symbols=test_built.symbols,
    edge_types=test_built.edge_types,
    literal_specs=test_built.literal_specs,
    per_sample_active_literals=audit_built_active,
    meta={**test_built.meta, "audit_subset_size": len(audit_indices)},
  )
  audit = audit_clauses(
    clauses=union_clauses,
    built=audit_built,
    protected=audit_protected,
    attribute=primary_attr,
    suspect_threshold=0.1,
  )
  logger.log("fairness.audit", audit.to_dict())

  intervention_dict = None
  if audit.suspect_clause_ids:
    delete_n = min(50, len(audit.suspect_clause_ids))
    intervention = apply_clause_deletion(
      summaries=all_summaries,
      labels=audit_labels,
      protected=audit_protected,
      attribute=primary_attr,
      deleted_clause_ids=audit.suspect_clause_ids[:delete_n],
      favourable_class_id=fav,
      n_classes=ds.n_classes,
    )
    intervention_dict = intervention.to_dict()
    logger.log("fairness.intervention", intervention_dict)

  def _pct(idx: int) -> float:
    if not recourse_latencies:
      return 0.0
    s = sorted(recourse_latencies)
    return s[min(len(s) - 1, int(idx * (len(s) - 1)))]

  summary_payload = {
    "dataset": dataset,
    "seed": seed,
    "backend_kind": backend_kind,
    "n_train": len(ds_train),
    "n_test": n_test,
    "n_literals": train_built.n_literals,
    "accuracy": report.accuracy,
    "macro_f1": report.macro_f1,
    "fit_seconds": fit_seconds,
    "demographic_parity_gap": fairness.demographic_parity_gap,
    "equal_opportunity_gap": fairness.equal_opportunity_gap,
    "equalized_odds_gap": fairness.equalized_odds_gap,
    "recourse_attempts": recourse_attempts,
    "recourse_success": recourse_success,
    "recourse_success_rate": (
      recourse_success / recourse_attempts if recourse_attempts else 0.0
    ),
    "recourse_p50_ms": _pct(0.5),
    "recourse_p95_ms": _pct(0.95),
    "faithfulness_witnesses": faithfulness_total,
    "faithfulness_strict_hits": faithfulness_hits,
    "faithfulness_strict_rate": (
      faithfulness_hits / faithfulness_total if faithfulness_total else 0.0
    ),
    "n_suspect_clauses": len(audit.suspect_clause_ids),
    "accuracy_after_intervention": (
      intervention_dict["accuracy_after"] if intervention_dict else report.accuracy
    ),
    "wall_clock_s": time.perf_counter() - t0,
  }
  (out_dir / "summary.json").write_text(
    json.dumps(summary_payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
  )
  logger.log("summary", summary_payload)
  logger.close()
  return summary_payload


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", required=True,
            choices=["synthetic", "bias_in_bios", "adult_income", "hr_analytics"])
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--out-dir", default="experiments/paper_d_synthetic")
  parser.add_argument("--max-train", type=int, default=None)
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--n-clauses", type=int, default=200)
  parser.add_argument("--T", type=int, default=100)
  parser.add_argument("--s", type=float, default=5.0)
  parser.add_argument("--depth", type=int, default=2)
  parser.add_argument("--ensemble", action="store_true")
  parser.add_argument("--n-seeds", type=int, default=5)
  parser.add_argument("--allow-fallback", action="store_true", default=False)
  parser.add_argument("--graph-backend", default="auto",
            choices=["auto", "stub"])
  parser.add_argument("--max-explain", type=int, default=50)
  parser.add_argument("--max-recourse-samples", type=int, default=50)
  args = parser.parse_args()
  out = run(
    dataset=args.dataset, seed=args.seed, out_dir=args.out_dir,
    max_train=args.max_train, n_epochs=args.n_epochs,
    n_clauses=args.n_clauses, T=args.T, s=args.s, depth=args.depth,
    ensemble=args.ensemble, n_seeds=args.n_seeds,
    allow_fallback=args.allow_fallback,
    graph_backend=args.graph_backend,
    max_explain_samples=args.max_explain,
    max_recourse_samples=args.max_recourse_samples,
  )
  print(json.dumps(out, indent=2))


if __name__ == "__main__":
  main()
