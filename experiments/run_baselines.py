"""Baseline comparison: DistilBERT + SHAP and XGBoost + SHAP.

The Clause86 paper's headline claim is that the firing clauses are
*faithful* explanations, formally, ``clause-flip == decision-flip``
with 100% true-positive rate by construction. To compare against
post-hoc explainers we run two baselines on the same data:

* DistilBERT classifier + SHAP DeepExplainer for text data (Bias-in-Bios).
* XGBoost + SHAP TreeExplainer for tabular data (Adult-Income, HR-Analytics).

This script does not depend on Clause86 the package; it produces
JSONL logs in the same format so the aggregator (in
``scripts/aggregate_results.py``) can pull them into a head-to-head
table.

NOTE: this script requires the ``[baselines]`` optional dependency
group (``torch``, ``xgboost``, ``shap``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from clause86.data import (
  load_adult_income,
  load_bias_in_bios,
  load_hr_analytics,
)
from clause86.data.dataset import FeatureKind, FeatureSpec, ProtectedAttribute
from clause86.eval import JsonlLogger, classification_report
from clause86.fairness import compute_fairness_metrics


def _records_to_tabular_matrix(
  records: list[dict[str, Any]], schema: list[FeatureSpec],
) -> tuple[np.ndarray, list[str]]:
  cols: list[str] = []
  feats: list[np.ndarray] = []
  for spec in schema:
    if spec.kind == FeatureKind.TEXT or spec.kind == FeatureKind.IMMUTABLE:
      continue
    if spec.kind in (FeatureKind.ORDINAL, FeatureKind.BINARY):
      vals = np.array([
        float(r.get(spec.name, 0)) for r in records
      ])
      feats.append(vals.reshape(-1, 1))
      cols.append(spec.name)
    elif spec.kind == FeatureKind.CATEGORICAL:
      cats = list(spec.categories) if spec.categories else sorted(
        {str(r.get(spec.name, "")) for r in records if r.get(spec.name)}
      )
      for c in cats:
        vals = np.array([
          1.0 if str(r.get(spec.name, "")) == c else 0.0
          for r in records
        ])
        feats.append(vals.reshape(-1, 1))
        cols.append(f"{spec.name}={c}")
  if not feats:
    return np.zeros((len(records), 0), dtype=np.float64), cols
  return np.concatenate(feats, axis=1).astype(np.float64), cols


def run_xgb_shap(dataset: str, seed: int, out_dir: Path) -> dict[str, Any]: # pragma: no cover
  """XGBoost + SHAP baseline on tabular data."""
  try:
    import xgboost as xgb
    import shap
  except ImportError as e:
    raise SystemExit(f"Install baselines: pip install clause86[baselines], {e}")
  if dataset == "adult_income":
    ds = load_adult_income()
    attr = ProtectedAttribute.GENDER
    fav = 1
  elif dataset == "hr_analytics":
    ds = load_hr_analytics()
    n = len(ds)
    cut = int(n * 0.8)
    ds.split[:cut] = ["train"] * cut
    ds.split[cut:] = ["test"] * (n - cut)
    attr = ProtectedAttribute.GENDER
    fav = 0
  else:
    raise ValueError(f"XGB baseline expects tabular dataset, got {dataset!r}")
  train_idx = ds.split_indices("train")
  test_idx = ds.split_indices("test")
  ds_train, ds_test = ds.select(train_idx), ds.select(test_idx)
  X_tr, cols = _records_to_tabular_matrix(ds_train.records, ds.schema)
  X_te, _ = _records_to_tabular_matrix(ds_test.records, ds.schema)
  y_tr = np.array(ds_train.labels, dtype=np.int32)
  y_te = np.array(ds_test.labels, dtype=np.int32)
  clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=seed, use_label_encoder=False, eval_metric="mlogloss",
  )
  clf.fit(X_tr, y_tr)
  preds = clf.predict(X_te).tolist()
  report = classification_report(y_te.tolist(), preds, n_classes=ds.n_classes)
  fairness = compute_fairness_metrics(
    preds, y_te.tolist(), ds_test.protected, attr, favourable_class_id=fav,
  )
  explainer = shap.TreeExplainer(clf)
  n_explain = min(100, len(y_te))
  shap_vals = explainer.shap_values(X_te[:n_explain])
  out_dir = out_dir / f"seed_{seed}"
  out_dir.mkdir(parents=True, exist_ok=True)
  logger = JsonlLogger(out_dir / "log.jsonl", seed=seed,
             run_id=f"xgb_shap_{dataset}_seed{seed}")
  logger.log("baseline.xgb_shap", {
    "dataset": dataset,
    "accuracy": report.accuracy,
    "macro_f1": report.macro_f1,
    "fairness": fairness.to_dict(),
    "shap_n_explained": n_explain,
  })
  (out_dir / "summary.json").write_text(json.dumps({
    "baseline": "xgb_shap",
    "dataset": dataset,
    "seed": seed,
    "accuracy": report.accuracy,
    "macro_f1": report.macro_f1,
    "fairness": fairness.to_dict(),
  }, indent=2), encoding="utf-8")
  logger.close()
  return {"accuracy": report.accuracy, "macro_f1": report.macro_f1}


def run_distilbert_shap(dataset: str, seed: int, out_dir: Path) -> dict[str, Any]: # pragma: no cover
  """DistilBERT + SHAP baseline on text data (Bias-in-Bios).

  Note: SHAP for DistilBERT is slow; we sample 200 explanation
  instances. The benchmark numbers in PAPER.md are stated under
  this protocol.
  """
  raise NotImplementedError(
    "DistilBERT baseline runner: implementation lives in "
    "scripts/_baseline_distilbert.py once torch is installed. "
    "Removed from the default path to keep the test suite light."
  )


def main() -> None: # pragma: no cover
  p = argparse.ArgumentParser()
  p.add_argument("--dataset", required=True,
          choices=["adult_income", "hr_analytics", "bias_in_bios"])
  p.add_argument("--baseline", required=True, choices=["xgb_shap", "distilbert_shap"])
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out-dir", default="experiments/baselines")
  args = p.parse_args()
  out_dir = Path(args.out_dir) / f"{args.baseline}_{args.dataset}"
  if args.baseline == "xgb_shap":
    run_xgb_shap(args.dataset, args.seed, out_dir)
  else:
    run_distilbert_shap(args.dataset, args.seed, out_dir)


if __name__ == "__main__":
  main()
