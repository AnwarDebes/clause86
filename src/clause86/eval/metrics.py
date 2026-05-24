"""Accuracy + macro-F1 + per-class precision-recall.

Implemented in pure NumPy so the module has no scikit-learn
dependency on the hot inference path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
  if len(y_true) != len(y_pred):
    raise ValueError("y_true and y_pred length mismatch.")
  if not y_true:
    return 0.0
  return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _confusion(
  y_true: list[int], y_pred: list[int], n_classes: int,
) -> np.ndarray:
  m = np.zeros((n_classes, n_classes), dtype=np.int64)
  for t, p in zip(y_true, y_pred):
    if 0 <= t < n_classes and 0 <= p < n_classes:
      m[t, p] += 1
  return m


def macro_f1_score(
  y_true: list[int],
  y_pred: list[int],
  n_classes: int | None = None,
) -> float:
  if n_classes is None:
    n_classes = max(max(y_true, default=-1), max(y_pred, default=-1)) + 1
  if n_classes <= 0 or not y_true:
    return 0.0
  cm = _confusion(y_true, y_pred, n_classes)
  f1s: list[float] = []
  for c in range(n_classes):
    tp = float(cm[c, c])
    fp = float(cm[:, c].sum() - cm[c, c])
    fn = float(cm[c, :].sum() - cm[c, c])
    if tp + fp == 0 or tp + fn == 0:
      f1 = 0.0
    else:
      p = tp / (tp + fp)
      r = tp / (tp + fn)
      f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    f1s.append(f1)
  return float(np.mean(f1s)) if f1s else 0.0


@dataclass
class ClassificationReport:
  accuracy: float
  macro_f1: float
  n: int
  per_class: dict[int, dict[str, float]] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      "accuracy": float(self.accuracy),
      "macro_f1": float(self.macro_f1),
      "n": int(self.n),
      "per_class": {
        int(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in self.per_class.items()
      },
    }


def classification_report(
  y_true: list[int],
  y_pred: list[int],
  n_classes: int | None = None,
) -> ClassificationReport:
  if n_classes is None:
    n_classes = max(max(y_true, default=-1), max(y_pred, default=-1)) + 1
  n_classes = max(n_classes, 1)
  cm = _confusion(y_true, y_pred, n_classes)
  per_class: dict[int, dict[str, float]] = {}
  for c in range(n_classes):
    tp = float(cm[c, c])
    fp = float(cm[:, c].sum() - cm[c, c])
    fn = float(cm[c, :].sum() - cm[c, c])
    support = float(cm[c, :].sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": support}
  return ClassificationReport(
    accuracy=accuracy_score(y_true, y_pred),
    macro_f1=macro_f1_score(y_true, y_pred, n_classes=n_classes),
    n=len(y_true),
    per_class=per_class,
  )
