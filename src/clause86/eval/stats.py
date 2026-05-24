"""Paired bootstrap, Wilcoxon signed-rank, Holm-Bonferroni.

Matches the statistics-reporting conventions across Anwar's repos so
the 5-seed aggregator scripts can be re-used.

All functions are pure NumPy / pure Python and do not depend on
scipy beyond the optional ``wilcoxon_signed_rank`` (we fall back to
a Z-approximation when scipy is missing).
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


def paired_bootstrap_ci(
  deltas: list[float],
  *,
  n_resamples: int = 10_000,
  confidence: float = 0.95,
  seed: int = 42,
) -> tuple[float, float, float]:
  """Bias-corrected percentile CI for the mean of paired deltas."""
  if not deltas:
    return 0.0, 0.0, 0.0
  rng = random.Random(seed)
  obs = np.asarray(deltas, dtype=np.float64)
  n = len(obs)
  if n_resamples <= 0:
    n_resamples = 1
  sample_means = np.empty(n_resamples, dtype=np.float64)
  for i in range(n_resamples):
    idx = [rng.randrange(n) for _ in range(n)]
    sample_means[i] = float(obs[idx].mean())
  alpha = 1.0 - confidence
  lo = float(np.quantile(sample_means, alpha / 2))
  hi = float(np.quantile(sample_means, 1.0 - alpha / 2))
  return float(obs.mean()), lo, hi


def wilcoxon_signed_rank(
  a: list[float], b: list[float],
) -> dict[str, float]:
  """Two-sided Wilcoxon signed-rank for paired observations."""
  if len(a) != len(b):
    raise ValueError("paired lengths differ")
  diffs = [x - y for x, y in zip(a, b) if x != y]
  n = len(diffs)
  if n == 0:
    return {"W": 0.0, "p_value": 1.0, "n": 0.0}
  try: # pragma: no cover - depends on env
    from scipy.stats import wilcoxon # type: ignore
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {"W": float(stat), "p_value": float(p), "n": float(n)}
  except Exception:
    pass
  # Approximate via normal: see Wilcoxon (1945) signed-rank Z form.
  ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
  abs_ranks = [0.0] * n
  for r, idx in enumerate(ranks, start=1):
    abs_ranks[idx] = float(r)
  W_plus = sum(abs_ranks[i] for i in range(n) if diffs[i] > 0)
  mean = n * (n + 1) / 4.0
  var = n * (n + 1) * (2 * n + 1) / 24.0
  z = (W_plus - mean) / math.sqrt(var) if var > 0 else 0.0
  p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
  return {"W": float(W_plus), "p_value": float(p), "n": float(n)}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
  """Holm step-down correction. Returns a list of reject/non-reject."""
  n = len(p_values)
  order = sorted(range(n), key=lambda i: p_values[i])
  reject = [False] * n
  for rank, idx in enumerate(order):
    p = p_values[idx]
    threshold = alpha / (n - rank)
    if p <= threshold:
      reject[idx] = True
    else:
      break
  return reject
