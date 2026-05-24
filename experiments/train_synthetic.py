"""Synthetic-dataset smoke runner.

Wraps :mod:`experiments.full_pipeline` for the synthetic dataset.
Used by tests and as the demo path that runs in <5 seconds on a
single CPU with no GraphTM backend.
"""
from __future__ import annotations

import argparse

from experiments.full_pipeline import run


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out-dir", default="experiments/paper_d_synthetic")
  p.add_argument("--n-train", type=int, default=400)
  p.add_argument("--n-epochs", type=int, default=10)
  p.add_argument("--allow-fallback", action="store_true", default=False)
  args = p.parse_args()
  run(
    dataset="synthetic", seed=args.seed, out_dir=args.out_dir,
    max_train=args.n_train, n_epochs=args.n_epochs,
    allow_fallback=args.allow_fallback,
    # Sweet-spot hyper-params for synthetic (calibrated 2026-05-24).
    n_clauses=100, T=50, s=3.9, depth=1,
    max_recourse=3, max_recourse_samples=30, max_explain_samples=30,
  )


if __name__ == "__main__":
  main()
