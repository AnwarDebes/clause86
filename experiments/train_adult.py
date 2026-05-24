"""Adult-Income training entry point (single seed).

Tabular UCI Adult-Income; race + sex protected. Used as a fairness
benchmark to compare Clause86 against the canonical fairness
literature (Hardt-Price-Srebro 2016 EqOdds, Zafar et al., reductions
approach).
"""
from __future__ import annotations

import argparse

from experiments.full_pipeline import run


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--n-epochs", type=int, default=30)
  p.add_argument("--n-clauses", type=int, default=300)
  p.add_argument("--T", type=int, default=150)
  p.add_argument("--s", type=float, default=8.0)
  p.add_argument("--depth", type=int, default=2)
  p.add_argument("--out-dir", default="experiments/paper_d_adult")
  p.add_argument("--ensemble", action="store_true")
  args = p.parse_args()
  run(
    dataset="adult_income", seed=args.seed, out_dir=args.out_dir,
    n_epochs=args.n_epochs, n_clauses=args.n_clauses, T=args.T,
    s=args.s, depth=args.depth, ensemble=args.ensemble,
    allow_fallback=False,
  )


if __name__ == "__main__":
  main()
