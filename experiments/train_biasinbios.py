"""Bias-in-Bios training entry point (single seed).

Loads De-Arteaga et al.'s 400k bios dataset, trains a Graph TM on
subword+dependency graphs (uses CAIR's tokenizer + spaCy + GraphTM
backend if installed), writes the full Article-86 explanation +
SAT receipt + fairness audit log.

Reproducing the headline 5-seed result::

  bash scripts/run_all_seeds.sh bias_in_bios
"""
from __future__ import annotations

import argparse

from experiments.full_pipeline import run


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--max-train", type=int, default=20_000,
          help="Cap per-split sample count; default lets a single V100 finish in <2h.")
  p.add_argument("--n-epochs", type=int, default=20)
  p.add_argument("--n-clauses", type=int, default=600)
  p.add_argument("--T", type=int, default=400)
  p.add_argument("--s", type=float, default=12.0)
  p.add_argument("--depth", type=int, default=3)
  p.add_argument("--out-dir", default="experiments/paper_d_biasinbios")
  p.add_argument("--ensemble", action="store_true")
  args = p.parse_args()
  run(
    dataset="bias_in_bios", seed=args.seed, out_dir=args.out_dir,
    max_train=args.max_train, n_epochs=args.n_epochs,
    n_clauses=args.n_clauses, T=args.T, s=args.s, depth=args.depth,
    ensemble=args.ensemble,
    # Refuse to fall back during a real benchmarking run.
    allow_fallback=False,
  )


if __name__ == "__main__":
  main()
