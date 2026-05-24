"""Aggregate per-seed summary.json files into a single results JSON.

Pulled into the same shape as Anwar's other repos so downstream
notebooks / paper-figure scripts work unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
  if not items:
    return {}
  keys = sorted({k for item in items for k in item if not isinstance(item.get(k), (dict, list))})
  out: dict[str, Any] = {}
  for k in keys:
    vals = [float(item[k]) for item in items if k in item and isinstance(item[k], (int, float))]
    if not vals:
      continue
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
      var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
      std = math.sqrt(var)
    else:
      std = 0.0
    out[k] = {
      "mean": mean,
      "std": std,
      "n": len(vals),
      "min": min(vals),
      "max": max(vals),
      "per_seed": vals,
    }
  return out


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--in-dir", required=True,
          help="Directory containing seed_<n>/summary.json files.")
  p.add_argument("--out", required=True,
          help="Output JSON path.")
  args = p.parse_args()
  in_dir = Path(args.in_dir)
  summaries: list[dict[str, Any]] = []
  for seed_dir in sorted(in_dir.glob("seed_*")):
    sp = seed_dir / "summary.json"
    if sp.exists():
      summaries.append(json.loads(sp.read_text(encoding="utf-8")))
  agg = _aggregate(summaries)
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  Path(args.out).write_text(json.dumps({
    "dataset": (summaries[0].get("dataset") if summaries else ""),
    "n_seeds": len(summaries),
    "aggregate": agg,
    "per_seed_summary": summaries,
  }, indent=2), encoding="utf-8")
  print(f"Wrote aggregate of {len(summaries)} seeds to {args.out}")


if __name__ == "__main__":
  main()
