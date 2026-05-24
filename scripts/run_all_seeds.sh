#!/usr/bin/env bash
# Run the 5-seed protocol for one dataset.
# Usage: scripts/run_all_seeds.sh <dataset>
# <dataset> is one of: synthetic, bias_in_bios, adult_income, hr_analytics
set -euo pipefail

DATASET="${1:?usage: $0 <dataset>}"
SEEDS=(42 123 456 789 1337)

OUT_DIR="experiments/paper_d_${DATASET}"
mkdir -p "${OUT_DIR}"

for s in "${SEEDS[@]}"; do
  echo "=== seed=${s} ==="
  case "${DATASET}" in
    synthetic)
      python experiments/train_synthetic.py --seed "${s}" --out-dir "${OUT_DIR}"
      ;;
    bias_in_bios)
      python experiments/train_biasinbios.py --seed "${s}" --out-dir "${OUT_DIR}"
      ;;
    adult_income)
      python experiments/train_adult.py --seed "${s}" --out-dir "${OUT_DIR}"
      ;;
    hr_analytics)
      python experiments/full_pipeline.py --dataset hr_analytics --seed "${s}" --out-dir "${OUT_DIR}"
      ;;
    *)
      echo "Unknown dataset: ${DATASET}" >&2
      exit 2
      ;;
  esac
done

echo "=== aggregating ==="
python scripts/aggregate_results.py --in-dir "${OUT_DIR}" --out "results/paper_d_${DATASET}.json"
