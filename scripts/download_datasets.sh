#!/usr/bin/env bash
# Download the public datasets Clause86 evaluates against.
# Usage: scripts/download_datasets.sh
#
# - Bias-in-Bios: via HuggingFace `datasets` (LabHC/bias_in_bios).
#          Run `python -c "import datasets; datasets.load_dataset('LabHC/bias_in_bios')"`
#          on first use; the cache lives under ~/.cache/huggingface/.
# - Adult Income: UCI archive (anonymous download).
# - HR Analytics: Kaggle (user must authenticate; see comment).
set -euo pipefail

mkdir -p data/raw/adult_income data/raw/bias_in_bios data/raw/hr_analytics

echo "=== Adult Income ==="
if [[ ! -f data/raw/adult_income/adult.data ]]; then
  curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" \
    -o data/raw/adult_income/adult.data
fi
if [[ ! -f data/raw/adult_income/adult.test ]]; then
  curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" \
    -o data/raw/adult_income/adult.test
fi

echo "=== Bias-in-Bios ==="
echo "Bias-in-Bios is pulled lazily by the HuggingFace loader."
echo " python -c \"from clause86.data import load_bias_in_bios; load_bias_in_bios(max_samples_per_split=200)\""

echo "=== HR Analytics ==="
echo "Kaggle does not allow anonymous downloads. Authenticate kaggle CLI then:"
echo " kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/raw/hr_analytics --unzip"

echo "Done."
