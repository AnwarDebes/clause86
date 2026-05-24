"""Adult Income (UCI) loader, canonical algorithmic-fairness benchmark.

About 50k tabular records from the 1994 US Census; binary income
target (``>$50k``); race + sex protected. The dataset is small enough
to bundle as a CSV but we still treat it as an external file because
column-encoding choices change across distributions.

Schema (column names as in UCI ``adult.names``):

  age, workclass, fnlwgt, education, education_num,
  marital_status, occupation, relationship, race, sex,
  capital_gain, capital_loss, hours_per_week, native_country,
  income (target)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import (
  FeatureKind,
  FeatureSpec,
  HiringDataset,
  ProtectedAttribute,
)

# Columns we expose to the TM. Race and sex are dropped from the
# feature schema (passed through ``protected`` only) so the model
# trains blind to them and the fairness audit can attach them later.
_TM_FEATURES: list[FeatureSpec] = [
  FeatureSpec(name="age", kind=FeatureKind.ORDINAL, n_buckets=6,
        nl_template="age >= {bucket}"),
  FeatureSpec(name="workclass", kind=FeatureKind.CATEGORICAL,
        categories=(
          "Private", "Self-emp-not-inc", "Self-emp-inc",
          "Federal-gov", "Local-gov", "State-gov",
          "Without-pay", "Never-worked", "Unknown",
        ),
        nl_template="workclass = {value}"),
  FeatureSpec(name="education_num", kind=FeatureKind.ORDINAL, n_buckets=8,
        nl_template="years of education >= {bucket}"),
  FeatureSpec(name="marital_status", kind=FeatureKind.CATEGORICAL,
        categories=(
          "Never-married", "Married-civ-spouse",
          "Married-spouse-absent", "Married-AF-spouse",
          "Divorced", "Separated", "Widowed",
        ),
        nl_template="marital status = {value}"),
  FeatureSpec(name="occupation", kind=FeatureKind.CATEGORICAL,
        categories=(
          "Tech-support", "Craft-repair", "Other-service", "Sales",
          "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
          "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
          "Transport-moving", "Priv-house-serv", "Protective-serv",
          "Armed-Forces", "Unknown",
        ),
        nl_template="occupation = {value}"),
  FeatureSpec(name="hours_per_week", kind=FeatureKind.ORDINAL, n_buckets=5,
        nl_template="hours per week >= {bucket}"),
  FeatureSpec(name="capital_gain_positive", kind=FeatureKind.BINARY,
        nl_template="positive capital gain reported"),
  FeatureSpec(name="capital_loss_positive", kind=FeatureKind.BINARY,
        nl_template="positive capital loss reported"),
  # Race + sex are intentionally OMITTED from the TM features.
]


_COLS = [
  "age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "sex",
  "capital_gain", "capital_loss", "hours_per_week", "native_country",
  "income",
]


def _read_uci_csv(path: Path) -> list[list[str]]:
  """Return rows from the UCI Adult CSV, stripped of trailing dots and spaces."""
  rows: list[list[str]] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip().rstrip(".")
      if not line or line.startswith("|"):
        continue
      parts = [p.strip() for p in line.split(",")]
      if len(parts) != len(_COLS):
        continue
      rows.append(parts)
  return rows


def load_adult_income(
  raw_dir: str | Path = "data/raw/adult_income",
  splits: tuple[str, ...] = ("train", "test"),
) -> HiringDataset:
  """Load the UCI Adult dataset.

  Expects ``adult.data`` (train split) and ``adult.test`` (test
  split) inside ``raw_dir``, in the canonical UCI CSV format. The
  public URL is https://archive.ics.uci.edu/dataset/2/adult.
  """
  raw_path = Path(raw_dir)
  records: list[dict[str, Any]] = []
  labels: list[int] = []
  protected: list[dict[ProtectedAttribute, str]] = []
  split_tags: list[str] = []

  files = {"train": "adult.data", "test": "adult.test"}
  for split in splits:
    if split not in files:
      raise ValueError(f"Unknown split: {split!r}")
    path = raw_path / files[split]
    if not path.exists():
      raise FileNotFoundError(
        f"Adult Income split {split!r} not found at {path}. "
        f"Download adult.data and adult.test from UCI and place "
        f"them under {raw_path}."
      )
    for parts in _read_uci_csv(path):
      row = dict(zip(_COLS, parts))
      try:
        age = int(row["age"])
        edu = int(row["education_num"])
        hrs = int(row["hours_per_week"])
        cap_g = int(row["capital_gain"])
        cap_l = int(row["capital_loss"])
      except ValueError:
        continue
      workclass = row["workclass"] if row["workclass"] != "?" else "Unknown"
      occupation = row["occupation"] if row["occupation"] != "?" else "Unknown"
      label = 1 if row["income"].startswith(">") else 0
      records.append({
        "age": age,
        "workclass": workclass,
        "education_num": edu,
        "marital_status": row["marital_status"],
        "occupation": occupation,
        "hours_per_week": hrs,
        "capital_gain_positive": int(cap_g > 0),
        "capital_loss_positive": int(cap_l > 0),
      })
      labels.append(label)
      protected.append({
        ProtectedAttribute.RACE: row["race"],
        ProtectedAttribute.GENDER: row["sex"],
      })
      split_tags.append(split)

  return HiringDataset(
    name="adult_income",
    records=records,
    labels=labels,
    class_names=["leq_50k", "gt_50k"],
    schema=_TM_FEATURES,
    protected=protected,
    split=split_tags,
    meta={
      "citation": "Kohavi, KDD 1996; UCI Machine Learning Repository.",
      "n_loaded": len(records),
    },
  )
