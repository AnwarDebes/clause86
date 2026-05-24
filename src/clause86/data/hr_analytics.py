"""IBM HR-Analytics employee-attrition loader.

About 1.5k tabular employee records, binary attrition target. Useful
as a third dataset for the Article-86 portfolio because it explicitly
treats HR decisions (predicting which employees will leave) and is
small enough that the TM trains in seconds.

The original CSV is hosted at:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The user must download the CSV manually (Kaggle T&C); we then read it
with sane column subsetting and bucketing.
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


_TM_FEATURES: list[FeatureSpec] = [
  FeatureSpec(name="age", kind=FeatureKind.ORDINAL, n_buckets=5,
        nl_template="age >= {bucket}"),
  FeatureSpec(name="business_travel", kind=FeatureKind.CATEGORICAL,
        categories=("Non-Travel", "Travel_Rarely", "Travel_Frequently"),
        nl_template="business travel = {value}"),
  FeatureSpec(name="department", kind=FeatureKind.CATEGORICAL,
        categories=("Sales", "Research & Development", "Human Resources"),
        nl_template="department = {value}"),
  FeatureSpec(name="distance_from_home", kind=FeatureKind.ORDINAL, n_buckets=4,
        nl_template="distance from home >= {bucket}"),
  FeatureSpec(name="education", kind=FeatureKind.ORDINAL, n_buckets=5,
        nl_template="education level >= {bucket}"),
  FeatureSpec(name="job_role", kind=FeatureKind.CATEGORICAL,
        categories=(
          "Sales Executive", "Research Scientist", "Laboratory Technician",
          "Manufacturing Director", "Healthcare Representative", "Manager",
          "Sales Representative", "Research Director", "Human Resources",
        ),
        nl_template="job role = {value}"),
  FeatureSpec(name="job_satisfaction", kind=FeatureKind.ORDINAL, n_buckets=4,
        nl_template="job satisfaction >= {bucket}"),
  FeatureSpec(name="monthly_income", kind=FeatureKind.ORDINAL, n_buckets=6,
        nl_template="monthly income >= {bucket}"),
  FeatureSpec(name="overtime", kind=FeatureKind.BINARY,
        nl_template="works overtime"),
  FeatureSpec(name="total_working_years", kind=FeatureKind.ORDINAL, n_buckets=5,
        nl_template="total working years >= {bucket}"),
  FeatureSpec(name="years_at_company", kind=FeatureKind.ORDINAL, n_buckets=5,
        nl_template="years at company >= {bucket}"),
]


def load_hr_analytics(
  csv_path: str | Path = "data/raw/hr_analytics/WA_Fn-UseC_-HR-Employee-Attrition.csv",
) -> HiringDataset:
  """Load the IBM HR-Analytics CSV into a :class:`HiringDataset`.

  Note: we deliberately do NOT do a held-out test split here; the
  canonical HR-Analytics split is k-fold on the full table. The
  user should fold-split downstream.
  """
  import pandas as pd

  csv_path = Path(csv_path)
  if not csv_path.exists():
    raise FileNotFoundError(
      f"HR-Analytics CSV not found at {csv_path}. Download from Kaggle: "
      "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset"
    )
  df = pd.read_csv(csv_path)
  records: list[dict[str, Any]] = []
  labels: list[int] = []
  protected: list[dict[ProtectedAttribute, str]] = []
  splits: list[str] = []
  for _, ex in df.iterrows():
    records.append({
      "age": int(ex["Age"]),
      "business_travel": str(ex["BusinessTravel"]),
      "department": str(ex["Department"]),
      "distance_from_home": int(ex["DistanceFromHome"]),
      "education": int(ex["Education"]),
      "job_role": str(ex["JobRole"]),
      "job_satisfaction": int(ex["JobSatisfaction"]),
      "monthly_income": int(ex["MonthlyIncome"]),
      "overtime": int(ex["OverTime"] == "Yes"),
      "total_working_years": int(ex["TotalWorkingYears"]),
      "years_at_company": int(ex["YearsAtCompany"]),
    })
    labels.append(int(ex["Attrition"] == "Yes"))
    protected.append({
      ProtectedAttribute.GENDER: str(ex["Gender"]),
      ProtectedAttribute.AGE_GROUP: _age_to_group(int(ex["Age"])),
    })
    splits.append("train") # fold-splitting is downstream

  return HiringDataset(
    name="hr_analytics",
    records=records,
    labels=labels,
    class_names=["stay", "attrit"],
    schema=_TM_FEATURES,
    protected=protected,
    split=splits,
    meta={
      "citation": "IBM HR Analytics Employee Attrition (Kaggle).",
      "n_loaded": len(records),
    },
  )


def _age_to_group(age: int) -> str:
  if age < 26:
    return "18-25"
  if age < 41:
    return "26-40"
  if age < 61:
    return "41-60"
  return "60+"
