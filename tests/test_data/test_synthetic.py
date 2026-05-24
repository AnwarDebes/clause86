from clause86.data import (
  HiringDataset,
  ProtectedAttribute,
  load_synthetic_hiring,
)
from clause86.data.dataset import FeatureKind


def test_load_synthetic_basic():
  ds = load_synthetic_hiring(n_train=10, n_dev=5, n_test=5, seed=42)
  assert isinstance(ds, HiringDataset)
  assert len(ds) == 20
  assert ds.n_classes == 2
  assert ds.class_names == ["reject", "hire"]
  assert set(ds.split) == {"train", "dev", "test"}


def test_synthetic_ground_truth_consistency():
  ds = load_synthetic_hiring(n_train=200, n_dev=0, n_test=0, seed=42, label_noise=0.0)
  # Decision rule:
  # HIRE iff (years>=3 AND has_degree AND not empty) OR (years>=10 AND not empty)
  for r, y in zip(ds.records, ds.labels):
    years = r["years_experience"]
    has_degree = bool(r["has_degree"])
    empty = bool(r["empty_resume"])
    expected = 0
    if not empty:
      if years >= 10:
        expected = 1
      elif years >= 3 and has_degree:
        expected = 1
    assert y == expected


def test_synthetic_label_noise():
  ds_clean = load_synthetic_hiring(n_train=500, n_dev=0, n_test=0, seed=42, label_noise=0.0)
  ds_noisy = load_synthetic_hiring(n_train=500, n_dev=0, n_test=0, seed=42, label_noise=0.2)
  # Same records and labels under same seed but noisy variant should
  # disagree on some labels.
  n_flipped = sum(
    1 for a, b in zip(ds_clean.labels, ds_noisy.labels) if a != b
  )
  assert 30 < n_flipped < 200 # generous bound around the expected 100


def test_protected_attribute_present():
  ds = load_synthetic_hiring(n_train=10, n_dev=0, n_test=0, seed=42)
  for p in ds.protected:
    assert ProtectedAttribute.GENDER in p
    assert ProtectedAttribute.AGE_GROUP in p


def test_select_returns_new_dataset():
  ds = load_synthetic_hiring(n_train=20, n_dev=10, n_test=10, seed=42)
  train = ds.select(ds.split_indices("train"))
  assert len(train) == 20
  assert all(t == "train" for t in train.split)


def test_schema_uses_known_kinds():
  ds = load_synthetic_hiring(n_train=10, n_dev=0, n_test=0, seed=42)
  kinds = {s.kind for s in ds.schema}
  assert kinds.issubset({k for k in FeatureKind})


def test_schema_records_consistency():
  # The HiringDataset post-init must reject mismatched schemas.
  import pytest

  ds = load_synthetic_hiring(n_train=5, n_dev=0, n_test=0, seed=42)
  with pytest.raises(ValueError):
    HiringDataset(
      name="bad",
      records=ds.records,
      labels=ds.labels,
      class_names=ds.class_names,
      schema=ds.schema[:1], # drop most features
      protected=ds.protected,
      split=ds.split,
    )
