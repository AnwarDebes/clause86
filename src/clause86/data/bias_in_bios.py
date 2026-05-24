"""Bias-in-Bios loader (De-Arteaga et al., FAT* 2019).

About 400k Common-Crawl biographies labelled with occupation (28
classes); gender as the protected attribute. Hosted under ``LabHC/bias_in_bios``
on the HuggingFace Hub and mirrored as a parquet snapshot in the
BiosBias S3 bucket.

The loader is deliberately tolerant: if ``datasets`` (or network)
isn't available we fall back to a *very small* parquet shard the user
must drop at ``data/raw/bias_in_bios/<split>.parquet``. The schema is:

  hard_text: str      # the bio text
  gender:  int (0=M,1=F)
  profession:int (0..27)

The full citation:

  De-Arteaga, M., Romanov, A., Wallach, H., Chayes, J., Borgs, C.,
  Chouldechova, A., Geyik, S., Kenthapadi, K., & Kalai, A. (2019).
  *Bias in Bios: A Case Study of Semantic Representation Bias in a
  High-Stakes Setting.* FAT* 2019.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .dataset import (
  FeatureKind,
  FeatureSpec,
  HiringDataset,
  ProtectedAttribute,
)

# Official 28-class occupation label set (canonical order from the
# LabHC HuggingFace dataset card).
PROFESSIONS = [
  "accountant", "architect", "attorney", "chiropractor", "comedian",
  "composer", "dentist", "dietitian", "dj", "filmmaker",
  "interior_designer", "journalist", "model", "nurse", "painter",
  "paralegal", "pastor", "personal_trainer", "photographer",
  "physician", "poet", "professor", "psychologist", "rapper",
  "software_engineer", "surgeon", "teacher", "yoga_teacher",
]


_SCHEMA: list[FeatureSpec] = [
  FeatureSpec(
    name="bio", kind=FeatureKind.TEXT,
    nl_template="bio mentions {value!r}",
  ),
  FeatureSpec(
    name="gender", kind=FeatureKind.IMMUTABLE,
    categories=("M", "F"),
    nl_template="recorded gender is {value}",
  ),
]


def _load_from_huggingface(split: str) -> list[dict[str, Any]]:
  """Pull the split from the Hub; raises ImportError if datasets is missing."""
  from datasets import load_dataset # local import, optional dep

  ds = load_dataset("LabHC/bias_in_bios", split=split)
  rows: list[dict[str, Any]] = []
  for ex in ds:
    rows.append({
      "bio": ex["hard_text"],
      "gender_code": int(ex["gender"]),
      "profession_code": int(ex["profession"]),
    })
  return rows


def _load_from_parquet(split: str, raw_dir: Path) -> list[dict[str, Any]]:
  """Local parquet fallback. Schema described in the module docstring."""
  import pandas as pd

  path = raw_dir / f"{split}.parquet"
  if not path.exists():
    raise FileNotFoundError(
      f"Bias-in-Bios fallback shard not found: {path}. "
      f"Either install `datasets` and let the HF loader run, or "
      f"drop the parquet shards under {raw_dir}/."
    )
  df = pd.read_parquet(path)
  cols = set(df.columns)
  if "hard_text" not in cols and "bio" in cols:
    df = df.rename(columns={"bio": "hard_text"})
  rows: list[dict[str, Any]] = []
  for _, ex in df.iterrows():
    rows.append({
      "bio": str(ex["hard_text"]),
      "gender_code": int(ex["gender"]),
      "profession_code": int(ex["profession"]),
    })
  return rows


def load_bias_in_bios(
  raw_dir: str | Path = "data/raw/bias_in_bios",
  splits: tuple[str, ...] = ("train", "dev", "test"),
  max_samples_per_split: int | None = None,
  prefer_local: bool = False,
) -> HiringDataset:
  """Load the Bias-in-Bios dataset into a :class:`HiringDataset`.

  Args:
    raw_dir: directory containing fallback parquet shards
      (one per split). Used only if the HuggingFace loader
      cannot be imported or ``prefer_local=True``.
    splits: which splits to include.
    max_samples_per_split: optional per-split cap. Useful for
      CI / dev runs.
    prefer_local: if True, skip the HuggingFace loader and go
      straight to local parquet shards.
  """
  raw_path = Path(raw_dir)
  records: list[dict[str, Any]] = []
  labels: list[int] = []
  protected: list[dict[ProtectedAttribute, str]] = []
  split_tags: list[str] = []

  for split in splits:
    rows: list[dict[str, Any]] | None = None
    if not prefer_local:
      try:
        rows = _load_from_huggingface(split)
      except Exception: # pragma: no cover - depends on env
        rows = None
    if rows is None:
      rows = _load_from_parquet(split, raw_path)
    if max_samples_per_split is not None:
      rows = rows[:max_samples_per_split]
    for ex in rows:
      g = "F" if ex["gender_code"] == 1 else "M"
      records.append({"bio": ex["bio"], "gender": g})
      labels.append(ex["profession_code"])
      protected.append({ProtectedAttribute.GENDER: g})
      split_tags.append(split)

  return HiringDataset(
    name="bias_in_bios",
    records=records,
    labels=labels,
    class_names=list(PROFESSIONS),
    schema=_SCHEMA,
    protected=protected,
    split=split_tags,
    meta={
      "citation": "De-Arteaga et al., FAT* 2019",
      "huggingface_id": "LabHC/bias_in_bios",
      "n_loaded": len(records),
      "max_samples_per_split": max_samples_per_split,
    },
  )
