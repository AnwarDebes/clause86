"""HiringDataset: the uniform container every loader returns.

Why a single container: the graph / TM / fairness / recourse modules
should not care whether the source is a corpus of resumes
(text-heavy) or a tabular HR record. The container holds:

* a list of per-sample feature records (``records``),
* canonical class labels (``labels``),
* per-sample protected attributes for the fairness audit
 (``protected``), with explicit codes for *immutable* attributes
 (recourse may not target them) vs. mutable proxies,
* a feature schema (``schema``) declaring which features are
 textual, ordinal, categorical, or immutable, used by the graph
 builder and the recourse-candidate generator.

This is intentionally schema-driven rather than DataFrame-typed so
that adding a new dataset is a 50-line file, not a refactor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FeatureKind(str, Enum):
  """How a feature is encoded into TM literals + how recourse may edit it."""

  TEXT = "text" # free-text (resume bio) -> subword graph
  CATEGORICAL = "categorical" # discrete bucket -> one-hot literals
  ORDINAL = "ordinal" # bucketed integer -> thermometer literals
  BINARY = "binary" # 0/1 literal as-is
  IMMUTABLE = "immutable" # demographic / immutable; never editable by recourse


class ProtectedAttribute(str, Enum):
  """Protected attributes the fairness audit groups by.

  These are stored separately from the input features so that the
  deployer can run the model fully blind to them; the audit module
  re-attaches them only to compute group-conditional metrics.
  """

  GENDER = "gender"
  RACE = "race"
  AGE_GROUP = "age_group"
  NATIONALITY = "nationality"
  DISABILITY = "disability"


@dataclass(frozen=True)
class FeatureSpec:
  """Schema entry for one feature column.

  Attributes:
    name: stable column name used by the literal mapper.
    kind: how to encode + whether recourse may edit.
    n_buckets: thermometer bucket count for ordinal features
      (e.g., ``years_experience`` may use 6 buckets at 0,1,2,5,10,20).
    categories: closed vocabulary for categorical features.
    nl_template: natural-language template used by the explainer
      when this feature appears in a firing clause. Receives
      ``{value}`` and ``{bucket}``.
  """

  name: str
  kind: FeatureKind
  n_buckets: int = 0
  categories: tuple[str, ...] = ()
  nl_template: str = "{name} = {value}"


@dataclass
class HiringDataset:
  """A loaded hiring/HR dataset, ready for the graph + TM pipeline.

  Attributes:
    name: dataset identifier (e.g., ``"bias_in_bios"``).
    records: per-sample dict of feature_name -> raw value.
    labels: per-sample int class label.
    class_names: human-readable class names indexed by label.
    schema: list of :class:`FeatureSpec` defining the columns.
    protected: per-sample dict of :class:`ProtectedAttribute` ->
      group code (e.g., ``ProtectedAttribute.GENDER -> "F"``).
    split: per-sample split tag (``"train"``, ``"dev"``, ``"test"``).
    meta: free-form metadata (source URL, license, sample count, ...).
  """

  name: str
  records: list[dict[str, Any]]
  labels: list[int]
  class_names: list[str]
  schema: list[FeatureSpec]
  protected: list[dict[ProtectedAttribute, str]]
  split: list[str]
  meta: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    n = len(self.records)
    if not (n == len(self.labels) == len(self.protected) == len(self.split)):
      raise ValueError(
        f"Length mismatch: records={len(self.records)} "
        f"labels={len(self.labels)} protected={len(self.protected)} "
        f"split={len(self.split)}"
      )
    names_in_records: set[str] = set()
    for r in self.records:
      names_in_records.update(r.keys())
    schema_names = {s.name for s in self.schema}
    unknown = names_in_records - schema_names
    if unknown:
      raise ValueError(f"records contain features not in schema: {sorted(unknown)}")

  def split_indices(self, tag: str) -> list[int]:
    """Return row indices for the given split tag."""
    return [i for i, s in enumerate(self.split) if s == tag]

  def select(self, indices: list[int]) -> "HiringDataset":
    """Return a new HiringDataset restricted to ``indices``."""
    return HiringDataset(
      name=self.name,
      records=[self.records[i] for i in indices],
      labels=[self.labels[i] for i in indices],
      class_names=list(self.class_names),
      schema=list(self.schema),
      protected=[self.protected[i] for i in indices],
      split=[self.split[i] for i in indices],
      meta=dict(self.meta),
    )

  @property
  def n_classes(self) -> int:
    return len(self.class_names)

  def __len__(self) -> int:
    return len(self.records)
