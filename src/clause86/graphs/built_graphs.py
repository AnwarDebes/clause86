"""Common payload for any graph builder.

This file defines the on-the-wire format the TM, explainer, recourse,
and verify modules all consume. Keeping the format here (and not
inside any specific builder) lets us swap the builder without
touching downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LiteralSpec:
  """Maps an integer literal id to its human-meaningful description.

  The TM sees only integer literal ids; the explainer and recourse
  modules need to project them back to (feature, bucket / category)
  to render Article 86 explanations and propose feasible edits.

  Attributes:
    literal_id: position in the literal vector (0..n_literals-1).
    feature_name: column in the dataset schema (e.g.,
      ``"years_experience"``).
    bucket_or_category: thermometer bucket ("years_experience>=3")
      or categorical value ("department=Sales") or the subword
      token surface form ("##MSc").
    feature_kind: from
      :class:`clause86.data.FeatureKind`, the explainer +
      recourse use this to choose templates and feasibility.
    nl_template: the dataset's per-feature template, copied here
      so the explainer does not need to re-read the schema.
    immutable: True if recourse may NOT edit this literal. Set
      for every literal whose feature has
      ``kind=FeatureKind.IMMUTABLE``.
  """

  literal_id: int
  feature_name: str
  bucket_or_category: str
  feature_kind: str
  nl_template: str
  immutable: bool = False


@dataclass
class BuiltGraphs:
  """Output of a graph builder.

  Attributes:
    graphs: ``GraphTsetlinMachine.graphs.Graphs`` object (or, in
      unit tests, a lightweight stub).
    n_samples: how many samples are in ``graphs``.
    symbols: the global symbol vocabulary used by ``graphs``.
    edge_types: the global edge-type vocabulary.
    literal_specs: list of :class:`LiteralSpec`, one per literal
      id, in literal-id order. ``len(literal_specs) ==
      n_literals``.
    per_sample_active_literals: list (one per sample) of the
      literal ids that are 1 for that sample. Stored as a list
      of tuples of ints so it round-trips through JSON.
    meta: free-form (which builder produced this, what params).
  """

  graphs: Any
  n_samples: int
  symbols: list[str]
  edge_types: list[str]
  literal_specs: list[LiteralSpec]
  per_sample_active_literals: list[tuple[int, ...]]
  meta: dict[str, Any] = field(default_factory=dict)

  @property
  def n_literals(self) -> int:
    return len(self.literal_specs)

  def get_active_literal_vector(self, sample_idx: int) -> list[int]:
    """Return the 0/1 literal vector for one sample.

    Used by the SAT encoder to fill in unit clauses for the
    receipt. Built lazily here so the BuiltGraphs payload stays
    small in memory.
    """
    n = self.n_literals
    vec = [0] * n
    for lid in self.per_sample_active_literals[sample_idx]:
      if 0 <= lid < n:
        vec[lid] = 1
    return vec
