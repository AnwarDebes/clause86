"""FeasibilityChecker, which literal flips are real-world meaningful.

Three hard rules:

1. **Immutable literals never flip.** Any literal attached to a
  feature with ``kind=FeatureKind.IMMUTABLE`` (gender, race,
  age_group, nationality, disability) is permanently locked.
2. **Ordinal literals are monotone upward.** "years_experience >= 5"
  may flip from 0 to 1 (gain experience) but not from 1 to 0
  ("lose 5 years of experience" is not a recourse).
3. **Binary literals coding negative attributes flip toward removal.**
  For ``empty_resume``, we allow 1->0 (add a resume) but not 0->1
  (removing your resume is not recourse).

The checker is dataset-agnostic: it inspects ``feature_kind`` +
``feature_name`` + ``bucket_or_category`` on the :class:`LiteralSpec`.
Project-specific overrides go in ``per_feature_policy``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

from ..graphs.built_graphs import LiteralSpec


# Names of binary features where flipping ``0 -> 1`` is *not* an
# improvement for the candidate, so recourse should only allow ``1 -> 0``.
_NEGATIVE_BINARY_FEATURES = frozenset({"empty_resume", "overtime"})


@dataclass
class FeasibilityChecker:
  """Decide whether a single (literal, from, to) flip is feasible.

  Args:
    literal_specs: vocabulary the literals refer to.
    per_feature_policy: optional override of the default rules.
      ``per_feature_policy[feature_name]`` is a callable
      ``(spec, from_v, to_v) -> bool`` returning whether the
      flip is allowed.
  """

  literal_specs: list[LiteralSpec]
  per_feature_policy: Mapping[str, Callable[[LiteralSpec, int, int], bool]] = field(
    default_factory=dict
  )

  def may_flip(self, literal_id: int, *, from_v: int, to_v: int) -> bool:
    if not (0 <= literal_id < len(self.literal_specs)):
      return False
    spec = self.literal_specs[literal_id]
    if spec.immutable:
      return False
    # Per-feature override wins.
    if spec.feature_name in self.per_feature_policy:
      return self.per_feature_policy[spec.feature_name](spec, from_v, to_v)
    if spec.feature_kind == "ordinal":
      # Monotone upward only (gain experience, not lose it).
      return from_v == 0 and to_v == 1
    if spec.feature_kind == "binary":
      if spec.feature_name in _NEGATIVE_BINARY_FEATURES:
        return from_v == 1 and to_v == 0
      return True
    if spec.feature_kind == "categorical":
      return True
    if spec.feature_kind == "text":
      # Allow toggling presence/absence of a subword token in
      # the bio. Adding tokens (0 -> 1) is the realistic direction
      # for recourse; removing is rarely meaningful.
      return from_v == 0 and to_v == 1
    return False
