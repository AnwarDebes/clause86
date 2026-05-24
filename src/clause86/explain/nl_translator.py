"""Translate a single literal or a whole clause into natural language.

Templates come from the dataset's :class:`FeatureSpec.nl_template`.
The translator falls back to a generic "feature_name = value" form if
a template is missing.
"""
from __future__ import annotations

from ..graphs.built_graphs import LiteralSpec
from ..tm.firing import FiringClauseSpec


def literal_to_nl(spec: LiteralSpec, *, negated: bool = False) -> str:
  """Render one literal as a short noun-phrase clause."""
  template = spec.nl_template or "{name} = {value}"
  bucket = spec.bucket_or_category
  # Extract numeric bucket for thermometer templates (">=3.0" -> 3.0).
  bucket_num: float | None = None
  if bucket.startswith(">="):
    try:
      bucket_num = float(bucket[2:])
    except ValueError:
      bucket_num = None
  try:
    body = template.format(
      name=spec.feature_name,
      value=bucket,
      bucket=(f"{bucket_num:g}" if bucket_num is not None else bucket),
    )
  except (KeyError, IndexError):
    body = f"{spec.feature_name} = {bucket}"
  if negated:
    # Prepend "NOT" but try to lower-case the first character so the
    # rendered sentence flows.
    return f"NOT {body}"
  return body


def render_clause_nl(
  clause: FiringClauseSpec,
  literal_specs: list[LiteralSpec],
) -> str:
  """Render a firing clause as a single readable sentence.

  The clause is rendered as a conjunction of literal premises plus
  a vote: ``[A AND B AND NOT C] -> +1 for class K``.
  """
  parts: list[str] = []
  for lid in clause.positive_literal_ids:
    if 0 <= lid < len(literal_specs):
      parts.append(literal_to_nl(literal_specs[lid], negated=False))
  for lid in clause.negative_literal_ids:
    if 0 <= lid < len(literal_specs):
      parts.append(literal_to_nl(literal_specs[lid], negated=True))
  body = " AND ".join(parts) if parts else "TRUE"
  vote = clause.polarity * clause.weight
  sign = "+" if vote >= 0 else "-"
  return f"[{body}] -> {sign}{abs(vote)} vote for class {clause.class_id}"
