"""Article-86 explanation generator.

The text of Article 86 (Right to Explanation of Individual
Decision-Making) requires the deployer to provide:

 "clear and meaningful explanations of the role of the AI system
  in the decision-making procedure and the main elements of the
  decision taken"

In the Clause86 architecture:

* "the role of the AI system" -> a fixed disclosure paragraph
 (the system was used to score this candidate's resume + structured
 features against a learned set of Boolean rules).
* "the main elements of the decision taken" -> the FIRING clauses
 for the predicted class, rendered as conjunctive literal
 premises and weighted by their voting contribution. Negative-
 polarity clauses that fire for *other* classes are also
 rendered, because they too are decision elements ("you were
 NOT classified as X because clause #q fired AGAINST X").

The explanation is JSON-serialisable so it can be stored alongside
the SAT receipt for downstream auditing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graphs.built_graphs import BuiltGraphs, LiteralSpec
from ..tm.firing import ClassScoreSummary, FiringClauseSpec
from .nl_translator import literal_to_nl, render_clause_nl


# Fixed Article 86 disclosure paragraph. The phrasing follows
# Article 86(1) and Recital 171; the deployer name and date are
# substituted at render time.
_DISCLOSURE_TEMPLATE = (
  "This decision was produced by an AI system (Clause86) on behalf of "
  "{deployer}. The system scored your application against a finite, "
  "auditable set of Boolean rules learned from historical data. No "
  "post-hoc approximation is used; the rules below ARE the decision. "
  "You may contest the decision and, where applicable, request human "
  "review under Article 14 of the AI Act."
)


@dataclass
class Article86Explanation:
  """Self-contained, auditor-replayable Art 86 explanation."""

  sample_idx: int
  deployer_name: str
  predicted_class: int
  predicted_class_name: str
  class_scores: tuple[int, ...]
  margin: int
  n_firing_clauses: int
  main_elements: list[dict[str, Any]] = field(default_factory=list)
  counter_elements: list[dict[str, Any]] = field(default_factory=list)
  disclosure: str = ""

  def render_nl(self, max_elements: int = 5) -> str:
    """Render a plain-language explanation for the affected person.

    Args:
      max_elements: cap the number of clauses rendered into the
        NL summary; the rest stay in the structured artefact
        for the auditor. Five is the default because the
        affected person rarely needs more.
    """
    lines = [self.disclosure, "", f"Decision: {self.predicted_class_name}."]
    if self.main_elements:
      lines.append("")
      lines.append("Main elements that supported this decision:")
      for el in self.main_elements[:max_elements]:
        lines.append(f" - {el['nl']}")
    if self.counter_elements:
      lines.append("")
      lines.append("Elements that argued against other outcomes:")
      for el in self.counter_elements[:max_elements]:
        lines.append(f" - {el['nl']}")
    return "\n".join(lines)

  def faithfulness_witness(
    self,
    literal_specs: list[LiteralSpec],
    literal_vector: list[int],
  ) -> list[dict[str, Any]]:
    """Return per-clause flip suggestions to validate faithfulness.

    For each main-element clause we propose flipping ONE of its
    positive literals to 0 (or negated literal to 1). The
    prediction is hypothesised to drop by at least
    ``polarity * weight`` after the flip. Downstream
    :mod:`clause86.eval.faithfulness` runs the TM on the perturbed
    vector and checks the actual margin change.
    """
    out: list[dict[str, Any]] = []
    for el in self.main_elements:
      clause = el["clause"]
      if clause.positive_literal_ids:
        lid = clause.positive_literal_ids[0]
        cur = literal_vector[lid] if lid < len(literal_vector) else 0
        out.append({
          "clause_id": clause.clause_id,
          "flip_literal": lid,
          "from_value": cur,
          "to_value": 1 - cur,
          "expected_score_delta": -clause.polarity * clause.weight,
          "literal_nl": literal_to_nl(literal_specs[lid]) if lid < len(literal_specs) else "",
        })
      elif clause.negative_literal_ids:
        lid = clause.negative_literal_ids[0]
        cur = literal_vector[lid] if lid < len(literal_vector) else 0
        out.append({
          "clause_id": clause.clause_id,
          "flip_literal": lid,
          "from_value": cur,
          "to_value": 1 - cur,
          "expected_score_delta": -clause.polarity * clause.weight,
          "literal_nl": literal_to_nl(literal_specs[lid], negated=True) if lid < len(literal_specs) else "",
        })
    return out

  def to_dict(self) -> dict[str, Any]:
    return {
      "sample_idx": self.sample_idx,
      "deployer_name": self.deployer_name,
      "predicted_class": self.predicted_class,
      "predicted_class_name": self.predicted_class_name,
      "class_scores": list(self.class_scores),
      "margin": self.margin,
      "n_firing_clauses": self.n_firing_clauses,
      "main_elements": [
        {
          "clause_id": el["clause"].clause_id,
          "class_id": el["clause"].class_id,
          "polarity": el["clause"].polarity,
          "weight": el["clause"].weight,
          "positive_literal_ids": list(el["clause"].positive_literal_ids),
          "negative_literal_ids": list(el["clause"].negative_literal_ids),
          "nl": el["nl"],
        }
        for el in self.main_elements
      ],
      "counter_elements": [
        {
          "clause_id": el["clause"].clause_id,
          "class_id": el["clause"].class_id,
          "polarity": el["clause"].polarity,
          "weight": el["clause"].weight,
          "positive_literal_ids": list(el["clause"].positive_literal_ids),
          "negative_literal_ids": list(el["clause"].negative_literal_ids),
          "nl": el["nl"],
        }
        for el in self.counter_elements
      ],
      "disclosure": self.disclosure,
    }


@dataclass
class Article86Explainer:
  """Build :class:`Article86Explanation` objects from TM summaries.

  Args:
    class_names: mapping label_id -> human-readable class name.
    deployer_name: name to substitute into the disclosure paragraph.
    top_k_main_elements: cap the structured main-element list.
    top_k_counter_elements: cap the counter-element list.
  """

  class_names: list[str]
  deployer_name: str = "Deployer"
  top_k_main_elements: int = 10
  top_k_counter_elements: int = 5

  def explain(
    self,
    summary: ClassScoreSummary,
    built: BuiltGraphs,
    sample_idx: int,
  ) -> Article86Explanation:
    literal_specs = built.literal_specs
    main: list[dict[str, Any]] = []
    counter: list[dict[str, Any]] = []
    for clause in summary.firing_clauses:
      nl = render_clause_nl(clause, literal_specs)
      entry = {"clause": clause, "nl": nl}
      if clause.class_id == summary.argmax_label:
        if clause.polarity > 0:
          main.append(entry)
        else:
          counter.append(entry)
      else:
        if clause.polarity > 0:
          # Positive vote for a non-winning class -> counter element.
          counter.append(entry)
        else:
          # Negative-polarity for a non-winning class -> *supporting* the winner.
          main.append(entry)
    main.sort(key=lambda e: -abs(e["clause"].polarity * e["clause"].weight))
    counter.sort(key=lambda e: -abs(e["clause"].polarity * e["clause"].weight))
    cname = (
      self.class_names[summary.argmax_label]
      if 0 <= summary.argmax_label < len(self.class_names)
      else f"class_{summary.argmax_label}"
    )
    return Article86Explanation(
      sample_idx=sample_idx,
      deployer_name=self.deployer_name,
      predicted_class=summary.argmax_label,
      predicted_class_name=cname,
      class_scores=summary.scores,
      margin=summary.margin,
      n_firing_clauses=len(summary.firing_clauses),
      main_elements=main[: self.top_k_main_elements],
      counter_elements=counter[: self.top_k_counter_elements],
      disclosure=_DISCLOSURE_TEMPLATE.format(deployer=self.deployer_name),
    )
