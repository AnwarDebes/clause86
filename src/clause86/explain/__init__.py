"""Article-86-compliant explanation generation.

The EU AI Act Article 86 (effective 2 August 2026) grants any
person affected by a high-risk AI decision the right to obtain
"clear and meaningful explanations of the role of the AI system
in the decision-making procedure and the main elements of the
decision taken."

This module emits, for each prediction:

* **Structured explanation** (:class:`Article86Explanation`): the
 fired clauses, their literal premises, the per-class vote
 arithmetic that produced the decision.
* **Natural-language explanation**
 (:meth:`Article86Explanation.render_nl`): a paragraph the affected
 individual receives.
* **Faithfulness witness**
 (:meth:`Article86Explanation.faithfulness_witness`): literal flips
 that, according to the explanation, would change the decision.
 Verified by :mod:`clause86.eval.faithfulness`.
"""

from .article86 import Article86Explanation, Article86Explainer
from .nl_translator import literal_to_nl, render_clause_nl

__all__ = [
  "Article86Explanation",
  "Article86Explainer",
  "literal_to_nl",
  "render_clause_nl",
]
