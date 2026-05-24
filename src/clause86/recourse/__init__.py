"""Counterfactual recourse for hiring decisions.

For any adverse prediction (e.g., "reject", or a class lower than
the candidate's preferred class) the module proposes the **minimum
set of feature edits** the affected person could make so that the
TM would re-classify them favourably.

Key design choices (mirroring ``graphtm-cbr``):

* Greedy minimum-edit search (no BFS).
* Feasibility constraints baked in (immutable literals never flip;
 ordinal literals are monotone upward; binary "negative" literals
 flip toward removal only).
* Domain-aware candidate generation that prunes losing edits.
"""

from .candidates import RecourseEdit, generate_candidates
from .feasibility import FeasibilityChecker
from .greedy import HireRecourseReport, greedy_minimum_edit

__all__ = [
  "RecourseEdit",
  "generate_candidates",
  "FeasibilityChecker",
  "HireRecourseReport",
  "greedy_minimum_edit",
]
