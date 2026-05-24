"""Graph builders for Clause86.

Two builders share a common ``BuiltGraphs`` payload that the TM module
consumes:

* :class:`SubwordDepGraphBuilder`, BPE subword nodes + spaCy
 dependency edges + sequential edges. Re-ported from
 ``subword-dep-graphtm`` (Paper A). Used for the resume / Bias-in-Bios
 text features.

* :class:`TabularBipartiteGraphBuilder`, one node per feature value
 + one shared per-sample sink; edges typed by feature name. Used
 for Adult-Income / HR-Analytics.

* :class:`CompositeGraphBuilder`, composes both when a dataset has
 both text and tabular features (the synthetic + bio-with-meta cases).

A :class:`BuiltGraphs` carries:

* the list of per-document graphs (node properties + edge types),
* the global symbol vocabulary,
* the per-sample literal indexing (feature_name + bucket -> literal id),
 used by the SAT receipt and the explainer to translate firing
 clauses back into natural language.
"""

from .built_graphs import BuiltGraphs, LiteralSpec
from .text_graph import SubwordDepGraphBuilder, SEQUENTIAL_EDGE_TYPES, DEP_TYPES
from .tabular_graph import TabularBipartiteGraphBuilder
from .builders import CompositeGraphBuilder, GraphBuilder

__all__ = [
  "BuiltGraphs",
  "LiteralSpec",
  "GraphBuilder",
  "SubwordDepGraphBuilder",
  "TabularBipartiteGraphBuilder",
  "CompositeGraphBuilder",
  "SEQUENTIAL_EDGE_TYPES",
  "DEP_TYPES",
]
