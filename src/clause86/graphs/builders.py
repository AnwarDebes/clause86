"""Composite graph builder + lightweight Protocol.

When a dataset has both a free-text field (``bio``) and tabular
features (``years_experience``, ``has_degree``, ...) we want a single
BuiltGraphs payload that exposes literals from both sides.

The Composite builder uses :class:`SubwordDepGraphBuilder` for the
text features and :class:`TabularBipartiteGraphBuilder` for the rest,
then merges their literal specs and per-sample active sets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..data.dataset import FeatureKind, FeatureSpec, HiringDataset
from .built_graphs import BuiltGraphs, LiteralSpec
from .tabular_graph import TabularBipartiteGraphBuilder
from .text_graph import SubwordDepGraphBuilder


class GraphBuilder(Protocol):
  """Common Protocol for any builder consumed by the TM module."""

  def fit(self, records_or_texts: Any) -> None: ...

  def transform(self, records_or_texts: Any, *, graph_backend: str = "auto") -> BuiltGraphs: ...


@dataclass
class CompositeGraphBuilder:
  """Combines text + tabular into one payload.

  Args:
    schema: list of FeatureSpec from the dataset.
    text_field: which column holds free-text. Default ``"bio"``.
    max_subwords: passed through to :class:`SubwordDepGraphBuilder`.
    tokenizer_name: passed through to the text builder.
    spacy_model: passed through to the text builder.
  """

  schema: list[FeatureSpec]
  text_field: str = "bio"
  max_subwords: int = 128
  tokenizer_name: str | None = "bert-base-uncased"
  spacy_model: str | None = "en_core_web_sm"
  use_dep_edges: bool = True
  use_sequential_edges: bool = True
  _text_builder: SubwordDepGraphBuilder | None = field(default=None, init=False)
  _tab_builder: TabularBipartiteGraphBuilder | None = field(default=None, init=False)

  def __post_init__(self) -> None:
    if any(s.name == self.text_field and s.kind == FeatureKind.TEXT
        for s in self.schema):
      self._text_builder = SubwordDepGraphBuilder(
        tokenizer_name=self.tokenizer_name,
        spacy_model=self.spacy_model,
        max_subwords=self.max_subwords,
        use_dep_edges=self.use_dep_edges,
        use_sequential_edges=self.use_sequential_edges,
      )
    else:
      self._text_builder = None
    non_text = [s for s in self.schema if s.kind != FeatureKind.TEXT]
    self._tab_builder = TabularBipartiteGraphBuilder(schema=non_text)

  # ------------------------------------------------------------------
  def fit(self, dataset: HiringDataset) -> None:
    if self._text_builder is not None:
      texts = [r.get(self.text_field, "") for r in dataset.records]
      self._text_builder.fit(texts)
    if self._tab_builder is not None:
      self._tab_builder.fit(dataset.records)

  def transform(
    self,
    dataset: HiringDataset,
    *,
    graph_backend: str = "auto",
    init_with: tuple[Any, Any] | None = None,
  ) -> BuiltGraphs:
    """Transform a dataset to BuiltGraphs.

    ``init_with`` is a ``(text_graphs, tabular_graphs)`` pair from
    a previously-built training BuiltGraphs.graphs. Used so the
    test set shares the symbol vocab + hypervector encoding of
    the training set (mandatory under the CAIR Graphs API).
    """
    init_text = init_with[0] if init_with else None
    init_tab = init_with[1] if init_with else None
    if self._text_builder is not None:
      texts = [r.get(self.text_field, "") for r in dataset.records]
      text_built = self._text_builder.transform(
        texts, graph_backend=graph_backend, init_with=init_text,
      )
    else:
      text_built = None
    if self._tab_builder is not None:
      tab_built = self._tab_builder.transform(
        dataset.records, graph_backend=graph_backend, init_with=init_tab,
      )
    else:
      tab_built = None

    if text_built is not None and tab_built is None:
      return text_built
    if text_built is None and tab_built is not None:
      return tab_built
    if text_built is None and tab_built is None: # pragma: no cover
      raise RuntimeError("No graph builder applies to this dataset.")
    return _merge(text_built, tab_built)


def _merge(text_b: BuiltGraphs, tab_b: BuiltGraphs) -> BuiltGraphs:
  """Merge two BuiltGraphs into one.

  The TM backend can only consume one Graphs object, so we keep the
  tabular backend's graph (more compact, more decisive features for
  the regulator-facing explanation) and drop the text-graph
  backend's graph object. Both sides still contribute literal specs
  and active-literal sets, so the explainer / recourse / SAT receipt
  see a unified literal space. The TM itself trains on the tabular
  nodes only in this merged-builder path; deployers who want the TM
  to train on the text dependency graph should instantiate
  ``SubwordDepGraphBuilder`` directly (see
  ``experiments/full_pipeline.py:_build_for_dataset``).
  """
  assert text_b.n_samples == tab_b.n_samples
  n_text = text_b.n_literals
  merged_specs: list[LiteralSpec] = list(text_b.literal_specs)
  for spec in tab_b.literal_specs:
    merged_specs.append(LiteralSpec(
      literal_id=spec.literal_id + n_text,
      feature_name=spec.feature_name,
      bucket_or_category=spec.bucket_or_category,
      feature_kind=spec.feature_kind,
      nl_template=spec.nl_template,
      immutable=spec.immutable,
    ))
  merged_active: list[tuple[int, ...]] = []
  for i in range(text_b.n_samples):
    t_active = set(text_b.per_sample_active_literals[i])
    tab_active = {lid + n_text for lid in tab_b.per_sample_active_literals[i]}
    merged_active.append(tuple(sorted(t_active | tab_active)))
  merged_symbols = list(text_b.symbols) + list(tab_b.symbols)
  merged_edges = list(set(text_b.edge_types) | set(tab_b.edge_types))
  return BuiltGraphs(
    graphs=tab_b.graphs, # tabular only; see docstring
    n_samples=text_b.n_samples,
    symbols=merged_symbols,
    edge_types=merged_edges,
    literal_specs=merged_specs,
    per_sample_active_literals=merged_active,
    meta={
      "builder": "CompositeGraphBuilder",
      "text_meta": text_b.meta,
      "tabular_meta": tab_b.meta,
      "note": (
        "tabular-only backend used for TM training; "
        "text literals contribute to explanations + recourse only."
      ),
    },
  )
