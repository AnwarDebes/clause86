"""Bipartite tabular graph builder for HR / Adult-Income features.

For tabular data we use a single-node graph per sample: every sample
becomes one node carrying its bucketed/categorical literals. This
gives a fair point of comparison with a vanilla multiclass TM while
keeping the same GraphTM pipeline that the text variant uses.

Edges are not strictly required for tabular (it degenerates to a TM
over node properties), but we expose an ``add_self_loop`` option that
adds a single ``self`` edge so clauses can still reference edge-type
literals if the user wants.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..data.dataset import FeatureKind, FeatureSpec
from .built_graphs import BuiltGraphs, LiteralSpec


@dataclass
class TabularBipartiteGraphBuilder:
  """Build tabular bipartite graphs from a list of dict records.

  Args:
    schema: list of FeatureSpec describing the tabular columns.
    add_self_loop: emit a ``self`` edge so clauses may reference
      edge-type literals.
    ordinal_thresholds: dict mapping feature_name -> list of cut
      points for thermometer encoding. If unset, derived from
      the data on ``fit``.
  """

  schema: list[FeatureSpec]
  add_self_loop: bool = True
  ordinal_thresholds: dict[str, list[float]] | None = None

  def __post_init__(self) -> None:
    self.ordinal_thresholds = dict(self.ordinal_thresholds or {})
    self._fitted = False
    self._literal_specs: list[LiteralSpec] = []
    self._literal_lookup: dict[tuple[str, str], int] = {}
    self._symbols: list[str] = []
    self._symbol_set: set[str] = set()
    self._edge_types: list[str] = ["self"] if self.add_self_loop else []

  # ------------------------------------------------------------------
  # Fit
  # ------------------------------------------------------------------
  def fit(self, records: Iterable[dict[str, Any]]) -> None:
    """Learn ordinal thresholds + categorical vocab from data."""
    records = list(records)
    self._literal_specs = []
    self._literal_lookup = {}
    self._symbols = []
    self._symbol_set = set()

    for spec in self.schema:
      if spec.kind == FeatureKind.IMMUTABLE:
        # Immutable features still get literals, recourse must
        # not edit them, but explainer + audit may reference them.
        self._fit_categorical(spec, records, immutable=True)
      elif spec.kind == FeatureKind.CATEGORICAL:
        self._fit_categorical(spec, records, immutable=False)
      elif spec.kind == FeatureKind.ORDINAL:
        self._fit_ordinal(spec, records)
      elif spec.kind == FeatureKind.BINARY:
        self._fit_binary(spec)
      elif spec.kind == FeatureKind.TEXT:
        continue # Handled by SubwordDepGraphBuilder, not here.
      else: # pragma: no cover - exhaustive enum
        raise ValueError(f"Unhandled feature kind: {spec.kind}")
    self._fitted = True

  def _add_literal(
    self,
    spec: FeatureSpec,
    bucket_or_category: str,
    nl_template: str | None = None,
    immutable: bool | None = None,
  ) -> int:
    key = (spec.name, bucket_or_category)
    if key in self._literal_lookup:
      return self._literal_lookup[key]
    lid = len(self._literal_specs)
    is_immut = immutable if immutable is not None else (
      spec.kind == FeatureKind.IMMUTABLE
    )
    self._literal_specs.append(LiteralSpec(
      literal_id=lid,
      feature_name=spec.name,
      bucket_or_category=bucket_or_category,
      feature_kind=spec.kind.value,
      nl_template=nl_template or spec.nl_template,
      immutable=is_immut,
    ))
    self._literal_lookup[key] = lid
    symbol = f"{spec.name}={bucket_or_category}"
    if symbol not in self._symbol_set:
      self._symbols.append(symbol)
      self._symbol_set.add(symbol)
    return lid

  def _fit_categorical(
    self,
    spec: FeatureSpec,
    records: list[dict[str, Any]],
    *,
    immutable: bool,
  ) -> None:
    cats = list(spec.categories) if spec.categories else []
    if not cats:
      seen: list[str] = []
      seen_set: set[str] = set()
      for r in records:
        v = str(r.get(spec.name, ""))
        if v and v not in seen_set:
          seen.append(v)
          seen_set.add(v)
      cats = seen
    for cat in cats:
      self._add_literal(spec, cat, immutable=immutable)

  def _fit_ordinal(
    self,
    spec: FeatureSpec,
    records: list[dict[str, Any]],
  ) -> None:
    thresholds = self.ordinal_thresholds.get(spec.name)
    if thresholds is None:
      values = [r.get(spec.name) for r in records if r.get(spec.name) is not None]
      values = [float(v) for v in values if v is not None]
      if not values:
        thresholds = []
      else:
        n = max(spec.n_buckets, 2)
        values = sorted(values)
        step = max(1, len(values) // n)
        thresholds = sorted({float(values[i]) for i in range(0, len(values), step)})
        if len(thresholds) > n:
          thresholds = thresholds[:n]
      self.ordinal_thresholds[spec.name] = thresholds
    for t in thresholds:
      label = f">={t:g}"
      self._add_literal(spec, label)

  def _fit_binary(self, spec: FeatureSpec) -> None:
    self._add_literal(spec, "1")

  # ------------------------------------------------------------------
  # Transform
  # ------------------------------------------------------------------
  def transform(
    self,
    records: list[dict[str, Any]],
    *,
    graph_backend: str = "auto",
    init_with: Any = None,
  ) -> BuiltGraphs:
    if not self._fitted:
      self.fit(records)

    per_sample_active: list[tuple[int, ...]] = []
    per_sample_node_symbols: list[list[str]] = []

    for r in records:
      active: set[int] = set()
      node_symbols: list[str] = []
      for spec in self.schema:
        v = r.get(spec.name)
        if spec.kind in (FeatureKind.CATEGORICAL, FeatureKind.IMMUTABLE):
          if v is None:
            continue
          key = (spec.name, str(v))
          if key in self._literal_lookup:
            lid = self._literal_lookup[key]
            active.add(lid)
            node_symbols.append(f"{spec.name}={v}")
        elif spec.kind == FeatureKind.ORDINAL:
          if v is None:
            continue
          try:
            vf = float(v)
          except (TypeError, ValueError):
            continue
          for t in self.ordinal_thresholds.get(spec.name, []):
            if vf >= t:
              label = f">={t:g}"
              key = (spec.name, label)
              if key in self._literal_lookup:
                active.add(self._literal_lookup[key])
                node_symbols.append(f"{spec.name}={label}")
        elif spec.kind == FeatureKind.BINARY:
          if v:
            key = (spec.name, "1")
            if key in self._literal_lookup:
              active.add(self._literal_lookup[key])
              node_symbols.append(f"{spec.name}=1")
        elif spec.kind == FeatureKind.TEXT:
          continue
      per_sample_active.append(tuple(sorted(active)))
      per_sample_node_symbols.append(node_symbols)

    backend = _instantiate_tabular_backend(
      per_sample_node_symbols,
      symbols=self._symbols,
      edge_types=self._edge_types,
      add_self_loop=self.add_self_loop,
      graph_backend=graph_backend,
      init_with=init_with,
    )

    return BuiltGraphs(
      graphs=backend,
      n_samples=len(records),
      symbols=list(self._symbols),
      edge_types=list(self._edge_types),
      literal_specs=list(self._literal_specs),
      per_sample_active_literals=per_sample_active,
      meta={
        "builder": "TabularBipartiteGraphBuilder",
        "add_self_loop": self.add_self_loop,
        "ordinal_thresholds": {
          k: list(v) for k, v in self.ordinal_thresholds.items()
        },
      },
    )


def _instantiate_tabular_backend(
  per_sample_node_symbols: list[list[str]],
  *,
  symbols: list[str],
  edge_types: list[str],
  add_self_loop: bool,
  graph_backend: str,
  init_with: Any = None,
) -> Any:
  """Pack per-sample symbol sets into a graph backend.

  Layout (matches the CAIR pattern verified on the synthetic task):

  * One node per sample carrying ALL active symbol properties.
  * Optional self-edge (kept for compatibility with the depth>=1
   message-passing path; harmless when depth=1).
  * ``init_with`` shares the symbol vocab + hypervectors with a
   previously-built training Graphs (mandatory for train/test
   symbol-id identity under the CAIR API).
  """
  if graph_backend == "stub":
    return _StubTabular(
      per_sample_node_symbols=per_sample_node_symbols,
      symbols=symbols,
      edge_types=edge_types,
    )
  try: # pragma: no cover - depends on env
    from GraphTsetlinMachine.graphs import Graphs as GTMGraphs # type: ignore
  except Exception:
    return _StubTabular(
      per_sample_node_symbols=per_sample_node_symbols,
      symbols=symbols,
      edge_types=edge_types,
    )
  n = len(per_sample_node_symbols)
  if init_with is not None:
    g = GTMGraphs(number_of_graphs=n, init_with=init_with)
  else:
    g = GTMGraphs(
      number_of_graphs=n,
      symbols=symbols if symbols else ["<empty>"],
      hypervector_size=128,
      hypervector_bits=2,
    )
  # Single node per sample.
  for gid in range(n):
    g.set_number_of_graph_nodes(gid, 1)
  g.prepare_node_configuration()
  for gid in range(n):
    n_edges = 1 if add_self_loop else 0
    g.add_graph_node(gid, "n0", n_edges)
  g.prepare_edge_configuration()
  for gid, syms in enumerate(per_sample_node_symbols):
    if add_self_loop and "self" in edge_types:
      try:
        g.add_graph_node_edge(gid, "n0", "n0", "self")
      except Exception:
        pass
    for sym in (syms or []):
      if sym in symbols:
        try:
          g.add_graph_node_property(gid, "n0", sym)
        except Exception:
          pass
  g.encode()
  return g


@dataclass
class _StubTabular:
  per_sample_node_symbols: list[list[str]]
  symbols: list[str]
  edge_types: list[str]
