"""Subword + dependency multigraph builder for resume / bio text.

Re-ported from ``subword-dep-graphtm`` (Paper A). Differences:

* The builder additionally records the literal-id mapping needed by
 the SAT receipt and the explainer.
* The HuggingFace tokenizer + spaCy model are lazily imported and
 guarded so the smoke-test path works without them (the builder
 raises a precise error message instead of importing).
* Unicode safety: emoji / multi-byte chars in resume bios are
 handled via ``return_offsets_mapping``-aware alignment.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .built_graphs import BuiltGraphs, LiteralSpec


# Dependency relation types tracked.
DEP_TYPES = [
  "nsubj", "obj", "iobj",
  "amod", "advmod", "compound",
  "prep", "pobj",
  "det", "aux",
  "conj", "cc",
  "mark", "advcl", "relcl",
  "ROOT",
]

SEQUENTIAL_EDGE_TYPES = ["seq_next", "seq_prev"]
DEP_EDGE_TYPES = (
  [f"dep:{d}" for d in DEP_TYPES] + [f"dep:{d}_inv" for d in DEP_TYPES]
)
ALL_TEXT_EDGE_TYPES = SEQUENTIAL_EDGE_TYPES + DEP_EDGE_TYPES


@dataclass
class _DocGraph:
  """Internal record for one document's graph (pre-Graphs-object)."""
  subword_tokens: list[str]
  edges: list[tuple[int, int, str]] # (src, dst, edge_type)


class SubwordDepGraphBuilder:
  """Build per-document subword+dependency multigraphs.

  Tokenizer + parser are loaded on first use. Set
  ``tokenizer_name=None`` to skip BERT BPE and use whitespace
  splitting (useful for unit tests).

  Usage::

    builder = SubwordDepGraphBuilder(max_subwords=128)
    builder.fit(train_texts)           # learn vocab
    built = builder.transform(train_texts)    # BuiltGraphs
    builder.transform_partial(test_texts, built) # extend in place
  """

  def __init__(
    self,
    tokenizer_name: str | None = "bert-base-uncased",
    spacy_model: str | None = "en_core_web_sm",
    max_subwords: int = 128,
    use_sequential_edges: bool = True,
    use_dep_edges: bool = True,
    min_token_freq: int = 2,
    max_vocab: int = 10_000,
  ) -> None:
    self.tokenizer_name = tokenizer_name
    self.spacy_model = spacy_model
    self.max_subwords = max_subwords
    self.use_sequential = use_sequential_edges
    self.use_deps = use_dep_edges
    self.min_token_freq = min_token_freq
    self.max_vocab = max_vocab

    self._tokenizer: Any = None
    self._nlp: Any = None
    self._vocab: set[str] | None = None
    self._frozen_symbols: list[str] | None = None

  # ------------------------------------------------------------------
  # Lazy-loaded backends
  # ------------------------------------------------------------------
  def _get_tokenizer(self) -> Any:
    if self._tokenizer is not None:
      return self._tokenizer
    if self.tokenizer_name is None:
      self._tokenizer = _WhitespaceTokenizer()
      return self._tokenizer
    try:
      from transformers import AutoTokenizer
    except ImportError as exc: # pragma: no cover - optional dep
      raise ImportError(
        "Subword graph requires `transformers`. Install with "
        "`pip install clause86[text]` or pass tokenizer_name=None "
        "for whitespace fallback."
      ) from exc
    self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    return self._tokenizer

  def _get_nlp(self) -> Any:
    if self._nlp is not None:
      return self._nlp
    if self.spacy_model is None or not self.use_deps:
      return None
    try:
      import spacy # type: ignore
    except ImportError as exc: # pragma: no cover - optional dep
      raise ImportError(
        "Dependency edges require `spacy`. Install with "
        "`pip install clause86[text]` or pass spacy_model=None / "
        "use_dep_edges=False."
      ) from exc
    try:
      self._nlp = spacy.load(self.spacy_model)
    except OSError: # pragma: no cover - depends on env
      self._nlp = spacy.blank("en")
    return self._nlp

  # ------------------------------------------------------------------
  # Vocabulary
  # ------------------------------------------------------------------
  def fit(self, texts: list[str]) -> None:
    """Learn the symbol vocabulary from ``texts``."""
    counter: Counter[str] = Counter()
    for txt in texts:
      for tok in self._subwordize(txt):
        counter[tok] += 1
    common = [t for t, c in counter.most_common(self.max_vocab) if c >= self.min_token_freq]
    self._vocab = set(common)
    self._frozen_symbols = sorted(self._vocab)

  @property
  def symbols(self) -> list[str]:
    if self._frozen_symbols is None:
      raise RuntimeError("Call fit() before reading symbols.")
    return list(self._frozen_symbols)

  @property
  def edge_types(self) -> list[str]:
    out: list[str] = []
    if self.use_sequential:
      out.extend(SEQUENTIAL_EDGE_TYPES)
    if self.use_deps and self._nlp is not None:
      out.extend(DEP_EDGE_TYPES)
    return out

  # ------------------------------------------------------------------
  # Per-document
  # ------------------------------------------------------------------
  def _subwordize(self, text: str) -> list[str]:
    tok = self._get_tokenizer()
    return tok.tokenize(text)[: self.max_subwords]

  def _build_one(self, text: str) -> _DocGraph:
    tokens = self._subwordize(text or "")
    if not tokens:
      tokens = ["<empty>"]
    edges: list[tuple[int, int, str]] = []

    if self.use_sequential:
      for i in range(len(tokens) - 1):
        edges.append((i, i + 1, "seq_next"))
        edges.append((i + 1, i, "seq_prev"))

    # Dependency edges: align spaCy word-token relations to subwords.
    nlp = self._get_nlp()
    if nlp is not None and self.use_deps and text:
      doc = nlp(text)
      tok = self._get_tokenizer()
      # Tokenizer offsets -> subword index
      encoding = tok(
        text, truncation=True, max_length=self.max_subwords,
        return_offsets_mapping=True, add_special_tokens=False,
      ) if hasattr(tok, "_offsets") is False and hasattr(tok, "__call__") else None
      offsets = encoding["offset_mapping"] if encoding is not None else None

      for word in doc:
        if word.head.i == word.i:
          dep = "ROOT"
        else:
          dep = word.dep_
        if dep not in DEP_TYPES:
          continue
        src_sw = _find_subword_for_char(word.idx, offsets) if offsets else None
        dst_sw = _find_subword_for_char(word.head.idx, offsets) if offsets else None
        if src_sw is None or dst_sw is None:
          continue
        if src_sw == dst_sw:
          continue
        if src_sw >= len(tokens) or dst_sw >= len(tokens):
          continue
        edges.append((src_sw, dst_sw, f"dep:{dep}"))
        edges.append((dst_sw, src_sw, f"dep:{dep}_inv"))

    return _DocGraph(subword_tokens=tokens, edges=edges)

  # ------------------------------------------------------------------
  # Public transform
  # ------------------------------------------------------------------
  def transform(
    self,
    texts: list[str],
    *,
    graph_backend: str = "auto",
    init_with: Any = None,
  ) -> BuiltGraphs:
    """Build BuiltGraphs from a list of texts.

    ``graph_backend``:

    * ``"auto"``, use GraphTsetlinMachine.graphs.Graphs if
     importable, else the lightweight stub.
    * ``"stub"``, always use the stub (used by tests).

    ``init_with`` lets a test-set transform re-use the symbol
    vocabulary / hypervector encoding of a previously-built
    training Graphs object (the CAIR API quirk).
    """
    if self._frozen_symbols is None:
      # Fit on the fly with a permissive vocab.
      self.fit(texts)
    per_doc = [self._build_one(t) for t in texts]
    return _assemble_text_graphs(
      self, per_doc, graph_backend=graph_backend, init_with=init_with,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _find_subword_for_char(char_idx: int, offsets) -> int | None:
  """Map a character offset to the subword index that covers it."""
  if offsets is None:
    return None
  for i, (start, end) in enumerate(offsets):
    if start == 0 and end == 0:
      continue
    if start <= char_idx < end:
      return i
  return None


class _WhitespaceTokenizer:
  """Tiny fallback tokenizer: whitespace + lowercase."""

  def tokenize(self, text: str) -> list[str]:
    return (text or "").lower().split()

  def __call__(self, *args: Any, **kwargs: Any) -> Any: # pragma: no cover
    raise NotImplementedError("Whitespace fallback does not support offsets.")


def _assemble_text_graphs(
  builder: SubwordDepGraphBuilder,
  per_doc: list[_DocGraph],
  *,
  graph_backend: str = "auto",
  init_with: Any = None,
) -> BuiltGraphs:
  """Build the BuiltGraphs payload from per-document _DocGraphs."""
  symbols = builder.symbols
  edge_types = builder.edge_types
  sym_to_id = {s: i for i, s in enumerate(symbols)}
  et_to_id = {e: i for i, e in enumerate(edge_types)}

  # Literal vocabulary: one literal per (symbol present in this doc)
  # plus one literal per (edge_type present in this doc). This keeps
  # the literal count tractable while still exposing the structural
  # information clauses can learn over.
  literal_specs: list[LiteralSpec] = []
  sym_literal_id: dict[str, int] = {}
  for i, s in enumerate(symbols):
    lid = len(literal_specs)
    literal_specs.append(LiteralSpec(
      literal_id=lid,
      feature_name="bio",
      bucket_or_category=s,
      feature_kind="text",
      nl_template="bio mentions {value!r}",
      immutable=False,
    ))
    sym_literal_id[s] = lid
  et_literal_id: dict[str, int] = {}
  for i, e in enumerate(edge_types):
    lid = len(literal_specs)
    literal_specs.append(LiteralSpec(
      literal_id=lid,
      feature_name="bio",
      bucket_or_category=f"edge:{e}",
      feature_kind="text",
      nl_template="bio has a {value!r} dependency edge",
      immutable=False,
    ))
    et_literal_id[e] = lid

  per_sample_active: list[tuple[int, ...]] = []
  for dg in per_doc:
    active: set[int] = set()
    for tok in dg.subword_tokens:
      if tok in sym_literal_id:
        active.add(sym_literal_id[tok])
    for _src, _dst, et in dg.edges:
      if et in et_literal_id:
        active.add(et_literal_id[et])
    per_sample_active.append(tuple(sorted(active)))

  backend = _instantiate_graph_backend(
    per_doc, symbols, edge_types, graph_backend=graph_backend,
    init_with=init_with,
  )

  return BuiltGraphs(
    graphs=backend,
    n_samples=len(per_doc),
    symbols=symbols,
    edge_types=edge_types,
    literal_specs=literal_specs,
    per_sample_active_literals=per_sample_active,
    meta={
      "builder": "SubwordDepGraphBuilder",
      "tokenizer_name": builder.tokenizer_name,
      "spacy_model": builder.spacy_model,
      "max_subwords": builder.max_subwords,
    },
  )


def _instantiate_graph_backend(
  per_doc: list[_DocGraph],
  symbols: list[str],
  edge_types: list[str],
  *,
  graph_backend: str,
  init_with: Any = None,
) -> Any:
  """Pack per-document graphs into a backend object.

  CAIR ``Graphs`` API quirks honoured:

  * Node names are strings (we use ``"n{i}"``).
  * Edge types are registered implicitly via ``add_graph_node_edge``.
  * Vocab sharing across train/test is done with ``init_with=train_graphs``;
   pass through here if the caller has a pre-built training Graphs.
  """
  if graph_backend == "stub":
    return _StubGraphs(per_doc=per_doc, symbols=symbols, edge_types=edge_types)
  try: # pragma: no cover - depends on env
    from GraphTsetlinMachine.graphs import Graphs as GTMGraphs # type: ignore
  except Exception:
    return _StubGraphs(per_doc=per_doc, symbols=symbols, edge_types=edge_types)

  n = len(per_doc)
  if init_with is not None:
    g = GTMGraphs(number_of_graphs=n, init_with=init_with)
  else:
    g = GTMGraphs(
      number_of_graphs=n,
      symbols=symbols if symbols else ["<empty>"],
      hypervector_size=128,
      hypervector_bits=2,
    )
  for gid, dg in enumerate(per_doc):
    g.set_number_of_graph_nodes(gid, max(len(dg.subword_tokens), 1))
  g.prepare_node_configuration()
  for gid, dg in enumerate(per_doc):
    toks = dg.subword_tokens or ["<empty>"]
    for nid, _tok in enumerate(toks):
      n_edges = sum(1 for s, _, _ in dg.edges if s == nid)
      g.add_graph_node(gid, f"n{nid}", n_edges)
  g.prepare_edge_configuration()
  for gid, dg in enumerate(per_doc):
    toks = dg.subword_tokens or ["<empty>"]
    for src, dst, et in dg.edges:
      if src >= len(toks) or dst >= len(toks):
        continue
      try:
        g.add_graph_node_edge(gid, f"n{src}", f"n{dst}", et)
      except Exception:
        continue
    for nid, tok in enumerate(toks):
      if tok in symbols:
        try:
          g.add_graph_node_property(gid, f"n{nid}", tok)
        except Exception:
          continue
  g.encode()
  return g


@dataclass
class _StubGraphs:
  """Lightweight stand-in used when GraphTsetlinMachine is unavailable."""
  per_doc: list[_DocGraph]
  symbols: list[str]
  edge_types: list[str]

  @property
  def n(self) -> int:
    return len(self.per_doc)
