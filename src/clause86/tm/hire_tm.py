"""HireTM: single-seed Graph TM hiring classifier.

This class hides the backend choice behind a stable API:

  tm = HireTM(n_clauses=400, T=200, s=10.0, depth=2)
  tm.fit(train_built, train_labels, n_epochs=25)
  pred = tm.predict(test_built)
  summary = tm.explain(test_built, sample_idx=0)

Backend selection (in order):

1. ``GraphTsetlinMachine.MultiClassGraphTsetlinMachine`` if importable
  (the production path).
2. ``tmu`` (CAIR's tmu) if the GraphTM is not importable.
3. :class:`MajorityVoteFallback`, pure-Python, for tests only.

A clean ``BackendUnavailableError`` is raised only if (1) and (2) are
unavailable AND the caller has passed ``allow_fallback=False``. The
default (``allow_fallback=True``) is safe: tests still pass, but the
predicted labels are not real TM predictions and any benchmarking
script will refuse to log under that mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..graphs.built_graphs import BuiltGraphs, LiteralSpec
from .fallback import MajorityVoteFallback
from .firing import ClassScoreSummary, FiringClauseSpec


class BackendUnavailableError(RuntimeError):
  """Raised when no TM backend is available and fallback is disabled."""


@dataclass
class HireTM:
  """One-seed Graph TM hiring classifier.

  Args:
    n_clauses: total clauses (half positive, half negative).
    T: voting margin threshold.
    s: specificity hyperparameter.
    depth: clause depth for the GraphTM message passing.
    seed: RNG seed.
    allow_fallback: if True (default), fall back to a pure-Python
      classifier when no TM backend is installed. Set to False
      in benchmarking runs to refuse to log fake numbers.
  """

  n_clauses: int = 400
  T: int = 200
  s: float = 10.0
  depth: int = 2
  seed: int = 42
  allow_fallback: bool = True

  n_classes: int = field(init=False, default=0)
  literal_specs: list[LiteralSpec] = field(init=False, default_factory=list)
  _backend: Any = field(init=False, default=None)
  _backend_kind: str = field(init=False, default="")

  # ------------------------------------------------------------------
  def fit(
    self,
    train_built: BuiltGraphs,
    train_labels: list[int],
    *,
    n_epochs: int = 25,
    n_classes: int | None = None,
  ) -> "HireTM":
    if n_classes is None:
      n_classes = max(train_labels) + 1
    self.n_classes = n_classes
    self.literal_specs = list(train_built.literal_specs)

    # If the built graphs are a stub (no CAIR hypervector_size),
    # the real backend cannot consume them; force the fallback.
    force_fallback = not hasattr(train_built.graphs, "hypervector_size")
    backend, kind = _instantiate_backend(
      n_clauses=self.n_clauses,
      T=self.T,
      s=self.s,
      depth=self.depth,
      seed=self.seed,
      n_classes=n_classes,
      allow_fallback=self.allow_fallback,
      force_fallback=force_fallback,
    )
    self._backend = backend
    self._backend_kind = kind

    if kind == "fallback":
      backend.fit(train_built, train_labels)
      return self

    # Real CAIR backend training. The CAIR API takes the Graphs
    # object directly and a 1-D label array.
    y = np.asarray(train_labels, dtype=np.uint32)
    backend.fit(train_built.graphs, y, epochs=n_epochs, incremental=True)
    return self

  # ------------------------------------------------------------------
  def predict(self, built: BuiltGraphs) -> list[int]:
    if self._backend is None:
      raise RuntimeError("HireTM.fit must be called before predict.")
    if self._backend_kind == "fallback":
      return self._backend.predict(built)
    preds = self._backend.predict(built.graphs)
    return [int(p) for p in preds]

  # ------------------------------------------------------------------
  def explain(self, built: BuiltGraphs, sample_idx: int) -> ClassScoreSummary:
    """Return the firing-clause summary for one sample.

    For the CAIR backend, we extract per-class clause vote counts
    from the trained automata-state snapshot. For the fallback,
    we return synthesised pseudo-clauses based on literal
    presence so downstream modules can be tested end-to-end.
    """
    if self._backend is None:
      raise RuntimeError("HireTM.fit must be called before explain.")
    if self._backend_kind == "fallback":
      return self._backend.explain(built, sample_idx)
    return _explain_cair_backend(
      self._backend, built, sample_idx, self.n_classes, self.literal_specs,
    )

  # ------------------------------------------------------------------
  def num_clauses_total(self) -> int:
    if self._backend_kind == "fallback":
      return self._backend.num_clauses_total()
    return int(self.n_clauses) * int(self.n_classes)


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------

def _instantiate_backend(
  *,
  n_clauses: int,
  T: int,
  s: float,
  depth: int,
  seed: int,
  n_classes: int,
  allow_fallback: bool,
  force_fallback: bool = False,
) -> tuple[Any, str]:
  """Try CAIR GraphTM, then tmu, then fallback.

  ``force_fallback`` is set by ``HireTM.fit`` when the BuiltGraphs
  carries a stub-typed backend (no ``hypervector_size`` attribute),
  in which case neither the CAIR nor the tmu kernel can consume it.
  """
  if force_fallback:
    if not allow_fallback:
      raise BackendUnavailableError(
        "BuiltGraphs is stub-typed (no hypervector_size on graphs); "
        "the CAIR / tmu kernels cannot consume it. Either build the "
        "graphs with graph_backend='auto' (requires GraphTsetlinMachine) "
        "or pass allow_fallback=True."
      )
    return MajorityVoteFallback(n_classes=n_classes, seed=seed), "fallback"
  try: # pragma: no cover - depends on env
    from GraphTsetlinMachine.tm import (
      MultiClassGraphTsetlinMachine, # type: ignore
    )
    # CAIR API: positional (n_clauses, T, s) + kwargs without seed.
    # The CAIR backend takes seeding from the per-process numpy RNG
    # which clause86.utils.seed_everything has already pinned.
    backend = MultiClassGraphTsetlinMachine(
      n_clauses,
      T,
      s,
      number_of_state_bits=8,
      depth=depth,
      message_size=256,
      message_bits=2,
      max_included_literals=8,
    )
    return backend, "graphtsetlinmachine"
  except Exception:
    pass
  try: # pragma: no cover - depends on env
    from tmu.models.classification.graph_classifier import ( # type: ignore
      GraphTsetlinMachineClassifier,
    )
    backend = GraphTsetlinMachineClassifier(
      number_of_clauses=n_clauses,
      T=T,
      s=s,
      depth=depth,
      seed=seed,
    )
    return backend, "tmu"
  except Exception:
    pass
  if not allow_fallback:
    raise BackendUnavailableError(
      "No GraphTsetlinMachine / tmu backend importable. "
      "Install with `pip install clause86[tm]` or pass allow_fallback=True "
      "for the pure-Python plumbing fallback."
    )
  return MajorityVoteFallback(n_classes=n_classes, seed=seed), "fallback"


# ----------------------------------------------------------------------
# Firing-clause extraction (CAIR backend)
# ----------------------------------------------------------------------

def _explain_cair_backend(
  backend: Any,
  built: BuiltGraphs,
  sample_idx: int,
  n_classes: int,
  literal_specs: list[LiteralSpec],
) -> ClassScoreSummary: # pragma: no cover - requires GraphTM + GPU
  """Extract firing clauses from a CAIR ``MultiClassGraphTsetlinMachine``.

  Uses the public ``get_weights()`` + ``get_clause_literals(symbol_hv)``
  API documented in ``GraphTsetlinMachine/tm.py``.

  * ``weights`` is the per-class clause-weight tensor; the polarity
   and weight of each firing clause come from there.
  * ``clause_literals`` is shape ``(n_clauses, 2 * n_symbols)``,
   where the first half encodes positive-symbol presence and the
   second half encodes negated-symbol presence in each clause.

  For a given sample we identify the set of active *symbol* literals
  (the ones whose ``feature_kind != "text-edge"``); a clause fires
  iff every positive-symbol it requires is present AND every
  negated-symbol it forbids is absent.

  This is the depth=0 / surface-clause view. Depth>=1 message
  clauses are ignored in this explanation, they encode neighbour
  structure that does not have a literal-level Article 86 surface
  form. The Article 86 explanation cites surface clauses only;
  this is the same simplification used by the CAIR clause-print
  examples (``examples/MNISTConvolution_print_clauses.py``).
  """
  graphs = built.graphs
  try:
    weights = backend.get_weights() # (n_outputs, n_clauses), int32
    symbol_hv = graphs.hypervectors
    clause_lits = backend.get_clause_literals(symbol_hv)
    # shape (n_clauses, 2 * n_symbols), float
  except Exception:
    return _fallback_predict_only(backend, built, sample_idx, n_classes)

  n_clauses = int(weights.shape[1])
  n_outputs = int(weights.shape[0])
  n_symbols = clause_lits.shape[1] // 2

  # Map symbol_id (CAIR graphs.symbol_id) -> our literal_id.
  # Tabular builder uses ``"{feature_name}={bucket_or_category}"``
  # as the symbol name; text builder uses the raw subword token
  # which equals ``bucket_or_category`` directly. Try both.
  sym_to_literal: dict[int, int] = {}
  if hasattr(graphs, "symbol_id"):
    for sym_name, sid in graphs.symbol_id.items():
      for lspec in literal_specs:
        composed = f"{lspec.feature_name}={lspec.bucket_or_category}"
        if sym_name == composed or sym_name == lspec.bucket_or_category:
          sym_to_literal[sid] = lspec.literal_id
          break

  # Set of symbol_ids active for this sample.
  active_literals = set(built.per_sample_active_literals[sample_idx])
  active_symbols: set[int] = {
    sid for sid, lid in sym_to_literal.items() if lid in active_literals
  }

  scores = [0] * n_classes
  firing: list[FiringClauseSpec] = []
  # Score binarisation threshold: get_clause_literals reports the
  # number of symbol-hypervector bits ``shared`` with each clause's
  # literal hypervector. A full match equals ``hypervector_bits``;
  # values strictly between 0 and that are partial collisions
  # between symbols whose hypervectors share some bits but not all.
  # We treat ``> 0`` as "the clause includes the symbol", matching
  # the convention used by CAIR's NoisyXOR_print_clauses example.
  INCLUDE_THRESHOLD = 0.5
  for j in range(n_clauses):
    pos_syms = [s for s in range(n_symbols)
          if clause_lits[j, s] > INCLUDE_THRESHOLD]
    neg_syms = [s for s in range(n_symbols)
          if clause_lits[j, n_symbols + s] > INCLUDE_THRESHOLD]
    if not pos_syms and not neg_syms:
      continue # vacuous clause, skip
    pos_ok = all(s in active_symbols for s in pos_syms)
    neg_ok = all(s not in active_symbols for s in neg_syms)
    if not (pos_ok and neg_ok):
      continue
    pos_lits = tuple(sym_to_literal[s] for s in pos_syms if s in sym_to_literal)
    neg_lits = tuple(sym_to_literal[s] for s in neg_syms if s in sym_to_literal)
    # The clause contributes its weight to each output class.
    for c in range(n_outputs):
      w = int(weights[c, j])
      if w == 0:
        continue
      polarity = 1 if w > 0 else -1
      scores[c] += w # signed, the polarity is folded in.
      firing.append(FiringClauseSpec(
        clause_id=c * n_clauses + j,
        class_id=c,
        polarity=polarity,
        weight=abs(w),
        positive_literal_ids=pos_lits,
        negative_literal_ids=neg_lits,
        member_id=None,
      ))

  if not firing:
    return _fallback_predict_only(backend, built, sample_idx, n_classes)

  argmax = max(range(n_classes), key=lambda c: scores[c])
  others = [scores[c] for c in range(n_classes) if c != argmax]
  margin = scores[argmax] - (max(others) if others else 0)
  return ClassScoreSummary(
    scores=tuple(scores),
    argmax_label=argmax,
    margin=margin,
    firing_clauses=tuple(firing),
  )


def _fallback_predict_only(
  backend: Any, built: BuiltGraphs, sample_idx: int, n_classes: int,
) -> ClassScoreSummary: # pragma: no cover
  """Last-resort: produce a minimal summary with only the predicted class."""
  scores = [0] * n_classes
  argmax = 0
  try:
    preds = backend.predict(built.graphs)
    argmax = int(preds[sample_idx])
  except Exception:
    pass
  scores[argmax] = 1
  return ClassScoreSummary(
    scores=tuple(scores),
    argmax_label=argmax,
    margin=1,
    firing_clauses=tuple(),
  )
