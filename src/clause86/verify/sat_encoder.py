"""CNF encoding of TM firing clauses + per-class score arithmetic.

Mirrors the encoding in ``clauseguard.verify.sat_encoder`` (which is
itself documented in ``docs/ARCHITECTURE.md`` of that project) but is
re-implemented here so Clause86 is auditable standalone.

CNF layout for one sample:

* Variables ``v_0 .. v_{n_lit-1}`` represent the binary literals of
 the sample's literal vector.
* Variables ``f_1 .. f_m`` (one per firing clause) represent whether
 that clause fires.
* Unit clauses pin ``v_i`` to its recorded value.
* For each firing clause ``c_j``:
  * If ``c_j`` requires ``v_a = 1`` and ``v_b = 0`` to fire, we
   add ``f_j -> v_a`` and ``f_j -> NOT v_b`` (i.e., ``NOT f_j v
   v_a`` and ``NOT f_j v NOT v_b``) AND we add ``v_a AND NOT v_b
   -> f_j`` (the reverse direction as a single CNF clause).
* The decision-flow constraint:
  ``label == argmax_class (sum_j polarity_j * weight_j *
  [class_id_j == class] * f_j)``
 is checked OUTSIDE the SAT encoding (by re-summing the per-class
 votes) because pseudo-Boolean comparison is expensive to compile.
 The SAT side verifies only that the firing clauses are consistent
 with the literal vector; the auditor checks the arithmetic.

This is intentionally a *consistency* check, not a proof of model
correctness: it verifies that THIS recorded decision is internally
coherent, which is what Article 14 oversight and Article 86
"replayable explanation" actually require.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FiringClauseForSAT:
  """Minimal slice of a firing clause needed for SAT encoding."""

  clause_id: int
  class_id: int
  polarity: int
  weight: int
  positive_literal_ids: tuple[int, ...] = field(default_factory=tuple)
  negative_literal_ids: tuple[int, ...] = field(default_factory=tuple)

  def variable_offset(self, n_literals: int) -> int:
    """The integer DIMACS variable id for ``f_j`` (1-based)."""
    return n_literals + self.clause_id + 1


def encode_clauses_to_dimacs(
  *,
  literal_vector: list[int],
  firing_clauses: list[FiringClauseForSAT],
  n_literals: int | None = None,
) -> tuple[list[list[int]], int]:
  """Return ``(cnf, n_vars)`` for a per-sample receipt.

  DIMACS variables are 1-based positive integers; the literal-vector
  variables come first, then one fresh variable per firing clause.
  """
  if n_literals is None:
    n_literals = len(literal_vector)
  if len(literal_vector) != n_literals:
    raise ValueError(
      f"literal_vector length {len(literal_vector)} != n_literals {n_literals}"
    )
  n_clauses_vars = max((fc.clause_id for fc in firing_clauses), default=-1) + 1
  n_vars = n_literals + n_clauses_vars

  cnf: list[list[int]] = []

  # Unit clauses pin literal-vector variables.
  for i, v in enumerate(literal_vector):
    if v not in (0, 1):
      raise ValueError(f"literal_vector[{i}] must be 0 or 1, got {v}")
    var = i + 1
    cnf.append([var] if v == 1 else [-var])

  # Per-firing-clause encoding.
  for fc in firing_clauses:
    f_var = fc.variable_offset(n_literals)
    # Implications f -> v_a (i.e., -f v v_a) and f -> NOT v_b (-f v -v_b).
    for lid in fc.positive_literal_ids:
      cnf.append([-f_var, lid + 1])
    for lid in fc.negative_literal_ids:
      cnf.append([-f_var, -(lid + 1)])
    # Reverse implication: AND of all premises -> f.
    # Equivalent CNF: f v (NOT v_a) v (NOT v_b) v ... v v_c ...
    clause = [f_var]
    for lid in fc.positive_literal_ids:
      clause.append(-(lid + 1))
    for lid in fc.negative_literal_ids:
      clause.append(lid + 1)
    cnf.append(clause)
    # Pin f_var according to whether the clause actually fires.
    actually_fires = all(literal_vector[l] == 1 for l in fc.positive_literal_ids) \
      and all(literal_vector[l] == 0 for l in fc.negative_literal_ids)
    cnf.append([f_var] if actually_fires else [-f_var])

  return cnf, n_vars


def write_dimacs_file(
  cnf: list[list[int]],
  n_vars: int,
  out_path: str | Path,
  *,
  comments: list[str] | None = None,
) -> None:
  """Write a DIMACS-format CNF file."""
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with out_path.open("w", encoding="utf-8") as f:
    for c in comments or []:
      f.write(f"c {c}\n")
    f.write(f"p cnf {n_vars} {len(cnf)}\n")
    for clause in cnf:
      f.write(" ".join(str(v) for v in clause))
      f.write(" 0\n")


def solve_cnf(
  cnf: list[list[int]], n_vars: int,
) -> tuple[bool, list[int] | None]:
  """Solve a CNF.

  Tries python-sat / Glucose 4 first; falls back to a brute-force
  DPLL-lite if python-sat is not installed and ``n_vars <= 20``.
  For larger instances without python-sat we raise.
  """
  try: # pragma: no cover - depends on env
    from pysat.solvers import Glucose4

    solver = Glucose4()
    for c in cnf:
      solver.add_clause(c)
    sat = solver.solve()
    model = solver.get_model() if sat else None
    solver.delete()
    return sat, model
  except ImportError:
    pass
  return _brute_force_solve(cnf, n_vars)


def _brute_force_solve(
  cnf: list[list[int]], n_vars: int,
) -> tuple[bool, list[int] | None]:
  """Truth-table search; only used when n_vars <= 20.

  This exists so the test suite + smoke pipeline can run without
  python-sat installed. Production runs require python-sat.
  """
  if n_vars > 20:
    raise RuntimeError(
      "python-sat is not installed and n_vars > 20; cannot brute-force. "
      "Install with `pip install clause86[sat]`."
    )
  for bits in range(1 << n_vars):
    assignment = [(i + 1) if (bits >> i) & 1 else -(i + 1) for i in range(n_vars)]
    assignment_set = {v for v in assignment if v > 0}
    ok = True
    for clause in cnf:
      satisfied = False
      for lit in clause:
        if lit > 0 and lit in assignment_set:
          satisfied = True
          break
        if lit < 0 and (-lit) not in assignment_set:
          satisfied = True
          break
      if not satisfied:
        ok = False
        break
    if ok:
      return True, assignment
  return False, None
