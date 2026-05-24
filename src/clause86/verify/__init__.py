"""SAT-verified clause receipts.

A :class:`ClauseReceipt` is the auditor-replayable artefact attached
to every Clause86 decision. It records the firing clauses, literal
vector, class scores, recourse trace, and an HMAC-SHA256 signature.

Verification (:func:`verify_receipt`) re-encodes the firing clauses
into DIMACS CNF, runs python-sat / Glucose 4 (or a brute-force
fallback for tiny instances), and re-runs the per-class score
arithmetic.
"""

from .sat_encoder import (
  FiringClauseForSAT,
  encode_clauses_to_dimacs,
  write_dimacs_file,
  solve_cnf,
)
from .receipt import (
  ClauseReceipt,
  ReceiptVerification,
  build_receipt,
  verify_receipt,
  ENV_HMAC_KEY,
)

__all__ = [
  "FiringClauseForSAT",
  "encode_clauses_to_dimacs",
  "write_dimacs_file",
  "solve_cnf",
  "ClauseReceipt",
  "ReceiptVerification",
  "build_receipt",
  "verify_receipt",
  "ENV_HMAC_KEY",
]
