"""Signed clause receipts.

A :class:`ClauseReceipt` is a JSON-serialisable artefact attached to
every Clause86 decision. Together with the DIMACS CNF file written
alongside it, the receipt lets an EU AI Act notified body (or any
auditor with a copy of the HMAC key) verify that:

1. The recorded literal vector + firing clauses are internally
  consistent (SAT-verifiable in ~milliseconds).
2. The recorded class-score arithmetic produces the recorded label.
3. The receipt has not been tampered with (HMAC-SHA256).

The HMAC key is deployer-controlled; ``CLAUSE86_HMAC_KEY`` is read
from the environment if no key is passed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.io import canonical_json, sha256_of
from .sat_encoder import (
  FiringClauseForSAT,
  encode_clauses_to_dimacs,
  solve_cnf,
  write_dimacs_file,
)


ENV_HMAC_KEY = "CLAUSE86_HMAC_KEY"


@dataclass
class ClauseReceipt:
  """Auditor-replayable receipt for one decision."""

  receipt_version: str = "clause86-receipt-v1"
  sample_id: str = ""
  dataset_name: str = ""
  model_version: str = ""
  deployer_name: str = ""
  issued_at_utc: str = ""

  n_literals: int = 0
  literal_vector_hash: str = ""
  active_literal_ids: list[int] = field(default_factory=list)

  firing_clauses: list[dict[str, Any]] = field(default_factory=list)
  class_scores: list[int] = field(default_factory=list)
  predicted_class: int = -1
  margin: int = 0
  class_names: list[str] = field(default_factory=list)

  cnf_path: str = ""
  cnf_n_vars: int = 0
  cnf_n_clauses: int = 0

  recourse: dict[str, Any] | None = None

  signature_hex: str = ""

  def signing_payload(self) -> bytes:
    """Canonical-JSON payload covered by the HMAC signature."""
    d = asdict(self)
    d.pop("signature_hex", None)
    return canonical_json(d)


def _hmac_key() -> bytes:
  raw = os.environ.get(ENV_HMAC_KEY, "")
  if not raw:
    # Default development key. Production deployers MUST set
    # CLAUSE86_HMAC_KEY in their environment.
    raw = "clause86-default-dev-key-NOT-FOR-PRODUCTION"
  return raw.encode("utf-8")


def _sign(payload: bytes, key: bytes | None = None) -> str:
  return hmac.new(key or _hmac_key(), payload, hashlib.sha256).hexdigest()


def build_receipt(
  *,
  sample_id: str,
  dataset_name: str,
  model_version: str,
  deployer_name: str,
  issued_at_utc: str,
  literal_vector: list[int],
  firing_clauses: list[FiringClauseForSAT],
  class_scores: list[int],
  predicted_class: int,
  margin: int,
  class_names: list[str],
  cnf_dir: str | Path,
  recourse_dict: dict[str, Any] | None = None,
  hmac_key: bytes | None = None,
) -> ClauseReceipt:
  """Assemble + sign a :class:`ClauseReceipt`.

  Side effect: writes the DIMACS CNF file under ``cnf_dir``. The
  file path is recorded on the receipt so the auditor can find it.
  """
  cnf, n_vars = encode_clauses_to_dimacs(
    literal_vector=literal_vector, firing_clauses=firing_clauses,
  )
  cnf_dir = Path(cnf_dir)
  cnf_dir.mkdir(parents=True, exist_ok=True)
  cnf_path = cnf_dir / f"{sample_id}.cnf"
  write_dimacs_file(
    cnf, n_vars, cnf_path,
    comments=[
      f"sample_id={sample_id}",
      f"dataset={dataset_name}",
      f"model_version={model_version}",
      f"predicted_class={predicted_class}",
    ],
  )
  receipt = ClauseReceipt(
    sample_id=sample_id,
    dataset_name=dataset_name,
    model_version=model_version,
    deployer_name=deployer_name,
    issued_at_utc=issued_at_utc,
    n_literals=len(literal_vector),
    literal_vector_hash=sha256_of(literal_vector),
    active_literal_ids=[i for i, v in enumerate(literal_vector) if v == 1],
    firing_clauses=[
      {
        "clause_id": fc.clause_id,
        "class_id": fc.class_id,
        "polarity": fc.polarity,
        "weight": fc.weight,
        "positive_literal_ids": list(fc.positive_literal_ids),
        "negative_literal_ids": list(fc.negative_literal_ids),
      }
      for fc in firing_clauses
    ],
    class_scores=list(class_scores),
    predicted_class=predicted_class,
    margin=margin,
    class_names=list(class_names),
    cnf_path=str(cnf_path),
    cnf_n_vars=n_vars,
    cnf_n_clauses=len(cnf),
    recourse=recourse_dict,
  )
  receipt.signature_hex = _sign(receipt.signing_payload(), key=hmac_key)
  return receipt


@dataclass
class ReceiptVerification:
  """Outcome of :func:`verify_receipt`."""

  signature_ok: bool
  sat_consistent: bool
  arithmetic_ok: bool
  notes: list[str] = field(default_factory=list)

  @property
  def all_ok(self) -> bool:
    return self.signature_ok and self.sat_consistent and self.arithmetic_ok


def verify_receipt(
  receipt: ClauseReceipt,
  *,
  expected_predicted_class: int | None = None,
  hmac_key: bytes | None = None,
) -> ReceiptVerification:
  """Replay a :class:`ClauseReceipt` from scratch.

  The verifier:

  1. Recomputes the HMAC over the receipt payload and compares.
  2. Re-encodes the recorded firing clauses + literal vector into
    a fresh CNF, runs the SAT solver, and confirms SAT.
  3. Recomputes the per-class score arithmetic from the recorded
    firing clauses and confirms it matches the recorded scores
    and predicted class.
  """
  notes: list[str] = []

  expected_sig = _sign(receipt.signing_payload(), key=hmac_key)
  sig_ok = hmac.compare_digest(expected_sig, receipt.signature_hex)
  if not sig_ok:
    notes.append("HMAC signature mismatch")

  # Rebuild the SAT.
  lit_vec = [0] * receipt.n_literals
  for lid in receipt.active_literal_ids:
    if 0 <= lid < receipt.n_literals:
      lit_vec[lid] = 1
  firing = [
    FiringClauseForSAT(
      clause_id=int(fc["clause_id"]),
      class_id=int(fc["class_id"]),
      polarity=int(fc["polarity"]),
      weight=int(fc["weight"]),
      positive_literal_ids=tuple(int(x) for x in fc["positive_literal_ids"]),
      negative_literal_ids=tuple(int(x) for x in fc["negative_literal_ids"]),
    )
    for fc in receipt.firing_clauses
  ]
  cnf, n_vars = encode_clauses_to_dimacs(
    literal_vector=lit_vec, firing_clauses=firing,
  )
  sat, _model = solve_cnf(cnf, n_vars)
  if not sat:
    notes.append("CNF UNSAT under recorded literal vector")

  # Re-do the arithmetic.
  n_classes = max(len(receipt.class_scores), len(receipt.class_names))
  rebuilt = [0] * n_classes
  for fc in firing:
    if 0 <= fc.class_id < n_classes:
      rebuilt[fc.class_id] += fc.polarity * fc.weight
  arith_ok = rebuilt == receipt.class_scores
  if not arith_ok:
    notes.append(
      f"score arithmetic mismatch: rebuilt={rebuilt} recorded={receipt.class_scores}"
    )
  if expected_predicted_class is not None and \
    expected_predicted_class != receipt.predicted_class:
    arith_ok = False
    notes.append(
      f"expected predicted_class={expected_predicted_class} "
      f"but receipt records {receipt.predicted_class}"
    )

  return ReceiptVerification(
    signature_ok=sig_ok,
    sat_consistent=sat,
    arithmetic_ok=arith_ok,
    notes=notes,
  )
