import json
import tempfile
import time
from pathlib import Path

from clause86.verify import (
  FiringClauseForSAT,
  build_receipt,
  verify_receipt,
)


def _firing():
  return [
    FiringClauseForSAT(
      clause_id=0, class_id=1, polarity=1, weight=1,
      positive_literal_ids=(0, 2),
      negative_literal_ids=(1,),
    ),
    FiringClauseForSAT(
      clause_id=1, class_id=0, polarity=1, weight=1,
      positive_literal_ids=(1,),
    ),
  ]


def test_receipt_round_trip_passes_verifier():
  literal_vec = [1, 0, 1]
  firing = _firing()
  class_scores = [1, 1] # both clauses fire under the recorded vec
  with tempfile.TemporaryDirectory() as td:
    receipt = build_receipt(
      sample_id="t1",
      dataset_name="unit",
      model_version="vtest",
      deployer_name="UnitTest",
      issued_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      literal_vector=literal_vec,
      firing_clauses=firing,
      class_scores=class_scores,
      predicted_class=0, # tie-breaking arbitrary, test we record what we pass.
      margin=0,
      class_names=["reject", "hire"],
      cnf_dir=td,
    )
    assert Path(receipt.cnf_path).exists()
    ver = verify_receipt(receipt)
    assert ver.signature_ok is True
    assert ver.sat_consistent is True
    assert ver.arithmetic_ok is True
    assert ver.all_ok is True


def test_receipt_detects_tampered_scores():
  literal_vec = [1, 0, 1]
  firing = _firing()
  with tempfile.TemporaryDirectory() as td:
    receipt = build_receipt(
      sample_id="t2", dataset_name="unit", model_version="vtest",
      deployer_name="UnitTest",
      issued_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      literal_vector=literal_vec, firing_clauses=firing,
      class_scores=[1, 1], predicted_class=1, margin=0,
      class_names=["reject", "hire"], cnf_dir=td,
    )
    # Tamper with the receipt without recomputing the HMAC.
    receipt.class_scores = [99, 1]
    ver = verify_receipt(receipt)
    assert ver.signature_ok is False # HMAC mismatch
    assert ver.all_ok is False


def test_receipt_with_wrong_expected_class_fails():
  literal_vec = [1, 0, 1]
  firing = _firing()
  with tempfile.TemporaryDirectory() as td:
    receipt = build_receipt(
      sample_id="t3", dataset_name="unit", model_version="vtest",
      deployer_name="UnitTest",
      issued_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      literal_vector=literal_vec, firing_clauses=firing,
      class_scores=[1, 1], predicted_class=0, margin=0,
      class_names=["reject", "hire"], cnf_dir=td,
    )
    ver = verify_receipt(receipt, expected_predicted_class=1)
    assert ver.arithmetic_ok is False
    assert any("expected predicted_class" in n for n in ver.notes)


def test_receipt_json_serialisable():
  literal_vec = [1, 0]
  firing = [FiringClauseForSAT(clause_id=0, class_id=0, polarity=1, weight=1,
                 positive_literal_ids=(0,))]
  with tempfile.TemporaryDirectory() as td:
    receipt = build_receipt(
      sample_id="t4", dataset_name="unit", model_version="vtest",
      deployer_name="UnitTest",
      issued_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      literal_vector=literal_vec, firing_clauses=firing,
      class_scores=[1, 0], predicted_class=0, margin=1,
      class_names=["reject", "hire"], cnf_dir=td,
    )
    s = json.dumps(receipt.__dict__, default=str)
    assert "signature_hex" in s
