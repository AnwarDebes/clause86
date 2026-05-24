from clause86.verify import FiringClauseForSAT, encode_clauses_to_dimacs
from clause86.verify.sat_encoder import solve_cnf


def test_encoding_for_satisfied_clause_is_sat():
  literal_vec = [1, 0, 1]
  firing = [
    FiringClauseForSAT(
      clause_id=0, class_id=0, polarity=1, weight=1,
      positive_literal_ids=(0, 2),
      negative_literal_ids=(1,),
    ),
  ]
  cnf, n_vars = encode_clauses_to_dimacs(
    literal_vector=literal_vec, firing_clauses=firing,
  )
  assert n_vars == len(literal_vec) + 1 # 3 vars + 1 firing var
  sat, model = solve_cnf(cnf, n_vars)
  assert sat is True
  # The firing variable (var 4) should be True under the recorded vector.
  assert 4 in model


def test_encoding_for_violated_clause_pins_f_false():
  literal_vec = [0, 0, 1]
  firing = [
    FiringClauseForSAT(
      clause_id=0, class_id=0, polarity=1, weight=1,
      positive_literal_ids=(0,), # but literal 0 is 0
    ),
  ]
  cnf, n_vars = encode_clauses_to_dimacs(
    literal_vector=literal_vec, firing_clauses=firing,
  )
  sat, model = solve_cnf(cnf, n_vars)
  assert sat is True
  # f_0 (var 4) is forced negative because the clause does NOT
  # fire under the recorded vector.
  assert -4 in model


def test_encoding_rejects_wrong_length_literal_vector():
  import pytest

  with pytest.raises(ValueError):
    encode_clauses_to_dimacs(
      literal_vector=[0, 1, 1], firing_clauses=[], n_literals=4,
    )


def test_encoding_rejects_non_binary_literals():
  import pytest

  with pytest.raises(ValueError):
    encode_clauses_to_dimacs(
      literal_vector=[1, 2, 0], firing_clauses=[],
    )
