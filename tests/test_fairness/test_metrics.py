from clause86.data.dataset import ProtectedAttribute
from clause86.fairness import compute_fairness_metrics


def _toy_data():
  # Build a 6-sample slice: 3 Female (2 positive, 1 negative; pred 1,0,0)
  #             3 Male  (2 positive, 1 negative; pred 1,1,0)
  labels = [1, 1, 0, 1, 1, 0]
  preds = [1, 0, 0, 1, 1, 0]
  protected = [
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "M"},
    {ProtectedAttribute.GENDER: "M"},
    {ProtectedAttribute.GENDER: "M"},
  ]
  return preds, labels, protected


def test_per_group_rates_present():
  preds, labels, protected = _toy_data()
  out = compute_fairness_metrics(preds, labels, protected,
                  ProtectedAttribute.GENDER, favourable_class_id=1)
  assert "F" in out.per_group and "M" in out.per_group
  assert out.per_group["F"]["n"] == 3.0
  assert out.per_group["M"]["n"] == 3.0


def test_dp_gap_nonnegative():
  preds, labels, protected = _toy_data()
  out = compute_fairness_metrics(preds, labels, protected,
                  ProtectedAttribute.GENDER, favourable_class_id=1)
  assert out.demographic_parity_gap >= 0.0
  assert out.equal_opportunity_gap >= 0.0
  assert out.equalized_odds_gap >= 0.0


def test_perfect_parity_is_zero():
  labels = [1, 0, 1, 0]
  preds = [1, 0, 1, 0]
  protected = [
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "F"},
    {ProtectedAttribute.GENDER: "M"},
    {ProtectedAttribute.GENDER: "M"},
  ]
  out = compute_fairness_metrics(preds, labels, protected,
                  ProtectedAttribute.GENDER, favourable_class_id=1)
  assert out.demographic_parity_gap == 0.0
