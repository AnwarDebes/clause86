from clause86.eval import accuracy_score, classification_report, macro_f1_score


def test_accuracy_basic():
  assert accuracy_score([0, 1, 1], [0, 1, 0]) == 2 / 3


def test_macro_f1_perfect_is_one():
  assert macro_f1_score([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0


def test_classification_report_returns_per_class():
  rep = classification_report([0, 1, 1, 0], [0, 1, 0, 0], n_classes=2)
  d = rep.to_dict()
  assert d["accuracy"] == 0.75
  assert 0 in d["per_class"] and 1 in d["per_class"]
  assert d["per_class"][1]["recall"] == 0.5
