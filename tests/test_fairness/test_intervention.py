from clause86.data.dataset import ProtectedAttribute
from clause86.fairness import apply_clause_deletion
from clause86.tm.firing import ClassScoreSummary, FiringClauseSpec


def _make_summary(predicted_class: int, with_clause_id: int) -> ClassScoreSummary:
  firing = (
    FiringClauseSpec(clause_id=with_clause_id, class_id=predicted_class,
             polarity=1, weight=2,
             positive_literal_ids=(0,)),
  )
  scores = [0, 0]
  scores[predicted_class] = 2
  return ClassScoreSummary(
    scores=tuple(scores), argmax_label=predicted_class, margin=2,
    firing_clauses=firing,
  )


def test_deletion_changes_prediction_when_only_clause_voted():
  summaries = [_make_summary(1, with_clause_id=99) for _ in range(3)]
  labels = [0, 0, 0]
  protected = [{ProtectedAttribute.GENDER: "F"}] * 3
  result = apply_clause_deletion(
    summaries=summaries, labels=labels, protected=protected,
    attribute=ProtectedAttribute.GENDER, deleted_clause_ids=[99],
    favourable_class_id=1, n_classes=2,
  )
  # Before: all predicted 1. After: clause 99 deleted, scores all
  # 0-0 -> argmax falls back to class 0.
  assert all(p == 1 for p in result.before_predictions)
  assert all(p == 0 for p in result.after_predictions)


def test_deletion_with_no_overlap_is_no_op():
  summaries = [_make_summary(1, with_clause_id=99)]
  labels = [1]
  protected = [{ProtectedAttribute.GENDER: "F"}]
  result = apply_clause_deletion(
    summaries=summaries, labels=labels, protected=protected,
    attribute=ProtectedAttribute.GENDER, deleted_clause_ids=[7],
    favourable_class_id=1, n_classes=2,
  )
  assert result.before_predictions == result.after_predictions
  assert result.accuracy_before == result.accuracy_after
