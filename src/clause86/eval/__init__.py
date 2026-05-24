"""Evaluation, metrics, logging, statistics.

Modules:

* :mod:`clause86.eval.metrics`, accuracy / macro-F1 / per-class
 precision-recall.
* :mod:`clause86.eval.faithfulness`, the "clause-flip ==
 decision-flip" check that distinguishes a faithful clause-based
 explanation from a post-hoc approximation.
* :mod:`clause86.eval.logger`, JSONL per-sample logger.
* :mod:`clause86.eval.stats`, paired bootstrap CI + Wilcoxon +
 Holm-Bonferroni.
"""

from .metrics import (
  ClassificationReport,
  classification_report,
  accuracy_score,
  macro_f1_score,
)
from .faithfulness import (
  FaithfulnessResult,
  check_faithfulness,
)
from .logger import JsonlLogger
from .stats import (
  paired_bootstrap_ci,
  wilcoxon_signed_rank,
  holm_bonferroni,
)

__all__ = [
  "ClassificationReport",
  "classification_report",
  "accuracy_score",
  "macro_f1_score",
  "FaithfulnessResult",
  "check_faithfulness",
  "JsonlLogger",
  "paired_bootstrap_ci",
  "wilcoxon_signed_rank",
  "holm_bonferroni",
]
