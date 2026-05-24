"""Dataset loaders for Clause86.

Each loader returns a :class:`HiringDataset`, a uniform container
the downstream graph / TM / fairness modules expect.

Available datasets:

* :func:`load_bias_in_bios`, 400k biographies -> occupation label,
 with gender as the protected attribute. Headline benchmark.
* :func:`load_adult_income`, 50k tabular records -> ">$50k" binary
 label, with race + sex protected. Canonical fairness benchmark.
* :func:`load_hr_analytics`, IBM employee-attrition tabular dataset
 (~1.5k rows), gender + age protected. Small sanity dataset.
* :func:`load_synthetic_hiring`, in-process synthetic dataset with
 known ground-truth decision rule. Used for smoke tests and to
 validate faithfulness metrics.
"""

from .dataset import HiringDataset, ProtectedAttribute, FeatureKind, FeatureSpec
from .synthetic import load_synthetic_hiring
from .bias_in_bios import load_bias_in_bios
from .adult_income import load_adult_income
from .hr_analytics import load_hr_analytics

__all__ = [
  "HiringDataset",
  "ProtectedAttribute",
  "FeatureKind",
  "FeatureSpec",
  "load_synthetic_hiring",
  "load_bias_in_bios",
  "load_adult_income",
  "load_hr_analytics",
]
