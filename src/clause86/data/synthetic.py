"""Synthetic hiring dataset with a KNOWN ground-truth decision rule.

This dataset exists for two reasons:

1. **Smoke tests** that run in milliseconds without any external
  downloads, so the full pipeline (graph -> TM -> explain -> recourse
  -> SAT verify -> fairness audit) can be tested in CI.

2. **Faithfulness validation**: the explainer's claim is that "the
  firing clauses ARE the main elements of the decision." We can
  only audit this claim if we know what those elements ARE. With a
  synthetic dataset the ground-truth rule is hand-written and the
  faithfulness metric (clause-flip == decision-flip) has an absolute
  reference.

Decision rule (deliberately readable):

  HIRE iff (years_experience >= 3)
      AND (has_degree)
      AND (NOT empty_resume)
    OR (years_experience >= 10)
      AND (NOT empty_resume)

Gender is included as the protected attribute but DOES NOT influence
the rule, so a faithful TM should learn clauses that do not
reference it; an unfaithful or biased TM may pick up spurious
correlations the audit then catches.

The text bio field is generated from a templated grammar so that the
subword-dependency graph builder has something to chew on.
"""
from __future__ import annotations

import random
from typing import Any

from .dataset import (
  FeatureKind,
  FeatureSpec,
  HiringDataset,
  ProtectedAttribute,
)


_SCHEMA: list[FeatureSpec] = [
  FeatureSpec(name="bio", kind=FeatureKind.TEXT,
        nl_template="bio mentions {value!r}"),
  FeatureSpec(name="years_experience", kind=FeatureKind.ORDINAL,
        n_buckets=6,
        nl_template="years of experience >= {bucket}"),
  FeatureSpec(name="has_degree", kind=FeatureKind.BINARY,
        nl_template="has university degree"),
  FeatureSpec(name="empty_resume", kind=FeatureKind.BINARY,
        nl_template="resume is empty"),
  FeatureSpec(name="age_group", kind=FeatureKind.IMMUTABLE,
        categories=("18-25", "26-40", "41-60", "60+"),
        nl_template="age group is {value}"),
  FeatureSpec(name="gender", kind=FeatureKind.IMMUTABLE,
        categories=("F", "M", "X"),
        nl_template="gender is {value}"),
]


_BIO_TEMPLATES = [
  "{years} years of {field} experience, {degree_phrase}.",
  "Worked as {role} for {years} years. {degree_phrase}.",
  "{degree_phrase} Specialised in {field}. {years} years on the job.",
  "{role}. {years} years building {field} systems. {degree_phrase}",
]
_FIELDS = ["software", "data", "ML", "infrastructure", "security", "frontend"]
_ROLES = ["engineer", "developer", "architect", "researcher", "consultant"]


def _bio(years: int, has_degree: bool, empty: bool, rng: random.Random) -> str:
  if empty:
    return ""
  template = rng.choice(_BIO_TEMPLATES)
  return template.format(
    years=years,
    field=rng.choice(_FIELDS),
    role=rng.choice(_ROLES),
    degree_phrase=("MSc holder" if has_degree else "Self-taught"),
  )


def _ground_truth(years: int, has_degree: bool, empty: bool) -> int:
  if empty:
    return 0
  if years >= 10:
    return 1
  if years >= 3 and has_degree:
    return 1
  return 0


def load_synthetic_hiring(
  n_train: int = 200,
  n_dev: int = 50,
  n_test: int = 50,
  seed: int = 42,
  label_noise: float = 0.0,
) -> HiringDataset:
  """Build the synthetic hiring dataset in memory.

  Args:
    n_train: training-split size.
    n_dev: dev-split size.
    n_test: test-split size.
    seed: RNG seed, only this seed is consumed; the function
      does not mutate global RNG state.
    label_noise: probability of flipping the ground-truth label
      per sample. Use 0 for the smoke-test path; use 0.05 to
      stress-test the TM's resource-allocation dynamics.
  """
  rng = random.Random(seed)
  records: list[dict[str, Any]] = []
  labels: list[int] = []
  protected: list[dict[ProtectedAttribute, str]] = []
  splits: list[str] = []

  def gen(n: int, tag: str) -> None:
    for _ in range(n):
      years = rng.choices([0, 1, 2, 3, 5, 8, 12, 20], k=1)[0]
      has_degree = rng.random() < 0.6
      empty = rng.random() < 0.05
      age = rng.choice(["18-25", "26-40", "41-60", "60+"])
      gender = rng.choice(["F", "M", "X"])
      bio = _bio(years, has_degree, empty, rng)
      label = _ground_truth(years, has_degree, empty)
      # Always consume the noise random draw so the RNG state
      # is identical across runs that differ only in label_noise.
      noise_draw = rng.random()
      if label_noise > 0 and noise_draw < label_noise:
        label = 1 - label
      records.append({
        "bio": bio,
        "years_experience": years,
        "has_degree": int(has_degree),
        "empty_resume": int(empty),
        "age_group": age,
        "gender": gender,
      })
      labels.append(label)
      protected.append({
        ProtectedAttribute.GENDER: gender,
        ProtectedAttribute.AGE_GROUP: age,
      })
      splits.append(tag)

  gen(n_train, "train")
  gen(n_dev, "dev")
  gen(n_test, "test")

  return HiringDataset(
    name="synthetic_hiring",
    records=records,
    labels=labels,
    class_names=["reject", "hire"],
    schema=_SCHEMA,
    protected=protected,
    split=splits,
    meta={
      "ground_truth": (
        "HIRE iff (years>=3 AND has_degree AND not empty) "
        "OR (years>=10 AND not empty)"
      ),
      "label_noise": label_noise,
      "rng_seed": seed,
    },
  )
