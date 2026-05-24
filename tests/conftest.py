"""Shared pytest fixtures.

These fixtures provide a tiny but exercised synthetic pipeline so the
test suite can validate every cross-module invariant without any
external dependency installed.
"""
from __future__ import annotations

import pytest

from clause86.data import load_synthetic_hiring
from clause86.graphs import CompositeGraphBuilder
from clause86.tm import HireTM


@pytest.fixture(scope="session")
def tiny_dataset():
  return load_synthetic_hiring(n_train=120, n_dev=30, n_test=30, seed=42)


@pytest.fixture(scope="session")
def tiny_builder(tiny_dataset):
  b = CompositeGraphBuilder(
    schema=tiny_dataset.schema,
    text_field="bio",
    tokenizer_name=None,
    spacy_model=None,
    use_dep_edges=False,
    use_sequential_edges=False,
  )
  b.fit(tiny_dataset)
  return b


@pytest.fixture(scope="session")
def tiny_built_train(tiny_dataset, tiny_builder):
  train = tiny_dataset.select(tiny_dataset.split_indices("train"))
  return tiny_builder.transform(train, graph_backend="stub"), train


@pytest.fixture(scope="session")
def tiny_built_test(tiny_dataset, tiny_builder):
  test = tiny_dataset.select(tiny_dataset.split_indices("test"))
  return tiny_builder.transform(test, graph_backend="stub"), test


@pytest.fixture(scope="session")
def fitted_tm(tiny_built_train, tiny_dataset):
  train_built, train = tiny_built_train
  tm = HireTM(allow_fallback=True, seed=42, n_clauses=32, T=20, s=3.9)
  tm.fit(train_built, train.labels, n_classes=tiny_dataset.n_classes, n_epochs=2)
  return tm
