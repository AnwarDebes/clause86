from clause86.graphs import SubwordDepGraphBuilder


def test_whitespace_tokenizer_fits_vocab():
  b = SubwordDepGraphBuilder(
    tokenizer_name=None, spacy_model=None,
    use_dep_edges=False, use_sequential_edges=True,
    min_token_freq=1,
  )
  texts = ["alpha bravo charlie", "alpha delta", "bravo charlie"]
  b.fit(texts)
  assert "alpha" in b.symbols
  assert "bravo" in b.symbols
  assert "delta" in b.symbols


def test_transform_emits_text_literals():
  b = SubwordDepGraphBuilder(
    tokenizer_name=None, spacy_model=None,
    use_dep_edges=False, use_sequential_edges=True,
    min_token_freq=1,
  )
  b.fit(["alpha bravo charlie", "bravo delta", "alpha alpha"])
  built = b.transform(["alpha bravo", "delta delta"], graph_backend="stub")
  assert built.n_samples == 2
  assert built.n_literals > 0
  # Two of three vocab tokens appear in sample 0.
  n_text_active = sum(
    1 for lid in built.per_sample_active_literals[0]
    if built.literal_specs[lid].feature_kind == "text"
    and not built.literal_specs[lid].bucket_or_category.startswith("edge:")
  )
  assert n_text_active >= 2


def test_empty_text_is_safe():
  b = SubwordDepGraphBuilder(
    tokenizer_name=None, spacy_model=None,
    use_dep_edges=False, use_sequential_edges=False,
    min_token_freq=1,
  )
  b.fit(["alpha"])
  built = b.transform(["", ""], graph_backend="stub")
  assert built.n_samples == 2
  # Empty text -> no active literals.
  for active in built.per_sample_active_literals:
    assert active == ()
