"""Canonical JSON, SHA-256 hashing, and JSONL streaming.

Why canonical JSON: receipts are HMAC-signed; signature must be
bit-stable across runs. Sorted keys + compact separators + utf-8
encoding give us that.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> bytes:
  """Encode ``obj`` to a deterministic byte-string.

  Uses sorted keys, no whitespace, NaN forbidden. The output is the
  canonical pre-image of any HMAC or hash over the object.
  """
  return json.dumps(
    obj,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")


def sha256_of(obj: Any) -> str:
  """Hex SHA-256 digest of ``canonical_json(obj)``."""
  return hashlib.sha256(canonical_json(obj)).hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
  """Stream-write ``rows`` to a JSONL file. Returns row count.

  The file is opened in append mode so per-sample logging from
  multiple stages can co-exist within the same experiment.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  with path.open("a", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
      f.write("\n")
      n += 1
  return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
  """Yield each row from a JSONL file."""
  path = Path(path)
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        yield json.loads(line)
