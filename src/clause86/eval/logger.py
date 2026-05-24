"""Per-sample JSONL logger.

Same shape as the loggers in Anwar's other repos: one record per
sample per stage (predict / explain / recourse / verify / fairness),
each stamped with the seed, sample_id, timestamp, and the JSON
payload.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
  """Append-mode JSONL writer with timestamps + per-call flush."""

  def __init__(self, path: str | Path, *, seed: int | None = None, run_id: str | None = None) -> None:
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.seed = seed
    self.run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    self._fh = self.path.open("a", encoding="utf-8")

  def log(self, stage: str, payload: dict[str, Any]) -> None:
    record = {
      "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "seed": self.seed,
      "run_id": self.run_id,
      "stage": stage,
      "payload": payload,
    }
    self._fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
    self._fh.write("\n")
    self._fh.flush()

  def close(self) -> None:
    try:
      self._fh.close()
    except Exception: # pragma: no cover - safety net
      pass

  def __enter__(self) -> "JsonlLogger":
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.close()
