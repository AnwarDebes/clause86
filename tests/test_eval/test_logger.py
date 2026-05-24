import json
import tempfile
from pathlib import Path

from clause86.eval import JsonlLogger


def test_logger_writes_rows():
  with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "out.jsonl"
    with JsonlLogger(path, seed=42, run_id="r1") as log:
      log.log("stage_a", {"x": 1})
      log.log("stage_b", {"y": 2})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["stage"] == "stage_a"
    assert rows[0]["payload"]["x"] == 1
    assert rows[0]["seed"] == 42
