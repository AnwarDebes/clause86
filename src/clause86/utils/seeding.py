"""Centralised RNG seeding.

Why centralised: any module-level ``np.random.seed`` makes seed audit
brittle. Every entry point calls ``seed_everything(seed)`` once and we
never seed again outside that.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
  """Seed Python ``random``, NumPy, and (if importable) PyTorch.

  PyTorch is optional; the import is guarded so the data/graph/TM
  pipeline does not pay the torch import cost on hot paths.
  """
  if seed is None or seed < 0:
    raise ValueError(f"seed must be non-negative int, got {seed!r}")
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  try: # pragma: no cover - torch is optional
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  except ImportError:
    pass
