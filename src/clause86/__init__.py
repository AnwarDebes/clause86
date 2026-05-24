"""Clause86, Article-86-compliant hiring decisions via a Graph Tsetlin Machine.

Public API entry points are grouped by sub-package:

* ``clause86.data``, Bias-in-Bios, Adult-Income, HR-Analytics, synthetic loaders.
* ``clause86.graphs``, subword-dependency multigraph + tabular bipartite graph builders.
* ``clause86.tm``, Graph TM classifier + K-seed ensemble wrapper.
* ``clause86.explain``, Article 86 main-elements explainer.
* ``clause86.recourse``, counterfactual minimum-edit search with feasibility constraints.
* ``clause86.verify``, SAT-encoded clause receipt + HMAC signature.
* ``clause86.fairness``, demographic parity / equal opportunity / equalized odds
 + clause-deletion fairness intervention.
* ``clause86.eval``, faithfulness, accuracy, JSONL logger, statistics.

Each sub-package exposes a stable interface frozen in ``docs/ARCHITECTURE.md``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
