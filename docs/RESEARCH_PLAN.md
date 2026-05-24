# Research plan

This document is the experiment-phase + compute-budget plan. Lock
the numbers in PAPER.md only after the corresponding phase below
completes.

## Phase 0, Implementation (done)

* Eight modules implemented, frozen interface in `docs/ARCHITECTURE.md`.
* 63 unit + integration tests passing on the pure-Python fallback
 path.
* Synthetic-dataset end-to-end smoke test completes in <100 ms.

## Phase 1, Bias-in-Bios full run (planned, 14 days V100)

Bias-in-Bios is the headline benchmark because:

* It is the largest public dataset adjacent to recruitment text
 (~400k biographies).
* It carries gender as the protected attribute and is the canonical
 benchmark for the FAccT community (De-Arteaga et al., FAT* 2019).
* Its 28-class occupation prediction task is a credible proxy for a
 resume->role-assignment hiring decision.

### Data

* Source: HuggingFace `LabHC/bias_in_bios`, mirrored S3 parquet
 fallback under `data/raw/bias_in_bios/`.
* Split: official train (255k) / dev (40k) / test (100k) split from
 De-Arteaga et al.
* Preprocessing: `clause86.data.bias_in_bios.load_bias_in_bios` with
 no caps on test; train capped at 200k for V100-time budget reasons
 in phase 1 (full 255k in phase 1b if time permits).

### Model

* `HireTMEnsemble` with `seeds=(42, 123, 456, 789, 1337)`,
 `n_clauses=600`, `T=400`, `s=12.0`, `depth=3`, `n_epochs=20`.
* CAIR `MultiClassGraphTsetlinMachine` backend; `allow_fallback=False`
 so the run aborts if the kernel is unavailable.
* GPU: single Tesla V100 (32 GB).

### Compute budget

* Per seed: ~10 GPU-hours (200k train, 20 epochs, ~10k subword
 literals).
* 5 seeds: ~50 GPU-hours.
* With overlap (CPU graph-building can run during GPU training of
 the prior seed) the wall-clock target is **~48 hours per seed
 including 5-seed protocol overlap**, totalling **~14 days
 wall-clock**.
* Inference: ~1 sample / ms on V100 (target), so the 100k test set
 finishes in ~100 seconds per seed.

### Outputs

* `experiments/paper_d_biasinbios/seed_<n>/summary.json` (5 files).
* `experiments/paper_d_biasinbios/seed_<n>/log.jsonl` (per-sample).
* `experiments/paper_d_biasinbios/seed_<n>/receipts/*.cnf` +
 `*.receipt.json`.
* `results/paper_d_bias_in_bios.json` aggregated (mean +/- std
 across 5 seeds).

### Targets (NOT YET MEASURED)

* macro-F1 >= 0.78 (vs DistilBERT ~0.83-0.85).
* TPR gap (gender) on the test set <= 0.18 raw; <= 0.10 after
 clause-deletion intervention with accuracy drop <= 3 macro-F1
 points.
* Faithfulness strict score = 1.00 (by construction; any value
 below 1.00 is a bug, not a measurement).
* Recourse success rate on adverse decisions (top-200 ambiguous
 test samples) >= 90% within 3 edits, p50 latency <= 250 ms.
* SAT verification latency p50 <= 50 ms (n_literals <= 12k, n_firing
 per sample <= 50).

## Phase 2, Adult-Income (planned, 1 day V100)

Adult-Income is the canonical fairness benchmark. We run it second
because:

* It validates Clause86's tabular path independent of the text-graph
 builder.
* It's the head-to-head against EBM, XGBoost+SHAP, and Hardt-style
 fairness-corrected classifiers.

### Data

* Source: UCI archive; downloaded by `scripts/download_datasets.sh`.
* Split: standard UCI train (32k) / test (16k).

### Model

* `HireTMEnsemble`, `seeds=(42, 123, 456, 789, 1337)`,
 `n_clauses=300`, `T=150`, `s=8.0`, `depth=2`, `n_epochs=30`.

### Budget

* Per seed ~30 minutes V100. 5 seeds ~3 hours wall-clock.

### Baselines

* Logistic Regression (sklearn), no SHAP needed; coefficients are
 the "explanation".
* Random Forest + TreeSHAP, standard tabular baseline.
* XGBoost + TreeSHAP, strongest standard baseline.
* EBM (interpret-ml), explainability competitor.

### Targets (NOT YET MEASURED)

* Accuracy >= 0.83 (vs XGBoost ~0.87, EBM ~0.85).
* DP gap <= 0.18 raw; <= 0.10 after clause-deletion intervention
 with accuracy drop <= 2 percentage points.

## Phase 3, HR-Analytics (planned, <1 hour)

The IBM HR-Analytics dataset (~1.5k rows). Smallest dataset; serves
as a sanity check that the pipeline handles tiny tabular data and
as a robustness test for clause overfitting.

* Per seed: a few minutes CPU.
* Targets: accuracy >= 0.84 (5-fold mean), DP gap <= 0.10 after
 intervention.

## Phase 4, Recourse stress test (planned, 1 day)

* Adversarial perturbations to the literal vector (~1k samples
 drawn from the test set).
* Measure: recourse success rate when the candidate is one literal
 flip away from the boundary.
* Output: recourse robustness JSON.

## Phase 5, Auditor user study (planned, 14 days)

The strongest Article-86 evidence is "the affected person can
predict, from the explanation alone, what would flip the decision."
Plan:

* n >= 10 cross-disciplinary participants (5 SWE / 3 HR / 2 law).
* Each participant sees 20 randomly-sampled Clause86 explanations
 with the recourse hidden, and is asked: "name one change that
 would flip the decision."
* Score: fraction of participants for whom their named change is
 among the recourse candidates Clause86 found.
* Compare against DistilBERT + SHAP on the same 20 samples (with
 feature attributions instead of clauses).
* Target: Clause86 score >= 80% vs DistilBERT+SHAP score ~50%.

## Phase 6, Lock numbers + paper draft (planned, 5 days)

* Aggregate all per-seed results.
* Run Wilcoxon + bootstrap CIs vs each baseline.
* Apply Holm-Bonferroni across the comparison family.
* Update PAPER.md with measured numbers; freeze.

## Phase 7, Submission (planned, 7 days)

* Target venues: FAccT 2027 main track, AIES 2026 late-breaking,
 ECML-PKDD ADS 2026, EU AI Act-aligned law-tech venue
 (e.g., Computer Law & Security Review).
* Sequence: submit FAccT first (deadline Sep 2026); arxiv pre-print
 Aug 2026 to land within the Article-86 enforcement window.

---

## Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| GraphTM CUDA backend instability on 200k train | Medium | Fall back to tmu CPU backend; capped to 50k train if needed. Document in paper. |
| Bias-in-Bios accuracy ceiling too far below DistilBERT | Medium | Frame as *interpretable+auditable accuracy*; report SHAP faithfulness gap as the primary contrast. |
| Clause-deletion drops accuracy by > 5 points before fixing the gap | Medium | Report Pareto curve, not a single point. Recommend deployer-controlled trade-off slider. |
| python-sat unavailable on deployer infra | Low | Brute-force fallback covers n_literals <= 20; document the dependency clearly. |
| Auditor user study delayed by IRB / consent | High | Pre-register protocol; offer a paper version with the user-study moved to Phase 1b if blocked. |
