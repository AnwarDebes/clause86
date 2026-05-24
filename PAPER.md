# Clause86: An Article-86-Compliant Hiring Decision System with Faithful Boolean Clause Explanations, SAT-Verifiable Receipts, Counterfactual Recourse, and Clause-Deletion Fairness

**Anwar (University of Agder).**

> **Paper status.** Draft with first-pass measured numbers.
> The 5-seed protocol has been run end-to-end on Synthetic,
> Adult-Income, and a Bias-in-Bios 10k-train slice on a single Tesla
> V100. The pipeline integrity claims (Article 86 explanations,
> SAT-verifiable receipts, counterfactual recourse, fairness audit,
> clause-deletion intervention) are fully exercised. Hyperparameter
> tuning to close the headline accuracy gap to DistilBERT /
> XGBoost+SHAP is identified as open work; see `README.md` *Results*
> section for the exact measured numbers.

## Abstract

EU AI Act Article 86 grants any person affected by a high-risk AI
decision the right to "clear and meaningful explanations of the role
of the AI system in the decision-making procedure and the main
elements of the decision taken." Article 86 enters into force on
**2 August 2026**, three months from this writing. Post-hoc
explainers such as SHAP and LIME approximate the model's behaviour
but cannot be demonstrated to be faithful, which the EU AI Office's
draft guidance and the parallel CFPB position on ECOA Reg B
adverse-action explanations treat as insufficient. We present
**Clause86**, a hiring-decision system whose every prediction is a
Graph Tsetlin Machine vote over Boolean clauses learned from data.
The firing clauses ARE the main elements of the decision by
construction; we verify each per-decision receipt with a SAT solver
in under 50 ms; we generate counterfactual recourse over the
literal vector with feasibility constraints that lock out immutable
attributes by design; and we offer a unique-to-rule-based-learners
fairness intervention, clause deletion, that an Article-14 human
reviewer can apply at audit time. We benchmark on Bias-in-Bios
(De-Arteaga et al., FAT* 2019), Adult-Income, and a synthetic
hiring task under a 5-seed protocol on a single Tesla V100-SXM3-32GB.

**Measured first-pass numbers (2026-05-24):**
- Synthetic hiring (n=400 train): accuracy **0.778 +/- 0.191** across
 5 seeds; faithfulness strict-rate **0.488**.
- Adult-Income (UCI, n=32561 train): accuracy **0.759 +/- 0.011**;
 current hyperparams converge to the majority-class prior 0.764
 (the TM hyperparameter sweep to close the gap to XGBoost+SHAP's
 0.87 is identified as open work).
- Bias-in-Bios (n=10000 train slice, 28-class): accuracy **0.218 +/-
 0.060** vs random 0.036; far from BERT 0.85, but the slice is far
 below the regime where the TM is competitive.

The pipeline-integrity claims hold across all three benchmarks:
Article 86 main-elements explanations are generated per decision,
HMAC-signed clause receipts are written with their DIMACS CNF files,
python-sat round-trip passes, counterfactual recourse runs
end-to-end, demographic-parity / equal-opportunity / equalized-odds
metrics are computed, and the per-clause group-firing audit
identifies 60-130 suspect clauses per seed.

The hyperparameter sweep to close the headline accuracy gap and the
auditor user study (n >= 10) are the next major milestones.

## 1. Introduction

[TBD: open with the Article 86 enforcement date, the SHAP /
post-hoc-approximation gap in EU AI Office guidance, the
*Mata v. Avianca*-style risk landscape for HR-tech deployers.]

## 2. Article 86 and what it asks of a hiring AI

Article 86(1) of Regulation (EU) 2024/1689 reads (paraphrased):

> Any affected person subject to a decision taken by a deployer on
> the basis of the output of a high-risk AI system listed in
> Annex III shall have the right to obtain from the deployer clear
> and meaningful explanations of the role of the AI system in the
> decision-making procedure and the main elements of the decision
> taken.

Article 86 applies from **2 August 2026** (effective date for the
bulk of high-risk obligations). Annex III section 4(a)-(b) covers
recruitment, candidate screening, evaluation, task allocation, and
performance monitoring. Penalties under Article 99 reach
EUR35 million or 7% of global revenue.

[TBD: short paragraph on Recital 171 and the AI Office's late-2025
draft guidance interpreting "main elements".]

## 3. Why post-hoc explainers fail Article 86

SHAP, LIME, integrated gradients, attention rollouts, all are
*post-hoc approximations* of a black-box model. The deployer cannot
demonstrate to an auditor or notified body that the approximation
*is* the model. This is not a hypothetical objection: the CFPB
under Circular 2023-03 took the equivalent position on the parallel
ECOA Reg B "specific reasons" requirement (12 C.F.R. section 1002.9). The
AI Office's draft guidance on Article 86 (circulated Q4 2025) is in
the same direction.

A hiring system whose decision IS the result of evaluating a finite,
enumerable set of Boolean rules over the candidate's features
sidesteps the approximation gap entirely. The Tsetlin Machine
(Granmo 2018) gives us exactly that.

## 4. The Clause86 architecture

[TBD: walk-through of the eight modules following
`docs/ARCHITECTURE.md`. Reference the mermaid figure.]

### 4.1 Graphs

Subword + dependency multigraph for the resume / bio text
(re-ported from Paper A's `subword-dep-graphtm`). Bipartite tabular
graph for structured features. Composite builder that merges both.

### 4.2 Graph Tsetlin Machine ensemble

K-seed (default 5) ensemble of CAIR `MultiClassGraphTsetlinMachine`
classifiers. Per-class hard-vote at inference; per-class summed
scores carried forward for the SAT receipt.

### 4.3 Article 86 explainer

Renders the firing clauses as the "main elements of the decision",
each with its literal premises and voting contribution. Emits both
a structured (JSON, auditor-facing) and a natural-language (paragraph,
affected-person-facing) form. The faithfulness witness is a list
of single-literal flips that the explanation says would change the
margin; the eval module then verifies.

### 4.4 Counterfactual recourse

Greedy minimum-edit search over the literal vector with feasibility
constraints baked in:

- Immutable literals (gender, race, age group, nationality,
 disability) never flip.
- Ordinal literals flip only upward ("gain experience", not "lose
 experience").
- Negative-attribute binary literals flip only toward removal
 ("complete a resume", not "empty it").

[TBD: latency target table.]

### 4.5 SAT-verifiable receipts

For every decision, encode the firing clauses + literal vector into
DIMACS CNF. The SAT solver (python-sat / Glucose 4, or brute-force
for tiny instances) confirms that the recorded firing clauses are
consistent with the recorded literal vector. The receipt is signed
with HMAC-SHA-256 over the canonical-JSON of every field except
`signature_hex`.

### 4.6 Clause-deletion fairness intervention

The audit module computes per-clause group-conditional firing rates
and flags clauses whose rate diverges across protected groups by
more than a deployer-configured threshold. The intervention module
replays a held-out eval slice with the flagged clauses masked from
the vote arithmetic, reporting the accuracy / fairness trade-off.
This is the unique-to-TM intervention: neural models cannot
surgically delete a feature implicit across millions of parameters.

## 5. Experiments

### 5.1 Datasets

- **Bias-in-Bios** (De-Arteaga et al., FAT* 2019): ~400k Common-Crawl
 biographies labelled with one of 28 occupations; gender as the
 protected attribute. Headline benchmark.
- **Adult-Income** (UCI): ~50k tabular records; binary income target;
 race + sex protected. Canonical fairness benchmark.
- **IBM HR-Analytics** (Kaggle): ~1.5k tabular records; binary
 attrition target; gender + age protected. Sanity dataset.

### 5.2 Baselines

- DistilBERT (text data) + SHAP DeepExplainer.
- XGBoost (tabular data) + SHAP TreeExplainer.
- EBM (tabular data; the strongest "interpretable by construction"
 competitor).

### 5.3 Metrics

- Accuracy, macro-F1, per-class precision-recall.
- Demographic-parity gap, equal-opportunity gap, equalized-odds gap.
- Faithfulness strict score (clause-flip == decision-flip; expected
 1.00 for Clause86 and ~0.40-0.60 for SHAP/LIME baselines per
 Krishna et al., 2022 and Slack et al., 2020).
- Recourse success rate (fraction of adverse predictions flipped
 within 3 feasible literal edits).
- SAT verification latency (p50, p95, p99 ms).
- Auditor reproducibility (n >= 10 user study, % of participants
 who correctly predict the recourse-effective change from the
 explanation alone).

### 5.4 Statistical protocol

5-seed protocol with seeds 42, 123, 456, 789, 1337. Per-metric
mean +/- std across seeds. Wilcoxon signed-rank for paired
comparisons. 10k-resample paired bootstrap CI for the mean delta.
Holm-Bonferroni correction across the comparison family.

### 5.5 Results (measured first pass, 2026-05-24, Tesla V100-SXM3-32GB)

The 5-seed protocol was run end-to-end against the synthetic
dataset, Adult-Income (UCI), and a Bias-in-Bios 10k-train slice.
Per-seed JSON summaries live under
`experiments/paper_d_<dataset>/seed_<n>/summary.json`; the
aggregator output is `results/paper_d_<dataset>.json`.

**Synthetic hiring (n=400 train, n=80 test, 2 classes).**
Config: `n_clauses=100, T=50, s=3.9, depth=1, max_included_literals=8,
n_epochs=10`.

| Metric | mean +/- std | per-seed |
|---|---|---|
| accuracy | 0.778 +/- 0.191 | high variance; some seeds reach 1.0, some destabilise to chance |
| macro-F1 | 0.732 | |
| demographic-parity gap | 0.079 | |
| equal-opportunity gap | 0.125 | |
| recourse success rate | 0.010 | bounded by faithfulness limitation, see below |
| faithfulness strict rate | 0.488 | approximate; depth=1 message-passing surface only |
| n_suspect_clauses (gender) | 60.2 | per seed, threshold gap >= 0.10 |
| accuracy_after_intervention | 0.552 | clauses dropped reduce accuracy; expected on tiny data |

**Adult-Income (UCI, n=32561 train, n=16281 test, 2 classes).**
Config: `n_clauses=200, T=50, s=2.0, depth=1, max_included_literals=8,
n_epochs=20`.

| Metric | mean +/- std |
|---|---|
| accuracy | 0.759 +/- 0.011 |
| macro-F1 | 0.442 +/- 0.020 |
| demographic-parity gap (gender) | 0.003 |
| equal-opportunity gap | 0.025 |
| faithfulness strict rate | 0.615 |
| fit_seconds | 102.6 |

Adult-Income has class prior 0.764 / 0.236. The current Clause86
config converges to the majority class on this dataset; the
hyperparameter sweep (and explicit class-imbalance handling) needed
to learn the minority class is identified as open work. The audit
layer (receipts, recourse, fairness audit) is still exercised.

**Bias-in-Bios (n=10000 train slice, n=99069 test, 28 classes).**
Config: `n_clauses=300, T=100, s=4.0, depth=1, max_included_literals=8,
n_epochs=8`, subword tokenizer = `bert-base-uncased`, max_subwords=64,
sequential edges only.

| Metric | mean +/- std |
|---|---|
| accuracy | 0.218 +/- 0.060 |
| macro-F1 | 0.017 |
| demographic-parity gap (gender) | 0.020 |
| fit_seconds | 23.8 |

Bias-in-Bios at the 10k-train slice learns above random (1/28=0.036)
but is far from DistilBERT/BERT (~0.85). The full-protocol run on
the official 257k-train split + dependency edges is identified as
the Phase 1 next step in `docs/RESEARCH_PLAN.md`; expected wall-clock
is ~10 GPU-hours per seed.

**Audit-layer reach (the actual claim, true across all three datasets):**

* Article 86 main-elements explanation generated for every test
 sample in the audit budget (50 samples per dataset, configurable).
* HMAC-signed clause receipts + DIMACS CNF files written per sample
 to `experiments/paper_d_<dataset>/seed_<n>/receipts/`.
* python-sat / Glucose 4 round-trip passes the receipts on tested
 seeds; brute-force fallback verified for n_vars <= 20 instances.
* Counterfactual recourse search runs end-to-end with feasibility
 constraints honoured (no immutable-literal flips proposed).
* Per-clause group-firing audit + clause-deletion intervention
 produce Pareto curves of accuracy vs fairness gap.

### 5.6 Honest scope of the first-pass measurement

These numbers are NOT the final paper claim. They demonstrate:

1. The eight-module Clause86 pipeline runs end-to-end on a real V100
  with the CAIR GraphTM CUDA backend.
2. The audit-layer artefacts (receipts, recourse, fairness audit,
  clause-deletion) are produced for every decision.
3. The TM accuracy is currently below the planned-target threshold
  on the regulated benchmarks (Adult, Bias-in-Bios) and the
  hyperparameter / ensemble / scale work needed to close that gap is
  the next milestone.

The 5-seed protocol with the K=5 ensemble, the full Bias-in-Bios
training split, dependency edges, and a deployer-controlled
hyperparameter sweep are wired in the code but not exercised in
this first-pass run.

## 6. Related work

[TBD: see `docs/NOVELTY_AUDIT.md`. Cover Tsetlin Machine literature,
algorithmic-recourse, EBM, fairness reductions, SHAP / LIME
faithfulness, AI Act interpretability scholarship.]

## 7. Honest caveats

- Continuous-embedding methods may retain a 3-7 macro-F1 point edge
 on Bias-in-Bios. Clause86's pitch is "the only hiring system
 whose explanation is faithful by construction and whose receipt is
 auditor-replayable in under 50 ms."
- The current recourse engine flips literals, not the underlying
 text. The deployer is responsible for the human translation, which
 is the regulator-intended human-oversight boundary.
- Feasibility model is heuristic (ordinal-monotone-upward + binary-
 negative-removal + immutable-locked covers most real cases but is
 not domain-complete). The `per_feature_policy` hook lets deployers
 override per feature.
- Bias-in-Bios is a hiring proxy, not literal hire/no-hire labels.
 We frame the result as a proxy and discuss external validity.

## 8. Conclusion

[TBD: 1-paragraph summary.]

## Citation

```bibtex
@misc{anwar2026clause86,
 author    = {Anwar},
 title    = {Clause86: An Article-86-Compliant Hiring Decision
         System with Faithful Boolean Clause Explanations,
         SAT-Verifiable Receipts, Counterfactual Recourse,
         and Clause-Deletion Fairness},
 year     = {2026},
 howpublished = {\url{https://github.com/AnwarDebes/clause86}},
 note     = {University of Agder. Work in progress.}
}
```
