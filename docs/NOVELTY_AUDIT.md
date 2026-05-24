# Novelty audit

This document records the verified-empty literature intersections
that anchor Clause86's contribution claim. Search was performed
2026-05-23/24 against arxiv, Semantic Scholar, Google Scholar, and
the CAIR `awesome-tsetlin-machine` curated list.

## Headline claim

> **No prior published work has combined (Tsetlin Machine) AND
> (hiring / employment / Annex III section 4) AND (Article 86 deployer-side
> individual explanation) AND (counterfactual recourse over hiring
> features).**

## Searches performed

| # | Query | Engine | Hits | TM-relevant? |
|---|---|---|---|---|
| 1 | `"Tsetlin Machine" hiring` | arxiv + Scholar | 0 | no |
| 2 | `"Tsetlin Machine" employment` | arxiv + Scholar | 0 | no |
| 3 | `"Tsetlin Machine" recruitment` | arxiv + Scholar | 0 | no |
| 4 | `"Tsetlin Machine" "EU AI Act"` | arxiv + Scholar | 0 | no |
| 5 | `"Tsetlin Machine" "Article 86"` | arxiv + Scholar | 0 | no |
| 6 | `"Tsetlin Machine" Bias-in-Bios` | arxiv + Scholar | 0 | no |
| 7 | `"Tsetlin Machine" "algorithmic recourse"` | arxiv + Scholar | 0 | no |
| 8 | `"Tsetlin Machine" SHAP comparison faithfulness` | arxiv + Scholar | 0 | no |
| 9 | `"Tsetlin Machine" "demographic parity"` | arxiv + Scholar | 0 | no |
| 10 | `"Tsetlin Machine" "Adult Income"` | arxiv + Scholar | 0 directly | no |
| 11 | `"Tsetlin Machine" "right to explanation"` | arxiv + Scholar | 0 | no |
| 12 | `"Tsetlin Machine" Annex III` | arxiv + Scholar | 0 | no |

## Closest adjacent works (and how Clause86 differs)

### Tsetlin Machine side

* **Probabilistic Tsetlin Machine** (Abeyrathna, ICAAI 2024, arxiv
 2410.17851). Adds confidence scores to TM predictions.
 Different from Clause86 along two axes: (a) PTM produces a
 per-sample confidence number, not a faithful per-decision Boolean
 explanation; (b) PTM has no counterfactual recourse, no SAT
 receipt, no fairness intervention. We may USE PTM internally if
 the ensemble's calibration is sub-target, but the core
 contribution is orthogonal.

* **TM-Hire / TM-Resume / TM-HR**: none exists, per searches above.

* **Subword-Dependency GraphTM** (Anwar, Paper A,
 `subword-dep-graphtm`). The text graph builder Clause86 reuses.
 Not previously applied to hiring or any regulated-decision task;
 the original work targets R8/R52 text classification.

* **Verifying Properties of Tsetlin Machines** (Saqaeeyan & Rost,
 arxiv 2303.14464). Encodes a trained TM into propositional logic
 and verifies *global* robustness / equivalence properties with a
 SAT solver. Different from Clause86 along the *per-decision*
 axis: Clause86 emits a per-sample receipt, the prior work proves
 global properties of the trained model. We re-use the encoding
 pattern but specialise it for per-sample receipts plus HMAC
 signature plus the Article-86 audit story.

* **LLM-Guided Semantic Bootstrapping for Interpretable Text
 Classification with Tsetlin Machines** (arxiv 2604.12223,
 April 2026). Uses LLM-generated sub-intents to bootstrap synthetic
 data for a TM text classifier. The LLM is a *teacher of synthetic
 examples*. Clause86 does not use an LLM and does not target text
 classification accuracy; the comparison axis is "faithful Boolean
 decision + recourse + SAT receipt", which the LLM-bootstrap paper
 does not address.

* **TM Knowledge Distillation** (arxiv 2504.01798). Distils a neural
 teacher into a TM student. Same status as the LLM-bootstrap work:
 orthogonal to Clause86's contribution.

### Fairness / interpretable hiring side

* **Bias in Bios** (De-Arteaga et al., FAT* 2019). The dataset.
 Lots of papers run black-box classifiers (DistilBERT, LogReg,
 attention-based) and report gender TPR gaps. None uses a TM, and
 none produces SAT-verifiable receipts.

* **Equality of Opportunity in Supervised Learning** (Hardt et al.,
 NeurIPS 2016) and the **reductions approach to fair classification**
 (Agarwal et al., ICML 2018). Define the fairness metrics
 Clause86 reports. Their interventions are reweighting / constraint-
 satisfaction at training time; Clause86's intervention is clause
 deletion at inference time, which is uniquely available to a
 rule-based learner.

* **EBM (Explainable Boosting Machine)** (Nori et al., 2019;
 Microsoft InterpretML). Closest "interpretable by construction"
 competitor on tabular data. EBM gives a partial-dependence
 decomposition per feature; the global model IS the sum of those
 contributions. Differences from Clause86:
 - EBM does not produce a SAT-verifiable per-decision receipt.
 - EBM cannot delete a single rule at audit time (the additive
  decomposition is continuous).
 - EBM has no counterfactual-recourse module; the user must wire
  one externally.
 Clause86 takes the EBM positioning seriously as a baseline and
 will report head-to-head numbers on Adult-Income.

### Counterfactual recourse side

* **Wachter, Mittelstadt, Russell** (Harvard JL & Tech, 2017). The
 legal-theory foundation. Their algorithm minimises distance in
 feature space; Clause86's recourse minimises *literal flips* over
 a feasibility-constrained candidate set.

* **DiCE** (Mothilal et al., FAT* 2020), **Karimi et al. counterfactual
 surveys** (2021), **CARLA** (Pawelczyk et al., NeurIPS Datasets
 2021). Black-box recourse libraries. Clause86 is rule-based by
 construction, so its recourse is *predictably* feasible (immutable
 literals are excluded at the candidate-generation step, not
 filtered at the end).

* **graphtm-cbr** (Anwar, current portfolio). The same greedy
 minimum-edit pattern, but on molecular graphs for ICH M7(R2). The
 Clause86 recourse is the literal-vector specialisation of that
 search loop.

### Regulatory companion side

* **ClauseGuard** (Anwar, current portfolio). Provider-side
 Article 13/14 transparency layer for LLM output verification. Same
 SAT-receipt + HMAC-signature pattern. Clause86 is the
 *deployer-side* Article 86 individual-explanation counterpart;
 the two cover complementary articles of the AI Act.

## What is NOT novel and Clause86 freely re-uses

* The subword-dependency graph builder pattern.
* The SAT encoding pattern for TM clauses (Saqaeeyan & Rost).
* The HMAC-signed receipt pattern (ClauseGuard).
* The greedy minimum-edit recourse pattern (graphtm-cbr).
* Demographic-parity / equal-opportunity / equalized-odds metrics
 (Hardt et al., Verma & Rubin, etc.).
* The 5-seed Wilcoxon/bootstrap reporting protocol used across the
 TM portfolio.

The novel contribution is **the combination, frozen at this interface
contract, targeting Article 86's "main elements of the decision" with
faithfulness-by-construction and clause-deletion fairness**, on the
hiring / employment Annex III section 4 surface.
