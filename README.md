# Optimizing LLMs for Causality Assessment in Pharmacovigilance

**Developing a Performance Metric as Objective for Bayesian Hyperparameter Optimization**

Nicole Sonne Heckmann, Arnault-Quentin Vermillet, Gerard Ompad, Søren Norlin Mølgaard, Lars Melskens, Christina Tvermoes, Maurizio Sessa
*Department of Drug Design and Pharmacology, University of Copenhagen · Novo Nordisk*

This project investigates whether **inference-time temperature optimization** can improve a large language model's agreement with expert pharmacovigilance assessors on the **Naranjo causality algorithm**, applied to FDA Adverse Event Reporting System (FAERS) reports — and develops the Gaussian-Process–compatible performance metrics needed to make that optimization possible.

---

## Key findings

- **Temperature has no universal optimum.** Across a grid sweep, three GP experiments, and an independent TPE optimizer, mean metric scores were flat across temperature ∈ [0, 2] (CWCS varied by only 0.023 over the full range).
- **Case-specific temperature selection still helps.** EWACS-guided optimization lifted Naranjo classification agreement from **45.0% → 72.0%** (+27.0 pp), with the largest gains on borderline *Doubtful* (+42.9 pp) and *Possible* (+18.5 pp) cases.
- **Score agreement, not reasoning similarity, drives classification.** Expert reasoning is template-dominated for 8/9 questions (0.3–1.5% unique) and the embedding space collapses onto one dimension (PC1 = 91.8%), so cosine similarity adds little independent signal except on question 5.
- **EWACS is the best metric for the endpoint.** Re-running optimization with the bounded multiplicative CWCS objective gave lower classification agreement (64.0% GP / 61.0% TPE) than EWACS (72.0%).

## The optimization metrics

| Metric | Form | Notes |
|---|---|---|
| **WCS** | mean of `cos(·) − (1 − a_q)/2` | subtractive, equal weights |
| **IWAS** | `Σ agree(q)·w(q)`, `w(q)=H(q)/ΣH` | entropy-weighted binary agreement |
| **EWACS** | `Σ H(q)·[cos(·) − (1 − a_q)/2]` | entropy-weighted WCS — **best for classification** |
| **CWCS** | `Σ w_q·λ_q·((cos(·)+1)/2)`, `λ_q = 1 − |ŷ_q − y_q|`, `w_q = M_q/ΣM_j` | bounded [0,1], multiplicative, soft agreement — **best-behaved GP target** |

## Classification agreement (same 100 cases)

| Condition | Overall | Doubtful | Possible |
|---|---|---|---|
| Baseline (T = 0) | 45.0% | 22.9% | 56.9% |
| **EWACS-optimized (GP)** | **72.0%** | **65.7%** | **75.4%** |
| CWCS-optimized (GP) | 64.0% | 77.1% | 56.9% |
| CWCS-optimized (TPE) | 61.0% | 80.0% | 50.8% |

---

## Repository structure

```
llm-naranjo-bo/
├── index.html                      # project page (GitHub Pages)
├── README.md
├── notebooks/
│   ├── naranjo_bo_pipeline.ipynb    # metrics, GP & TPE optimization, all figures
│   └── anisotropy_diagnostic.ipynb  # embedding anisotropy / whitening check
├── manuscript/
│   ├── Manuscript_tracked.docx
│   └── Supplementary_Materials.docx
└── figures/                         # exported result figures
```

## Quick start

```bash
pip install pandas numpy scipy scikit-learn sentence-transformers \
            torch gpytorch botorch optuna openpyxl openai
jupyter lab notebooks/naranjo_bo_pipeline.ipynb
```

An OpenAI API key (GPT-5.2) is required to regenerate raw model outputs; the notebooks also support resuming from cached intermediate result files, so the metric and optimization analyses can be reproduced without re-querying the model.

## Pipeline at a glance

- **Data** — 2.58M FAERS reports (2025 Q1–Q2) → stratified down-sample → 723 expert-assessed cases.
- **Model** — GPT-5.2 with chain-of-thought prompting, one call per Naranjo question (score + reasoning).
- **Reasoning encoder** — `paraphrase-multilingual-mpnet-base-v2`.
- **Optimization** — GP surrogate (Matérn 5/2) + Probability of Improvement, cross-checked with a TPE sampler, over temperature ∈ [0, 2].

---
