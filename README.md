# Robustness Under Stress

**An Automated Black-Box Framework for Auditing Tabular Machine Learning Models**

This project investigates the trade-off between a machine learning model's *architectural complexity* and its *operational robustness*: how gracefully its performance degrades when the input data is corrupted by realistic noise.

> **Research question:** What is the relationship between a model's architectural complexity and its rate of performance degradation under black-box feature perturbation?

> **Hypothesis:** Complex, non-linear ensemble models (e.g., XGBoost) that achieve superior accuracy on pristine data will exhibit a significantly steeper degradation curve under unseen feature noise than simpler, robust linear baselines (e.g., Logistic Regression).

## Why this matters

In production, trained models are frozen, but the world around them is not. Data drift, covariate shift, and upstream engineering bugs (e.g., a feature silently switching from monthly to annual units) corrupt model inputs long after deployment. Models then fail *silently*: no errors are raised, predictions simply get worse. This project builds a **domain-independent, black-box auditing tool** that quantifies a model's "safety margin" before deployment by measuring exactly how fast its performance decays as input noise increases.

## Experimental design

The hypothesis is tested on a 2×2 matrix, testing two model complexities across two tabular domains with very different class distributions:

|                          | **Domain A: Customer Churn** (moderately imbalanced, ~73/27) | **Domain B: Credit Card Fraud** (extremely imbalanced, ~0.1% positive) |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------------ |
| **Logistic Regression**  | low complexity baseline                          | low complexity baseline                                       |
| **XGBoost**              | high complexity ensemble                         | high complexity ensemble                                      |

The auditing engine treats every trained model as a pure **black box**: it only sends test inputs and reads probability outputs. It applies controlled, stepped perturbations to the held-out test set:

- **Numerical features:** multiplicative noise at ±5%, ±10%, and ±20%
- **Categorical features:** random value swaps drawn from each feature's empirical distribution

and tracks metric decay (F1, ROC-AUC, PR-AUC) across noise levels, producing **model degradation curves**.

## Project status

🚧 **Work in progress**. The project is being built incrementally. Current stage: **both domains audited, cross-domain analysis and final report up next**.

- [x] Environment and repository setup
- [x] Automated dataset download (Telco Customer Churn)
- [x] Exploratory data analysis
- [x] Preprocessing and baseline models (Logistic Regression, XGBoost)
- [x] Black-box perturbation engine
- [x] Audit loop and degradation metrics
- [x] Second domain (Credit Card Fraud)
- [ ] Cross-domain analysis and final report

### Results so far

**Domain A (churn): the hypothesis holds.** At 20% noise, XGBoost loses 8.9% ROC-AUC and 27% F1 against its clean baseline, while Logistic Regression loses only 4.1% and 3.4%. The complex ensemble degrades roughly 2x faster on every metric.

**Domain B (fraud): the hypothesis does not hold.** Both models are essentially noise-immune: every metric drops less than 3% at 20% noise (LR ROC-AUC −0.1%, XGBoost F1 −2.5%), and XGBoost's ROC-AUC and PR-AUC even improve marginally under the strongest perturbation. Degradation under noise appears to be **domain-dependent rather than purely a function of model complexity** — a more interesting finding than a clean confirmation, and the focus of the upcoming analysis.

Methodological note: churn F1 scores use a fixed 0.5 decision threshold, while fraud F1 scores use a per-model threshold tuned for optimal F1 on the clean test set (necessary given the ~0.1% positive rate). The F1 columns of the two domains are therefore not directly comparable; ROC-AUC and PR-AUC are threshold-free and comparable.

## Repository structure

```
robustness-under-stress/
├── data/
│   ├── raw/                  # Datasets (gitignored, downloaded automatically)
│   └── processed/            # Cleaned data (gitignored)
├── notebooks/
│   ├── explore_churn.ipynb   # Churn: data download + exploratory analysis
│   ├── train_churn.ipynb     # Churn: LR and XGBoost baselines, frozen to outputs/models
│   ├── audit_churn.ipynb     # Churn: perturbation engine, audit loop, degradation curves
│   ├── explore_fraud.ipynb   # Fraud: data download + exploratory analysis
│   ├── train_fraud.ipynb     # Fraud: LR and XGBoost baselines, frozen to outputs/models
│   └── audit_fraud.ipynb     # Fraud: black-box audit and degradation curves
├── outputs/                  # Frozen models, results tables, figures (gitignored)
├── requirements.txt          # Python dependencies
└── README.md
```

The structure grows with the project; source code will be promoted from notebooks into a `src/` package as components stabilise.

## Getting started

Requires **Python 3.10+**.

```bash
git clone <this-repo-url>
cd robustness-under-stress

# Create and activate a virtual environment
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# Windows (CMD / PowerShell):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order, each one top to bottom (per domain: explore → train → audit):

1. `notebooks/explore_churn.ipynb` downloads and cleans the data (**no manual download needed**, the first cells fetch the Telco dataset into `data/raw/`)
2. `notebooks/train_churn.ipynb` trains the baselines and freezes them to `outputs/models/`
3. `notebooks/audit_churn.ipynb` runs the black box audit and produces the degradation curves
4. `notebooks/explore_fraud.ipynb`, `notebooks/train_fraud.ipynb`, `notebooks/audit_fraud.ipynb` repeat the same pipeline on the fraud dataset (auto-fetched from OpenML)

Note: the notebooks use relative paths (`../data`, `../outputs`), so run them with `notebooks/` as the working directory (the default when launched via `jupyter notebook` from the repo root).

## Data

| Dataset | Domain | Source | Acquisition |
| --- | --- | --- | --- |
| Telco Customer Churn | churn (moderately imbalanced, ~73/27) | [IBM sample dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) | auto-downloaded by the notebook |
| Credit Card Fraud | fraud (extremely imbalanced, ~0.1% positive) | [ULB / OpenML](https://www.openml.org/d/1597) | auto-downloaded via `sklearn.datasets.fetch_openml` |

Note: the Telco dataset is a *fictional* sample published by IBM and widely used as a community benchmark. This is acceptable here because the research question concerns model degradation behaviour, not telco business insight. The fraud dataset contains real (anonymised) transactions.

## About

An independent research project in AI safety and empirical model evaluation, with the goal of publishing the findings as a short research paper.
