# Robustness Under Stress

**An Automated Black-Box Framework for Auditing Tabular Machine Learning Models**

This project investigates the trade-off between a machine learning model's *architectural complexity* and its *operational robustness*: how gracefully its performance degrades when the input data is corrupted by realistic noise.

> **Research question:** What is the relationship between a model's architectural complexity and its rate of performance degradation under black-box feature perturbation?

> **Hypothesis:** Complex, non-linear ensemble models (e.g., XGBoost) that achieve superior accuracy on pristine data will exhibit a significantly steeper degradation curve under unseen feature noise than simpler, robust linear baselines (e.g., Logistic Regression).

## Why this matters

In production, trained models are frozen, but the world around them is not. Data drift, covariate shift, and upstream engineering bugs (e.g., a feature silently switching from monthly to annual units) corrupt model inputs long after deployment. Models then fail *silently*: no errors are raised, predictions simply get worse. This project builds a **domain-independent, black-box auditing tool** that quantifies a model's "safety margin" before deployment by measuring exactly how fast its performance decays as input noise increases.

## Experimental design

The hypothesis is tested on a 2×2 matrix, testing two model complexities across two tabular domains with very different class distributions:

|                          | **Domain A: Customer Churn** (balanced, ~50/50) | **Domain B: Credit Card Fraud** (imbalanced, ~0.1% positive) |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------------------ |
| **Logistic Regression**  | low complexity baseline                          | low complexity baseline                                       |
| **XGBoost**              | high complexity ensemble                         | high complexity ensemble                                      |

The auditing engine treats every trained model as a pure **black box**: it only sends test inputs and reads probability outputs. It applies controlled, stepped perturbations to the held-out test set:

- **Numerical features:** multiplicative noise at ±5%, ±10%, and ±20%
- **Categorical features:** random value swaps drawn from each feature's empirical distribution

and tracks metric decay (F1, ROC-AUC, PR-AUC) across noise levels, producing **model degradation curves**.

## Project status

🚧 **Work in progress**. The project is being built incrementally. Current stage: **Domain A audited, Domain B (fraud) up next**.

- [x] Environment and repository setup
- [x] Automated dataset download (Telco Customer Churn)
- [x] Exploratory data analysis
- [x] Preprocessing and baseline models (Logistic Regression, XGBoost)
- [x] Black-box perturbation engine
- [x] Audit loop and degradation metrics
- [ ] Second domain (Credit Card Fraud)
- [ ] Degradation curves, analysis, and final report

**First result (Domain A):** the hypothesis holds on churn. At 20% noise, XGBoost loses 8.9% ROC-AUC and 27% F1 against its clean baseline, while Logistic Regression loses only 4.1% and 3.4%. The complex ensemble degrades roughly 2x faster on every metric.

## Repository structure

```
robustness-under-stress/
├── data/
│   └── raw/              # Datasets (gitignored, downloaded automatically)
├── notebooks/
│   └── explore.ipynb     # Data download + exploratory analysis
├── requirements.txt      # Python dependencies
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

Open `notebooks/explore.ipynb` and run it top to bottom. **No manual data download is needed**: the first cells fetch the Telco Customer Churn dataset automatically into `data/raw/`.

## Data

| Dataset | Domain | Source | Acquisition |
| --- | --- | --- | --- |
| Telco Customer Churn | churn (balanced) | [IBM sample dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) | auto-downloaded by the notebook |
| Credit Card Fraud | fraud (imbalanced) | [ULB / OpenML](https://www.openml.org/d/1597) | planned, fetched via `sklearn.datasets.fetch_openml` |

Note: the Telco dataset is a *fictional* sample published by IBM and widely used as a community benchmark. This is acceptable here because the research question concerns model degradation behaviour, not telco business insight. The fraud dataset contains real (anonymised) transactions.

## About

An independent research project in AI safety and empirical model evaluation, with the goal of publishing the findings as a short research paper.
