# Robustness Under Stress

**An Automated Black-Box Framework for Auditing Tabular Machine Learning Models**

## Research Question
What is the relationship between a model's architectural complexity and its rate of
performance degradation under black-box feature perturbation?

## Hypothesis
High-complexity ensembles (XGBoost) that dominate on pristine data exhibit steeper
degradation curves under feature noise than simple linear baselines (Logistic Regression).

## Experimental Design (2x2 Matrix)

|                        | Logistic Regression | XGBoost |
|------------------------|---------------------|---------|
| Churn (balanced)       | LR-Churn            | XGB-Churn |
| Fraud (imbalanced)     | LR-Fraud            | XGB-Fraud |

Perturbations:
- **Numerical**: multiplicative noise at ±5%, ±10%, ±20%
- **Categorical**: uniform random swaps sampled from the empirical category distribution

Metrics: ROC-AUC (primary), F1-Score, PR-AUC (fraud domain).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 1. Place raw CSVs in data/raw/ (see configs/experiment_config.yaml)
# 2. Train frozen models
python -m src.train --config configs/experiment_config.yaml

# 3. Run the black-box audit
python run_audit.py --config configs/experiment_config.yaml

# 4. Figures land in outputs/figures/, tables in outputs/results/
```

## Repository Layout

```
configs/        Experiment configuration (single source of truth)
data/raw        Original immutable datasets (never modified, gitignored)
data/processed  Train/test splits, fitted preprocessors
notebooks/      EDA and results analysis (exploration only, no core logic)
src/            Production pipeline code
src/auditor/    The black-box auditing engine (domain-independent)
tests/          Unit tests (pytest)
outputs/        Models, degradation curves, result tables (gitignored)
reports/        Final paper drafts and figures selected for publication
```

## Team Rules
- All core logic lives in `src/`; notebooks import from `src`, never the reverse.
- The auditor only ever calls `model.predict_proba(X)` — no access to internals.
- Every experiment is reproducible from `configs/experiment_config.yaml` + a seed.
