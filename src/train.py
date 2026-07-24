"""Phase 1 -- Train and freeze the four models of the 2x2 matrix.

Usage:
    python -m src.train --config configs/experiment_config.yaml

Produces one artifact per (domain, model) cell in outputs/models/, e.g.
    churn__logistic_regression.joblib
    fraud__xgboost.joblib

CRITICAL DESIGN DECISION:
Each artifact is a full sklearn Pipeline(preprocessor -> classifier).
The preprocessor (scaling + one-hot encoding) is INSIDE the black box.
This means the auditor perturbs interpretable raw features (income, tenure,
contract type), not scaled/encoded internals -- exactly like a production API.
"""
from __future__ import annotations

import argparse

import joblib  # noqa: F401
from sklearn.compose import ColumnTransformer  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: F401
from xgboost import XGBClassifier  # noqa: F401


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> "ColumnTransformer":
    """StandardScaler on numericals, OneHotEncoder(handle_unknown='ignore') on categoricals."""
    raise NotImplementedError  # TODO(Phase 1)


def build_model(name: str, params: dict, y_train=None):
    """Return an unfitted classifier.

    TODO(Phase 1): for XGBoost on fraud, set
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    """
    raise NotImplementedError


def main(config_path: str) -> None:
    """Loop over datasets x models, fit Pipeline, report clean-data metrics,
    save frozen artifacts + the held-out test split to data/processed/."""
    raise NotImplementedError  # TODO(Phase 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    main(parser.parse_args().config)
