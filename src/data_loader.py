"""Phase 1 -- Dataset loading and stratified splitting.

Owns: reading raw CSVs, target binarization, stratified train/test split.
Never applies noise; the auditor perturbs *raw feature space* downstream.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path: str, target: str, positive_label) -> tuple[pd.DataFrame, pd.Series]:
    """Load a raw CSV and return (X, y) with y binarized to {0, 1}.

    TODO(Phase 1):
    - read CSV, drop obvious ID columns (e.g. customerID)
    - coerce numeric-looking object columns (Telco's TotalCharges is a string!)
    - map target to 0/1 using positive_label
    """
    raise NotImplementedError


def split(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Stratified split. Stratification is mandatory for the fraud domain."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
