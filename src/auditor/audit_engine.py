"""Phase 3 -- The Black-Box Auditor core loop.

CONTRACT: the auditor sees the model ONLY through `predict_proba(X_raw)`.
No coefficients, no trees, no gradients. If you find yourself importing
xgboost in this file, you have broken the research design.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

from src.auditor.feature_typing import infer_feature_types  # noqa: F401
from src.auditor.metrics import score  # noqa: F401
from src.auditor.perturbations import (  # noqa: F401
    perturb_categorical,
    perturb_numerical,
)


class BlackBoxModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class BlackBoxAuditor:
    """Runs the full stress-test grid for one frozen model.

    Parameters
    ----------
    model : any object exposing predict_proba
    noise_levels : e.g. [0.0, 0.05, 0.10, 0.20]  (0.0 = clean baseline row)
    n_trials : Monte Carlo repetitions per level (report mean +/- std)
    metrics : list of metric names understood by metrics.score
    seed : master seed; trial t at level l uses a deterministic child seed
    """

    def __init__(self, model: BlackBoxModel, noise_levels: list[float],
                 n_trials: int, metrics: list[str], seed: int = 42):
        self.model = model
        self.noise_levels = noise_levels
        self.n_trials = n_trials
        self.metrics = metrics
        self.seed = seed

    def run(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Return tidy long-format results:

            columns = [noise_level, trial, metric, value, feature_mode]

        Algorithm (TODO Phase 3):
        1. types = infer_feature_types(X_test)
        2. record clean-data scores (noise_level=0, trial=0)
        3. for level in noise_levels, for trial in range(n_trials):
             X_pert = perturb_numerical(...) then perturb_categorical(...)
             proba = self.model.predict_proba(X_pert)[:, 1]
             append one row per metric
        4. Bonus ablation (feature_mode): perturb numerical-only vs
           categorical-only vs both -- isolates which feature family
           drives fragility.
        """
        raise NotImplementedError
