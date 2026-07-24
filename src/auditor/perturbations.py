"""Phase 2 -- Controlled noise injection in raw feature space.

Pure functions: take a DataFrame, return a perturbed COPY. Never mutate input.
All randomness flows through an explicit numpy Generator for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def perturb_numerical(
    X: pd.DataFrame,
    columns: list[str],
    level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Multiplicative uniform noise: x' = x * (1 + U(-level, +level)).

    level=0.05 corresponds to the ±5% audit step.
    TODO(Phase 2): implement; must be vectorized (fraud set has 285k rows).
    """
    raise NotImplementedError


def perturb_categorical(
    X: pd.DataFrame,
    columns: list[str],
    swap_prob: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """With probability swap_prob per cell, replace the category with a value
    drawn from that column's EMPIRICAL category distribution (so noise stays
    realistic -- rare categories stay rare).

    TODO(Phase 2): implement; test that (a) ~swap_prob fraction changes,
    (b) marginal distribution is approximately preserved.
    """
    raise NotImplementedError
