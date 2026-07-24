"""Phase 3 -- Metric computation and degradation quantification."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (  # noqa: F401
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def score(y_true, y_proba, metric: str) -> float:
    """Compute one metric from probability outputs.

    metric in {"roc_auc", "pr_auc", "f1"}.
    For f1, threshold at 0.5 (document this choice in the paper).
    TODO(Phase 3): implement.
    """
    raise NotImplementedError


def degradation_slope(results: pd.DataFrame, metric: str) -> float:
    """The headline number of the whole project.

    Fit OLS of mean metric vs. noise level; return the slope
    (performance lost per unit of noise). Steeper (more negative)
    slope = more fragile model.
    TODO(Phase 3): implement with np.polyfit; also report
    normalized degradation: (clean - noisy@20%) / clean.
    """
    raise NotImplementedError
