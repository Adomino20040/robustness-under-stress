"""Phase 4 -- Automated degradation curve generation.

Every figure must be publication-ready straight out of the pipeline:
axis labels, units, legend, error bands, vector PDF + PNG.
"""
from __future__ import annotations

import pandas as pd


def plot_degradation_curves(
    results: pd.DataFrame,
    metric: str,
    title: str,
    out_path: str,
) -> None:
    """One panel: x = noise level, y = metric, one line per model,
    shaded band = +/- 1 std over Monte Carlo trials.

    TODO(Phase 4): implement with matplotlib; save PNG (300 dpi) and PDF.
    """
    raise NotImplementedError


def plot_2x2_summary(all_results: dict[str, pd.DataFrame], out_path: str) -> None:
    """The money figure of the paper: 2x2 grid (domain x metric) showing
    LR vs XGB degradation curves side by side.
    TODO(Phase 4)."""
    raise NotImplementedError
