"""Phase 2 -- Automatic feature-type inference (domain independence).

The auditor must work on ANY tabular dataset with zero manual schema input.
"""
from __future__ import annotations

import pandas as pd


def infer_feature_types(
    X: pd.DataFrame, max_cardinality: int = 15
) -> dict[str, list[str]]:
    """Return {"numerical": [...], "categorical": [...]}.

    Rules (document these in the paper -- they're a methodological contribution):
    - object / category / bool dtype        -> categorical
    - numeric dtype with nunique <= max_cardinality -> categorical
      (catches integer-encoded categories like SeniorCitizen 0/1)
    - everything else numeric               -> numerical

    TODO(Phase 2): implement + unit test against both datasets.
    """
    raise NotImplementedError
