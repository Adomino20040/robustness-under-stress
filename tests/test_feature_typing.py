import numpy as np
import pandas as pd

from src.auditor.feature_typing import infer_feature_types


def test_basic_inference():
    df = pd.DataFrame({
        "income": np.random.rand(100) * 1e5,          # numerical
        "contract": ["a", "b"] * 50,                   # categorical (object)
        "senior": [0, 1] * 50,                         # categorical (low-card int)
        "tenure": np.arange(100),                      # numerical (high-card int)
    })
    types = infer_feature_types(df, max_cardinality=15)
    assert set(types["numerical"]) == {"income", "tenure"}
    assert set(types["categorical"]) == {"contract", "senior"}


def test_partition_is_complete_and_disjoint():
    df = pd.DataFrame({"a": np.random.rand(50), "b": ["x"] * 50})
    types = infer_feature_types(df)
    assert sorted(types["numerical"] + types["categorical"]) == sorted(df.columns)
