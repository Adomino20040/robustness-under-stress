"""Unit tests for the noise engine. These tests define correctness --
write them BEFORE finishing perturbations.py (TDD)."""
import numpy as np
import pandas as pd
import pytest

from src.auditor.perturbations import perturb_categorical, perturb_numerical


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def df():
    return pd.DataFrame({
        "income": np.linspace(1_000, 100_000, 500),
        "tenure": np.arange(500).astype(float),
        "contract": np.random.default_rng(1).choice(
            ["monthly", "yearly", "two_year"], size=500, p=[0.6, 0.3, 0.1]
        ),
    })


class TestNumerical:
    def test_input_not_mutated(self, df, rng):
        original = df.copy()
        perturb_numerical(df, ["income"], 0.10, rng)
        pd.testing.assert_frame_equal(df, original)

    def test_noise_bounded(self, df, rng):
        out = perturb_numerical(df, ["income"], 0.10, rng)
        ratio = out["income"] / df["income"]
        assert ratio.between(0.9, 1.1).all()

    def test_zero_level_is_identity(self, df, rng):
        out = perturb_numerical(df, ["income"], 0.0, rng)
        pd.testing.assert_series_equal(out["income"], df["income"])

    def test_untouched_columns_unchanged(self, df, rng):
        out = perturb_numerical(df, ["income"], 0.20, rng)
        pd.testing.assert_series_equal(out["tenure"], df["tenure"])

    def test_reproducible_with_same_seed(self, df):
        a = perturb_numerical(df, ["income"], 0.10, np.random.default_rng(7))
        b = perturb_numerical(df, ["income"], 0.10, np.random.default_rng(7))
        pd.testing.assert_frame_equal(a, b)


class TestCategorical:
    def test_swap_fraction_close_to_prob(self, df, rng):
        out = perturb_categorical(df, ["contract"], 0.20, rng)
        changed = (out["contract"] != df["contract"]).mean()
        # replacement can redraw the same value, so changed <= swap_prob
        assert 0.05 < changed <= 0.20

    def test_no_new_categories_invented(self, df, rng):
        out = perturb_categorical(df, ["contract"], 0.20, rng)
        assert set(out["contract"]) <= set(df["contract"])

    def test_zero_prob_is_identity(self, df, rng):
        out = perturb_categorical(df, ["contract"], 0.0, rng)
        pd.testing.assert_series_equal(out["contract"], df["contract"])
