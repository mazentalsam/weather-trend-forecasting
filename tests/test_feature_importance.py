"""Tests for feature importance analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from src.models.feature_importance import (
    HAS_SHAP,
    compute_correlation_importance,
    compute_perm_importance,
    compute_rf_importance,
    compute_shap_values,
)


@pytest.fixture()
def trained_rf():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "lag_1": rng.normal(20, 5, n),
        "lag_2": rng.normal(20, 5, n),
        "rolling_mean_7": rng.normal(20, 3, n),
        "sin_day": np.sin(np.linspace(0, 4 * np.pi, n)),
        "noise": rng.normal(0, 1, n),
    })
    y = 0.6 * X["lag_1"] + 0.3 * X["rolling_mean_7"] + rng.normal(0, 0.5, n)
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestRFImportance:
    def test_returns_series(self, trained_rf):
        model, X, _ = trained_rf
        imp = compute_rf_importance(model, list(X.columns))
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(X.columns)

    def test_sorted_descending(self, trained_rf):
        model, X, _ = trained_rf
        imp = compute_rf_importance(model, list(X.columns))
        assert list(imp.values) == sorted(imp.values, reverse=True)

    def test_values_non_negative(self, trained_rf):
        model, X, _ = trained_rf
        imp = compute_rf_importance(model, list(X.columns))
        assert (imp >= 0).all()

    def test_sum_to_one(self, trained_rf):
        model, X, _ = trained_rf
        imp = compute_rf_importance(model, list(X.columns))
        assert abs(imp.sum() - 1.0) < 1e-6

    def test_lag1_most_important(self, trained_rf):
        model, X, _ = trained_rf
        imp = compute_rf_importance(model, list(X.columns))
        assert imp.index[0] == "lag_1"


class TestPermutationImportance:
    def test_returns_series(self, trained_rf):
        model, X, y = trained_rf
        imp = compute_perm_importance(model, X, y, list(X.columns), n_repeats=5)
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(X.columns)

    def test_sorted_descending(self, trained_rf):
        model, X, y = trained_rf
        imp = compute_perm_importance(model, X, y, list(X.columns), n_repeats=5)
        assert list(imp.values) == sorted(imp.values, reverse=True)

    def test_important_features_positive(self, trained_rf):
        model, X, y = trained_rf
        imp = compute_perm_importance(model, X, y, list(X.columns), n_repeats=5)
        assert imp["lag_1"] > 0


class TestCorrelationImportance:
    def test_returns_series(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "target": [1.1, 2.1, 3.1, 4.1, 5.1],
        })
        imp = compute_correlation_importance(df, "target")
        assert isinstance(imp, pd.Series)
        assert "target" not in imp.index

    def test_values_between_0_and_1(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 100),
            "y": rng.normal(0, 1, 100),
            "target": rng.normal(0, 1, 100),
        })
        imp = compute_correlation_importance(df, "target")
        assert (imp >= 0).all()
        assert (imp <= 1).all()

    def test_missing_target_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
        imp = compute_correlation_importance(df, "nonexistent")
        assert len(imp) == 0

    def test_perfect_correlation_ranked_first(self):
        df = pd.DataFrame({
            "perfect": [1, 2, 3, 4, 5],
            "noise": [3, 1, 4, 1, 5],
            "target": [1, 2, 3, 4, 5],
        })
        imp = compute_correlation_importance(df, "target")
        assert imp.index[0] == "perfect"


@pytest.mark.skipif(not HAS_SHAP, reason="shap not installed")
class TestSHAPValues:
    def test_returns_three_items(self, trained_rf):
        model, X, _ = trained_rf
        mean_abs, shap_vals, sample = compute_shap_values(model, X, max_samples=50)
        assert isinstance(mean_abs, pd.Series)
        assert isinstance(shap_vals, np.ndarray)
        assert isinstance(sample, pd.DataFrame)

    def test_shap_shape_matches_sample(self, trained_rf):
        model, X, _ = trained_rf
        _, shap_vals, sample = compute_shap_values(model, X, max_samples=50)
        assert shap_vals.shape == (len(sample), len(X.columns))

    def test_max_samples_respected(self, trained_rf):
        model, X, _ = trained_rf
        _, _, sample = compute_shap_values(model, X, max_samples=30)
        assert len(sample) == 30

    def test_sorted_descending(self, trained_rf):
        model, X, _ = trained_rf
        mean_abs, _, _ = compute_shap_values(model, X, max_samples=50)
        assert list(mean_abs.values) == sorted(mean_abs.values, reverse=True)
