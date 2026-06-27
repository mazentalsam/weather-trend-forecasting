"""Tests for ensemble methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.ensemble import build_ensemble


@pytest.fixture
def mock_forecasts():
    idx = pd.date_range("2024-04-01", periods=20, freq="D")
    actual = pd.Series(np.sin(np.arange(20) * 0.3) * 5 + 20, index=idx)
    forecasts = {
        "ModelA": actual + np.random.default_rng(1).normal(0, 0.5, 20),
        "ModelB": actual + np.random.default_rng(2).normal(0, 1.0, 20),
        "ModelC": actual + np.random.default_rng(3).normal(0, 0.8, 20),
    }
    for k in forecasts:
        forecasts[k] = pd.Series(forecasts[k], index=idx)
    return forecasts, actual


class TestBuildEnsemble:
    def test_returns_three_outputs(self, mock_forecasts):
        forecasts, actual = mock_forecasts
        simple, weighted, weights = build_ensemble(forecasts, actual)
        assert isinstance(simple, pd.Series)
        assert isinstance(weighted, pd.Series)
        assert isinstance(weights, dict)

    def test_weights_sum_to_one(self, mock_forecasts):
        forecasts, actual = mock_forecasts
        _, _, weights = build_ensemble(forecasts, actual)
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_all_models_have_weights(self, mock_forecasts):
        forecasts, actual = mock_forecasts
        _, _, weights = build_ensemble(forecasts, actual)
        assert set(weights.keys()) == set(forecasts.keys())

    def test_better_model_gets_higher_weight(self, mock_forecasts):
        forecasts, actual = mock_forecasts
        _, _, weights = build_ensemble(forecasts, actual)
        assert weights["ModelA"] > weights["ModelB"]

    def test_ensemble_length_matches(self, mock_forecasts):
        forecasts, actual = mock_forecasts
        simple, weighted, _ = build_ensemble(forecasts, actual)
        assert len(simple) == len(actual)
        assert len(weighted) == len(actual)

    def test_simple_is_mean_of_forecasts(self, mock_forecasts):
        forecasts, actual = mock_forecasts
        simple, _, _ = build_ensemble(forecasts, actual)
        expected = sum(v for v in forecasts.values()) / len(forecasts)
        pd.testing.assert_series_equal(simple, expected, check_names=False)

    def test_mismatched_indices(self):
        rng = np.random.default_rng(42)
        idx_a = pd.date_range("2024-04-01", periods=20, freq="D")
        idx_b = pd.date_range("2024-04-05", periods=20, freq="D")
        actual = pd.Series(rng.normal(20, 2, 24), index=pd.date_range("2024-04-01", periods=24, freq="D"))
        forecasts = {
            "A": pd.Series(rng.normal(20, 2, 20), index=idx_a),
            "B": pd.Series(rng.normal(20, 2, 20), index=idx_b),
        }
        simple, weighted, _ = build_ensemble(forecasts, actual)
        assert len(simple) == 16
