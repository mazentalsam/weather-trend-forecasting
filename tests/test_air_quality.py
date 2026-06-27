"""Tests for air quality analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.analysis.air_quality import compute_aq_weather_correlation


class TestAQWeatherCorrelation:
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "aq_pm25": np.random.randn(50),
            "aq_co": np.random.randn(50),
            "temperature": np.random.randn(50),
            "humidity": np.random.randn(50),
        })
        corr = compute_aq_weather_correlation(
            df, ["aq_pm25", "aq_co"], ["temperature", "humidity"],
        )
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (2, 2)

    def test_values_in_range(self):
        n = 100
        df = pd.DataFrame({
            "aq": np.random.randn(n),
            "temp": np.random.randn(n),
        })
        corr = compute_aq_weather_correlation(df, ["aq"], ["temp"])
        assert (corr.abs() <= 1.0).all().all()

    def test_perfect_correlation(self):
        df = pd.DataFrame({
            "aq": np.arange(50, dtype=float),
            "temp": np.arange(50, dtype=float),
        })
        corr = compute_aq_weather_correlation(df, ["aq"], ["temp"])
        assert abs(corr.iloc[0, 0] - 1.0) < 1e-10

    def test_missing_column_handled(self):
        df = pd.DataFrame({"aq": [1, 2, 3], "temp": [4, 5, 6]})
        corr = compute_aq_weather_correlation(df, ["aq", "nonexistent"], ["temp"])
        assert "nonexistent" not in corr.index

    def test_symmetric_submatrix(self):
        n = 100
        data = np.random.default_rng(42).normal(size=(n, 3))
        df = pd.DataFrame(data, columns=["aq1", "weather1", "weather2"])
        corr = compute_aq_weather_correlation(df, ["aq1"], ["weather1", "weather2"])
        assert corr.shape == (1, 2)
