"""Tests for statistical forecasting models."""

from __future__ import annotations

import pandas as pd
from src.models.statistical import fit_arima, fit_holt_winters, fit_prophet, fit_sarima


class TestFitArima:
    def test_returns_series_and_label(self, train_test_split):
        train, test = train_test_split
        pred, label = fit_arima(train, len(test), test.index)
        assert isinstance(pred, pd.Series)
        assert isinstance(label, str)
        assert "ARIMA" in label

    def test_forecast_length_matches(self, train_test_split):
        train, test = train_test_split
        pred, _ = fit_arima(train, len(test), test.index)
        assert len(pred) == len(test)

    def test_forecast_index_matches(self, train_test_split):
        train, test = train_test_split
        pred, _ = fit_arima(train, len(test), test.index)
        assert (pred.index == test.index).all()

    def test_config_passthrough(self, train_test_split):
        train, test = train_test_split
        cfg = {"max_p": 3, "max_q": 3, "max_d": 1}
        pred, _ = fit_arima(train, len(test), test.index, cfg)
        assert len(pred) == len(test)


class TestFitSarima:
    def test_returns_series_and_label(self, train_test_split):
        train, test = train_test_split
        pred, label = fit_sarima(train, len(test), test.index)
        assert isinstance(pred, pd.Series)
        assert "SARIMA" in label

    def test_forecast_length(self, train_test_split):
        train, test = train_test_split
        pred, _ = fit_sarima(train, len(test), test.index)
        assert len(pred) == len(test)


class TestFitHoltWinters:
    def test_returns_series(self, train_test_split):
        train, test = train_test_split
        pred, label = fit_holt_winters(train, len(test), test.index)
        assert isinstance(pred, pd.Series)
        assert len(pred) == len(test)

    def test_short_series_fallback(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        train = pd.Series(range(10), index=idx, dtype=float)
        future_idx = pd.date_range("2024-01-11", periods=3, freq="D")
        pred, label = fit_holt_winters(train, 3, future_idx, seasonal_period=7)
        assert "Holt" in label
        assert len(pred) == 3


class TestFitProphet:
    def test_returns_series(self, train_test_split):
        train, test = train_test_split
        pred, label = fit_prophet(train, len(test), test.index)
        assert isinstance(pred, pd.Series)
        assert label == "Prophet"
        assert len(pred) == len(test)

    def test_values_are_finite(self, train_test_split):
        train, test = train_test_split
        pred, _ = fit_prophet(train, len(test), test.index)
        assert pred.notna().all()
