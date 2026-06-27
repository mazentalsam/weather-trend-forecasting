"""Tests for model evaluation module."""

import numpy as np
import pandas as pd
from src.models.evaluation import ForecastMetrics, compute_prediction_interval, evaluate_forecast


class TestEvaluateForecast:
    def test_perfect_prediction(self):
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        m = evaluate_forecast(actual, actual, "perfect")
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.r2 == 1.0

    def test_returns_forecast_metrics(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        pred = pd.Series([1.1, 2.2, 2.8])
        m = evaluate_forecast(actual, pred, "test_model")
        assert isinstance(m, ForecastMetrics)
        assert m.model == "test_model"
        assert m.mae > 0
        assert m.rmse > 0

    def test_to_dict(self):
        actual = pd.Series([1.0, 2.0, 3.0])
        pred = pd.Series([1.1, 2.2, 2.8])
        d = evaluate_forecast(actual, pred, "test").to_dict()
        assert "Model" in d
        assert "MAE" in d
        assert "RMSE" in d
        assert "MAPE (%)" in d
        assert "R²" in d


class TestComputePredictionInterval:
    def test_interval_contains_prediction(self):
        actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = pd.Series([1.1, 2.2, 2.8, 4.1, 4.9])
        lower, upper, std = compute_prediction_interval(actual, pred)
        assert (upper >= pred).all()
        assert (lower <= pred).all()
        assert std > 0

    def test_wider_at_lower_confidence(self):
        actual = pd.Series(np.random.randn(100))
        pred = pd.Series(np.random.randn(100))
        _, upper_99, _ = compute_prediction_interval(actual, pred, confidence=0.99)
        _, upper_90, _ = compute_prediction_interval(actual, pred, confidence=0.90)
        assert upper_99.mean() > upper_90.mean()
