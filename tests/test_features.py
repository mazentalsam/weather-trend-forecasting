"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest
from src.features.engineering import create_lag_features, prepare_feature_matrix


@pytest.fixture
def sample_series():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.Series(np.sin(np.arange(60) * 2 * np.pi / 30) * 10 + 20, index=idx, name="temp")


class TestCreateLagFeatures:
    def test_output_columns(self, sample_series):
        result = create_lag_features(sample_series, n_lags=7)
        assert "target" in result.columns
        assert "Yesterday" in result.columns
        assert "7 Days Ago" in result.columns
        assert "Avg Temp (Last 7 Days)" in result.columns
        assert "Season (sin)" in result.columns
        assert "Season (cos)" in result.columns

    def test_no_nans(self, sample_series):
        result = create_lag_features(sample_series, n_lags=7)
        assert result.isna().sum().sum() == 0

    def test_fewer_rows_than_input(self, sample_series):
        result = create_lag_features(sample_series, n_lags=14)
        assert len(result) < len(sample_series)

    def test_lag_values_correct(self, sample_series):
        result = create_lag_features(sample_series, n_lags=3)
        valid_idx = result.index[0]
        assert result.loc[valid_idx, "Yesterday"] == sample_series.shift(1).loc[valid_idx]


class TestPrepareFeatureMatrix:
    def test_excludes_target(self):
        df = pd.DataFrame({
            "temp": [1, 2, 3, 4, 5],
            "humidity": [60, 65, 70, 75, 80],
            "wind": [10, 12, 8, 15, 9],
        })
        X, y = prepare_feature_matrix(df, "temp")
        assert "temp" not in X.columns
        assert len(y) == len(X)

    def test_excludes_patterns(self):
        df = pd.DataFrame({
            "temp": [1, 2, 3],
            "feels_like_c": [1, 2, 3],
            "humidity": [60, 65, 70],
        })
        X, _ = prepare_feature_matrix(df, "temp")
        assert "feels_like_c" not in X.columns
