"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "temperature_celsius": np.sin(np.arange(n) * 2 * np.pi / 30) * 10 + 20,
        "precip_mm": np.random.exponential(2, n),
        "humidity": np.random.uniform(30, 90, n),
        "wind_kph": np.random.uniform(0, 40, n),
        "country": np.random.choice(["United States", "United Kingdom", "France", "Japan", "Brazil"], n),
        "location_name": np.random.choice(["CityA", "CityB", "CityC"], n),
        "latitude": np.random.uniform(-60, 60, n),
        "longitude": np.random.uniform(-180, 180, n),
        "last_updated": dates.astype(str),
        "air_quality_PM2.5": np.random.uniform(5, 80, n),
        "air_quality_CO": np.random.uniform(100, 500, n),
    })


@pytest.fixture
def daily_temperature_series() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    values = np.sin(np.arange(120) * 2 * np.pi / 30) * 10 + 20
    noise = np.random.default_rng(42).normal(0, 1, 120)
    return pd.Series(values + noise, index=idx, name="temperature")


@pytest.fixture
def train_test_split(daily_temperature_series):
    split = int(len(daily_temperature_series) * 0.8)
    train = daily_temperature_series[:split]
    test = daily_temperature_series[split:]
    return train, test


@pytest.fixture
def feature_matrix(daily_temperature_series):
    from src.features.engineering import create_lag_features
    df = create_lag_features(daily_temperature_series, n_lags=7)
    features = [c for c in df.columns if c != "target"]
    split = int(len(df) * 0.8)
    return (
        df.iloc[:split][features],
        df.iloc[:split]["target"],
        df.iloc[split:][features],
        df.iloc[split:]["target"],
        df.iloc[split:].index,
    )
