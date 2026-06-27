"""Tests for anomaly detection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.analysis.anomaly import detect_anomalies_isolation_forest, detect_anomalies_zscore


class TestIsolationForest:
    def test_returns_boolean_mask(self, sample_df):
        mask = detect_anomalies_isolation_forest(
            sample_df, ["temperature_celsius", "humidity"],
        )
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert len(mask) == len(sample_df)

    def test_contamination_controls_anomaly_rate(self, sample_df):
        features = ["temperature_celsius", "humidity"]
        mask_low = detect_anomalies_isolation_forest(sample_df, features, contamination=0.01)
        mask_high = detect_anomalies_isolation_forest(sample_df, features, contamination=0.15)
        assert mask_low.sum() <= mask_high.sum()

    def test_handles_missing_values(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 100],
            "b": [10, 20, 30, np.nan, 50, 60, 70, 80, 90, 1000],
        })
        mask = detect_anomalies_isolation_forest(df, ["a", "b"])
        assert len(mask) == len(df)

    def test_deterministic_with_seed(self, sample_df):
        features = ["temperature_celsius", "humidity"]
        mask1 = detect_anomalies_isolation_forest(sample_df, features, random_state=42)
        mask2 = detect_anomalies_isolation_forest(sample_df, features, random_state=42)
        pd.testing.assert_series_equal(mask1, mask2)


class TestZScore:
    def test_returns_boolean_mask(self):
        s = pd.Series(np.random.randn(100))
        mask = detect_anomalies_zscore(s)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool

    def test_detects_extreme_values(self):
        values = list(np.random.default_rng(42).normal(0, 1, 100))
        values.append(100.0)
        s = pd.Series(values)
        mask = detect_anomalies_zscore(s, threshold=3.0)
        assert mask.iloc[-1]

    def test_stricter_threshold_finds_fewer(self):
        s = pd.Series(np.random.default_rng(42).normal(0, 1, 500))
        mask_strict = detect_anomalies_zscore(s, threshold=4.0)
        mask_loose = detect_anomalies_zscore(s, threshold=2.0)
        assert mask_strict.sum() <= mask_loose.sum()

    def test_handles_nan(self):
        s = pd.Series([1, 2, np.nan, 4, 5, 100])
        mask = detect_anomalies_zscore(s)
        assert len(mask) == len(s)
