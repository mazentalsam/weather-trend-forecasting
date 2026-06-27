"""Tests for data cleaning module."""

import numpy as np
import pandas as pd
import pytest
from src.data.cleaning import (
    detect_outliers_iqr,
    impute_missing,
    normalize,
    parse_datetime_features,
    winsorize,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "temperature": [20.0, 22.5, np.nan, 25.0, 18.0, 30.0, 15.0, 28.0, np.nan, 21.0],
        "humidity": [60.0, 65.0, 70.0, np.nan, 55.0, 80.0, 50.0, 75.0, 68.0, np.nan],
        "country": ["US", "UK", None, "US", "UK", "US", None, "UK", "US", "UK"],
        "last_updated": pd.date_range("2024-01-01", periods=10, freq="D").astype(str),
    })


class TestImputeMissing:
    def test_no_missing_after_imputation(self, sample_df):
        result = impute_missing(sample_df)
        assert result.isna().sum().sum() == 0

    def test_numeric_uses_median(self, sample_df):
        result = impute_missing(sample_df)
        median_temp = sample_df["temperature"].median()
        originally_null = sample_df["temperature"].isna()
        assert all(result.loc[originally_null, "temperature"] == median_temp)

    def test_categorical_uses_mode(self, sample_df):
        result = impute_missing(sample_df)
        mode_country = sample_df["country"].mode().iloc[0]
        originally_null = sample_df["country"].isna()
        assert all(result.loc[originally_null, "country"] == mode_country)

    def test_does_not_modify_original(self, sample_df):
        original_nulls = sample_df.isna().sum().sum()
        impute_missing(sample_df)
        assert sample_df.isna().sum().sum() == original_nulls


class TestDetectOutliersIQR:
    def test_returns_mask_and_bounds(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])
        mask, lower, upper = detect_outliers_iqr(s)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool
        assert lower < upper

    def test_detects_extreme_values(self):
        s = pd.Series([1, 2, 3, 4, 5, 100])
        mask, _, _ = detect_outliers_iqr(s)
        assert mask.iloc[-1]

    def test_custom_multiplier(self):
        s = pd.Series([1, 2, 3, 4, 5, 10])
        mask_strict, _, _ = detect_outliers_iqr(s, multiplier=1.0)
        mask_loose, _, _ = detect_outliers_iqr(s, multiplier=3.0)
        assert mask_strict.sum() >= mask_loose.sum()


class TestWinsorize:
    def test_clips_values(self):
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5, 100, -50]})
        result = winsorize(df, ["val"], lower_pct=0.1, upper_pct=0.9)
        assert result["val"].max() <= df["val"].quantile(0.9)
        assert result["val"].min() >= df["val"].quantile(0.1)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"val": [1, 100]})
        winsorize(df, ["val"])
        assert df["val"].iloc[1] == 100


class TestNormalize:
    def test_output_shape(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        normed, scaler = normalize(df, ["a", "b"])
        assert normed.shape == (3, 2)
        assert list(normed.columns) == ["a_norm", "b_norm"]

    def test_zero_mean_unit_var(self):
        df = pd.DataFrame({"a": np.random.randn(100)})
        normed, _ = normalize(df, ["a"])
        assert abs(normed["a_norm"].mean()) < 0.01
        assert abs(normed["a_norm"].std() - 1.0) < 0.1


class TestParseDatetimeFeatures:
    def test_adds_temporal_columns(self, sample_df):
        result = parse_datetime_features(sample_df, "last_updated")
        assert "date" in result.columns
        assert "year" in result.columns
        assert "month" in result.columns
        assert "day_of_year" in result.columns
        assert "week" in result.columns
