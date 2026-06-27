"""Feature engineering for time-series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_lag_features(series: pd.Series, n_lags: int = 14) -> pd.DataFrame:
    df = pd.DataFrame({"target": series})

    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = series.shift(i)

    for w in (7, 14):
        df[f"rolling_mean_{w}"] = series.rolling(w).mean()
        df[f"rolling_std_{w}"] = series.rolling(w).std()

    df["rolling_min_7"] = series.rolling(7).min()
    df["rolling_max_7"] = series.rolling(7).max()

    df["day_of_year"] = series.index.dayofyear
    df["month"] = series.index.month
    df["day_of_week"] = series.index.dayofweek
    df["sin_day"] = np.sin(2 * np.pi * series.index.dayofyear / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * series.index.dayofyear / 365.25)

    return df.dropna()


def prepare_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    exclude_patterns: list[str] | None = None,
    min_non_null_frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series]:
    exclude_patterns = exclude_patterns or ["fahrenheit", "feels_like", "epoch"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c != target_col
        and c not in ("year", "month", "day_of_year", "week")
        and not any(p in c.lower() for p in exclude_patterns)
        and df[c].notna().sum() > len(df) * min_non_null_frac
    ]

    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
    return X, y
