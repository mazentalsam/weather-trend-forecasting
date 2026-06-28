"""Feature engineering for time-series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_lag_features(series: pd.Series, n_lags: int = 14) -> pd.DataFrame:
    df = pd.DataFrame({"target": series})

    day_labels = {
        1: "Yesterday", 2: "2 Days Ago", 3: "3 Days Ago",
    }
    for i in range(1, n_lags + 1):
        label = day_labels.get(i, f"{i} Days Ago")
        df[label] = series.shift(i)

    for w in (7, 14):
        df[f"Avg Temp (Last {w} Days)"] = series.rolling(w).mean()
        df[f"Temp Variability ({w}-Day)"] = series.rolling(w).std()

    df["Coldest in Last 7 Days"] = series.rolling(7).min()
    df["Hottest in Last 7 Days"] = series.rolling(7).max()

    df["Day of Year"] = series.index.dayofyear
    df["Month"] = series.index.month
    df["Day of Week"] = series.index.dayofweek
    df["Season (sin)"] = np.sin(2 * np.pi * series.index.dayofyear / 365.25)
    df["Season (cos)"] = np.cos(2 * np.pi * series.index.dayofyear / 365.25)

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
