"""Anomaly detection — Isolation Forest and Z-score methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    data = df[feature_cols].dropna()
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    labels = iso.fit_predict(data)
    result = pd.Series(False, index=df.index)
    result.loc[data.index] = labels == -1
    return result


def detect_anomalies_zscore(
    series: pd.Series, threshold: float = 3.0
) -> pd.Series:
    from scipy import stats
    z = np.abs(stats.zscore(series.dropna()))
    mask = pd.Series(False, index=series.index)
    mask.loc[series.dropna().index] = z > threshold
    return mask
