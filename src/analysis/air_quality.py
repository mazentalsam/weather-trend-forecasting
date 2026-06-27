"""Air quality vs weather correlation analysis."""

from __future__ import annotations

import pandas as pd


def compute_aq_weather_correlation(
    df: pd.DataFrame,
    aq_cols: list[str],
    weather_cols: list[str],
) -> pd.DataFrame:
    all_cols = [c for c in aq_cols + weather_cols if c in df.columns]
    corr = df[all_cols].corr()
    return corr.loc[
        [c for c in aq_cols if c in corr.index],
        [c for c in weather_cols if c in corr.columns],
    ]
