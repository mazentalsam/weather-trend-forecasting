"""Climate zone classification and hemisphere analysis."""

from __future__ import annotations

import pandas as pd


def classify_climate_zone(lat: float) -> str:
    abs_lat = abs(lat)
    if abs_lat <= 23.5:
        return "Tropical"
    elif abs_lat <= 35:
        return "Subtropical"
    elif abs_lat <= 55:
        return "Temperate"
    elif abs_lat <= 66.5:
        return "Subarctic"
    return "Polar"


def add_climate_features(df: pd.DataFrame, lat_col: str) -> pd.DataFrame:
    df = df.copy()
    df["climate_zone"] = df[lat_col].apply(classify_climate_zone)
    df["hemisphere"] = df[lat_col].apply(lambda x: "Northern" if x >= 0 else "Southern")
    return df
