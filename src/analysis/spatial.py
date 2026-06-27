"""Spatial analysis — continent mapping via pycountry_convert."""

from __future__ import annotations

import pandas as pd
import pycountry_convert as pc


def _get_continent(country_name: str) -> str:
    try:
        alpha2 = pc.country_name_to_country_alpha2(country_name)
        code = pc.country_alpha2_to_continent_code(alpha2)
        return pc.convert_continent_code_to_continent_name(code)
    except (KeyError, Exception):
        return "Unknown"


def map_continents(df: pd.DataFrame, country_col: str) -> pd.DataFrame:
    df = df.copy()
    unique = df[country_col].unique()
    mapping = {c: _get_continent(c) for c in unique}
    df["continent"] = df[country_col].map(mapping)
    return df
