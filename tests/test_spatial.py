"""Tests for spatial analysis module."""

from __future__ import annotations

import pandas as pd
import pytest
from src.analysis.spatial import _get_continent, map_continents


class TestGetContinent:
    @pytest.mark.parametrize("country,expected", [
        ("United States", "North America"),
        ("United Kingdom", "Europe"),
        ("Japan", "Asia"),
        ("Brazil", "South America"),
        ("Australia", "Oceania"),
        ("Nigeria", "Africa"),
    ])
    def test_known_countries(self, country, expected):
        assert _get_continent(country) == expected

    def test_unknown_country_returns_unknown(self):
        assert _get_continent("FakeCountryXYZ") == "Unknown"

    def test_empty_string(self):
        assert _get_continent("") == "Unknown"


class TestMapContinents:
    def test_adds_continent_column(self):
        df = pd.DataFrame({"country": ["United States", "Japan", "Brazil"]})
        result = map_continents(df, "country")
        assert "continent" in result.columns
        assert result["continent"].iloc[0] == "North America"

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"country": ["France"]})
        result = map_continents(df, "country")
        assert "continent" not in df.columns
        assert "continent" in result.columns

    def test_handles_unknown_countries(self):
        df = pd.DataFrame({"country": ["FakePlace", "United Kingdom"]})
        result = map_continents(df, "country")
        assert result["continent"].iloc[0] == "Unknown"
        assert result["continent"].iloc[1] == "Europe"
