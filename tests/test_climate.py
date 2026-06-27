"""Tests for climate analysis module."""

import pandas as pd
import pytest
from src.analysis.climate import add_climate_features, classify_climate_zone


class TestClassifyClimateZone:
    @pytest.mark.parametrize("lat,expected", [
        (0, "Tropical"),
        (10, "Tropical"),
        (-15, "Tropical"),
        (30, "Subtropical"),
        (-30, "Subtropical"),
        (45, "Temperate"),
        (-50, "Temperate"),
        (60, "Subarctic"),
        (-64, "Subarctic"),
        (70, "Polar"),
        (-85, "Polar"),
    ])
    def test_zone_classification(self, lat, expected):
        assert classify_climate_zone(lat) == expected


class TestAddClimateFeatures:
    def test_adds_columns(self):
        df = pd.DataFrame({"latitude": [10, -30, 50, 65, 80]})
        result = add_climate_features(df, "latitude")
        assert "climate_zone" in result.columns
        assert "hemisphere" in result.columns

    def test_hemisphere_assignment(self):
        df = pd.DataFrame({"latitude": [10, -10]})
        result = add_climate_features(df, "latitude")
        assert result.iloc[0]["hemisphere"] == "Northern"
        assert result.iloc[1]["hemisphere"] == "Southern"
