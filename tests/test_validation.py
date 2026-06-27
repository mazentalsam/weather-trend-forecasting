"""Tests for data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.data.validation import ValidationReport, validate_dataframe


class TestValidateDataframe:
    def test_clean_data_is_valid(self):
        df = pd.DataFrame({
            "temperature_celsius": [20.0, 22.0, 25.0],
            "humidity": [60.0, 65.0, 70.0],
            "last_updated": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        report = validate_dataframe(df)
        assert report.is_valid
        assert report.n_rows == 3

    def test_detects_high_missing_rate(self):
        df = pd.DataFrame({
            "temperature_celsius": [np.nan] * 8 + [20.0, 22.0],
            "last_updated": [f"2024-01-{i:02d}" for i in range(1, 11)],
        })
        report = validate_dataframe(df)
        assert not report.is_valid
        assert any("missing" in w.lower() for w in report.warnings)

    def test_detects_out_of_range(self):
        df = pd.DataFrame({
            "temperature_celsius": [20.0, 100.0, -200.0],
            "last_updated": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        report = validate_dataframe(df)
        assert "temperature_celsius" in report.out_of_range
        assert report.out_of_range["temperature_celsius"] == 2

    def test_detects_missing_temperature(self):
        df = pd.DataFrame({"humidity": [60.0], "wind": [10.0]})
        report = validate_dataframe(df)
        assert any("temperature" in w.lower() for w in report.warnings)

    def test_detects_duplicates(self):
        df = pd.DataFrame({
            "temperature_celsius": [20.0] * 20,
            "last_updated": ["2024-01-01"] * 20,
        })
        report = validate_dataframe(df)
        assert report.duplicate_rows > 0


class TestValidationReport:
    def test_summary_output(self):
        report = ValidationReport(
            n_rows=1000, n_columns=10,
            missing_pct={"col_a": 5.2},
            warnings=["test warning"],
        )
        text = report.summary()
        assert "1,000" in text
        assert "test warning" in text

    def test_is_valid_with_no_warnings(self):
        assert ValidationReport().is_valid

    def test_is_invalid_with_warnings(self):
        report = ValidationReport(warnings=["problem"])
        assert not report.is_valid
